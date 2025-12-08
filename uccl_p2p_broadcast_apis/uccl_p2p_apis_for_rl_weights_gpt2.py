# rl_training_rdma_gpt2.py
"""
RL Training with RDMA Weight Transfer using UCCL P2P
- Phase 1: PPO Training on training nodes
- Phase 2: RDMA weight transfer to inference nodes  
- Phase 3: Rollout generation on inference nodes
- Loop back to Phase 1
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import GPT2LMHeadModel, GPT2Config, GPT2Tokenizer
from uccl import p2p
import time
import math
from dataclasses import dataclass
from typing import List, Tuple

# ============ PARSE ARGUMENTS ============
parser = argparse.ArgumentParser(description='RL Training with RDMA Weight Transfer')
parser.add_argument('--num_training', type=int, default=8, help='Number of training GPUs')
parser.add_argument('--num_inference', type=int, default=8, help='Number of inference GPUs')
parser.add_argument('--gpus_per_node', type=int, default=8, help='GPUs per node')
parser.add_argument('--num_shards', type=int, default=8, help='Number of model shards')
parser.add_argument('--rl_iterations', type=int, default=5, help='Number of RL iterations')
parser.add_argument('--ppo_epochs', type=int, default=4, help='PPO epochs per iteration')
parser.add_argument('--rollouts_per_gpu', type=int, default=8, help='Rollouts per inference GPU')
parser.add_argument('--max_gen_length', type=int, default=64, help='Max generation length')
args = parser.parse_args()

# ============ CONFIGURATION ============
NUM_TRAINING_RANKS = args.num_training
NUM_INFERENCE_RANKS = args.num_inference
GPUS_PER_NODE = args.gpus_per_node
NUM_SHARDS = min(args.num_shards, NUM_TRAINING_RANKS, NUM_INFERENCE_RANKS)
RL_ITERATIONS = args.rl_iterations
PPO_EPOCHS = args.ppo_epochs
ROLLOUTS_PER_GPU = args.rollouts_per_gpu
MAX_GEN_LENGTH = args.max_gen_length

# PPO Hyperparameters
PPO_CLIP_EPSILON = 0.2
VALUE_LOSS_COEF = 0.5
ENTROPY_COEF = 0.01
LEARNING_RATE = 1e-5
MAX_GRAD_NORM = 1.0

rank = int(os.environ.get("RANK", 0))
local_rank = int(os.environ.get("LOCAL_RANK", 0))
world_size = int(os.environ.get("WORLD_SIZE", NUM_TRAINING_RANKS + NUM_INFERENCE_RANKS))

TRAIN_RANKS = list(range(0, NUM_TRAINING_RANKS))
INFERENCE_RANKS = list(range(NUM_TRAINING_RANKS, NUM_TRAINING_RANKS + NUM_INFERENCE_RANKS))

role = "training" if rank in TRAIN_RANKS else "inference"


def log(msg):
    print(f"[Rank {rank}] {msg}", flush=True)
    sys.stdout.flush()


# ============ VALUE HEAD ============
class ValueHead(nn.Module):
    """Value function head for PPO"""
    def __init__(self, hidden_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, 1)
    
    def forward(self, hidden_states):
        x = self.dense(hidden_states[:, -1, :])
        x = torch.tanh(x)
        return self.out(x).squeeze(-1)


# ============ REWARD FUNCTION ============
class RewardFunction:
    """Synthetic reward for RL training"""
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    def compute_reward(self, prompt: str, response: str) -> float:
        rewards = 0.0
        words = response.split()
        
        # Length reward
        if 10 < len(words) < 50:
            rewards += 0.3
        elif len(words) <= 10:
            rewards += 0.1 * len(words) / 10
        
        # No repetition reward
        if not self._has_repetition(response):
            rewards += 0.3
        
        # Ends with punctuation
        if response.strip().endswith(('.', '!', '?')):
            rewards += 0.2
        
        # Diversity
        if len(words) > 0:
            rewards += 0.2 * len(set(words)) / len(words)
        
        return rewards
    
    def _has_repetition(self, text: str, n: int = 3) -> bool:
        words = text.lower().split()
        if len(words) < n * 2:
            return False
        ngrams = [tuple(words[i:i+n]) for i in range(len(words) - n + 1)]
        return len(ngrams) != len(set(ngrams))


# ============ ROLLOUT STORAGE ============
@dataclass
class RolloutBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    actions: torch.Tensor
    old_log_probs: torch.Tensor
    rewards: torch.Tensor
    values: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor


# ============ ROLLOUT GENERATOR (FIXED) ============
class RolloutGenerator:
    """Generate rollouts for RL training"""
    
    def __init__(self, model, value_head, tokenizer, reward_fn, device):
        self.model = model
        self.value_head = value_head
        self.tokenizer = tokenizer
        self.reward_fn = reward_fn
        self.device = device
        
        self.prompts = [
            "The meaning of life is",
            "Artificial intelligence will",
            "The future of technology",
            "Science has shown that",
            "In the next decade",
            "The most important thing",
            "People often forget that",
            "History teaches us that",
            "The key to success is",
            "Innovation comes from",
            "The world needs more",
            "Understanding requires",
        ]
    
    @torch.no_grad()
    def generate_rollouts(self, num_rollouts: int, max_length: int = 64) -> RolloutBatch:
        self.model.eval()
        if self.value_head is not None:
            self.value_head.eval()
        
        all_input_ids = []
        all_attention_masks = []
        all_actions = []
        all_log_probs = []
        all_rewards = []
        all_values = []
        
        for i in range(num_rollouts):
            prompt = self.prompts[i % len(self.prompts)]
            
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
            input_ids = inputs["input_ids"].to(self.device)
            attention_mask = inputs["attention_mask"].to(self.device)
            prompt_len = input_ids.shape[1]
            
            if isinstance(self.model, DDP):
                model_to_use = self.model.module
            else:
                model_to_use = self.model
            
            # Generate tokens
            generated_ids = []  # List of ints
            log_probs_list = []  # List of scalar tensors
            current_ids = input_ids.clone()
            current_mask = attention_mask.clone()
            
            for _ in range(max_length - prompt_len):
                outputs = model_to_use(
                    input_ids=current_ids, 
                    attention_mask=current_mask, 
                    output_hidden_states=True
                )
                logits = outputs.logits[:, -1, :]
                
                # Sample next token
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                next_token_id = next_token.item()
                
                # Get log prob (FIXED: proper scalar extraction)
                log_prob = F.log_softmax(logits, dim=-1)
                token_log_prob = log_prob[0, next_token_id]
                
                # Store (FIXED: store as int, not tensor)
                generated_ids.append(next_token_id)
                log_probs_list.append(token_log_prob)
                
                # Update
                current_ids = torch.cat([current_ids, next_token], dim=1)
                current_mask = torch.cat([current_mask, torch.ones_like(next_token)], dim=1)
                
                if next_token_id == self.tokenizer.eos_token_id:
                    break
            
            # Compute value
            if self.value_head is not None and len(generated_ids) > 0:
                hidden = outputs.hidden_states[-1]
                value = self.value_head(hidden).squeeze()
            else:
                value = torch.tensor(0.0, device=self.device)
            
            # Compute reward
            full_text = self.tokenizer.decode(current_ids[0], skip_special_tokens=True)
            response = full_text[len(prompt):].strip()
            reward = self.reward_fn.compute_reward(prompt, response)
            
            # Create tensors (FIXED: proper tensor creation)
            if len(generated_ids) > 0:
                actions = torch.tensor(generated_ids, device=self.device, dtype=torch.long).unsqueeze(0)
                total_log_prob = torch.stack(log_probs_list).sum()
            else:
                actions = torch.zeros((1, 1), dtype=torch.long, device=self.device)
                total_log_prob = torch.tensor(0.0, device=self.device)
            
            all_input_ids.append(current_ids)
            all_attention_masks.append(current_mask)
            all_actions.append(actions)
            all_log_probs.append(total_log_prob)
            all_rewards.append(torch.tensor(reward, device=self.device, dtype=torch.float32))
            all_values.append(value.float() if isinstance(value, torch.Tensor) else torch.tensor(value, device=self.device, dtype=torch.float32))
        
        # Pad sequences
        max_len = max(ids.shape[1] for ids in all_input_ids)
        max_action_len = max(a.shape[1] for a in all_actions)
        
        padded_input_ids = []
        padded_attention_masks = []
        padded_actions = []
        
        for ids, mask, actions in zip(all_input_ids, all_attention_masks, all_actions):
            pad_len = max_len - ids.shape[1]
            action_pad_len = max_action_len - actions.shape[1]
            
            padded_input_ids.append(F.pad(ids, (0, pad_len), value=self.tokenizer.pad_token_id))
            padded_attention_masks.append(F.pad(mask, (0, pad_len), value=0))
            padded_actions.append(F.pad(actions, (0, action_pad_len), value=self.tokenizer.pad_token_id))
        
        # Stack
        input_ids = torch.cat(padded_input_ids, dim=0)
        attention_mask = torch.cat(padded_attention_masks, dim=0)
        actions = torch.cat(padded_actions, dim=0)
        old_log_probs = torch.stack(all_log_probs)
        rewards = torch.stack(all_rewards)
        values = torch.stack(all_values)
        
        # Advantages
        advantages = rewards - values
        if advantages.std() > 1e-8:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return RolloutBatch(
            input_ids=input_ids,
            attention_mask=attention_mask,
            actions=actions,
            old_log_probs=old_log_probs,
            rewards=rewards,
            values=values,
            advantages=advantages,
            returns=rewards
        )


# ============ PPO TRAINER ============
class PPOTrainer:
    def __init__(self, model, value_head, optimizer, device):
        self.model = model
        self.value_head = value_head
        self.optimizer = optimizer
        self.device = device
    
    def ppo_step(self, batch: RolloutBatch) -> dict:
        input_ids = batch.input_ids.to(self.device)
        attention_mask = batch.attention_mask.to(self.device)
        actions = batch.actions.to(self.device)
        old_log_probs = batch.old_log_probs.to(self.device)
        advantages = batch.advantages.to(self.device)
        returns = batch.returns.to(self.device)
        
        # Forward
        if isinstance(self.model, DDP):
            outputs = self.model.module(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        else:
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        
        logits = outputs.logits
        hidden_states = outputs.hidden_states[-1]
        
        # New log probs
        gen_len = actions.shape[1]
        if gen_len > 0 and logits.shape[1] > gen_len:
            action_logits = logits[:, -gen_len-1:-1, :]
            log_probs = F.log_softmax(action_logits, dim=-1)
            new_log_probs = torch.gather(log_probs, dim=-1, index=actions.unsqueeze(-1)).squeeze(-1)
            new_log_probs = new_log_probs.sum(dim=-1)
            entropy = -(F.softmax(action_logits, dim=-1) * log_probs).sum(dim=-1).mean()
        else:
            new_log_probs = old_log_probs
            entropy = torch.tensor(0.0, device=self.device)
        
        # Value
        values = self.value_head(hidden_states)
        
        # PPO loss
        ratio = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - PPO_CLIP_EPSILON, 1 + PPO_CLIP_EPSILON) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        value_loss = F.mse_loss(values, returns)
        loss = policy_loss + VALUE_LOSS_COEF * value_loss - ENTROPY_COEF * entropy
        
        self.optimizer.zero_grad()
        loss.backward()
        
        if isinstance(self.model, DDP):
            torch.nn.utils.clip_grad_norm_(self.model.module.parameters(), MAX_GRAD_NORM)
        else:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), MAX_GRAD_NORM)
        torch.nn.utils.clip_grad_norm_(self.value_head.parameters(), MAX_GRAD_NORM)
        
        self.optimizer.step()
        
        return {
            'loss': loss.item(),
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy.item() if isinstance(entropy, torch.Tensor) else entropy,
            'approx_kl': ((ratio - 1) - torch.log(ratio)).mean().item()
        }


# ============ SHARD PARAMETERS ============
def get_param_shards(model, num_shards):
    if isinstance(model, DDP):
        params = list(model.module.named_parameters())
    else:
        params = list(model.named_parameters())
    
    total_size = sum(p.numel() * p.element_size() for _, p in params)
    shard_size_target = total_size // num_shards
    
    shards = [[] for _ in range(num_shards)]
    shard_sizes = [0] * num_shards
    current_shard = 0
    current_size = 0
    
    for name, param in params:
        param_size = param.numel() * param.element_size()
        shards[current_shard].append((name, param))
        shard_sizes[current_shard] += param_size
        current_size += param_size
        
        if current_size >= shard_size_target and current_shard < num_shards - 1:
            current_shard += 1
            current_size = 0
    
    return shards, shard_sizes


# ============ MAIN ============

# Print topology
if rank == 0:
    log("")
    log("=" * 70)
    log("              RL TRAINING WITH RDMA WEIGHT TRANSFER")
    log("=" * 70)
    log(f"  Total GPUs:          {world_size}")
    log(f"  Training GPUs:       {NUM_TRAINING_RANKS} (Ranks 0-{NUM_TRAINING_RANKS-1})")
    log(f"  Inference GPUs:      {NUM_INFERENCE_RANKS} (Ranks {NUM_TRAINING_RANKS}-{NUM_TRAINING_RANKS+NUM_INFERENCE_RANKS-1})")
    log(f"  RL Iterations:       {RL_ITERATIONS}")
    log(f"  PPO Epochs:          {PPO_EPOCHS}")
    log("=" * 70)

# Step 1: GLOO Init
log("Step 1: GLOO init")
dist.init_process_group(backend="gloo")
torch.cuda.set_device(local_rank)

# Step 2: Load Model
log("Step 2: Load model")
config = GPT2Config.from_pretrained("gpt2")
model = GPT2LMHeadModel(config).to(f"cuda:{local_rank}")
value_head = ValueHead(config.n_embd).to(f"cuda:{local_rank}")

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

reward_fn = RewardFunction(tokenizer)

model_size_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
model_size_mb = model_size_bytes / 1e6

if rank == 0:
    log(f"Model size: {model_size_mb:.2f} MB")

# Step 3: DDP
log("Step 3: DDP setup")
TRAIN_GROUP = None

if rank in TRAIN_RANKS:
    TRAIN_GROUP = dist.new_group(ranks=TRAIN_RANKS, backend="nccl")
    model = DDP(model, device_ids=[local_rank], process_group=TRAIN_GROUP)
    log("DDP wrapped")
else:
    log("Inference rank - no DDP")

dist.barrier()

# Step 4: RDMA Setup
log("Step 4: RDMA setup")
ep = p2p.Endpoint(local_rank, 4)
local_metadata = ep.get_metadata()
ip, port, gpu = p2p.Endpoint.parse_metadata(local_metadata)
log(f"Endpoint: IP={ip}, Port={port}, GPU={gpu}")

connections = {}

if rank < NUM_SHARDS:
    for inf_rank in INFERENCE_RANKS:
        dist.send(torch.ByteTensor(list(local_metadata)), dst=inf_rank)
        remote_tensor = torch.zeros(len(local_metadata), dtype=torch.uint8)
        dist.recv(remote_tensor, src=inf_rank)
        remote_metadata = bytes(remote_tensor.tolist())
        
        r_ip, r_port, r_gpu = p2p.Endpoint.parse_metadata(remote_metadata)
        ok, conn_id = ep.connect(r_ip, r_gpu, remote_port=r_port)
        if ok:
            connections[inf_rank] = conn_id
    log(f"Connected to {len(connections)} inference ranks")

elif rank in INFERENCE_RANKS:
    for src_rank in range(NUM_SHARDS):
        remote_tensor = torch.zeros(len(local_metadata), dtype=torch.uint8)
        dist.recv(remote_tensor, src=src_rank)
        dist.send(torch.ByteTensor(list(local_metadata)), dst=src_rank)
        
        ok, c_ip, c_gpu, conn_id = ep.accept()
        if ok:
            connections[src_rank] = conn_id
    log(f"Accepted from {len(connections)} sender ranks")
else:
    log("No RDMA connections (idle during transfer)")

dist.barrier()

# Step 5: Initialize
optimizer = None
ppo_trainer = None
rollout_generator = None

if role == "training":
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(value_head.parameters()),
        lr=LEARNING_RATE
    )
    ppo_trainer = PPOTrainer(model, value_head, optimizer, f"cuda:{local_rank}")

if role == "inference":
    rollout_generator = RolloutGenerator(model, value_head, tokenizer, reward_fn, f"cuda:{local_rank}")

dist.barrier()

# Get shards
shards, shard_sizes = get_param_shards(model, num_shards=NUM_SHARDS)

# ============ PRINT ROUTING TABLE ============
if rank == 0:
    log("")
    log("=" * 70)
    log("                         ROUTING TABLE")
    log("=" * 70)
    log("")
    log("  Shard Distribution:")
    log("  " + "-" * 50)
    for i, size in enumerate(shard_sizes):
        log(f"    Shard {i}: {size/1e6:6.2f} MB (Sender: Rank {i})")
    log("  " + "-" * 50)
    log(f"    Total Model Size: {sum(shard_sizes)/1e6:.2f} MB")
    log("")
    log("  RDMA Connection Mapping (Training → Inference):")
    log("  " + "-" * 50)
    for sender_rank in range(NUM_SHARDS):
        receivers = [f"R{r}" for r in INFERENCE_RANKS]
        log(f"    Rank {sender_rank} [Shard {sender_rank}] ──► {', '.join(receivers)}")
    log("  " + "-" * 50)
    log(f"    Active Senders:    {NUM_SHARDS} (Ranks 0-{NUM_SHARDS-1})")
    log(f"    Idle Training:     {NUM_TRAINING_RANKS - NUM_SHARDS} (Ranks {NUM_SHARDS}-{NUM_TRAINING_RANKS-1})" if NUM_TRAINING_RANKS > NUM_SHARDS else "")
    log(f"    Receivers:         {NUM_INFERENCE_RANKS} (Ranks {INFERENCE_RANKS[0]}-{INFERENCE_RANKS[-1]})")
    log(f"    Total Connections: {NUM_SHARDS * NUM_INFERENCE_RANKS}")
    log("=" * 70)

# Metrics
total_training_time = 0
total_transfer_time = 0
final_reward = 0

# Per-iteration metrics storage
iter_metrics = []

# ============ MAIN RL LOOP ============
if rank == 0:
    log("")
    log("=" * 70)
    log("                    STARTING RL TRAINING LOOP")
    log("=" * 70)

for rl_iter in range(RL_ITERATIONS):
    iter_start_time = time.perf_counter()
    iter_training_time = 0
    iter_transfer_time = 0
    iter_inference_time = 0
    iter_reward = 0
    iter_aggregate_bw = 0
    
    if rank == 0:
        log("")
        log(f"{'='*70}")
        log(f"                    RL ITERATION {rl_iter + 1}/{RL_ITERATIONS}")
        log(f"{'='*70}")
    
    # PHASE 1: PPO Training
    if rank == 0:
        log("")
        log("PHASE 1: PPO TRAINING")
        log("-" * 50)
    
    if role == "training":
        model.train()
        value_head.train()
        
        rollout_gen = RolloutGenerator(model, value_head, tokenizer, reward_fn, f"cuda:{local_rank}")
        rollouts = rollout_gen.generate_rollouts(ROLLOUTS_PER_GPU, MAX_GEN_LENGTH)
        
        ppo_start = time.perf_counter()
        
        for ppo_epoch in range(PPO_EPOCHS):
            metrics = ppo_trainer.ppo_step(rollouts)
            
            if rank == 0:
                log(f"  PPO Epoch {ppo_epoch + 1}/{PPO_EPOCHS}: "
                    f"Loss={metrics['loss']:.4f}, Policy={metrics['policy_loss']:.4f}, "
                    f"Value={metrics['value_loss']:.4f}, KL={metrics['approx_kl']:.4f}")
        
        ppo_time = time.perf_counter() - ppo_start
        total_training_time += ppo_time
        iter_training_time = ppo_time
        final_reward = rollouts.rewards.mean().item()
        iter_reward = final_reward
        
        if rank == 0:
            log(f"  Time: {ppo_time:.2f}s | Reward: {final_reward:.4f}")
    else:
        log("Waiting for training...")
    
    dist.barrier()
    
    # PHASE 2: RDMA Transfer
    if rank == 0:
        log("")
        log("PHASE 2: RDMA WEIGHT TRANSFER")
        log("-" * 50)
    
    transfer_start = time.perf_counter()
    
    if rank < NUM_SHARDS:
        my_shard = shards[rank]
        
        registered_mrs = {}
        for name, param in my_shard:
            tensor = param.data.contiguous()
            ok, mr_id = ep.reg(tensor.data_ptr(), tensor.numel() * tensor.element_size())
            if ok:
                registered_mrs[name] = (mr_id, tensor.data_ptr(), tensor.numel() * tensor.element_size())
        
        torch.cuda.synchronize()
        dist.barrier()
        
        t0 = time.perf_counter()
        total_bytes = 0
        
        for inf_rank in INFERENCE_RANKS:
            conn_id = connections[inf_rank]
            for name, (mr_id, ptr, size) in registered_mrs.items():
                ok = ep.send(conn_id, mr_id, ptr, size)
                if ok:
                    total_bytes += size
        
        duration = time.perf_counter() - t0
        bw = (total_bytes / 1e9) / duration if duration > 0 else 0
        log(f"  Sent {total_bytes/1e6:.1f} MB | {duration:.4f}s | {bw:.2f} GB/s")
    
    elif rank in INFERENCE_RANKS:
        registered_mrs = {}
        for name, param in model.named_parameters():
            tensor = param.data.contiguous()
            ok, mr_id = ep.reg(tensor.data_ptr(), tensor.numel() * tensor.element_size())
            if ok:
                registered_mrs[name] = (mr_id, tensor.data_ptr(), tensor.numel() * tensor.element_size())
        
        torch.cuda.synchronize()
        dist.barrier()
        
        t0 = time.perf_counter()
        total_bytes = 0
        
        for src_rank in range(NUM_SHARDS):
            conn_id = connections[src_rank]
            src_shard = shards[src_rank]
            
            for name, _ in src_shard:
                if name in registered_mrs:
                    mr_id, ptr, size = registered_mrs[name]
                    ok = ep.recv(conn_id, mr_id, ptr, size)
                    if ok:
                        total_bytes += size
        
        duration = time.perf_counter() - t0
        bw = (total_bytes / 1e9) / duration if duration > 0 else 0
        log(f"  Received {total_bytes/1e6:.1f} MB | {duration:.4f}s | {bw:.2f} GB/s")
    else:
        torch.cuda.synchronize()
        dist.barrier()
    
    dist.barrier()
    transfer_time = time.perf_counter() - transfer_start
    total_transfer_time += transfer_time
    iter_transfer_time = transfer_time
    
    if rank == 0:
        aggregate_bw = (model_size_mb * NUM_INFERENCE_RANKS / 1e3) / transfer_time if transfer_time > 0 else 0
        iter_aggregate_bw = aggregate_bw
        log(f"  Total: {transfer_time:.4f}s | Aggregate: {aggregate_bw:.2f} GB/s")
    
    # PHASE 3: Inference
    if rank == 0:
        log("")
        log("PHASE 3: INFERENCE")
        log("-" * 50)
    
    if role == "inference":
        model.eval()
        
        inference_start = time.perf_counter()
        rollouts = rollout_generator.generate_rollouts(ROLLOUTS_PER_GPU, MAX_GEN_LENGTH)
        inference_time = time.perf_counter() - inference_start
        iter_inference_time = inference_time
        
        avg_reward = rollouts.rewards.mean().item()
        
        if rank == INFERENCE_RANKS[0]:
            log(f"  Rollouts: {ROLLOUTS_PER_GPU} | Time: {inference_time:.2f}s | Reward: {avg_reward:.4f}")
            sample = tokenizer.decode(rollouts.input_ids[0], skip_special_tokens=True)
            log(f"  Sample: \"{sample[:80]}...\"")
    else:
        log("Waiting for inference...")
    
    dist.barrier()
    
    # Per-iteration summary
    iter_total_time = time.perf_counter() - iter_start_time
    
    if rank == 0:
        iter_metrics.append({
            'iteration': rl_iter + 1,
            'training_time': iter_training_time,
            'transfer_time': iter_transfer_time,
            'aggregate_bw': iter_aggregate_bw,
            'reward': iter_reward,
            'total_time': iter_total_time
        })
        
        log("")
        log(f"  ┌{'─'*50}┐")
        log(f"  │ ITERATION {rl_iter + 1} SUMMARY{' '*32}│")
        log(f"  ├{'─'*50}┤")
        log(f"  │  Training Time:      {iter_training_time:>8.2f}s{' '*17}│")
        log(f"  │  Transfer Time:      {iter_transfer_time:>8.4f}s{' '*17}│")
        log(f"  │  Aggregate BW:       {iter_aggregate_bw:>8.2f} GB/s{' '*13}│")
        log(f"  │  Avg Reward:         {iter_reward:>8.4f}{' '*19}│")
        log(f"  │  Total Iter Time:    {iter_total_time:>8.2f}s{' '*17}│")
        log(f"  └{'─'*50}┘")

# Final Summary
if rank == 0:
    log("")
    log("=" * 70)
    log("                        FINAL SUMMARY")
    log("=" * 70)
    
    # Per-iteration metrics table
    log("")
    log("  PER-ITERATION METRICS:")
    log("  " + "─" * 66)
    log(f"  │ {'Iter':^4} │ {'Training':^10} │ {'Transfer':^10} │ {'Agg BW':^10} │ {'Reward':^8} │")
    log(f"  │ {'':^4} │ {'(s)':^10} │ {'(s)':^10} │ {'(GB/s)':^10} │ {'':^8} │")
    log("  " + "─" * 66)
    
    for m in iter_metrics:
        log(f"  │ {m['iteration']:^4} │ {m['training_time']:^10.2f} │ {m['transfer_time']:^10.4f} │ {m['aggregate_bw']:^10.2f} │ {m['reward']:^8.4f} │")
    
    log("  " + "─" * 66)
    
    # Averages
    avg_training = sum(m['training_time'] for m in iter_metrics) / len(iter_metrics)
    avg_transfer = sum(m['transfer_time'] for m in iter_metrics) / len(iter_metrics)
    avg_bw = sum(m['aggregate_bw'] for m in iter_metrics) / len(iter_metrics)
    avg_reward = sum(m['reward'] for m in iter_metrics) / len(iter_metrics)
    
    log(f"  │ {'AVG':^4} │ {avg_training:^10.2f} │ {avg_transfer:^10.4f} │ {avg_bw:^10.2f} │ {avg_reward:^8.4f} │")
    log("  " + "─" * 66)
    
    log("")
    log("  TOTALS:")
    log("  " + "-" * 50)
    log(f"    RL Iterations:       {RL_ITERATIONS}")
    log(f"    Total Training:      {total_training_time:.2f}s")
    log(f"    Total Transfer:      {total_transfer_time:.4f}s")
    log(f"    Final Reward:        {final_reward:.4f}")
    log(f"    Transfer Overhead:   {100 * total_transfer_time / (total_training_time + total_transfer_time + 0.001):.2f}%")
    log("  " + "-" * 50)
    log("")
    log("  CONFIGURATION:")
    log("  " + "-" * 50)
    log(f"    Training GPUs:       {NUM_TRAINING_RANKS}")
    log(f"    Inference GPUs:      {NUM_INFERENCE_RANKS}")
    log(f"    Active Senders:      {NUM_SHARDS}")
    log(f"    PPO Epochs/Iter:     {PPO_EPOCHS}")
    log(f"    Rollouts/GPU:        {ROLLOUTS_PER_GPU}")
    log("  " + "-" * 50)
    log("=" * 70)

dist.destroy_process_group()