# main_rdma_rl_qwen_fixed.py
import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader, DistributedSampler
from uccl import p2p
import time
import math

# ============ PARSE ARGUMENTS ============
parser = argparse.ArgumentParser(description='RL Training with Qwen 2.5 + RDMA Transfer')
parser.add_argument('--model_name', type=str, default='Qwen/Qwen2.5-0.5B', 
                    choices=['Qwen/Qwen2.5-0.5B', 'Qwen/Qwen2.5-1.5B', 'Qwen/Qwen2.5-3B', 'Qwen/Qwen2.5-7B'],
                    help='Qwen 2.5 model variant')
parser.add_argument('--num_training', type=int, default=24, help='Number of training GPUs')
parser.add_argument('--num_inference', type=int, default=8, help='Number of inference GPUs')
parser.add_argument('--gpus_per_node', type=int, default=8, help='GPUs per node')
parser.add_argument('--num_shards', type=int, default=8, help='Number of model shards')
parser.add_argument('--epochs', type=int, default=2, help='Training epochs')
parser.add_argument('--ppo_epochs', type=int, default=4, help='PPO update epochs per batch')
parser.add_argument('--batch_size', type=int, default=4, help='Batch size per GPU')
parser.add_argument('--max_length', type=int, default=512, help='Max sequence length')
parser.add_argument('--lr', type=float, default=1e-6, help='Learning rate')
parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
parser.add_argument('--clip_epsilon', type=float, default=0.2, help='PPO clip epsilon')
parser.add_argument('--kl_coef', type=float, default=0.1, help='KL penalty coefficient')
parser.add_argument('--value_coef', type=float, default=0.5, help='Value loss coefficient')
parser.add_argument('--use_fp32', action='store_true', help='Use float32 instead of bfloat16')
args = parser.parse_args()

# ============ CONFIGURATION ============
NUM_TRAINING_RANKS = args.num_training
NUM_INFERENCE_RANKS = args.num_inference
GPUS_PER_NODE = args.gpus_per_node
NUM_SHARDS = min(args.num_shards, NUM_TRAINING_RANKS, NUM_INFERENCE_RANKS)

rank = int(os.environ.get("RANK", 0))
local_rank = int(os.environ.get("LOCAL_RANK", 0))
world_size = int(os.environ.get("WORLD_SIZE", NUM_TRAINING_RANKS + NUM_INFERENCE_RANKS))

TRAIN_RANKS = list(range(0, NUM_TRAINING_RANKS))
INFERENCE_RANKS = list(range(NUM_TRAINING_RANKS, NUM_TRAINING_RANKS + NUM_INFERENCE_RANKS))

role = "training" if rank in TRAIN_RANKS else "inference"

# Set dtype
DTYPE = torch.float32 if args.use_fp32 else torch.bfloat16

def log(msg):
    print(f"[Rank {rank}] {msg}", flush=True)
    sys.stdout.flush()

# ============ VALUE HEAD (FIXED DTYPE) ============
class ValueHead(nn.Module):
    """Value head for PPO with correct dtype"""
    def __init__(self, hidden_size, dtype=torch.bfloat16):
        super().__init__()
        self.dtype = dtype
        self.linear1 = nn.Linear(hidden_size, hidden_size // 2)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size // 2, 1)
    
    def forward(self, hidden_states):
        # Cast input to match weights dtype
        x = hidden_states.to(self.dtype)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x.squeeze(-1)

# ============ POLICY MODEL WITH VALUE HEAD (FIXED) ============
class PolicyWithValueHead(nn.Module):
    """Qwen model with value head for PPO - Fixed for DDP and dtype"""
    def __init__(self, model_name, device, dtype=torch.bfloat16):
        super().__init__()
        self.dtype = dtype
        self.device = device
        
        # Load policy model
        self.policy = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            trust_remote_code=True
        ).to(device)
        
        # Create value head with matching dtype
        hidden_size = self.policy.config.hidden_size
        self.value_head = ValueHead(hidden_size, dtype=dtype).to(device).to(dtype)
        
    def forward(self, input_ids, attention_mask=None, return_value=False):
        outputs = self.policy(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        
        logits = outputs.logits
        
        if return_value:
            hidden_states = outputs.hidden_states[-1]
            values = self.value_head(hidden_states)
            return logits, values
        
        return logits
    
    def generate(self, *args, **kwargs):
        return self.policy.generate(*args, **kwargs)

# ============ PPO TRAINER (FIXED DTYPE) ============
class PPOTrainer:
    """PPO Trainer for RL with proper dtype handling"""
    def __init__(self, model, ref_model, tokenizer, optimizer, args, device, dtype=torch.bfloat16):
        self.model = model
        self.ref_model = ref_model
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.args = args
        self.device = device
        self.dtype = dtype
        
    def compute_rewards(self, response_ids, attention_mask):
        """Compute rewards - simple length-based reward for demo"""
        batch_size = response_ids.shape[0]
        rewards = torch.zeros(batch_size, device=self.device, dtype=self.dtype)
        
        for i in range(batch_size):
            response_len = attention_mask[i].sum().item()
            target_len = 100
            rewards[i] = -abs(response_len - target_len) / target_len + 1.0
        
        return rewards
    
    def compute_log_probs(self, logits, labels, attention_mask):
        """Compute log probabilities of actions"""
        # Ensure same dtype
        logits = logits.to(self.dtype)
        
        log_probs = torch.log_softmax(logits, dim=-1)
        
        shift_log_probs = log_probs[:, :-1, :]
        shift_labels = labels[:, 1:]
        shift_mask = attention_mask[:, 1:].to(self.dtype)
        
        token_log_probs = torch.gather(
            shift_log_probs, 
            dim=-1, 
            index=shift_labels.unsqueeze(-1)
        ).squeeze(-1)
        
        masked_log_probs = token_log_probs * shift_mask
        sequence_log_probs = masked_log_probs.sum(dim=-1) / shift_mask.sum(dim=-1).clamp(min=1)
        
        return sequence_log_probs
    
    def compute_kl_divergence(self, logits, ref_logits, attention_mask):
        """Compute KL divergence between policy and reference"""
        logits = logits.to(self.dtype)
        ref_logits = ref_logits.to(self.dtype)
        attention_mask = attention_mask.to(self.dtype)
        
        log_probs = torch.log_softmax(logits, dim=-1)
        ref_log_probs = torch.log_softmax(ref_logits, dim=-1)
        
        # Numerical stability
        probs = torch.exp(log_probs).clamp(min=1e-8)
        kl = (probs * (log_probs - ref_log_probs)).sum(dim=-1)
        kl = (kl * attention_mask).sum(dim=-1) / attention_mask.sum(dim=-1).clamp(min=1)
        
        return kl.mean()
    
    def compute_advantages(self, rewards, values, gamma=0.99, lam=0.95):
        """Compute GAE advantages"""
        rewards = rewards.to(self.dtype)
        values = values.to(self.dtype)
        
        advantages = torch.zeros_like(rewards)
        last_gae = torch.tensor(0.0, dtype=self.dtype, device=self.device)
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = torch.tensor(0.0, dtype=self.dtype, device=self.device)
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + gamma * next_value - values[t]
            last_gae = delta + gamma * lam * last_gae
            advantages[t] = last_gae
        
        returns = advantages + values
        return advantages, returns
    
    def ppo_step(self, batch):
        """Single PPO update step with proper dtype handling"""
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        
        # Forward pass
        logits, values = self.model(input_ids, attention_mask, return_value=True)
        logits = logits.to(self.dtype)
        values = values.to(self.dtype)
        
        # Reference model forward
        with torch.no_grad():
            ref_outputs = self.ref_model(input_ids, attention_mask=attention_mask)
            ref_logits = ref_outputs.logits.to(self.dtype)
        
        # Compute log probs
        log_probs = self.compute_log_probs(logits, input_ids, attention_mask)
        
        with torch.no_grad():
            old_log_probs = self.compute_log_probs(ref_logits, input_ids, attention_mask)
        
        # Compute rewards
        rewards = self.compute_rewards(input_ids, attention_mask)
        
        # Compute advantages
        with torch.no_grad():
            values_for_gae = values.mean(dim=-1).detach()
            advantages, returns = self.compute_advantages(rewards, values_for_gae)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO loss
        ratio = torch.exp(log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.args.clip_epsilon, 1 + self.args.clip_epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Value loss - ensure same dtype
        value_pred = values.mean(dim=-1)
        value_loss = nn.functional.mse_loss(value_pred, returns.to(value_pred.dtype))
        
        # KL penalty
        kl_loss = self.compute_kl_divergence(logits, ref_logits, attention_mask)
        
        # Total loss
        total_loss = policy_loss + self.args.value_coef * value_loss + self.args.kl_coef * kl_loss
        
        return total_loss, {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'kl_loss': kl_loss.item(),
            'reward_mean': rewards.float().mean().item()
        }

# ============ PRINT TOPOLOGY ============
if rank == 0:
    log("")
    log("=" * 70)
    log("              RL TRAINING WITH QWEN 2.5 + RDMA")
    log("=" * 70)
    log(f"  Model:               {args.model_name}")
    log(f"  Total GPUs:          {world_size}")
    log(f"  Training GPUs:       {NUM_TRAINING_RANKS} (Ranks 0-{NUM_TRAINING_RANKS-1})")
    log(f"  Inference GPUs:      {NUM_INFERENCE_RANKS} (Ranks {NUM_TRAINING_RANKS}-{NUM_TRAINING_RANKS+NUM_INFERENCE_RANKS-1})")
    log(f"  Model Shards:        {NUM_SHARDS}")
    log(f"  Dtype:               {DTYPE}")
    log(f"  PPO Epochs:          {args.ppo_epochs}")
    log(f"  Clip Epsilon:        {args.clip_epsilon}")
    log(f"  KL Coefficient:      {args.kl_coef}")
    log("=" * 70)
    log("")

# ============ STEP 1: GLOO INIT ============
log("Step 1: GLOO init")
dist.init_process_group(backend="gloo")
torch.cuda.set_device(local_rank)
device = f"cuda:{local_rank}"

# ============ STEP 2: LOAD MODEL ============
log(f"Step 2: Loading {args.model_name}")

tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

if role == "training":
    # Training: load policy with value head + reference model
    model = PolicyWithValueHead(args.model_name, device, dtype=DTYPE)
    
    # Reference model (frozen)
    ref_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=DTYPE,
        trust_remote_code=True
    ).to(device)
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False
    
    log("Policy + Reference models loaded")
else:
    # Inference: just load the model
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=DTYPE,
        trust_remote_code=True
    ).to(device)
    ref_model = None
    log("Inference model loaded")

# Calculate model size
model_size_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
model_size_mb = model_size_bytes / 1e6

if rank == 0:
    log(f"Model size: {model_size_mb:.2f} MB ({model_size_mb/1e3:.2f} GB)")

# ============ STEP 3: DDP FOR TRAINING ============
log("Step 3: DDP setup")
TRAIN_GROUP = None

if rank in TRAIN_RANKS:
    TRAIN_GROUP = dist.new_group(ranks=TRAIN_RANKS, backend="nccl")
    model = DDP(model, device_ids=[local_rank], process_group=TRAIN_GROUP, find_unused_parameters=True)
    log("DDP wrapped")

dist.barrier()

# ============ SHARD MODEL PARAMETERS ============
def get_param_shards(model, num_shards):
    """Split model parameters into shards"""
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

# ============ STEP 4: RDMA SETUP ============
log("Step 4: RDMA setup")
ep = p2p.Endpoint(local_rank, 4)
local_metadata = ep.get_metadata()
ip, port, gpu = p2p.Endpoint.parse_metadata(local_metadata)
log(f"Endpoint: IP={ip}, Port={port}, GPU={gpu}")

connections = {}

if rank < NUM_SHARDS:
    # Sender ranks: connect to all inference ranks
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
    # Inference ranks: accept from sender ranks
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

# ============ STEP 5: PREPARE DATA ============
dataloader = None
if role == "training":
    log("Step 5: Prepare RL dataset")
    
    dataset = load_dataset("Anthropic/hh-rlhf", split="train")
    dataset = dataset.select(range(min(1000, len(dataset))))
    
    def preprocess(examples):
        prompts = []
        for ex in examples["chosen"]:
            try:
                prompt = ex.split("Human:")[-1].split("Assistant:")[0].strip()
                if len(prompt) > 10:
                    prompts.append(prompt)
                else:
                    prompts.append("What is artificial intelligence?")
            except:
                prompts.append("What is artificial intelligence?")
        
        tokenized = tokenizer(
            prompts,
            truncation=True,
            max_length=args.max_length // 2,
            padding="max_length"
        )
        return tokenized
    
    tokenized = dataset.map(preprocess, batched=True, remove_columns=dataset.column_names)
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask"])
    
    sampler = DistributedSampler(tokenized, num_replicas=NUM_TRAINING_RANKS, rank=TRAIN_RANKS.index(rank))
    dataloader = DataLoader(tokenized, batch_size=args.batch_size, sampler=sampler)
    log(f"Dataset ready: {len(tokenized)} prompts")

dist.barrier()

# ============ STEP 6: RL TRAINING ============
total_training_time = 0
final_reward = 0
aggregate_bw = 0

if role == "training":
    if rank == 0:
        log("")
        log("=" * 70)
        log("                    PHASE 1: RL TRAINING (PPO)")
        log("=" * 70)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    
    if isinstance(model, DDP):
        ppo_trainer = PPOTrainer(model.module, ref_model, tokenizer, optimizer, args, device, dtype=DTYPE)
    else:
        ppo_trainer = PPOTrainer(model, ref_model, tokenizer, optimizer, args, device, dtype=DTYPE)
    
    model.train()
    training_start = time.perf_counter()
    
    global_step = 0
    epoch_rewards = []
    
    for epoch in range(args.epochs):
        epoch_start = time.perf_counter()
        dataloader.sampler.set_epoch(epoch)
        
        epoch_loss = 0
        epoch_policy_loss = 0
        epoch_value_loss = 0
        epoch_kl_loss = 0
        epoch_reward = 0
        epoch_steps = 0
        
        for batch in dataloader:
            for ppo_epoch in range(args.ppo_epochs):
                optimizer.zero_grad()
                
                try:
                    loss, metrics = ppo_trainer.ppo_step(batch)
                    
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                    epoch_policy_loss += metrics['policy_loss']
                    epoch_value_loss += metrics['value_loss']
                    epoch_kl_loss += metrics['kl_loss']
                    epoch_reward += metrics['reward_mean']
                    epoch_steps += 1
                except Exception as e:
                    if rank == 0:
                        log(f"Warning: PPO step failed: {e}")
                    continue
            
            global_step += 1
            
            if global_step % 10 == 0 and rank == 0:
                avg_reward = epoch_reward / max(epoch_steps, 1)
                avg_kl = epoch_kl_loss / max(epoch_steps, 1)
                log(f"Step {global_step}: Reward={avg_reward:.4f}, KL={avg_kl:.4f}")
        
        epoch_time = time.perf_counter() - epoch_start
        avg_loss = epoch_loss / max(epoch_steps, 1)
        avg_reward = epoch_reward / max(epoch_steps, 1)
        epoch_rewards.append(avg_reward)
        
        if rank == 0:
            log(f"Epoch {epoch+1}/{args.epochs}: "
                f"Loss={avg_loss:.4f}, Reward={avg_reward:.4f}, "
                f"Policy={epoch_policy_loss/max(epoch_steps,1):.4f}, "
                f"Value={epoch_value_loss/max(epoch_steps,1):.4f}, "
                f"KL={epoch_kl_loss/max(epoch_steps,1):.4f}, "
                f"Time={epoch_time:.2f}s")
    
    total_training_time = time.perf_counter() - training_start
    final_reward = epoch_rewards[-1] if epoch_rewards else 0
    
    if rank == 0:
        log("")
        log(f"RL Training Complete: {total_training_time:.2f}s")
        log(f"Final Reward: {final_reward:.4f}")
else:
    log("Waiting for RL training...")

dist.barrier()

# ============ STEP 7: RDMA WEIGHT TRANSFER ============
if rank == 0:
    log("")
    log("=" * 70)
    log("                PHASE 2: RDMA WEIGHT TRANSFER")
    log("=" * 70)

shards, shard_sizes = get_param_shards(model, num_shards=NUM_SHARDS)

if rank == 0:
    log("")
    log("Shard Distribution:")
    log("-" * 50)
    for i, size in enumerate(shard_sizes):
        log(f"  Shard {i}: {size/1e6:8.2f} MB")
    log("-" * 50)
    log(f"  Total:   {sum(shard_sizes)/1e6:.2f} MB")
    log("")

transfer_start = time.perf_counter()

if rank < NUM_SHARDS:
    # Sender
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
    
    log(f"UCCL Broadcast Complete. Time: {duration:.4f}s | BW: {bw:.2f} GB/s | "
        f"Sent: {total_bytes/1e6:.1f} MB to {NUM_INFERENCE_RANKS} receivers")

elif rank in INFERENCE_RANKS:
    # Receiver
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
    
    log(f"UCCL Receive Complete. Updated Model. Time: {duration:.4f}s | BW: {bw:.2f} GB/s | "
        f"Received: {total_bytes/1e6:.1f} MB from {NUM_SHARDS} senders")

else:
    # Idle ranks
    torch.cuda.synchronize()
    dist.barrier()

dist.barrier()
transfer_time = time.perf_counter() - transfer_start

if rank == 0:
    total_data_mb = model_size_mb * NUM_INFERENCE_RANKS
    aggregate_bw = (total_data_mb / 1e3) / transfer_time if transfer_time > 0 else 0
    
    log("")
    log("Transfer Summary:")
    log(f"  Wall Clock:    {transfer_time:.4f}s")
    log(f"  Aggregate BW:  {aggregate_bw:.2f} GB/s ({aggregate_bw*8:.2f} Gbps)")

# ============ STEP 8: INFERENCE ============
if role == "inference":
    if rank == INFERENCE_RANKS[0]:
        log("")
        log("=" * 70)
        log("                  PHASE 3: RL-TRAINED INFERENCE")
        log("=" * 70)
    
    model.eval()
    
    prompts = [
        "What is the meaning of life?",
        "How can I improve my productivity?",
        "Explain quantum computing in simple terms.",
        "What are the benefits of exercise?"
    ]
    
    inference_start = time.perf_counter()
    total_tokens = 0
    
    for i, prompt in enumerate(prompts):
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        input_len = input_ids.shape[1]
        
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                max_new_tokens=128,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id
            )
        
        tokens = outputs.shape[1] - input_len
        total_tokens += tokens
        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        if rank == INFERENCE_RANKS[0]:
            log(f"\nPrompt {i+1}: \"{prompt}\"")
            log(f"Response: {text}")
            log("-" * 50)
    
    inference_time = time.perf_counter() - inference_start
    
    if rank == INFERENCE_RANKS[0]:
        log(f"\nInference: {inference_time:.2f}s | {total_tokens/inference_time:.1f} tok/s")

dist.barrier()

# ============ FINAL SUMMARY ============
if rank == 0:
    log("")
    log("=" * 70)
    log("                        FINAL SUMMARY")
    log("=" * 70)
    log(f"  Model:           {args.model_name}")
    log(f"  Dtype:           {DTYPE}")
    log(f"  Training GPUs:   {NUM_TRAINING_RANKS}")
    log(f"  Inference GPUs:  {NUM_INFERENCE_RANKS}")
    log("")
    log(f"  RL Training:     {total_training_time:.2f}s")
    log(f"  Final Reward:    {final_reward:.4f}")
    log(f"  Transfer Time:   {transfer_time:.4f}s")
    log(f"  Transfer BW:     {aggregate_bw:.2f} GB/s ({aggregate_bw*8:.2f} Gbps)")
    log("=" * 70)

dist.destroy_process_group()