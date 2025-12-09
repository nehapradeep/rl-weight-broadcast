from __future__ import annotations
import os
import sys
import time
import math
import logging
import socket
import functools
import itertools
from datetime import datetime, timedelta
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers.models.gpt2.modeling_gpt2 import GPT2Block
from datasets import load_dataset

# FSDP Imports
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    StateDictType,
    FullStateDictConfig,
)
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
)

# -----------------------------------------------------------
# Configuration
# -----------------------------------------------------------
TRAIN_NODES = 2
GPUS_PER_NODE = 8
TRAIN_WORLD_SIZE = TRAIN_NODES * GPUS_PER_NODE
INFERENCE_MASTER_RANK = TRAIN_WORLD_SIZE 

# -----------------------------------------------------------
# Logging Setup
# -----------------------------------------------------------
def setup_logging(rank: int) -> str:
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"rank_{rank}_{timestamp}.log")

    logger = logging.getLogger()
    logger.setLevel(logging.INFO) 
    for h in list(logger.handlers):
        logger.removeHandler(h)

    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter(fmt="%(asctime)s | Rank %(rank)d | %(levelname)s | %(message)s", datefmt="%H:%M:%S")

    class RankFilter(logging.Filter):
        def filter(self, record):
            record.rank = rank
            return True

    fh.addFilter(RankFilter())
    ch.addFilter(RankFilter())
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return log_file

# -----------------------------------------------------------
# Dataset (WikiText-2 used as Prompts)
# -----------------------------------------------------------
def get_wikitext_dataset(tokenizer: GPT2Tokenizer, seq_len: int = 128):
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if local_rank != 0:
        import datasets
        datasets.logging.set_verbosity_error()
        
    logging.info("Loading WikiText-2 dataset...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

    def tokenize_function(examples):
        return tokenizer(examples["text"])

    tokenized_datasets = dataset.map(tokenize_function, batched=True, num_proc=4, remove_columns=["text"])
    all_input_ids = list(itertools.chain(*tokenized_datasets["input_ids"]))
    data_tensor = torch.tensor(all_input_ids, dtype=torch.long)
    
    class WikiTextDataset(torch.utils.data.Dataset):
        def __init__(self, data, seq_len):
            self.data = data
            self.seq_len = seq_len
            self.num_samples = (len(self.data) - 1) // self.seq_len
        def __len__(self): return self.num_samples
        def __getitem__(self, idx):
            start = idx * self.seq_len
            end = start + self.seq_len
            # In RL, we treat 'x' as the trajectory
            return self.data[start : end]

    logging.info(f"Dataset ready. Samples: {(len(data_tensor)-1)//seq_len}")
    return WikiTextDataset(data_tensor, seq_len)

# -----------------------------------------------------------
# Broadcast Helper
# -----------------------------------------------------------
def broadcast_model_to_inference(model, rank, bridge_group, is_sender=False):
    if bridge_group is None: return

    try:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        device = torch.device("cuda", local_rank)
    except: device = torch.device("cuda")

    if is_sender: logging.info("Broadcasting RL weights to Inference Node...")
    
    t0 = time.time()
    with torch.no_grad():
        for param in model.parameters():
            if param.device.type == "cpu":
                gpu_param = param.data.to(device)
                dist.broadcast(gpu_param, src=0, group=bridge_group)
            else:
                dist.broadcast(param.data, src=0, group=bridge_group)
    
    if is_sender:
        duration = time.time() - t0
        logging.info(f"Broadcast complete. Time taken: {duration:.2f}s")
    else:
        logging.info("RL Weights received from Training Cluster.")

# -----------------------------------------------------------
# Simple Reward Function
# -----------------------------------------------------------
def compute_reward(token_ids, tokenizer):
    """
    Simulated Reward Function.
    Goal: Encourage the usage of the word "the".
    Reward = +1.0 for every 'the' in the sequence.
    Reward = -0.1 baseline penalty to encourage density.
    """
    # Token ID for "the" (with space prefix usually) in GPT2 is 262
    target_id = 262 
    
    # Create reward tensor matching batch size
    # token_ids shape: [Batch, Seq]
    matches = (token_ids == target_id).float()
    
    # Sum matches per row to get reward per sample
    rewards = matches.sum(dim=1) 
    
    # Normalize rewards slightly to prevent explosion
    rewards = (rewards * 0.5) - 0.5
    
    return rewards

# -----------------------------------------------------------
# RL Trainer Loop (Policy Gradient)
# -----------------------------------------------------------
def run_trainer(rank, world_size, train_group, bridge_group):
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    logging.info("Initializing FSDP Actor Model...")
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    gpt2_auto_wrap_policy = functools.partial(transformer_auto_wrap_policy, transformer_layer_cls={GPT2Block})
    
    model = FSDP(model, auto_wrap_policy=gpt2_auto_wrap_policy, process_group=train_group, device_id=torch.cuda.current_device())
    
    dataset = get_wikitext_dataset(tokenizer, seq_len=128)
    sampler = DistributedSampler(dataset, num_replicas=dist.get_world_size(group=train_group), rank=dist.get_rank(group=train_group), shuffle=True)
    
    BATCH_SIZE = 4 
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5) # Lower LR for RL stability

    num_epochs = 2
    total_training_start = time.time()
    
    for epoch in range(1, num_epochs + 1):
        model.train()
        sampler.set_epoch(epoch)
        total_reward = 0.0
        
        epoch_start = time.time()
        
        for i, batch_ids in enumerate(dataloader):
            batch_ids = batch_ids.to(device)
            optimizer.zero_grad()
            
            # 1. Forward Pass (Get Logits)
            outputs = model(batch_ids, labels=batch_ids)
            logits = outputs.logits
            
            # 2. Calculate Log Probs of the sequence
            # Shift logits and labels for next-token prediction alignment
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = batch_ids[..., 1:].contiguous()
            
            # CrossEntropy is -log_prob. 
            # We want log_prob, so we take negative CrossEntropy (without reduction first)
            loss_fct = nn.CrossEntropyLoss(reduction='none')
            # Shape: [Batch, Seq_len]
            neg_log_probs = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            neg_log_probs = neg_log_probs.view(shift_labels.size())
            
            # log_probs = -neg_log_probs
            
            # 3. Calculate Reward (The RL Part)
            # We evaluate the batch_ids themselves as the "trajectory"
            rewards = compute_reward(batch_ids, tokenizer) # Shape: [Batch]
            
            # 4. Policy Gradient Loss
            # Loss = - (Log_Prob * Reward)
            # We need to broadcast reward to sequence length
            rewards_expanded = rewards.unsqueeze(1).expand_as(neg_log_probs)
            
            # Since neg_log_probs is POSITIVE (it's loss), minimizing (neg_log_probs * reward) 
            # is equivalent to Maximizing (log_probs * reward)
            # If reward is +5, we want to minimize loss (make log_probs higher).
            # If reward is -5, we want to maximize loss (make log_probs lower).
            rl_loss = (neg_log_probs * rewards_expanded).mean()
            
            rl_loss.backward()
            optimizer.step()
            
            total_reward += rewards.mean().item()
            
            if i % 20 == 0 and rank == 0:
                 logging.info(f"Batch {i}/{len(dataloader)} | RL Loss: {rl_loss.item():.4f} | Avg Reward: {rewards.mean().item():.2f}")

        epoch_duration = time.time() - epoch_start
        total_tokens_processed = len(dataloader) * BATCH_SIZE * 128 * dist.get_world_size(group=train_group)
        throughput = total_tokens_processed / epoch_duration

        if rank == 0:
            avg_reward = total_reward / len(dataloader)
            logging.info(f"--- RL EPOCH {epoch} STATS ---")
            logging.info(f"Duration: {epoch_duration:.2f} seconds")
            logging.info(f"Throughput: {throughput:.2f} tokens/sec")
            logging.info(f"Avg Reward: {avg_reward:.4f}")
            logging.info(f"-------------------------")
            
        # --- BRIDGE SYNC ---
        save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
            full_state = model.state_dict()

        if rank == 0:
            cpu_model = GPT2LMHeadModel.from_pretrained("gpt2")
            cpu_model.load_state_dict(full_state)
            broadcast_model_to_inference(cpu_model, rank, bridge_group, is_sender=True)
            del cpu_model

    total_training_time = time.time() - total_training_start
    if rank == 0:
        logging.info(f"Total Session Time: {total_training_time:.2f} seconds")

# -----------------------------------------------------------
# Inference Loop (With Better Sampling)
# -----------------------------------------------------------
def run_inference(rank, bridge_group):
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    logging.info("Initializing Inference Model...")
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    model.eval()
    
    num_epochs = 2 
    for epoch in range(1, num_epochs + 1):
        logging.info(f"Waiting for RL model update (Epoch {epoch})...")
        broadcast_model_to_inference(model, rank, bridge_group, is_sender=False)
        
        logging.info("Running inference test...")
        test_input = "The AI scientist discovered"
        inputs = tokenizer(test_input, return_tensors="pt").to(device)
        
        # IMPROVED GENERATION PARAMS
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=30,
                do_sample=True,        # Enable Sampling to fix loops
                temperature=0.8,       # Creativity
                repetition_penalty=1.2 # Stop "of of of of"
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logging.info(f"--- [INFERENCE RESULT EPOCH {epoch}] ---")
        logging.info(f"Output: {generated_text}")

# -----------------------------------------------------------
# Main
# -----------------------------------------------------------
def main():
    dist.init_process_group(backend="nccl", init_method="env://", timeout=timedelta(minutes=60))
    rank = dist.get_rank()
    log_file = setup_logging(rank)

    if rank == 0:
        logging.info(f"NCCL_SOCKET_IFNAME: {os.environ.get('NCCL_SOCKET_IFNAME', 'Not Set')}")

    train_ranks = list(range(0, TRAIN_WORLD_SIZE))
    train_group = dist.new_group(ranks=train_ranks)
    bridge_ranks = [0, INFERENCE_MASTER_RANK]
    bridge_group = dist.new_group(ranks=bridge_ranks)

    if rank in train_ranks:
        my_bridge = bridge_group if rank == 0 else None
        run_trainer(rank, TRAIN_WORLD_SIZE, train_group, my_bridge)
    elif rank == INFERENCE_MASTER_RANK:
        run_inference(rank, bridge_group)
    else:
        pass

    logging.info("Waiting for all ranks to complete...")
    dist.barrier()
    dist.destroy_process_group()

if __name__ == "__main__":
    main()