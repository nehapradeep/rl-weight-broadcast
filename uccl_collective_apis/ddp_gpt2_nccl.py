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

# --- DDP Import ---
from torch.nn.parallel import DistributedDataParallel as DDP

# --- SUPPRESS WARNINGS ---
import warnings
warnings.filterwarnings("ignore")
os.environ["TORCH_CPP_LOG_LEVEL"] = "ERROR"
os.environ["TORCH_DISTRIBUTED_DEBUG"] = "OFF"

# -----------------------------------------------------------
# Configuration
# -----------------------------------------------------------
TRAIN_NODES = 2
GPUS_PER_NODE = 8
TRAIN_WORLD_SIZE = TRAIN_NODES * GPUS_PER_NODE
INFERENCE_MASTER_RANK = TRAIN_WORLD_SIZE 

# -----------------------------------------------------------
# Logging Setup (Fixed Flushing)
# -----------------------------------------------------------
def setup_logging(rank: int) -> str:
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"rank_{rank}_{timestamp}.log")

    logger = logging.getLogger()
    logger.setLevel(logging.INFO) 
    
    # Clear existing handlers
    for h in list(logger.handlers):
        logger.removeHandler(h)

    # File Handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    
    # Console Handler with Force Flush
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    
    # Custom Formatter
    formatter = logging.Formatter(fmt="%(asctime)s | Rank %(rank)d | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    # Rank Filter
    class RankFilter(logging.Filter):
        def filter(self, record):
            record.rank = rank
            return True

    fh.addFilter(RankFilter())
    ch.addFilter(RankFilter())
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    return log_file

# -----------------------------------------------------------
# Dataset
# -----------------------------------------------------------
def get_wikitext_dataset(tokenizer: GPT2Tokenizer, seq_len: int = 128):
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    # Only log on rank 0 to avoid spam, unless error
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
            return self.data[start : end]

    logging.info(f"Dataset ready. Samples: {(len(data_tensor)-1)//seq_len}")
    return WikiTextDataset(data_tensor, seq_len)

# -----------------------------------------------------------
# Broadcast Helper (Standard PyTorch TCP/NCCL)
# -----------------------------------------------------------
def broadcast_model_to_inference(model, rank, bridge_group, is_sender=False):
    """
    Sends model weights using standard torch.distributed.broadcast.
    This works over whatever backend the bridge_group uses (likely Gloo or NCCL).
    """
    if bridge_group is None: return

    try:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        device = torch.device("cuda", local_rank)
    except: 
        device = torch.device("cuda")

    if is_sender: 
        logging.info("Broadcasting weights to Inference Node...")
    else:
        logging.info("Waiting for weights from Training Node...")
        # Force flush to ensure log appears before blocking operation
        sys.stdout.flush()

    t0 = time.time()
    
    # We iterate over parameters to ensure robust transfer
    with torch.no_grad():
        for param in model.parameters():
            # Ensure tensor is on GPU for NCCL broadcast (if using NCCL backend)
            # Or ensure consistent device for Gloo.
            if param.device.type == "cpu":
                gpu_param = param.data.to(device)
                dist.broadcast(gpu_param, src=0, group=bridge_group)
                # If receiver, move back to CPU if model requires it (rare)
            else:
                dist.broadcast(param.data, src=0, group=bridge_group)
    
    if is_sender:
        duration = time.time() - t0
        logging.info(f"Broadcast complete. Time: {duration:.2f}s")
    else:
        logging.info("Weights received.")
        sys.stdout.flush()

# -----------------------------------------------------------
# Reward Function
# -----------------------------------------------------------
def compute_reward(token_ids, tokenizer):
    target_id = 262 # "the"
    matches = (token_ids == target_id).float()
    rewards = matches.sum(dim=1) 
    rewards = (rewards * 0.5) - 0.5
    return rewards

# -----------------------------------------------------------
# Trainer Loop (DDP)
# -----------------------------------------------------------
def run_trainer(rank, world_size, train_group, bridge_group):
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    logging.info("Initializing DDP Actor Model...")
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    
    # --- DDP WRAPPER ---
    model = DDP(model, device_ids=[local_rank], output_device=local_rank, process_group=train_group)
    
    dataset = get_wikitext_dataset(tokenizer, seq_len=128)
    sampler = DistributedSampler(dataset, num_replicas=dist.get_world_size(group=train_group), rank=dist.get_rank(group=train_group), shuffle=True)
    
    BATCH_SIZE = 4 
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5) 

    num_epochs = 2
    
    for epoch in range(1, num_epochs + 1):
        model.train()
        sampler.set_epoch(epoch)
        epoch_start = time.time()
        total_reward = 0.0
        
        for i, batch_ids in enumerate(dataloader):
            # Debug limit
            # if i > 50: break 
            
            batch_ids = batch_ids.to(device)
            optimizer.zero_grad()
            
            outputs = model(batch_ids, labels=batch_ids)
            logits = outputs.logits
            
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = batch_ids[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss(reduction='none')
            neg_log_probs = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            neg_log_probs = neg_log_probs.view(shift_labels.size())
            
            rewards = compute_reward(batch_ids, tokenizer)
            rewards_expanded = rewards.unsqueeze(1).expand_as(neg_log_probs)
            rl_loss = (neg_log_probs * rewards_expanded).mean()
            
            rl_loss.backward()
            optimizer.step()
            
            total_reward += rewards.mean().item()
            
            if i % 20 == 0 and rank == 0:
                 logging.info(f"Batch {i}/{len(dataloader)} | RL Loss: {rl_loss.item():.4f} | Avg Reward: {rewards.mean().item():.2f}")

        epoch_duration = time.time() - epoch_start
        if rank == 0:
            avg_reward = total_reward / (i+1)
            logging.info(f"--- RL EPOCH {epoch} STATS ---")
            logging.info(f"Duration: {epoch_duration:.2f} seconds")
            logging.info(f"Avg Reward: {avg_reward:.4f}")
            logging.info(f"-------------------------")
            sys.stdout.flush()
            
        # --- BRIDGE SYNC ---
        if rank == 0:
            # DDP wraps model in .module
            broadcast_model_to_inference(model.module, rank, bridge_group, is_sender=True)

# -----------------------------------------------------------
# Inference Loop
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
        sys.stdout.flush()
        
        # This will block until Rank 0 calls broadcast
        broadcast_model_to_inference(model, rank, bridge_group, is_sender=False)
        
        logging.info("Running inference test...")
        test_input = "The AI scientist discovered"
        inputs = tokenizer(test_input, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=30,
                do_sample=True,
                temperature=0.8,
                repetition_penalty=1.2
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logging.info(f"--- [INFERENCE RESULT EPOCH {epoch}] ---")
        logging.info(f"Output: {generated_text}")
        logging.info("---------------------------------------")
        sys.stdout.flush()

# -----------------------------------------------------------
# Main
# -----------------------------------------------------------
def main():
    # Use NCCL everywhere for simplicity if hardware supports it, 
    # but GLOO is safer for the bridge group if network config is tricky.
    # Here we use NCCL globally for performance.
    dist.init_process_group(backend="nccl", init_method="env://", timeout=timedelta(minutes=60))
    rank = dist.get_rank()
    log_file = setup_logging(rank)

    if rank == 0:
        logging.info(f"NCCL_SOCKET_IFNAME: {os.environ.get('NCCL_SOCKET_IFNAME', 'Not Set')}")

    train_ranks = list(range(0, TRAIN_WORLD_SIZE))
    train_group = dist.new_group(ranks=train_ranks)
    
    bridge_ranks = [0, INFERENCE_MASTER_RANK]
    # Use NCCL for bridge too so we use RDMA/Broadcoms
    bridge_group = dist.new_group(ranks=bridge_ranks)

    if rank in train_ranks:
        my_bridge = bridge_group if rank == 0 else None
        run_trainer(rank, TRAIN_WORLD_SIZE, train_group, my_bridge)
    elif rank == INFERENCE_MASTER_RANK:
        run_inference(rank, bridge_group)
    else:
        logging.info("Idling...")
        pass

    logging.info("Waiting for all ranks to complete...")
    dist.barrier()
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
