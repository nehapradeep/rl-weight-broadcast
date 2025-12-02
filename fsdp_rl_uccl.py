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

# UCCL Import
try:
    from uccl import collective
except ImportError:
    print("Error: 'uccl' library not found. Please ensure it is installed.")
    sys.exit(1)

# --- SUPPRESS WARNINGS ---
import warnings
warnings.filterwarnings("ignore")
os.environ["TORCH_CPP_LOG_LEVEL"] = "ERROR"
os.environ["TORCH_DISTRIBUTED_DEBUG"] = "OFF"

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
    # Force flush to prevent missing logs
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
# Dataset
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
            return self.data[start : end]

    logging.info(f"Dataset ready. Samples: {(len(data_tensor)-1)//seq_len}")
    return WikiTextDataset(data_tensor, seq_len)

# -----------------------------------------------------------
# UCCL Collective Broadcast Helper
# -----------------------------------------------------------
def broadcast_model_uccl(model, rank, is_sender=False):
    """
    Uses uccl.collective to send/recv model weights.
    Rank 0 sends to Rank 16.
    """
    # Define Source and Destination
    SRC_RANK = 0
    DST_RANK = INFERENCE_MASTER_RANK

    # We only care if we are the sender or the receiver
    if rank != SRC_RANK and rank != DST_RANK:
        return

    # Determine Device for RDMA (must be GPU)
    try:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        device = torch.device("cuda", local_rank)
    except:
        device = torch.device("cuda")

    if is_sender:
        logging.info("UCCL: Starting Broadcast (Sender)...")
    else:
        logging.info("UCCL: Waiting for Broadcast (Receiver)...")

    state_dict = model.state_dict()
    items = list(state_dict.items())
    total_bytes = sum(t.numel() * t.element_size() for _, t in items)
    
    t0 = time.perf_counter()

    for name, tensor in items:
        # RDMA requires contiguous GPU memory
        # 1. Move to GPU if needed
        if not tensor.is_cuda:
            tensor_gpu = tensor.cuda(device, non_blocking=True)
        else:
            tensor_gpu = tensor

        # 2. Ensure contiguous
        if not tensor_gpu.is_contiguous():
            tensor_gpu = tensor_gpu.contiguous()

        # 3. Register (Pin) memory for max performance
        # Note: In a loop like this, registration adds overhead. 
        # Ideally, we register once, but FSDP creates new tensors per epoch.
        collective.register_tensor(tensor_gpu)

        if is_sender:
            # Send to Inference Node
            collective.send(tensor_gpu, dst=DST_RANK)
        else:
            # Receive from Training Node
            # We must recv into the GPU tensor, then copy back if the model expects CPU (rare for inference)
            collective.recv(tensor_gpu, src=SRC_RANK)
            
            # If the original model was on CPU (unlikely for inference), copy back
            if not tensor.is_cuda:
                tensor.copy_(tensor_gpu.cpu())
            elif tensor.data_ptr() != tensor_gpu.data_ptr():
                # If we created a new contiguous buffer, copy back to model param
                tensor.copy_(tensor_gpu)

    duration = time.perf_counter() - t0
    bw = (total_bytes / 1e9) / duration
    
    if is_sender:
        logging.info(f"UCCL Broadcast Complete. Time: {duration:.3f}s | BW: {bw:.2f} GB/s")
    else:
        logging.info(f"UCCL Receive Complete. Updated Model.")

# -----------------------------------------------------------
# RL Trainer Loop
# -----------------------------------------------------------
def compute_reward(token_ids, tokenizer):
    target_id = 262 # "the"
    matches = (token_ids == target_id).float()
    rewards = matches.sum(dim=1) 
    rewards = (rewards * 0.5) - 0.5
    return rewards

def run_trainer(rank, world_size, train_group):
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
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5) 

    num_epochs = 2
    total_training_start = time.time()
    
    for epoch in range(1, num_epochs + 1):
        model.train()
        sampler.set_epoch(epoch)
        total_reward = 0.0
        epoch_start = time.time()
        
        for i, batch_ids in enumerate(dataloader):
            if i > 40: break
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
        total_tokens_processed = len(dataloader) * BATCH_SIZE * 128 * dist.get_world_size(group=train_group)
        throughput = total_tokens_processed / epoch_duration

        if rank == 0:
            avg_reward = total_reward / len(dataloader)
            logging.info(f"--- RL EPOCH {epoch} STATS ---")
            logging.info(f"Duration: {epoch_duration:.2f} seconds")
            logging.info(f"Throughput: {throughput:.2f} tokens/sec")
            logging.info(f"Avg Reward: {avg_reward:.4f}")
            logging.info(f"-------------------------")
            
        # --- BRIDGE SYNC (UCCL Collective) ---
        # 1. Gather full model to CPU on Rank 0
        save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
            full_state = model.state_dict()

        # 2. Broadcast via UCCL
        if rank == 0:
            logging.info("Preparing weights for UCCL broadcast...")
            cpu_model = GPT2LMHeadModel.from_pretrained("gpt2")
            cpu_model.load_state_dict(full_state)
            
            broadcast_model_uccl(cpu_model, rank, is_sender=True)
            del cpu_model

    total_training_time = time.time() - total_training_start
    if rank == 0:
        logging.info(f"Total Session Time: {total_training_time:.2f} seconds")

# -----------------------------------------------------------
# Inference Loop
# -----------------------------------------------------------
def run_inference(rank):
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
        logging.info(f"Waiting for UCCL model update (Epoch {epoch})...")
        
        # Receive via UCCL Collective
        broadcast_model_uccl(model, rank, is_sender=False)
        
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

# -----------------------------------------------------------
# Main
# -----------------------------------------------------------
def main():
    # Setup GLOO interface to match NCCL if set (crucial for multi-NIC nodes)
    # UCCL needs GLOO to work, and GLOO needs to know which interface to use.
    if "NCCL_SOCKET_IFNAME" in os.environ and "GLOO_SOCKET_IFNAME" not in os.environ:
        os.environ["GLOO_SOCKET_IFNAME"] = os.environ["NCCL_SOCKET_IFNAME"]

    # 1. Init Standard Distributed (GLOO for UCCL Control Plane)
    # Changed from 'nccl' to 'gloo' because UCCL CollectiveContext requires it.
    dist.init_process_group(backend="gloo", init_method="env://", timeout=timedelta(minutes=60))
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    log_file = setup_logging(rank)

    if rank == 0:
        logging.info(f"NCCL_SOCKET_IFNAME: {os.environ.get('NCCL_SOCKET_IFNAME', 'Not Set')}")
        logging.info(f"GLOO_SOCKET_IFNAME: {os.environ.get('GLOO_SOCKET_IFNAME', 'Not Set')}")

    # 2. Init UCCL Collective (This wires up RDMA connections automatically)
    # This replaces the manual p2p handshake
    logging.info("Initializing UCCL Collective...")
    collective.init_collective(num_cpus=4)
    
    # 3. Setup Process Groups for Training (Explicit NCCL for GPU speed)
    # FSDP requires NCCL for performance, so we create a new group explicitly using NCCL backend
    # instead of inheriting the default GLOO backend.
    train_ranks = list(range(0, TRAIN_WORLD_SIZE))
    train_group = dist.new_group(ranks=train_ranks, backend="nccl")
    
    # 4. Branch Logic
    if rank in train_ranks:
        run_trainer(rank, TRAIN_WORLD_SIZE, train_group)
    elif rank == INFERENCE_MASTER_RANK:
        run_inference(rank)
    else:
        logging.info("Idling...")
        pass

    logging.info("Waiting for all ranks to complete...")
    dist.barrier()
    
    # Cleanup
    collective.finalize_collective()
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
