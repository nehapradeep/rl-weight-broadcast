from __future__ import annotations
import os
import sys
import time
import math
import logging
import socket
import functools
from datetime import datetime
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers.models.gpt2.modeling_gpt2 import GPT2Block

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
# Adjust these based on your cluster
TRAIN_NODES = 2
GPUS_PER_NODE = 8
TRAIN_WORLD_SIZE = TRAIN_NODES * GPUS_PER_NODE  # 16
# The Inference Node is the first rank after the training ranks
INFERENCE_MASTER_RANK = TRAIN_WORLD_SIZE  # Rank 16

# -----------------------------------------------------------
# Logging Setup
# -----------------------------------------------------------
def setup_logging(rank: int) -> str:
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"rank_{rank}_{timestamp}.log")

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    for h in list(logger.handlers):
        logger.removeHandler(h)

    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
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
def get_dummy_dataset(tokenizer: GPT2Tokenizer, seq_len: int = 64, dataset_size: int = 1024):
    text = "Hello world! This is a dummy dataset for testing distributed training. " * 100
    enc = tokenizer(text, return_tensors="pt")
    input_ids = enc["input_ids"][0]
    if len(input_ids) < seq_len:
        input_ids = input_ids.repeat(math.ceil(seq_len / len(input_ids)))

    samples = []
    for i in range(dataset_size):
        start = (i * seq_len) % (len(input_ids) - seq_len - 1)
        end = start + seq_len
        x = input_ids[start:end]
        y = input_ids[start + 1 : end + 1]
        samples.append((x, y))

    class DummyDataset(torch.utils.data.Dataset):
        def __init__(self, samples): self.samples = samples
        def __len__(self): return len(self.samples)
        def __getitem__(self, idx): return x.clone(), y.clone()
    return DummyDataset(samples)

# -----------------------------------------------------------
# Communication Helper: Send/Recv Weights
# -----------------------------------------------------------
def broadcast_model_to_inference(model, rank, bridge_group, is_sender=False):
    """
    Synchronizes the model weights from Training Master (Rank 0) 
    to Inference Master (Rank 16) using the bridge_group.
    """
    if bridge_group is None:
        return

    # Ensure model is on CPU for transfer to avoid GPU-GPU direct issues across unrelated nodes
    # In a real high-perf setup, you might keep on GPU, but CPU is safer for generic setups.
    if is_sender:
        logging.info("Broadcasting weights to Inference Node...")
    
    with torch.no_grad():
        # We iterate over the state dict in a deterministic order
        # Note: On sender, model is likely a CPU copy of the full weights (after FSDP gather)
        # On receiver, model is a standard GPT2LMHeadModel
        
        # Sender (Rank 0) and Receiver (Rank 16) must have same architecture
        for param in model.parameters():
            # Broadcast the tensor. Source is Rank 0 (global). 
            # Note: dist.broadcast uses global ranks if group is not specified, 
            # but with 'group', src is relative to the group? 
            # PyTorch dist.broadcast src is GLOBAL rank even when using a group.
            dist.broadcast(param.data, src=0, group=bridge_group)
            
    if is_sender:
        logging.info("Broadcast complete.")
    else:
        logging.info("Weights received from Training Cluster.")

# -----------------------------------------------------------
# Trainer Loop
# -----------------------------------------------------------
def run_trainer(rank, world_size, train_group, bridge_group):
    # Setup
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Init Model
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    
    # FSDP Wrap (Only for training ranks)
    gpt2_auto_wrap_policy = functools.partial(transformer_auto_wrap_policy, transformer_layer_cls={GPT2Block})
    
    # IMPORTANT: Pass the 'train_group' so FSDP only shards across Ranks 0-15
    model = FSDP(
        model,
        auto_wrap_policy=gpt2_auto_wrap_policy,
        process_group=train_group, 
        device_id=torch.cuda.current_device(),
    )
    
    dataset = get_dummy_dataset(tokenizer, seq_len=64, dataset_size=512)
    # Note: num_replicas is size of TRAIN group, not world size
    sampler = DistributedSampler(dataset, num_replicas=dist.get_world_size(group=train_group), rank=dist.get_rank(group=train_group), shuffle=True)
    dataloader = DataLoader(dataset, batch_size=4, sampler=sampler, num_workers=2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    num_epochs = 2
    
    for epoch in range(1, num_epochs + 1):
        # Training
        model.train()
        sampler.set_epoch(epoch)
        total_loss = 0.0
        
        logging.info(f"Starting Epoch {epoch}")
        for i, (ids, labels) in enumerate(dataloader):
            ids, labels = ids.to(device), labels.to(device)
            optimizer.zero_grad()
            output = model(ids, labels=labels)
            output.loss.backward()
            optimizer.step()
            total_loss += output.loss.item()
            
        avg_loss = total_loss / len(dataloader)
        if rank == 0:
            logging.info(f"Epoch {epoch} Done. Avg Loss: {avg_loss:.4f}")
            
        # --- SYNCHRONIZATION WITH INFERENCE NODE ---
        # 1. Gather full weights to Rank 0 (CPU)
        save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
            full_state = model.state_dict()

        # 2. Broadcast from Rank 0 to Rank 16
        # Only Rank 0 needs to participate in the 'send', but FSDP gather needs all train ranks
        if rank == 0:
            # We need a temporary model to load the state dict into for easy parameter iteration
            # or we can iterate the state dict tensors directly. 
            # For simplicity in this example, we iterate the CPU model structure.
            cpu_model = GPT2LMHeadModel.from_pretrained("gpt2")
            cpu_model.load_state_dict(full_state)
            
            # Signal Inference node that we are sending (Optional, implicit in the blocking broadcast)
            broadcast_model_to_inference(cpu_model, rank, bridge_group, is_sender=True)
            del cpu_model

# -----------------------------------------------------------
# Inference Loop
# -----------------------------------------------------------
def run_inference(rank, bridge_group):
    # Setup
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Standard Model (Not FSDP)
    logging.info("Initializing Inference Model...")
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    model.eval()
    
    # We loop indefinitely (or for known epochs) waiting for updates
    # In a real app, you'd have a signal to exit. Here we match the 2 epochs.
    num_epochs = 2 
    
    for epoch in range(1, num_epochs + 1):
        logging.info(f"Waiting for model update from Training Cluster (Epoch {epoch})...")
        
        # Receive weights (This blocks until Rank 0 sends)
        # We need the model on CPU or same device type as sender? 
        # dist.broadcast handles device mismatch usually, but let's be safe:
        # We will receive into the GPU model directly if possible.
        broadcast_model_to_inference(model, rank, bridge_group, is_sender=False)
        
        # Run Inference Test
        logging.info("Running inference test...")
        test_input = "The scientist discovered a new"
        inputs = tokenizer(test_input, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=20)
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logging.info(f"--- [INFERENCE EPOCH {epoch}] ---")
        logging.info(f"Input: {test_input}")
        logging.info(f"Output: {generated_text}")
        logging.info("---------------------------------")

# -----------------------------------------------------------
# Main
# -----------------------------------------------------------
def main():
    dist.init_process_group(backend="nccl", init_method="env://")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    log_file = setup_logging(rank)
    logging.info(f"Initialized Rank {rank}/{world_size}")

    # 1. Create Subgroups
    # Training Group: Ranks 0 to 15
    train_ranks = list(range(0, TRAIN_WORLD_SIZE))
    train_group = dist.new_group(ranks=train_ranks)
    
    # Bridge Group: Rank 0 (Train Master) and Rank 16 (Inference Master)
    # This channel is used to ship weights across the boundary
    bridge_ranks = [0, INFERENCE_MASTER_RANK]
    # Note: All processes must call new_group, even if they aren't in it, to keep IDs consistent
    bridge_group = dist.new_group(ranks=bridge_ranks)

    # 2. Branch logic
    if rank in train_ranks:
        # If I am in the bridge group (Rank 0), I pass it. Others pass None.
        my_bridge = bridge_group if rank == 0 else None
        run_trainer(rank, world_size, train_group, my_bridge)
    
    elif rank == INFERENCE_MASTER_RANK:
        # Run inference logic
        run_inference(rank, bridge_group)
        
    else:
        # Ranks 17-23 (if they exist) just idle
        logging.info("Idling...")
        # A real implementation would handle clean exit. 
        # For now, they wait for the process group to be destroyed.
        pass

    logging.info("Work complete. destroying process group.")
    dist.destroy_process_group()

if __name__ == "__main__":
    main()