from __future__ import annotations
import os
import sys
import time
import math
import logging
import socket
import functools
import itertools
from datetime import datetime
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
# Logging Setup
# -----------------------------------------------------------
def setup_logging(rank: int) -> str:
    """Setup logging to file (per rank) and console."""
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"rank_{rank}_{timestamp}.log")

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # Clear existing handlers
    for h in list(logger.handlers):
        logger.removeHandler(h)

    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)

    formatter = logging.Formatter(
        fmt="%(asctime)s | Rank %(rank)d | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

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
# GPU/Node Logging Helper
# -----------------------------------------------------------
def log_gpu_info():
    """Log how many GPUs each node has and how ranks map to GPUs."""
    try:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
    except (TypeError, ValueError):
        local_rank = 0
    node_rank = os.environ.get("NODE_RANK", "unknown")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    num_gpus = torch.cuda.device_count()
    hostname = socket.gethostname()

    # One summary line per node (LOCAL_RANK == 0)
    if local_rank == 0:
        logging.info(
            f"[NODE SUMMARY] host={hostname} node_rank={node_rank} "
            f"num_gpus_visible={num_gpus} world_size={world_size}"
        )

    # Detailed log for every rank
    logging.info(
        f"[RANK MAP] global_rank={rank} node_rank={node_rank} "
        f"local_rank={local_rank} cuda_device={local_rank} host={hostname}"
    )

# -----------------------------------------------------------
# Dataset: WikiText-2
# -----------------------------------------------------------
def get_wikitext_dataset(tokenizer: GPT2Tokenizer, seq_len: int = 128):
    """
    Loads WikiText-2, tokenizes it, and chunks it into sequences of seq_len.
    """
    # Only main process on each node should ideally manage downloads, 
    # but 'datasets' uses file locking, so it's safe to call on all ranks.
    # We suppress progress bars on non-zero local ranks to keep logs clean.
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if local_rank != 0:
        import datasets
        datasets.logging.set_verbosity_error()
        
    logging.info("Loading WikiText-2 dataset...")
    # Using 'wikitext-2-raw-v1' (no pre-processing)
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

    logging.info("Tokenizing WikiText-2...")
    
    def tokenize_function(examples):
        return tokenizer(examples["text"])

    # Tokenize the dataset
    tokenized_datasets = dataset.map(
        tokenize_function, 
        batched=True, 
        num_proc=4, 
        remove_columns=["text"]
    )

    logging.info("Flattening and chunking dataset...")
    # Flatten all input_ids into one long list
    all_input_ids = list(itertools.chain(*tokenized_datasets["input_ids"]))
    
    # Create tensor
    # We need to ensure we can create samples of (seq_len) plus 1 token for label
    total_tokens = len(all_input_ids)
    
    # Convert to tensor
    data_tensor = torch.tensor(all_input_ids, dtype=torch.long)
    
    class WikiTextDataset(torch.utils.data.Dataset):
        def __init__(self, data, seq_len):
            self.data = data
            self.seq_len = seq_len
            # Calculate how many full blocks we can make
            # We need block_size + 1 (for next token prediction)
            self.num_samples = (len(self.data) - 1) // self.seq_len

        def __len__(self):
            return self.num_samples

        def __getitem__(self, idx):
            start = idx * self.seq_len
            end = start + self.seq_len
            
            # x is input, y is target (shifted by 1)
            x = self.data[start : end]
            y = self.data[start+1 : end+1]
            return x, y

    logging.info(f"Dataset created: {total_tokens} tokens, {(total_tokens-1)//seq_len} samples.")
    return WikiTextDataset(data_tensor, seq_len)

# -----------------------------------------------------------
# Training Helpers
# -----------------------------------------------------------
def train_one_epoch(model, dataloader, optimizer, device, epoch, rank):
    model.train()
    total_loss = 0.0
    num_batches = len(dataloader)

    logging.info(f"Starting epoch {epoch} with {num_batches} batches")

    for batch_idx, (input_ids, labels) in enumerate(dataloader):
        input_ids = input_ids.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if batch_idx % 10 == 0:
            logging.info(
                f"Epoch {epoch} | Batch {batch_idx}/{num_batches} | "
                f"Loss: {loss.item():.4f}"
            )

    avg_loss = total_loss / max(1, num_batches)
    logging.info(f"Finished epoch {epoch} | Average loss: {avg_loss:.4f}")
    return avg_loss

def save_checkpoint_fsdp(model, optimizer, epoch, rank):
    """
    Saves a consolidated checkpoint on rank 0.
    FSDP requires a specific context manager to gather parameters 
    from all shards into a single state_dict.
    """
    # Policy: gather full state dict to CPU on rank 0
    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    
    # Use the context manager to gather the full state dict
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
        cpu_state = model.state_dict()
    
    # Optimizer saving is complex in FSDP (sharded vs full). 
    # For simplicity in this 'exact same functionality' port, we save the full model weights.
    # Saving full optimizer state requires FSDP.optim_state_dict + scattering logic on load,
    # so we stick to model weights primarily for this example.
    
    if rank == 0:
        ckpt_dir = "checkpoints"
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_path = os.path.join(ckpt_dir, f"fsdp_gpt2_wikitext2.pt")
        
        state = {
            "epoch": epoch,
            "model_state_dict": cpu_state,
            # Note: Standard optimizer.state_dict() may not work as expected in FSDP without specific calls.
            # We save it here for structure consistency, but restoring it requires FSDP.load_optim_state_dict
            "optimizer_state_dict": optimizer.state_dict(), 
        }
        torch.save(state, ckpt_path)
        logging.info(f"Checkpoint saved at: {ckpt_path}")

# -----------------------------------------------------------
# Main Training Loop
# -----------------------------------------------------------
def main():
    # Initialize process group
    dist.init_process_group(backend="nccl", init_method="env://")
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Setup logging
    log_file = setup_logging(rank)
    logging.info(f"Logging initialized. Log file: {log_file}")
    logging.info(f"Init: rank={rank}, world_size={world_size}")

    # log GPU/node info
    log_gpu_info()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/ROCm is required for this script.")

    # Assign GPU
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    logging.info(f"Using CUDA device index {local_rank}")

    # Load model + tokenizer
    logging.info("Loading GPT-2 model and tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # 1. Instantiate model on CPU (standard practice for FSDP to avoid GPU OOM on init, 
    #    though for GPT2-small it fits fine). We move it to device automatically via FSDP 
    #    or manually before depending on strategy. Here we move to device to match DDP flow 
    #    closely, but FSDP handles sharding.
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)

    # 2. Define Auto Wrapping Policy
    # This tells FSDP to wrap every GPT2Block individually, which enables
    # sharding parameters and clearing gradients layer-by-layer.
    gpt2_auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={GPT2Block},
    )

    # 3. Wrap with FSDP
    model = FSDP(
        model,
        auto_wrap_policy=gpt2_auto_wrap_policy,
        device_id=torch.cuda.current_device(), # Important: bind to specific GPU
    )

    logging.info(f"Model wrapped with FSDP.")

    # Dataset (UPDATED)
    dataset = get_wikitext_dataset(tokenizer, seq_len=128)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    dataloader = DataLoader(dataset, batch_size=4, sampler=sampler, num_workers=2)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    num_epochs = 2
    
    # Track total training time
    total_start_time = time.time()

    for epoch in range(1, num_epochs + 1):
        epoch_start_time = time.time()
        
        sampler.set_epoch(epoch)
        avg_loss = train_one_epoch(model, dataloader, optimizer, device, epoch, rank)

        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time

        if rank == 0:
            logging.info(
                f"Epoch {epoch} summary: avg_loss={avg_loss:.4f} | "
                f"Time: {epoch_duration:.2f}s"
            )

    total_end_time = time.time()
    total_duration = total_end_time - total_start_time
    
    if rank == 0:
        logging.info(f"Total training time: {total_duration:.2f}s")

    save_checkpoint_fsdp(model, optimizer, num_epochs, rank)

    logging.info("Destroying process group...")
    dist.destroy_process_group()
    logging.info("Process finished.")
    logging.info(f"Log file: {log_file}")

# -----------------------------------------------------------
# Entry Point
# -----------------------------------------------------------
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logging.warning("Interrupted by user")
    except Exception as e:
        logging.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)
        
# torchrun   --nnodes=2   --nproc_per_node=8   --node_rank=0   --master_addr=10.162.224.131   --master_port=29500   fsdp_training.py(amd 1)
# torchrun   --nnodes=2   --nproc_per_node=8   --node_rank=1   --master_addr=10.162.224.131   --master_port=29500   fsdp_training.py(amd 2)