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
# Dummy Dataset
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
        def __getitem__(self, idx):
            x, y = self.samples[idx]
            return x.clone(), y.clone()

    return DummyDataset(samples)

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

    # Dataset
    dataset = get_dummy_dataset(tokenizer, seq_len=64, dataset_size=512)
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