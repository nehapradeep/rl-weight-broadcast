from __future__ import annotations
import os
import sys
import time
import math
import logging
import socket
from datetime import datetime

import torch
import torch.nn as nn
import torch.distributed as dist

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from transformers import GPT2LMHeadModel, GPT2Tokenizer

#MODEL_NAME = os.environ.get("MODEL", "gpt2")
#model, tokenizer = load_model_and_tokenizer(MODEL_NAME)



#------------------------------------------------------------
# Auto setup model
#------------------------------------------------------------
def load_model_and_tokenizer(model_name: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logging.info(f"Loading model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16  # safer and faster
    )

    return model, tokenizer



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

        # ---------------------------
        # Start timing (rank 0 only)
        # ---------------------------
        if rank == 0:
            step_start = time.time()

        # Move to GPU
        input_ids = input_ids.to(device)
        labels = labels.to(device)

        # Forward/backward
        optimizer.zero_grad()
        outputs = model(input_ids, labels=labels)
        loss = outputs.loss

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # ---------------------------
        # End timing + throughput
        # ---------------------------
        if rank == 0:
            step_time = time.time() - step_start
            tokens = input_ids.numel()
            tokens_per_sec = tokens / step_time

            print(
                f"[BENCH] epoch={epoch} step={batch_idx} "
                f"step_time={step_time:.4f}s tokens_per_sec={tokens_per_sec:.1f}"
            )

        # Existing logging
        if batch_idx % 10 == 0:
            logging.info(
                f"Epoch {epoch} | Batch {batch_idx}/{num_batches} | "
                f"Loss: {loss.item():.4f}"
            )

    avg_loss = total_loss / max(1, num_batches)
    logging.info(f"Finished epoch {epoch} | Average loss: {avg_loss:.4f}")
    return avg_loss


def save_checkpoint(model, optimizer, epoch, rank):
    if rank != 0:
        return

    ckpt_dir = "checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, f"ddp_gpt2_wikitext2.pt")

    state = {
        "epoch": epoch,
        "model_state_dict": model.module.state_dict()
        if isinstance(model, nn.parallel.DistributedDataParallel)
        else model.state_dict(),
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

    # NEW: log GPU/node info
    log_gpu_info()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/ROCm is required for this script.")

    # Assign GPU
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    logging.info(f"Using CUDA device index {local_rank}")

    # Load model + tokenizer
#    logging.info("Loading GPT-2 model and tokenizer...")
#    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
#    tokenizer.pad_token = tokenizer.eos_token

#    model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
#    model = nn.parallel.DistributedDataParallel(
#        model,
#        device_ids=[local_rank],
#        output_device=local_rank,
#    )

    MODEL_NAME = os.environ.get("MODEL", "gpt2")

    logging.info(f"Loading model and tokenizer for: {MODEL_NAME}")
    model, tokenizer = load_model_and_tokenizer(MODEL_NAME)

    model = model.to(device)

    model = nn.parallel.DistributedDataParallel(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
    )



    # Dataset
    dataset = get_dummy_dataset(tokenizer, seq_len=64, dataset_size=512)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)

    dataloader = DataLoader(dataset, batch_size=4, sampler=sampler, num_workers=2)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    num_epochs = 2
    for epoch in range(1, num_epochs + 1):
        sampler.set_epoch(epoch)
        avg_loss = train_one_epoch(model, dataloader, optimizer, device, epoch, rank)

        if rank == 0:
            logging.info(
                f"Epoch {epoch} summary: avg_loss={avg_loss:.4f}"
            )

    save_checkpoint(model, optimizer, num_epochs, rank)

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
