# ddp_train_uccl_init.py
from __future__ import annotations
import os
import sys
import time
import math
import logging
from datetime import datetime

import torch
import torch.distributed as dist
import torch.nn as nn

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_dataset

from uccl import p2p


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def setup_logging(rank: int) -> str:
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"{log_dir}/trainer_rank_{rank}_{timestamp}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | Rank %(rank)d | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout),
        ],
    )

    old_factory = logging.getLogRecordFactory()

    def record_factory(*args, **kwargs):
        record = old_factory(*args, **kwargs)
        record.rank = rank
        return record

    logging.setLogRecordFactory(record_factory)
    logging.info(f"Logging initialized for trainer rank {rank}. Log file: {log_file}")
    return log_file


# ---------------------------------------------------------------------------
# UCCl recv helper
# ---------------------------------------------------------------------------
def recv_model_uccl(ep: p2p.Endpoint, conn_id, model: nn.Module, rank: int):
    """Receive model parameters from broadcaster via UCCl."""
    state_dict = model.state_dict()
    total_tensors = len(list(state_dict.items()))
    total_size_mb = sum(t.numel() * t.element_size() for t in state_dict.values()) / 1e6

    logging.info("=" * 80)
    logging.info("UCCl RECEIVE START (trainer root)")
    logging.info(f"Total tensors: {total_tensors}")
    logging.info(f"Total size: {total_size_mb:.2f} MB")
    logging.info("=" * 80)

    t0 = time.perf_counter()

    for idx, (name, tensor) in enumerate(state_dict.items(), 1):
        recv_tensor = torch.empty_like(tensor, device="cuda")
        size_bytes = recv_tensor.numel() * recv_tensor.element_size()
        ptr = recv_tensor.data_ptr()

        ok, mr_id = ep.reg(ptr, size_bytes)
        assert ok, f"UCCl: failed to register recv buffer for tensor {name}"

        ok = ep.recv(conn_id, mr_id, ptr, size_bytes)
        assert ok, f"UCCl: receive failed for tensor {name}"

        model.state_dict()[name].copy_(recv_tensor)

        if idx % 20 == 0 or idx == total_tensors:
            pct = (idx / total_tensors) * 100
            logging.info(f"UCCl progress: {pct:.1f}% ({idx}/{total_tensors})")

    total_time = time.perf_counter() - t0
    bw_gbps = (total_size_mb / 1000.0) / total_time if total_time > 0 else 0.0

    logging.info("=" * 80)
    logging.info("UCCl RECEIVE COMPLETE (trainer root)")
    logging.info(f"Total time: {total_time:.2f}s")
    logging.info(f"Average bandwidth: {bw_gbps:.2f} GB/s")
    logging.info("=" * 80)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
def prepare_dataset(tokenizer: GPT2Tokenizer, max_length: int = 128, num_samples: int = 200):
    logging.info("=" * 80)
    logging.info("LOADING DATASET (WikiText-2)")
    logging.info("=" * 80)

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    logging.info(f"Loaded {len(dataset)} raw examples")

    dataset = dataset.filter(lambda x: len(x["text"].strip()) > 0)
    logging.info(f"After filtering empty examples: {len(dataset)}")

    if len(dataset) > num_samples:
        dataset = dataset.select(range(num_samples))
        logging.info(f"Using subset of {num_samples} examples for faster training")

    logging.info("Tokenizing dataset...")

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )

    tokenized = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
    )
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask"])

    logging.info("Tokenization complete")
    logging.info(f"  - Examples: {len(tokenized)}")
    logging.info(f"  - Max length: {max_length}")
    logging.info("=" * 80)
    return tokenized


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def run_distributed_training(
    model: nn.Module,
    tokenizer: GPT2Tokenizer,
    device: torch.device,
    rank: int,
    world_size: int,
    num_epochs: int = 2,
    batch_size: int = 4,
    lr: float = 5e-5,
):
    logging.info(f"RANK {rank}: starting DDP training.")

    dataset = prepare_dataset(tokenizer, max_length=128, num_samples=200)

    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    total_steps = len(dataloader) * num_epochs
    logging.info(f"RANK {rank}: total steps = {total_steps}")

    global_step = 0
    total_train_start = time.perf_counter()

    for epoch in range(num_epochs):
        sampler.set_epoch(epoch)
        epoch_loss = 0.0
        epoch_perplexity = 0.0

        logging.info(f"RANK {rank}: EPOCH {epoch + 1}/{num_epochs} starting")

        for step, batch in enumerate(dataloader):
            global_step += 1
            step_start = time.perf_counter()

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids,
            )
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_value = loss.item()
            perplexity = math.exp(loss_value) if loss_value < 100 else float("inf")

            epoch_loss += loss_value
            epoch_perplexity += perplexity

            if step % 10 == 0 and rank == 0:
                step_time = time.perf_counter() - step_start
                logging.info(
                    f"[Epoch {epoch+1}/{num_epochs}] "
                    f"[Step {step}/{len(dataloader)} | Global {global_step}/{total_steps}] "
                    f"Loss: {loss_value:.4f} | Perplexity: {perplexity:.2f} "
                    f"| Time/step: {step_time:.3f}s"
                )

        avg_epoch_loss = epoch_loss / len(dataloader)
        avg_epoch_perplexity = epoch_perplexity / len(dataloader)

        if rank == 0:
            logging.info("=" * 80)
            logging.info(f"EPOCH {epoch + 1} SUMMARY (reported by rank {rank})")
            logging.info(f"  Avg loss      : {avg_epoch_loss:.4f}")
            logging.info(f"  Avg perplexity: {avg_epoch_perplexity:.2f}")
            logging.info("=" * 80)

    total_train_time = time.perf_counter() - total_train_start
    if rank == 0:
        avg_step_time = total_train_time / total_steps
        logging.info("=" * 80)
        logging.info("TRAINING COMPLETE (leader rank)")
        logging.info(f"  Total training time: {total_train_time:.2f}s")
        logging.info(f"  Total steps        : {total_steps}")
        logging.info(f"  Avg time/step      : {avg_step_time:.3f}s")
        logging.info(f"  Steps/sec          : {total_steps / total_train_time:.2f}")
        logging.info("=" * 80)

    if rank == 0:
        checkpoint_dir = "checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        save_model = model.module if isinstance(model, nn.parallel.DistributedDataParallel) else model
        ckpt_path = os.path.join(checkpoint_dir, "ddp_trained_model.pt")
        torch.save(
            {
                "model_state_dict": save_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            ckpt_path,
        )
        logging.info(f"Checkpoint saved at: {ckpt_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    # Default PG over trainer nodes only
    dist.init_process_group(backend="nccl", init_method="env://")
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    log_file = setup_logging(rank)
    logging.info(f"Trainer init: rank={rank}, world_size={world_size}")

    if world_size < 2:
        raise RuntimeError("Need at least 2 trainer ranks for DDP")

    # Trainer root is rank 0 in this job
    root_trainer = 0

    # Device assignment
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    logging.info(f"Rank {rank}: using CUDA device index {local_rank}")

    # UCCl metadata file path (shared filesystem between trainer nodes and broadcaster)
    metadata_path = os.environ.get("UCCL_MD_PATH", "uccl_root_md.bin")

    # Initialize UCCl endpoint on all ranks (only root uses it, but endpoint init is cheap)
    logging.info("Initializing UCCl endpoint on trainer rank...")
    ep = p2p.Endpoint(local_rank, 4)
    local_md = ep.get_metadata()
    logging.info(f"Rank {rank}: UCCl local metadata size: {len(local_md)} bytes")

    # Root trainer writes its UCCl metadata to a shared file and waits for broadcaster.
    if rank == root_trainer:
        logging.info(f"Root trainer writing UCCl metadata to file: {metadata_path}")
        try:
            with open(metadata_path, "wb") as f:
                f.write(local_md)
            logging.info("Root trainer wrote UCCl metadata file successfully.")
        except Exception as e:
            logging.error(f"Failed to write UCCl metadata file: {e}")
            raise

        logging.info("Root trainer waiting for UCCl connection from broadcaster...")
        ok, r_ip, r_gpu, conn_id = ep.accept()
        assert ok, "UCCl accept() failed on trainer root"
        logging.info(f"Root trainer connected to broadcaster via UCCl: ip={r_ip}, gpu={r_gpu}")
    else:
        conn_id = None

    # Build model + tokenizer on all ranks
    logging.info("Initializing model and tokenizer on trainer rank...")
    base_model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    logging.info("Model and tokenizer initialized")

    # Root trainer receives weights via UCCl
    if rank == root_trainer:
        logging.info("Root trainer receiving model weights over UCCl...")
        recv_model_uccl(ep, conn_id, base_model, rank)
        logging.info("Root trainer finished UCCl receive.")

    # Ensure UCCl receive is done before broadcasting
    dist.barrier()

    # Broadcast parameters from root trainer to all trainers via PyTorch
    logging.info("Synchronizing model weights from root trainer to all trainer ranks...")
    with torch.no_grad():
        for param in base_model.parameters():
            dist.broadcast(param.data, src=root_trainer)
    logging.info("All trainer ranks have synchronized model weights.")

    # Wrap in DDP
    ddp_model = nn.parallel.DistributedDataParallel(
        base_model,
        device_ids=[local_rank],
        output_device=local_rank,
    )

    # Now perform DDP training
    run_distributed_training(ddp_model, tokenizer, device, rank, world_size)

    logging.info("Destroying process group...")
    dist.destroy_process_group()
    logging.info("Trainer rank finished.")
    logging.info(f"Log file: {log_file}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logging.warning("Trainer interrupted by user")
        sys.exit(0)
    except Exception as e:
        logging.error(f"Trainer fatal error: {e}", exc_info=True)
        sys.exit(1)
