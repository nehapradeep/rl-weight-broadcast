from __future__ import annotations
import os
import sys
import time
import math
import logging
from datetime import datetime
import socket

import torch
import torch.nn as nn
import torch.distributed as dist

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_dataset

# UCCL P2P (same as in gpu_transfer_wikitext2.py)
from uccl import p2p


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def setup_logging(rank: int) -> str:
    """
    Configure logging so each rank logs to its own file.
    Rank 0 logs INFO to stdout; others log WARNING to stderr.
    """
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"{log_dir}/rank_{rank}_{timestamp}.log"

    logging.basicConfig(
        level=logging.INFO if rank == 0 else logging.WARNING,
        format="%(asctime)s | Rank %(rank)d | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout) if rank == 0 else logging.StreamHandler(sys.stderr),
        ],
    )

    # Inject rank into log records
    old_factory = logging.getLogRecordFactory()

    def record_factory(*args, **kwargs):
        record = old_factory(*args, **kwargs)
        record.rank = rank
        return record

    logging.setLogRecordFactory(record_factory)

    logging.info(f"Logging initialized. Log file: {log_file}")
    return log_file


def log_gpu_info():
    """
    Log how many GPUs are visible on this node and how ranks map to local GPUs.
    Call this after dist.init_process_group() and setup_logging().
    """
    try:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
    except (TypeError, ValueError):
        local_rank = 0

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    num_gpus = torch.cuda.device_count()
    hostname = socket.gethostname()
    node_rank = os.environ.get("NODE_RANK", "unknown")

    # One summary line per node (LOCAL_RANK == 0)
    if local_rank == 0:
        logging.info(
            f"[NODE SUMMARY] host={hostname} node_rank={node_rank} "
            f"num_gpus_visible={num_gpus} world_size={world_size}"
        )

    # Detailed per-rank mapping
    logging.info(
        f"[RANK MAP] global_rank={rank} node_rank={node_rank} "
        f"local_rank={local_rank} cuda_device={local_rank} host={hostname}"
    )


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
def prepare_dataset(tokenizer: GPT2Tokenizer, max_length: int = 128, num_samples: int = 200):
    """
    Load WikiText-2, filter empty lines, take a small subset, and tokenize.
    All sequences are truncated/padded to max_length (<= GPT-2 context window).
    """
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

    tokenizer.model_max_length = 1024  # GPT-2 context length

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
        desc="Tokenizing",
    )

    logging.info("Tokenization complete")
    logging.info(f"  - Examples: {len(tokenized)}")
    logging.info(f"  - Max length: {max_length}")
    logging.info("=" * 80)

    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask"])
    return tokenized


# ---------------------------------------------------------------------------
# UCCL model broadcast helpers (API copied from gpu_transfer_wikitext2.py style)
# ---------------------------------------------------------------------------
def broadcast_model(ep: p2p.Endpoint, conn_ids, model: nn.Module, rank: int):
    """
    Send model to one or more receivers with detailed logging using UCCL P2P.

    Uses:
      - ep.reg(ptr, size_bytes) -> (ok, mr_id)
      - ep.send(conn_id, mr_id, ptr, size_bytes)
    """
    state_dict = model.state_dict()
    total_tensors = len(list(state_dict.items()))
    total_size_mb = sum(t.numel() * t.element_size() for t in state_dict.values()) / 1e6

    logging.info("=" * 80)
    logging.info(f"BROADCAST START - Sending to {len(conn_ids)} receivers")
    logging.info(f"Total tensors: {total_tensors}")
    logging.info(f"Total size: {total_size_mb:.2f} MB")
    logging.info("=" * 80)

    broadcast_start = time.perf_counter()

    for idx, (name, tensor) in enumerate(state_dict.items(), 1):
        if not tensor.is_cuda:
            tensor = tensor.cuda()

        size_bytes = tensor.numel() * tensor.element_size()
        ptr = tensor.data_ptr()

        # Register memory
        ok, mr_id = ep.reg(ptr, size_bytes)
        assert ok, f"Failed to register tensor {name}"

        # Send to all receivers
        for receiver_idx, conn_id in enumerate(conn_ids, 1):
            ok = ep.send(conn_id, mr_id, ptr, size_bytes)
            assert ok, f"Send failed for {name} to receiver {receiver_idx}"

        if idx % 20 == 0 or idx == total_tensors:
            progress_pct = (idx / total_tensors) * 100
            logging.info(f"Progress: {progress_pct:.1f}% ({idx}/{total_tensors})")

    total_time = time.perf_counter() - broadcast_start
    avg_bandwidth = (total_size_mb / 1000) / total_time  # GB/s

    logging.info("=" * 80)
    logging.info("BROADCAST COMPLETE")
    logging.info(f"Total time: {total_time:.2f}s")
    logging.info(f"Average bandwidth: {avg_bandwidth:.2f} GB/s")
    logging.info("=" * 80)


def broadcast_trained_model_uccl(
    ddp_model: nn.Module,
    rank: int,
    local_rank: int,
    metadata_path: str = "uccl_trainer_metadata.bin",
    num_cpus: int = 4,
):
    """
    On rank 0, use UCCL P2P to broadcast the trained model weights to
    an inference node.

    Protocol:
      - Trainer (this script, rank 0) acts as UCCL "server":
          ep = Endpoint(local_rank, num_cpus)
          local_md = ep.get_metadata()
          write local_md to a file (metadata_path)
          accept() a connection from inference node
          broadcast_model(ep, [conn_id], core_model)
      - Inference node:
          reads metadata_path, parses via Endpoint.parse_metadata,
          creates its own Endpoint, connect() to trainer,
          recv_model(...)
    """
    if rank != 0:
        return

    core_model = ddp_model.module if isinstance(ddp_model, nn.parallel.DistributedDataParallel) else ddp_model

    logging.info("=" * 80)
    logging.info("UCCL BROADCAST: Trainer rank 0 preparing Endpoint")
    logging.info("=" * 80)

    ep = p2p.Endpoint(local_rank, num_cpus)
    local_md = ep.get_metadata()
    logging.info(f"UCCL local metadata obtained (size: {len(local_md)} bytes)")

    # Write metadata so the inference script can connect using parse_metadata
    with open(metadata_path, "wb") as f:
        f.write(local_md)
    logging.info(f"UCCL trainer metadata written to {metadata_path}")

    logging.info("Waiting for UCCL inference node to connect (ep.accept()) ...")
    ok, r_ip, r_gpu, conn_id = ep.accept()
    assert ok, "UCCL accept() failed"
    logging.info(f"UCCL connected: remote_ip={r_ip}, remote_gpu={r_gpu}, conn_id={conn_id}")

    # Broadcast the trained model to that single inference connection
    broadcast_model(ep, [conn_id], core_model, rank=rank)

    logging.info("UCCL broadcast of trained model complete")


# ---------------------------------------------------------------------------
# Training Loop
# ---------------------------------------------------------------------------
def train_ddp(
    model: nn.Module,
    tokenizer: GPT2Tokenizer,
    device: torch.device,
    rank: int,
    world_size: int,
    num_epochs: int = 2,
    batch_size: int = 4,
    lr: float = 5e-5,
):
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
        num_workers=2,
        pin_memory=True,
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

            optimizer.zero_grad(set_to_none=True)
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

    # Save checkpoint only on rank 0
    if rank == 0:
        checkpoint_dir = "checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        ckpt_path = os.path.join(checkpoint_dir, "ddp_gpt2_wikitext2.pt")

        state = {
            "epoch": num_epochs,
            "model_state_dict": model.module.state_dict()
            if isinstance(model, nn.parallel.DistributedDataParallel)
            else model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }

        torch.save(state, ckpt_path)
        logging.info(f"Checkpoint saved at: {ckpt_path}")


@torch.no_grad()
def run_quick_inference(model: nn.Module, tokenizer: GPT2Tokenizer, device: torch.device):
    """
    Simple sanity check: generate text from a fixed prompt using the trained model.
    Call only on rank 0.
    """
    model.eval()
    core_model = model.module if isinstance(model, nn.parallel.DistributedDataParallel) else model

    prompt = "The meaning of life is"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    output_ids = core_model.generate(
        **inputs,
        max_new_tokens=32,
        do_sample=True,
        top_k=50,
        top_p=0.95,
    )

    generated = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    logging.info("=== QUICK INFERENCE CHECK (trained model) ===")
    logging.info(f"Prompt: {prompt!r}")
    logging.info(f"Output: {generated!r}")
    logging.info("============================================")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    # Initialize default process group
    dist.init_process_group(backend="nccl", init_method="env://")
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    log_file = setup_logging(rank)
    logging.info(f"Init: rank={rank}, world_size={world_size}")

    # Log GPU/node mapping
    log_gpu_info()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/ROCm is required for this script.")

    # local_rank is set by torchrun
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    logging.info(f"Using CUDA device index {local_rank}")

    # Build model & tokenizer
    logging.info("Loading GPT-2 model and tokenizer...")
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = 1024  # explicit

    # Silence loss_type=None warning by setting it explicitly if present
    if hasattr(model.config, "loss_type"):
        try:
            model.config.loss_type = "ForCausalLMLoss"
        except Exception:
            pass

    logging.info("Model and tokenizer initialized")

    # Wrap with DDP
    ddp_model = nn.parallel.DistributedDataParallel(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
    )

    # Train
    train_ddp(
        ddp_model,
        tokenizer,
        device=device,
        rank=rank,
        world_size=world_size,
        num_epochs=2,
        batch_size=4,
        lr=5e-5,
    )

    # Simple inference sanity-check on rank 0
    if rank == 0:
        run_quick_inference(ddp_model, tokenizer, device)

    # OPTIONAL: UCCL broadcast of trained weights to inference node
    # Enable by setting USE_UCCL_BROADCAST=1 in your environment.
    if os.getenv("USE_UCCL_BROADCAST", "0") == "1":
        broadcast_trained_model_uccl(
            ddp_model,
            rank=rank,
            local_rank=local_rank,
            metadata_path="uccl_trainer_metadata.bin",
            num_cpus=4,
        )

    logging.info("Destroying process group...")
    dist.destroy_process_group()
    logging.info("Process finished.")
    logging.info(f"Log file: {log_file}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logging.warning("Interrupted by user")
        sys.exit(0)
    except Exception as e:
        logging.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)
