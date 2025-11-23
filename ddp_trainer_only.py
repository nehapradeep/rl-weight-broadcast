from __future__ import annotations
import os
import sys
import time
import math
import logging
from datetime import datetime

import torch
import torch.nn as nn
import torch.distributed as dist

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_dataset


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def setup_logging(rank: int) -> str:
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

    old_factory = logging.getLogRecordFactory()

    def record_factory(*args, **kwargs):
        record = old_factory(*args, **kwargs)
        record.rank = rank
        return record

    logging.setLogRecordFactory(record_factory)
    logging.info(f"Logging initialized. Log file: {log_file}")
    return log_file


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
def run_training(
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
            loss = outputs.loss  # scalar

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
        save_model = model.module if isinstance(model, nn.parallel.DistributedDataParallel) else model
        ckpt_path = os.path.join(checkpoint_dir, "ddp_gpt2_wikitext2.pt")
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
    # Initialize default process group
    dist.init_process_group(backend="nccl", init_method="env://")
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    log_file = setup_logging(rank)
    logging.info(f"Init: rank={rank}, world_size={world_size}")

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
    logging.info("Model and tokenizer initialized")

    # Wrap model in DDP
    ddp_model = nn.parallel.DistributedDataParallel(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
    )

    # Train
    run_training(
        ddp_model,
        tokenizer,
        device=device,
        rank=rank,
        world_size=world_size,
        num_epochs=2,
        batch_size=4,
        lr=5e-5,
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
