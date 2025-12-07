from __future__ import annotations
import os
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_dataset


def setup_dist():
    # torchrun sets these env vars for you
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    return rank, world_size, local_rank, device


def build_dataloader(tokenizer, rank, world_size, seq_len=256, batch_size=4):
    raw = load_dataset("wikitext", "wikitext-2-raw-v1")  # train/valid/test

    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=seq_len,
            padding="max_length",
        )

    tokenized = raw["train"].map(
        tokenize_fn,
        batched=True,
        remove_columns=["text"]
    )
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask"])

    sampler = DistributedSampler(
        tokenized,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        drop_last=True,
    )

    loader = DataLoader(
        tokenized,
        batch_size=batch_size,
        sampler=sampler,
        pin_memory=True,
    )
    return loader, sampler


def main():
    rank, world_size, local_rank, device = setup_dist()

    if rank == 0:
        print(f"World size = {world_size}, running FSDP on {world_size} GPUs")

    # 1) Build tokenizer & model on every rank
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token  # GPT2 has no pad by default

    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.to(device)

    # 2) Wrap with FSDP *before* creating optimizer
    model = FSDP(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    # 3) DataLoader + DistributedSampler
    train_loader, train_sampler = build_dataloader(
        tokenizer, rank, world_size, seq_len=128, batch_size=2
    )

    num_epochs = 2

    for epoch in range(num_epochs):
        # Ensure each rank gets a different shard each epoch
        train_sampler.set_epoch(epoch)

        model.train()
        total_loss = 0.0

        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids,  # standard LM training
            )
            loss = outputs.loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if step % 50 == 0 and rank == 0:
                avg_loss = total_loss / (step + 1)
                print(f"[epoch {epoch} step {step}] loss={avg_loss:.4f}")

        # Optionally save checkpoint only on rank 0
        if rank == 0:
            save_path = f"checkpoint_epoch{epoch}.pt"
            # unwrap FSDP module for state_dict
            state = model.state_dict()
            torch.save(state, save_path)
            print(f"Saved checkpoint to {save_path}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
