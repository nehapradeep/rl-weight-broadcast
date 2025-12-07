#!/usr/bin/env python3
"""Unified FSDP/DDP runner for Qwen and RL (GPT-2) with broadcast or UCCL transfer.

Example:
  torchrun --nproc_per_node=8 unified_runner.py --mode fsdp --model qwen --backend broadcast
  torchrun --nproc_per_node=8 unified_runner.py --mode ddp --model rl --backend uccl
"""
from __future__ import annotations
import argparse
import os
import sys
import time
import logging
import functools
import itertools
import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from datasets import load_dataset

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GPT2LMHeadModel,
    GPT2Tokenizer,
)
from transformers.models.gpt2.modeling_gpt2 import GPT2Block
try:
    from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer
except ImportError:
    Qwen2DecoderLayer = None

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    StateDictType,
    FullStateDictConfig,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

# Silence the noisy bits by default
warnings.filterwarnings("ignore")
os.environ.setdefault("TORCH_CPP_LOG_LEVEL", "ERROR")
os.environ.setdefault("TORCH_DISTRIBUTED_DEBUG", "OFF")


@dataclass
class RunConfig:
    mode: str                    # "fsdp" or "ddp"
    model_key: str               # "qwen" or "rl"
    backend: str                 # "broadcast" or "uccl"
    train_nodes: int
    gpus_per_node: int
    seq_len: int
    epochs: int
    max_batches: Optional[int]

    @property
    def train_world_size(self) -> int:
        return self.train_nodes * self.gpus_per_node

    @property
    def inference_rank(self) -> int:
        return self.train_world_size


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
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
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter(
        fmt="%(asctime)s | Rank %(rank)d | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
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


# ---------------------------------------------------------------------------
# Model + data helpers
# ---------------------------------------------------------------------------
def build_model_and_tokenizer(model_key: str, device: torch.device):
    model_key = model_key.lower()
    if model_key.startswith("qwen"):
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B").to(device)
        block_cls = Qwen2DecoderLayer
    else:
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
        block_cls = GPT2Block
    return tokenizer, model, block_cls


def get_dataset(tokenizer, seq_len: int = 128):
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if local_rank != 0:
        import datasets
        datasets.logging.set_verbosity_error()

    logging.info("Loading WikiText-2 dataset...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

    def tokenize_fn(examples):
        return tokenizer(examples["text"])

    tokenized = dataset.map(tokenize_fn, batched=True, num_proc=4, remove_columns=["text"])
    all_input_ids = list(itertools.chain(*tokenized["input_ids"]))
    data_tensor = torch.tensor(all_input_ids, dtype=torch.long)

    class WikiTextDataset(torch.utils.data.Dataset):
        def __init__(self, data, seq_len):
            self.data = data
            self.seq_len = seq_len
            self.num_samples = (len(self.data) - 1) // self.seq_len

        def __len__(self):
            return self.num_samples

        def __getitem__(self, idx):
            start = idx * self.seq_len
            end = start + self.seq_len
            return self.data[start:end]

    logging.info(f"Dataset ready. Samples: {(len(data_tensor) - 1) // seq_len}")
    return WikiTextDataset(data_tensor, seq_len)


def compute_reward(token_ids, tokenizer):
    target_id = tokenizer.encode("the", add_special_tokens=False)[0]
    matches = (token_ids == target_id).float()
    rewards = matches.sum(dim=1)
    rewards = (rewards * 0.5) - 0.5
    return rewards


def unwrap_model(model):
    return model.module if isinstance(model, DDP) else model


def gather_full_state_if_fsdp(model):
    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
        return model.state_dict()


# ---------------------------------------------------------------------------
# Weight transfer helpers
# ---------------------------------------------------------------------------
def torch_broadcast_model(model: torch.nn.Module, bridge_group, rank: int, is_sender: bool):
    if bridge_group is None:
        return

    try:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        device = torch.device("cuda", local_rank)
    except Exception:
        device = torch.device("cuda")

    if is_sender:
        logging.info("Broadcasting weights to inference rank...")
    else:
        logging.info("Waiting for weights from training rank...")
        sys.stdout.flush()

    t0 = time.time()
    with torch.no_grad():
        for param in model.parameters():
            tensor = param.data
            if tensor.device.type == "cpu":
                tensor = tensor.to(device)
            dist.broadcast(tensor, src=0, group=bridge_group)
            if tensor.data_ptr() != param.data_ptr():
                param.copy_(tensor.to(param.device))

    if is_sender:
        duration = time.time() - t0
        total_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
        bw = (total_bytes / 1e9) / duration if duration > 0 else 0
        logging.info(f"Broadcast complete. Time: {duration:.4f}s | BW: {bw:.2f} GB/s")
    else:
        logging.info("Weights received.")
        sys.stdout.flush()


def ensure_uccl_available():
    try:
        from uccl import collective  # noqa: F401
    except ImportError:
        raise SystemExit("Error: backend 'uccl' requested but uccl package is not available.")


def uccl_transfer_state(model: torch.nn.Module, rank: int, inference_rank: int, is_sender: bool):
    from uccl import collective  # type: ignore

    SRC_RANK = 0
    DST_RANK = inference_rank

    if rank != SRC_RANK and rank != DST_RANK:
        return

    try:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        device = torch.device("cuda", local_rank)
    except Exception:
        device = torch.device("cuda")

    if is_sender:
        logging.info("UCCL: starting broadcast (sender)...")
    else:
        logging.info("UCCL: waiting for weights (receiver)...")
        sys.stdout.flush()

    state_dict = model.state_dict()
    items = list(state_dict.items())
    total_bytes = sum(t.numel() * t.element_size() for _, t in items)

    t0 = time.perf_counter()
    for _, tensor in items:
        if not tensor.is_cuda:
            tensor_gpu = tensor.cuda(device, non_blocking=True)
        else:
            tensor_gpu = tensor

        if not tensor_gpu.is_contiguous():
            tensor_gpu = tensor_gpu.contiguous()

        collective.register_tensor(tensor_gpu)
        if is_sender:
            collective.send(tensor_gpu, dst=DST_RANK)
        else:
            collective.recv(tensor_gpu, src=SRC_RANK)
            if not tensor.is_cuda:
                tensor.copy_(tensor_gpu.cpu())
            elif tensor.data_ptr() != tensor_gpu.data_ptr():
                tensor.copy_(tensor_gpu)

    duration = time.perf_counter() - t0
    bw = (total_bytes / 1e9) / duration if duration > 0 else 0
    if is_sender:
        logging.info(f"UCCL broadcast complete. Time: {duration:.4f}s | BW: {bw:.2f} GB/s")
    else:
        logging.info("UCCL receive complete.")
        sys.stdout.flush()


def prepare_sender_model(cfg: RunConfig, model, model_key: str, device: torch.device):
    base_model = unwrap_model(model)
    if cfg.mode == "fsdp":
        full_state = gather_full_state_if_fsdp(base_model)
        cpu_model_device = torch.device("cpu") if device.type == "cuda" else device
        _, fresh_model, _ = build_model_and_tokenizer(model_key, cpu_model_device)
        fresh_model.load_state_dict(full_state)
        return fresh_model
    return base_model


# ---------------------------------------------------------------------------
# Training / inference
# ---------------------------------------------------------------------------
def run_trainer(cfg: RunConfig, rank: int, train_group, bridge_group):
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    tokenizer, model, block_cls = build_model_and_tokenizer(cfg.model_key, device)

    if cfg.mode == "fsdp":
        awp = (
            functools.partial(transformer_auto_wrap_policy, transformer_layer_cls={block_cls})
            if block_cls
            else None
        )
        model = FSDP(
            model,
            auto_wrap_policy=awp,
            process_group=train_group,
            device_id=torch.cuda.current_device(),
        )
    else:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            process_group=train_group,
        )

    dataset = get_dataset(tokenizer, seq_len=cfg.seq_len)
    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(group=train_group),
        rank=dist.get_rank(group=train_group),
        shuffle=True,
    )
    dataloader = DataLoader(dataset, batch_size=4, sampler=sampler, num_workers=2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        sampler.set_epoch(epoch)
        total_reward = 0.0
        epoch_start = time.time()

        for batch_idx, batch_ids in enumerate(dataloader):
            if cfg.max_batches is not None and batch_idx >= cfg.max_batches:
                break

            batch_ids = batch_ids.to(device)
            optimizer.zero_grad()

            outputs = model(batch_ids, labels=batch_ids)
            logits = outputs.logits

            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = batch_ids[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss(reduction="none")
            neg_log_probs = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            ).view(shift_labels.size())

            rewards = compute_reward(batch_ids, tokenizer)
            rewards_expanded = rewards.unsqueeze(1).expand_as(neg_log_probs)
            rl_loss = (neg_log_probs * rewards_expanded).mean()

            rl_loss.backward()
            optimizer.step()
            total_reward += rewards.mean().item()

            if batch_idx % 20 == 0 and rank == 0:
                logging.info(
                    f"Batch {batch_idx}/{len(dataloader)} | "
                    f"RL Loss: {rl_loss.item():.4f} | Avg Reward: {rewards.mean().item():.2f}"
                )

        epoch_duration = time.time() - epoch_start
        if rank == 0:
            avg_reward = total_reward / (batch_idx + 1)
            logging.info(f"--- EPOCH {epoch} STATS ---")
            logging.info(f"Duration: {epoch_duration:.2f}s | Avg Reward: {avg_reward:.4f}")
            logging.info("-------------------------")
            sys.stdout.flush()

        if rank == 0:
            sender_model = prepare_sender_model(cfg, model, cfg.model_key, device)
            if cfg.backend == "broadcast":
                torch_broadcast_model(sender_model, bridge_group, rank, is_sender=True)
            else:
                uccl_transfer_state(sender_model, rank, cfg.inference_rank, is_sender=True)


def run_inference(cfg: RunConfig, rank: int, bridge_group):
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    tokenizer, model, _ = build_model_and_tokenizer(cfg.model_key, device)
    model.eval()

    for epoch in range(1, cfg.epochs + 1):
        logging.info(f"Waiting for model update (Epoch {epoch})...")
        sys.stdout.flush()

        if cfg.backend == "broadcast":
            torch_broadcast_model(model, bridge_group, rank, is_sender=False)
        else:
            uccl_transfer_state(model, rank, cfg.inference_rank, is_sender=False)

        logging.info("Running inference sample...")
        prompt = "The AI scientist discovered"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=30,
                do_sample=True,
                temperature=0.8,
                repetition_penalty=1.2,
            )
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logging.info(f"[INFERENCE EPOCH {epoch}] {generated.replace(chr(10), ' ')}")


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Unified FSDP/DDP runner for Qwen and RL.")
    p.add_argument("--mode", required=True, choices=["fsdp", "ddp"], help="Training parallelism.")
    p.add_argument("--model", required=True, choices=["qwen", "rl"], help="Model family.")
    p.add_argument("--backend", required=True, choices=["broadcast", "uccl"], help="Weight transfer backend.")
    p.add_argument("--train-nodes", type=int, default=2, help="Number of training nodes.")
    p.add_argument("--gpus-per-node", type=int, default=8, help="GPUs per node.")
    p.add_argument("--seq-len", type=int, default=128, help="Sequence length.")
    p.add_argument("--epochs", type=int, default=2, help="Epochs to run.")
    p.add_argument("--max-batches", type=int, default=None, help="Optional cap on batches per epoch.")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = RunConfig(
        mode=args.mode.lower(),
        model_key=args.model.lower(),
        backend=args.backend.lower(),
        train_nodes=args.train_nodes,
        gpus_per_node=args.gpus_per_node,
        seq_len=args.seq_len,
        epochs=args.epochs,
        max_batches=args.max_batches,
    )

    if cfg.backend == "uccl":
        ensure_uccl_available()
        if "NCCL_SOCKET_IFNAME" in os.environ and "GLOO_SOCKET_IFNAME" not in os.environ:
            os.environ["GLOO_SOCKET_IFNAME"] = os.environ["NCCL_SOCKET_IFNAME"]

    init_backend = "gloo" if cfg.backend == "uccl" else "nccl"
    dist.init_process_group(backend=init_backend, init_method="env://", timeout=timedelta(minutes=60))
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    setup_logging(rank)

    if rank == 0:
        logging.info(f"init backend: {init_backend}")
        logging.info(f"NCCL_SOCKET_IFNAME: {os.environ.get('NCCL_SOCKET_IFNAME', 'unset')}")
        logging.info(f"GLOO_SOCKET_IFNAME: {os.environ.get('GLOO_SOCKET_IFNAME', 'unset')}")

    if world_size <= cfg.inference_rank:
        raise SystemExit(
            f"World size {world_size} too small for train_world_size={cfg.train_world_size} "
            f"+ inference rank {cfg.inference_rank}."
        )

    # Build process groups
    train_ranks = list(range(cfg.train_world_size))
    train_group = dist.new_group(ranks=train_ranks, backend="nccl" if cfg.backend == "uccl" else init_backend)
    bridge_group = None
    if cfg.backend == "broadcast":
        bridge_group = dist.new_group(ranks=[0, cfg.inference_rank], backend="nccl")

    if cfg.backend == "uccl":
        from uccl import collective  # type: ignore
        logging.info("Initializing UCCL collective...")
        collective.init_collective(num_cpus=4)

    if rank in train_ranks:
        run_trainer(cfg, rank, train_group, bridge_group)
    elif rank == cfg.inference_rank:
        run_inference(cfg, rank, bridge_group)
    else:
        logging.info("Idling...")

    logging.info("Waiting for all ranks to complete...")
    dist.barrier()

    if cfg.backend == "uccl":
        from uccl import collective  # type: ignore
        collective.finalize_collective()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
