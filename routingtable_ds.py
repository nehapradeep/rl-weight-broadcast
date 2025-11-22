from __future__ import annotations
import torch, time, os, sys
import torch.distributed as dist
import logging
from datetime import datetime
import math
from dataclasses import dataclass
from typing import Dict, Tuple, Any

from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader
from uccl import p2p

@dataclass(slots=True, frozen=True)
class Identity:
    base_name: str

    @property
    def weight_name(self) -> str:
        return self.base_name + ".weight"


@dataclass(slots=True, frozen=True)
class Quantization:
    base_name: str
    scale_suffix: str

    @property
    def weight_name(self) -> str:
        return self.base_name + ".weight"

    @property
    def scale_name(self) -> str:
        return self.base_name + self.scale_suffix

@dataclass(slots=True, frozen=True)
class WeightNameMapping:
    trainer: Identity | Quantization | ProjectionFusion
    rollout: Identity | Quantization
    do_quant: bool

@dataclass
class WeightTransferEntry:
    src_ptr: int
    src_size: int
    dst_ptr: int
    dst_size: int
    dst_mr: MemoryRegion

@dataclass(slots=True)
class WeightTransferEntry:
    name_mapping: WeightNameMapping
    rollout_workers: tuple[int, ...]

@dataclass(slots=True)
class WeightTransferGroup:
    mesh_group: set[Mesh]
    transfer_entries: list[WeightTransferEntry]

@dataclass(slots=True)
class WeightTransferRoutingTable:
    groups: list[WeightTransferGroup]

@dataclass(slots=True)
class WeightTransferSchedule:
    trainers: list[WeightTransferRoutingTable]
