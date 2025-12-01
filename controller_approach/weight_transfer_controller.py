"""
Weight Transfer Controller for Distributed Training/Inference
Implements routing table approach for efficient parameter updates
"""

import torch
import torch.distributed as dist
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import time
from collections import defaultdict


@dataclass
class ParamMeta:
    """Metadata for a single parameter"""
    name: str
    shape: Tuple[int, ...]
    dtype: torch.dtype
    numel: int
    rank: int  # Which GPU rank owns this parameter
    device: str


@dataclass
class WeightTransferEntry:
    """Single weight transfer instruction"""
    param_name: str
    src_rank: int  # Training GPU rank
    dst_rank: int  # Inference GPU rank
    shape: Tuple[int, ...]
    dtype: torch.dtype
    
    def __repr__(self):
        return f"Transfer({self.param_name}: rank{self.src_rank}→rank{self.dst_rank})"


@dataclass
class RoutingTable:
    """Routing table for a single training worker"""
    training_rank: int
    transfers: List[WeightTransferEntry] = field(default_factory=list)
    total_bytes: int = 0
    
    def add_transfer(self, entry: WeightTransferEntry):
        self.transfers.append(entry)
        element_size = torch.tensor([], dtype=entry.dtype).element_size()
        num_elements = 1
        for dim in entry.shape:
            num_elements *= dim
        self.total_bytes += num_elements * element_size
    
    def __repr__(self):
        return f"RoutingTable(rank={self.training_rank}, transfers={len(self.transfers)}, bytes={self.total_bytes/1e9:.2f}GB)"


class WeightTransferController:
    """
    Controller that computes and manages weight transfer routing tables
    """
    
    def __init__(
        self,
        training_ranks: List[int],
        inference_ranks: List[int],
        world_size: int
    ):
        self.training_ranks = training_ranks
        self.inference_ranks = inference_ranks
        self.world_size = world_size
        self.routing_tables: Dict[int, RoutingTable] = {}
        
    def collect_param_metadata(
        self,
        model: torch.nn.Module,
        rank: int
    ) -> Dict[str, ParamMeta]:
        """Collect parameter metadata from a model"""
        metadata = {}
        for name, param in model.named_parameters():
            metadata[name] = ParamMeta(
                name=name,
                shape=tuple(param.shape),
                dtype=param.dtype,
                numel=param.numel(),
                rank=rank,
                device=str(param.device)
            )
        return metadata
    
    def match_parameters(
        self,
        trainer_params: Dict[int, Dict[str, ParamMeta]],
        inference_params: Dict[int, Dict[str, ParamMeta]]
    ) -> List[Tuple[str, List[int], List[int]]]:
        """
        Match parameters between training and inference
        Returns: List of (param_name, training_ranks_with_param, inference_ranks_with_param)
        """
        # For GPT-2, parameters should match exactly by name
        # Get all unique parameter names from training
        all_param_names = set()
        for rank_params in trainer_params.values():
            all_param_names.update(rank_params.keys())
        
        matched = []
        for param_name in sorted(all_param_names):
            # Find which training ranks have this param
            train_ranks = [
                rank for rank, params in trainer_params.items()
                if param_name in params
            ]
            
            # Find which inference ranks need this param
            infer_ranks = [
                rank for rank, params in inference_params.items()
                if param_name in params
            ]
            
            if train_ranks and infer_ranks:
                matched.append((param_name, train_ranks, infer_ranks))
        
        return matched
    
    def compute_routing_tables(
        self,
        trainer_params: Dict[int, Dict[str, ParamMeta]],
        inference_params: Dict[int, Dict[str, ParamMeta]]
    ) -> Dict[int, RoutingTable]:
        """
        Compute routing tables for all training workers
        Uses load balancing to distribute transfers evenly
        """
        # Initialize routing tables
        routing_tables = {
            rank: RoutingTable(training_rank=rank)
            for rank in self.training_ranks
        }
        
        # Track bytes sent by each training rank (for load balancing)
        bytes_sent = {rank: 0 for rank in self.training_ranks}
        
        # Match parameters
        matched_params = self.match_parameters(trainer_params, inference_params)
        
        print(f"\n=== Matched {len(matched_params)} parameters ===")
        
        # For each parameter, assign transfers
        for param_name, train_ranks, infer_ranks in matched_params:
            # Get parameter metadata from first training rank that has it
            param_meta = trainer_params[train_ranks[0]][param_name]
            
            # For each inference rank that needs this parameter
            for infer_rank in infer_ranks:
                # Choose training rank with minimum bytes sent (load balancing)
                src_rank = min(
                    train_ranks,
                    key=lambda r: bytes_sent[r]
                )
                
                # Create transfer entry
                entry = WeightTransferEntry(
                    param_name=param_name,
                    src_rank=src_rank,
                    dst_rank=infer_rank,
                    shape=param_meta.shape,
                    dtype=param_meta.dtype
                )
                
                # Add to routing table
                routing_tables[src_rank].add_transfer(entry)
                
                # Update load balancing tracker
                bytes_sent[src_rank] += routing_tables[src_rank].transfers[-1].total_bytes
        
        # Print routing table summary
        print("\n=== Routing Table Summary ===")
        for rank in sorted(routing_tables.keys()):
            table = routing_tables[rank]
            print(f"Training Rank {rank}: {len(table.transfers)} transfers, {table.total_bytes/1e6:.2f} MB")
        
        total_bytes = sum(t.total_bytes for t in routing_tables.values())
        print(f"Total transfer: {total_bytes/1e6:.2f} MB")
        
        return routing_tables
    
    def print_routing_details(self, routing_tables: Dict[int, RoutingTable]):
        """Print detailed routing information"""
        print("\n=== Detailed Routing Plan ===")
        for rank, table in sorted(routing_tables.items()):
            print(f"\nTraining Rank {rank}:")
            for entry in table.transfers[:5]:  # Show first 5
                print(f"  {entry}")
            if len(table.transfers) > 5:
                print(f"  ... and {len(table.transfers) - 5} more transfers")


def simulate_parameter_distribution(
    model: torch.nn.Module,
    training_ranks: List[int],
    inference_ranks: List[int],
    use_data_parallel: bool = True
) -> Tuple[Dict[int, Dict[str, ParamMeta]], Dict[int, Dict[str, ParamMeta]]]:
    """
    Simulate how parameters would be distributed across ranks
    
    For simplicity:
    - Training: If use_data_parallel=True, all ranks have full model (replicated)
    - Inference: All ranks have full model (replicated)
    """
    trainer_params = {}
    inference_params = {}
    
    # Collect parameter metadata
    param_dict = {}
    for name, param in model.named_parameters():
        param_dict[name] = ParamMeta(
            name=name,
            shape=tuple(param.shape),
            dtype=param.dtype,
            numel=param.numel(),
            rank=-1,  # Will be set per rank
            device="cuda"
        )
    
    # Distribute to training ranks (replicated for DP)
    if use_data_parallel:
        for rank in training_ranks:
            trainer_params[rank] = {
                name: ParamMeta(
                    name=name,
                    shape=meta.shape,
                    dtype=meta.dtype,
                    numel=meta.numel(),
                    rank=rank,
                    device=f"cuda:{rank}"
                )
                for name, meta in param_dict.items()
            }
    
    # Distribute to inference ranks (replicated)
    for rank in inference_ranks:
        inference_params[rank] = {
            name: ParamMeta(
                name=name,
                shape=meta.shape,
                dtype=meta.dtype,
                numel=meta.numel(),
                rank=rank,
                device=f"cuda:{rank}"
            )
            for name, meta in param_dict.items()
        }
    
    return trainer_params, inference_params


# Example usage
if __name__ == "__main__":
    from transformers import GPT2LMHeadModel, GPT2Config
    
    # Setup: 4 nodes × 8 GPUs = 32 total ranks
    # Node 0 (ranks 0-7): Controller + Training
    # Node 1 (ranks 8-15): Training
    # Node 2 (ranks 16-23): Training (we'll use node 0-1 for training)
    # Node 3 (ranks 24-31): Inference
    
    # For this example:
    # Training ranks: 0-15 (nodes 0-1)
    # Inference ranks: 24-31 (node 3)
    
    training_ranks = list(range(0, 16))  # 16 training GPUs
    inference_ranks = list(range(24, 32))  # 8 inference GPUs
    world_size = 32
    
    print("=== Weight Transfer Controller Demo ===")
    print(f"Training ranks: {training_ranks}")
    print(f"Inference ranks: {inference_ranks}")
    
    # Create a GPT-2 model
    config = GPT2Config.from_pretrained("gpt2")
    model = GPT2LMHeadModel(config)
    
    print(f"\nModel: GPT-2 with {sum(p.numel() for p in model.parameters())/1e6:.1f}M parameters")
    
    # Simulate parameter distribution
    trainer_params, inference_params = simulate_parameter_distribution(
        model, training_ranks, inference_ranks, use_data_parallel=True
    )
    
    # Create controller
    controller = WeightTransferController(
        training_ranks=training_ranks,
        inference_ranks=inference_ranks,
        world_size=world_size
    )
    
    # Compute routing tables
    routing_tables = controller.compute_routing_tables(
        trainer_params, inference_params
    )
    
    # Print detailed routing
    controller.print_routing_details(routing_tables)
    
    print("\n=== Controller Setup Complete ===")
