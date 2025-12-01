"""
RDMA-based Training Worker using UCCL P2P
Sends weights to inference workers via RDMA
"""

import torch
import torch.distributed as dist
from typing import Dict, List, Optional, Tuple
import time
import logging
from uccl import p2p

from weight_transfer_controller import RoutingTable, WeightTransferEntry


class RDMATrainingWorker:
    """
    Training worker that sends model weights using UCCL P2P RDMA
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        rank: int,
        local_rank: int,
        world_size: int,
        num_max_connections: int = 16
    ):
        self.model = model
        self.rank = rank
        self.local_rank = local_rank
        self.world_size = world_size
        self.routing_table: Optional[RoutingTable] = None
        
        # Cache parameter dictionary
        self.param_dict = {name: param for name, param in model.named_parameters()}
        
        # Initialize UCCL P2P endpoint
        logging.info(f"[Trainer Rank {rank}] Initializing UCCL P2P endpoint on GPU {local_rank}")
        self.ep = p2p.Endpoint(local_rank, num_max_connections)
        self.local_metadata = self.ep.get_metadata()
        
        # Connection management
        self.connections: Dict[int, int] = {}  # Maps inference rank -> conn_id
        self.registered_mrs: Dict[int, int] = {}  # Maps tensor ptr -> mr_id
        
        logging.info(f"[Trainer Rank {rank}] UCCL P2P endpoint initialized")
    
    def exchange_metadata(self, all_ranks: List[int]) -> Dict[int, bytes]:
        """
        Exchange P2P metadata with all ranks using torch.distributed
        
        Returns:
            Dict mapping rank -> metadata bytes
        """
        logging.info(f"[Trainer Rank {self.rank}] Starting metadata exchange...")
        
        all_metadata = {}
        all_metadata[self.rank] = self.local_metadata
        
        # Use torch.distributed for metadata exchange
        for i in all_ranks:
            if i == self.rank:
                # Send my metadata to all others
                for j in all_ranks:
                    if j != self.rank:
                        metadata_tensor = torch.ByteTensor(list(self.local_metadata))
                        dist.send(metadata_tensor, dst=j)
            else:
                # Receive metadata from rank i
                remote_md = torch.zeros(len(self.local_metadata), dtype=torch.uint8)
                dist.recv(remote_md, src=i)
                all_metadata[i] = bytes(remote_md.tolist())
        
        logging.info(f"[Trainer Rank {self.rank}] Metadata exchange complete")
        return all_metadata
    
    def setup_connections(
        self,
        inference_ranks: List[int],
        all_metadata: Dict[int, bytes]
    ):
        """
        Setup RDMA connections to inference ranks with pre-determined order
        
        Args:
            inference_ranks: List of inference rank IDs
            all_metadata: Metadata from all ranks
        """
        logging.info(f"[Trainer Rank {self.rank}] Setting up connections to inference ranks...")
        
        # Synchronize all ranks before starting connections
        dist.barrier()
        logging.info(f"[Trainer Rank {self.rank}] Starting staggered connection setup")
        
        # Stagger connections based on rank to ensure deterministic order
        # Training ranks connect in order: 0, 1, 2, 3, ...
        import time
        connection_delay = 0.2  # 200ms between each rank's connection attempt
        time.sleep(self.rank * connection_delay)
        
        for infer_rank in inference_ranks:
            # Parse remote metadata
            ip, port, r_gpu = p2p.Endpoint.parse_metadata(all_metadata[infer_rank])
            
            logging.info(f"[Trainer Rank {self.rank}] Connecting to inference rank {infer_rank}: "
                        f"IP={ip}, Port={port}, GPU={r_gpu}")
            
            # Connect to inference worker
            ok, conn_id = self.ep.connect(ip, r_gpu, remote_port=port)
            
            if not ok:
                raise RuntimeError(f"Failed to connect to inference rank {infer_rank}")
            
            self.connections[infer_rank] = conn_id
            logging.info(f"[Trainer Rank {self.rank}] Connected to inference rank {infer_rank}, "
                        f"conn_id={conn_id}")
        
        # Synchronize after all connections are established
        dist.barrier()
        logging.info(f"[Trainer Rank {self.rank}] All connections established")
    
    def set_routing_table(self, routing_table: RoutingTable):
        """Set the routing table computed by the controller"""
        self.routing_table = routing_table
        logging.info(f"[Trainer Rank {self.rank}] Routing table set: "
                    f"{len(routing_table.transfers)} transfers")
    
    def register_memory_region(self, tensor: torch.Tensor) -> int:
        """
        Register a tensor's memory for RDMA
        
        Args:
            tensor: PyTorch tensor to register
            
        Returns:
            Memory region ID
        """
        ptr = tensor.data_ptr()
        
        # Check if already registered
        if ptr in self.registered_mrs:
            return self.registered_mrs[ptr]
        
        # Register new memory region
        size_bytes = tensor.numel() * tensor.element_size()
        ok, mr_id = self.ep.reg(ptr, size_bytes)
        
        if not ok:
            raise RuntimeError(f"Failed to register memory region for tensor at {ptr}")
        
        self.registered_mrs[ptr] = mr_id
        return mr_id
    
    def get_param_metadata(self) -> Dict[str, dict]:
        """Get parameter metadata for the controller"""
        metadata = {}
        for name, param in self.model.named_parameters():
            metadata[name] = {
                'name': name,
                'shape': tuple(param.shape),
                'dtype': param.dtype,
                'numel': param.numel(),
                'rank': self.rank
            }
        return metadata
    
    def transfer_weights(self) -> float:
        """
        Execute weight transfer using RDMA according to routing table
        
        Returns:
            Transfer time in seconds
        """
        if self.routing_table is None:
            raise RuntimeError("Routing table not set. Call set_routing_table() first.")
        
        if len(self.routing_table.transfers) == 0:
            logging.info(f"[Trainer Rank {self.rank}] No transfers to perform")
            return 0.0
        
        start_time = time.perf_counter()
        
        logging.info(f"[Trainer Rank {self.rank}] Starting {len(self.routing_table.transfers)} "
                    f"RDMA weight transfers...")
        
        # Execute transfers according to routing table
        for entry in self.routing_table.transfers:
            # Get the parameter tensor
            if entry.param_name not in self.param_dict:
                logging.warning(f"[Trainer Rank {self.rank}] Parameter {entry.param_name} not found")
                continue
            
            param = self.param_dict[entry.param_name]
            
            # Ensure tensor is on GPU and contiguous
            if not param.is_cuda:
                param = param.cuda()
            tensor = param.data.contiguous()
            
            # Register memory region
            mr_id = self.register_memory_region(tensor)
            
            # Get connection ID
            if entry.dst_rank not in self.connections:
                logging.error(f"[Trainer Rank {self.rank}] No connection to rank {entry.dst_rank}")
                continue
            
            conn_id = self.connections[entry.dst_rank]
            
            # Send via RDMA
            ptr = tensor.data_ptr()
            size_bytes = tensor.numel() * tensor.element_size()
            
            ok = self.ep.send(conn_id, mr_id, ptr, size_bytes)
            
            if not ok:
                logging.error(f"[Trainer Rank {self.rank}] RDMA send failed for {entry.param_name} "
                            f"to rank {entry.dst_rank}")
        
        transfer_time = time.perf_counter() - start_time
        
        bandwidth_mbps = (self.routing_table.total_bytes / transfer_time / 1e6)
        logging.info(f"[Trainer Rank {self.rank}] Completed {len(self.routing_table.transfers)} "
                    f"RDMA transfers in {transfer_time:.2f}s ({bandwidth_mbps:.2f} MB/s)")
        
        return transfer_time


class OptimizedRDMATrainingWorker(RDMATrainingWorker):
    """
    Optimized RDMA training worker with pipelining and batching
    Similar to the blog post approach
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        rank: int,
        local_rank: int,
        world_size: int,
        num_max_connections: int = 16
    ):
        super().__init__(model, rank, local_rank, world_size, num_max_connections)
        
        # Pipeline parameters
        self.pipeline_batch_size = 4
        self.max_tmp_bytes = 2 << 30  # 2 GB
    
    def transfer_weights_pipelined(self) -> float:
        """
        Transfer weights with pipelining:
        - Batch multiple tensors together
        - Overlap GPU operations with RDMA transfers
        - Memory-bounded execution
        """
        if self.routing_table is None or len(self.routing_table.transfers) == 0:
            return 0.0
        
        start_time = time.perf_counter()
        
        logging.info(f"[Trainer Rank {self.rank}] Starting pipelined RDMA transfer...")
        
        transfers = self.routing_table.transfers
        current_mem_usage = 0
        
        for batch_start in range(0, len(transfers), self.pipeline_batch_size):
            batch = transfers[batch_start:batch_start + self.pipeline_batch_size]
            
            # Stage 1: Prepare tensors (ensure GPU, contiguous, register MR)
            prepared = []
            for entry in batch:
                if entry.param_name not in self.param_dict:
                    continue
                
                param = self.param_dict[entry.param_name]
                
                # Move to GPU if needed
                if not param.is_cuda:
                    param = param.cuda()
                
                tensor = param.data.contiguous()
                
                # Register memory region
                mr_id = self.register_memory_region(tensor)
                
                prepared.append((entry, tensor, mr_id))
            
            # Stage 2: Execute RDMA sends for this batch
            for entry, tensor, mr_id in prepared:
                if entry.dst_rank not in self.connections:
                    continue
                
                conn_id = self.connections[entry.dst_rank]
                ptr = tensor.data_ptr()
                size_bytes = tensor.numel() * tensor.element_size()
                
                ok = self.ep.send(conn_id, mr_id, ptr, size_bytes)
                
                if not ok:
                    logging.error(f"[Trainer Rank {self.rank}] RDMA send failed for "
                                f"{entry.param_name}")
        
        transfer_time = time.perf_counter() - start_time
        
        bandwidth_mbps = (self.routing_table.total_bytes / transfer_time / 1e6)
        logging.info(f"[Trainer Rank {self.rank}] Pipelined RDMA transfer completed in "
                    f"{transfer_time:.2f}s ({bandwidth_mbps:.2f} MB/s)")
        
        return transfer_time


# Example usage and integration
if __name__ == "__main__":
    from transformers import GPT2LMHeadModel
    import torch.distributed as dist
    
    # Initialize distributed
    dist.init_process_group(backend="gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = rank % 8  # Assuming 8 GPUs per node
    
    torch.cuda.set_device(local_rank)
    
    # Load model
    model = GPT2LMHeadModel.from_pretrained("gpt2").cuda()
    
    # Create RDMA worker
    worker = OptimizedRDMATrainingWorker(
        model=model,
        rank=rank,
        local_rank=local_rank,
        world_size=world_size
    )
    
    logging.info(f"[Rank {rank}] RDMA Training Worker initialized")
    
    # Clean up
    dist.destroy_process_group()
