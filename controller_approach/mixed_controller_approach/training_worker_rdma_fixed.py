"""
RDMA-based Training Worker using UCCL P2P
Sends weights to inference workers via RDMA

FIXED: Uses GLOO-style send/recv for metadata exchange (like benchmark_uccl.py)
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
        
        # Log endpoint info
        ip, port, gpu = p2p.Endpoint.parse_metadata(self.local_metadata)
        logging.info(f"[Trainer Rank {rank}] Endpoint: IP={ip}, Port={port}, GPU={gpu}")
        
        # Connection management
        self.connections: Dict[int, int] = {}  # Maps inference rank -> conn_id
        self.registered_mrs: Dict[int, int] = {}  # Maps tensor ptr -> mr_id
        
        logging.info(f"[Trainer Rank {rank}] UCCL P2P endpoint initialized")
    
    def exchange_metadata(self, all_ranks: List[int], gloo_group=None) -> Dict[int, bytes]:
        """
        Exchange metadata using point-to-point send/recv with GLOO backend
        This matches benchmark_uccl.py which works reliably
        
        CRITICAL: Uses CPU tensors with GLOO backend (not NCCL GPU tensors)
        
        Args:
            all_ranks: List of all ranks participating in metadata exchange
            gloo_group: GLOO process group for CPU tensor communication
        """
        logging.info(f"[Trainer Rank {self.rank}] Starting metadata exchange with {len(all_ranks)} ranks (GLOO)...")
        
        all_metadata = {}
        local_metadata_list = list(self.local_metadata)
        metadata_len = len(local_metadata_list)
        
        # Store our own metadata
        all_metadata[self.rank] = self.local_metadata
        
        # Exchange with each rank using send/recv on CPU tensors (like benchmark_uccl.py)
        for other_rank in all_ranks:
            if other_rank == self.rank:
                continue
            
            # Use CPU tensors - this is the key difference from broken NCCL approach
            if self.rank < other_rank:
                # Send first, then receive
                send_tensor = torch.ByteTensor(local_metadata_list)
                dist.send(send_tensor, dst=other_rank, group=gloo_group)
                
                recv_tensor = torch.zeros(metadata_len, dtype=torch.uint8)
                dist.recv(recv_tensor, src=other_rank, group=gloo_group)
            else:
                # Receive first, then send
                recv_tensor = torch.zeros(metadata_len, dtype=torch.uint8)
                dist.recv(recv_tensor, src=other_rank, group=gloo_group)
                
                send_tensor = torch.ByteTensor(local_metadata_list)
                dist.send(send_tensor, dst=other_rank, group=gloo_group)
            
            # Convert to bytes
            all_metadata[other_rank] = bytes(recv_tensor.tolist())
            
            # Debug: verify metadata is correct
            try:
                ip, port, gpu = p2p.Endpoint.parse_metadata(all_metadata[other_rank])
                logging.info(f"[Trainer Rank {self.rank}] Got metadata from rank {other_rank}: IP={ip}, Port={port}, GPU={gpu}")
            except Exception as e:
                logging.error(f"[Trainer Rank {self.rank}] Failed to parse metadata from rank {other_rank}: {e}")
        
        logging.info(f"[Trainer Rank {self.rank}] Metadata exchange complete with {len(all_metadata)} ranks")
        return all_metadata
    
    def setup_connections(
        self,
        inference_ranks: List[int],
        all_metadata: Dict[int, bytes]
    ):
        """
        Setup RDMA connections to inference ranks with staggering
        """
        logging.info(f"[Trainer Rank {self.rank}] Setting up connections to inference ranks...")
        
        # Synchronize all ranks before starting connections
        dist.barrier()
        logging.info(f"[Trainer Rank {self.rank}] Starting staggered connection setup")
        
        # Stagger based on (rank % num_inference_ranks) to spread connections
        num_inference = len(inference_ranks)
        stagger_group = self.rank % num_inference
        connection_delay = 2.0
        
        logging.info(f"[Trainer Rank {self.rank}] Stagger group {stagger_group}, waiting {stagger_group * connection_delay}s...")
        time.sleep(stagger_group * connection_delay)
        
        for infer_rank in inference_ranks:
            # Parse remote metadata
            ip, port, r_gpu = p2p.Endpoint.parse_metadata(all_metadata[infer_rank])
            
            logging.info(f"[Trainer Rank {self.rank}] Connecting to inference rank {infer_rank}: "
                        f"IP={ip}, Port={port}, GPU={r_gpu}")
            
            # Connect with retries
            max_retries = 15
            ok = False
            for attempt in range(max_retries):
                ok, conn_id = self.ep.connect(ip, r_gpu, remote_port=port)
                if ok:
                    self.connections[infer_rank] = conn_id
                    logging.info(f"[Trainer Rank {self.rank}] Connected to inference rank {infer_rank}, conn_id={conn_id}")
                    break
                else:
                    if attempt < max_retries - 1:
                        wait_time = 3.0 + (attempt * 0.5)
                        logging.warning(f"[Trainer Rank {self.rank}] Connection attempt {attempt+1}/{max_retries} failed, "
                                      f"retrying in {wait_time}s...")
                        time.sleep(wait_time)
            
            if not ok:
                raise RuntimeError(f"Failed to connect to inference rank {infer_rank} after {max_retries} attempts")
            
            time.sleep(0.5)
        
        dist.barrier()
        logging.info(f"[Trainer Rank {self.rank}] All connections established")
    
    def set_routing_table(self, routing_table: RoutingTable):
        """Set the routing table computed by the controller"""
        self.routing_table = routing_table
        logging.info(f"[Trainer Rank {self.rank}] Routing table set: {len(routing_table.transfers)} transfers")
    
    def register_memory_region(self, tensor: torch.Tensor) -> int:
        """Register a tensor's memory for RDMA"""
        ptr = tensor.data_ptr()
        
        if ptr in self.registered_mrs:
            return self.registered_mrs[ptr]
        
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
        """Execute weight transfer using RDMA according to routing table"""
        if self.routing_table is None:
            raise RuntimeError("Routing table not set. Call set_routing_table() first.")
        
        if len(self.routing_table.transfers) == 0:
            logging.info(f"[Trainer Rank {self.rank}] No transfers to perform")
            return 0.0
        
        start_time = time.perf_counter()
        
        logging.info(f"[Trainer Rank {self.rank}] Starting {len(self.routing_table.transfers)} RDMA transfers...")
        
        for entry in self.routing_table.transfers:
            if entry.param_name not in self.param_dict:
                logging.warning(f"[Trainer Rank {self.rank}] Parameter {entry.param_name} not found")
                continue
            
            param = self.param_dict[entry.param_name]
            
            if not param.is_cuda:
                param = param.cuda()
            tensor = param.data.contiguous()
            
            mr_id = self.register_memory_region(tensor)
            
            if entry.dst_rank not in self.connections:
                logging.error(f"[Trainer Rank {self.rank}] No connection to rank {entry.dst_rank}")
                continue
            
            conn_id = self.connections[entry.dst_rank]
            ptr = tensor.data_ptr()
            size_bytes = tensor.numel() * tensor.element_size()
            
            ok = self.ep.send(conn_id, mr_id, ptr, size_bytes)
            
            if not ok:
                logging.error(f"[Trainer Rank {self.rank}] RDMA send failed for {entry.param_name}")
        
        transfer_time = time.perf_counter() - start_time
        bandwidth_mbps = (self.routing_table.total_bytes / transfer_time / 1e6)
        logging.info(f"[Trainer Rank {self.rank}] Completed transfers in {transfer_time:.2f}s ({bandwidth_mbps:.2f} MB/s)")
        
        return transfer_time


class OptimizedRDMATrainingWorker(RDMATrainingWorker):
    """Optimized RDMA training worker with pipelining"""
    
    def __init__(
        self,
        model: torch.nn.Module,
        rank: int,
        local_rank: int,
        world_size: int,
        num_max_connections: int = 16
    ):
        super().__init__(model, rank, local_rank, world_size, num_max_connections)
        self.pipeline_batch_size = 4
        self.max_tmp_bytes = 2 << 30
    
    def transfer_weights_pipelined(self) -> float:
        """Transfer weights with pipelining"""
        if self.routing_table is None or len(self.routing_table.transfers) == 0:
            return 0.0
        
        start_time = time.perf_counter()
        logging.info(f"[Trainer Rank {self.rank}] Starting pipelined RDMA transfer...")
        
        transfers = self.routing_table.transfers
        
        for batch_start in range(0, len(transfers), self.pipeline_batch_size):
            batch = transfers[batch_start:batch_start + self.pipeline_batch_size]
            
            prepared = []
            for entry in batch:
                if entry.param_name not in self.param_dict:
                    continue
                
                param = self.param_dict[entry.param_name]
                if not param.is_cuda:
                    param = param.cuda()
                
                tensor = param.data.contiguous()
                mr_id = self.register_memory_region(tensor)
                prepared.append((entry, tensor, mr_id))
            
            for entry, tensor, mr_id in prepared:
                if entry.dst_rank not in self.connections:
                    continue
                
                conn_id = self.connections[entry.dst_rank]
                ptr = tensor.data_ptr()
                size_bytes = tensor.numel() * tensor.element_size()
                
                ok = self.ep.send(conn_id, mr_id, ptr, size_bytes)
                if not ok:
                    logging.error(f"[Trainer Rank {self.rank}] RDMA send failed for {entry.param_name}")
        
        transfer_time = time.perf_counter() - start_time
        bandwidth_mbps = (self.routing_table.total_bytes / transfer_time / 1e6)
        logging.info(f"[Trainer Rank {self.rank}] Pipelined transfer completed in {transfer_time:.2f}s ({bandwidth_mbps:.2f} MB/s)")
        
        return transfer_time