"""
RDMA-based Inference Worker using UCCL P2P
Receives weights from training workers via RDMA

FIXED: Uses GLOO-style send/recv for metadata exchange (like benchmark_uccl.py)
"""

import torch
import torch.distributed as dist
from typing import Dict, List, Optional
import time
import logging
from uccl import p2p


class RDMAInferenceWorker:
    """
    Inference worker that receives model weights using UCCL P2P RDMA
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        rank: int,
        local_rank: int,
        world_size: int,
        training_ranks: List[int],
        num_max_connections: int = 16
    ):
        self.model = model
        self.rank = rank
        self.local_rank = local_rank
        self.world_size = world_size
        self.training_ranks = training_ranks
        
        # Cache parameter dictionary
        self.param_dict = {name: param for name, param in model.named_parameters()}
        
        # Initialize UCCL P2P endpoint
        logging.info(f"[Inference Rank {rank}] Initializing UCCL P2P endpoint on GPU {local_rank}")
        self.ep = p2p.Endpoint(local_rank, num_max_connections)
        self.local_metadata = self.ep.get_metadata()
        
        # Log endpoint info
        ip, port, gpu = p2p.Endpoint.parse_metadata(self.local_metadata)
        logging.info(f"[Inference Rank {rank}] Endpoint: IP={ip}, Port={port}, GPU={gpu}")
        
        # Connection management
        self.connections: Dict[int, int] = {}  # Maps training rank -> conn_id
        self.registered_mrs: Dict[int, int] = {}  # Maps tensor ptr -> mr_id
        
        # Track which parameters to expect from which training rank
        self.param_sources: Dict[str, int] = {}
        
        logging.info(f"[Inference Rank {rank}] UCCL P2P endpoint initialized")
    
    def exchange_metadata(self, all_ranks: List[int], gloo_group=None) -> Dict[int, bytes]:
        """
        Exchange metadata using point-to-point send/recv with GLOO backend
        This matches benchmark_uccl.py which works reliably
        
        CRITICAL: Uses CPU tensors with GLOO backend (not NCCL GPU tensors)
        
        Args:
            all_ranks: List of all ranks participating in metadata exchange
            gloo_group: GLOO process group for CPU tensor communication
        """
        logging.info(f"[Inference Rank {self.rank}] Starting metadata exchange with {len(all_ranks)} ranks (GLOO)...")
        
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
                logging.info(f"[Inference Rank {self.rank}] Got metadata from rank {other_rank}: IP={ip}, Port={port}, GPU={gpu}")
            except Exception as e:
                logging.error(f"[Inference Rank {self.rank}] Failed to parse metadata from rank {other_rank}: {e}")
        
        logging.info(f"[Inference Rank {self.rank}] Metadata exchange complete with {len(all_metadata)} ranks")
        return all_metadata
    
    def setup_connections(self, training_ranks: List[int] = None):
        """Accept RDMA connections from training ranks"""
        if training_ranks is None:
            training_ranks = self.training_ranks
        
        num_expected_connections = len(training_ranks)
        
        logging.info(f"[Inference Rank {self.rank}] Waiting for {num_expected_connections} connections...")
        
        dist.barrier()
        logging.info(f"[Inference Rank {self.rank}] Starting connection acceptance")
        
        for i in range(num_expected_connections):
            expected_train_rank = training_ranks[i]
            
            ok, r_ip, r_gpu, conn_id = self.ep.accept()
            
            if not ok:
                raise RuntimeError(f"Failed to accept connection {i+1}/{num_expected_connections}")
            
            self.connections[expected_train_rank] = conn_id
            
            logging.info(f"[Inference Rank {self.rank}] Accepted {i+1}/{num_expected_connections}: "
                        f"IP={r_ip}, GPU={r_gpu}, conn_id={conn_id}, mapped to Training Rank {expected_train_rank}")
        
        dist.barrier()
        logging.info(f"[Inference Rank {self.rank}] All connections accepted")
        
        for train_rank, conn_id in sorted(self.connections.items()):
            logging.info(f"  Training Rank {train_rank} -> conn_id {conn_id}")
    
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
    
    def set_param_sources(self, param_sources: Dict[str, int]):
        """Set which training rank will send each parameter"""
        self.param_sources = param_sources
        logging.info(f"[Inference Rank {self.rank}] Parameter sources set: {len(param_sources)} parameters")
    
    def receive_weights(self) -> float:
        """Receive weight updates via RDMA"""
        start_time = time.perf_counter()
        
        total_params = len(self.param_sources) if self.param_sources else len(self.param_dict)
        logging.info(f"[Inference Rank {self.rank}] Receiving {total_params} parameters via RDMA...")
        
        received_count = 0
        total_bytes = 0
        
        if self.param_sources:
            for param_name, src_rank in self.param_sources.items():
                if param_name not in self.param_dict:
                    logging.warning(f"[Inference Rank {self.rank}] Unexpected parameter {param_name}")
                    continue
                
                param = self.param_dict[param_name]
                
                if not param.is_cuda:
                    param = param.cuda()
                
                mr_id = self.register_memory_region(param.data)
                
                if src_rank not in self.connections:
                    logging.error(f"[Inference Rank {self.rank}] No connection from rank {src_rank}")
                    continue
                
                conn_id = self.connections[src_rank]
                ptr = param.data.data_ptr()
                size_bytes = param.numel() * param.element_size()
                
                ok = self.ep.recv(conn_id, mr_id, ptr, size_bytes)
                
                if not ok:
                    logging.error(f"[Inference Rank {self.rank}] RDMA recv failed for {param_name}")
                else:
                    received_count += 1
                    total_bytes += size_bytes
        else:
            for param_name, param in self.param_dict.items():
                recv_tensor = torch.empty_like(param.data, device="cuda")
                mr_id = self.register_memory_region(recv_tensor)
                
                if not self.connections:
                    logging.error(f"[Inference Rank {self.rank}] No connections available")
                    break
                
                conn_id = list(self.connections.values())[0]
                ptr = recv_tensor.data_ptr()
                size_bytes = recv_tensor.numel() * recv_tensor.element_size()
                
                ok = self.ep.recv(conn_id, mr_id, ptr, size_bytes)
                
                if ok:
                    param.data.copy_(recv_tensor)
                    received_count += 1
                    total_bytes += size_bytes
        
        receive_time = time.perf_counter() - start_time
        bandwidth_mbps = (total_bytes / receive_time / 1e6) if receive_time > 0 else 0
        logging.info(f"[Inference Rank {self.rank}] Received {received_count} parameters in "
                    f"{receive_time:.2f}s ({bandwidth_mbps:.2f} MB/s)")
        
        return receive_time
    
    @torch.no_grad()
    def run_inference(self, input_ids: torch.Tensor, **kwargs) -> torch.Tensor:
        """Run inference with the current model weights"""
        self.model.eval()
        
        if input_ids.device != next(self.model.parameters()).device:
            input_ids = input_ids.to(next(self.model.parameters()).device)
        
        outputs = self.model.generate(
            input_ids=input_ids,
            max_length=kwargs.get('max_length', 100),
            do_sample=kwargs.get('do_sample', True),
            temperature=kwargs.get('temperature', 1.0),
            top_p=kwargs.get('top_p', 0.9),
        )
        
        return outputs
    
    def verify_weights_updated(self, previous_params: Dict[str, torch.Tensor]) -> bool:
        """Verify that weights were actually updated"""
        changed = 0
        unchanged = 0
        
        for name, param in self.param_dict.items():
            if name in previous_params:
                if not torch.equal(param.data, previous_params[name]):
                    changed += 1
                else:
                    unchanged += 1
        
        logging.info(f"[Inference Rank {self.rank}] Weight verification: {changed} changed, {unchanged} unchanged")
        return changed > 0