"""
RDMA-based Inference Worker using UCCL P2P
Receives weights from training workers via RDMA
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
        
        # Connection management
        self.connections: Dict[int, int] = {}  # Maps training rank -> conn_id
        self.registered_mrs: Dict[int, int] = {}  # Maps tensor ptr -> mr_id
        
        # Track which parameters to expect from which training rank
        self.param_sources: Dict[str, int] = {}  # param_name -> training_rank
        
        logging.info(f"[Inference Rank {rank}] UCCL P2P endpoint initialized")
    
    def exchange_metadata(self, all_ranks: List[int], metadata_group=None) -> Dict[int, bytes]:
        """Non-blocking metadata exchange using all_gather"""
        logging.info(f"[Inference Rank {self.rank}] Starting metadata exchange...")
        
        # Convert metadata to tensor and move to GPU for NCCL
        device = torch.device(f"cuda:{self.local_rank}")
        metadata_tensor = torch.ByteTensor(list(self.local_metadata)).to(device)
        
        # Gather all metadata at once (non-blocking collective)
        gathered = [torch.zeros_like(metadata_tensor).to(device) for _ in all_ranks]
        
        # Use metadata_group if provided, otherwise default group
        if metadata_group is not None:
            dist.all_gather(gathered, metadata_tensor, group=metadata_group)
        else:
            dist.all_gather(gathered, metadata_tensor)
        
        # Convert back to dict (move to CPU for processing)
        all_metadata = {
            rank: bytes(tensor.cpu().tolist()) 
            for rank, tensor in zip(all_ranks, gathered)
        }
        
        logging.info(f"[Inference Rank {self.rank}] Metadata exchange complete")
        return all_metadata
    
    def setup_connections(self, training_ranks: List[int] = None):
        """
        Accept RDMA connections from training ranks in pre-determined order
        
        Args:
            training_ranks: List of training ranks to accept from (in order)
        """
        if training_ranks is None:
            training_ranks = self.training_ranks
        
        num_expected_connections = len(training_ranks)
        
        logging.info(f"[Inference Rank {self.rank}] Waiting for {num_expected_connections} "
                    f"connections from training ranks...")
        
        # Synchronize all ranks before starting connections
        dist.barrier()
        logging.info(f"[Inference Rank {self.rank}] Starting connection acceptance")
        
        # Accept connections in order - training ranks connect in order (0, 1, 2, 3, ...)
        # So we can safely map the i-th accepted connection to training_ranks[i]
        for i in range(num_expected_connections):
            expected_train_rank = training_ranks[i]
            
            # Accept connection
            ok, r_ip, r_gpu, conn_id = self.ep.accept()
            
            if not ok:
                raise RuntimeError(f"Failed to accept connection {i+1}/{num_expected_connections}")
            
            # Map this connection to the expected training rank
            # This works because training ranks connect in deterministic order
            self.connections[expected_train_rank] = conn_id
            
            logging.info(f"[Inference Rank {self.rank}] Accepted connection {i+1}/{num_expected_connections}: "
                        f"IP={r_ip}, GPU={r_gpu}, conn_id={conn_id}, "
                        f"mapped to Training Rank {expected_train_rank}")
        
        # Synchronize after all connections are accepted
        dist.barrier()
        logging.info(f"[Inference Rank {self.rank}] All connections accepted and mapped")
        
        # Log the complete connection mapping
        logging.info(f"[Inference Rank {self.rank}] Connection mapping:")
        for train_rank, conn_id in sorted(self.connections.items()):
            logging.info(f"  Training Rank {train_rank} -> conn_id {conn_id}")
    
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
    
    def set_param_sources(self, param_sources: Dict[str, int]):
        """
        Set which training rank will send each parameter
        
        Args:
            param_sources: Dict mapping parameter name -> training rank
        """
        self.param_sources = param_sources
        logging.info(f"[Inference Rank {self.rank}] Parameter sources set: "
                    f"{len(param_sources)} parameters")
    
    def receive_weights(self) -> float:
        """
        Receive weight updates via RDMA
        
        Returns:
            Time taken to receive weights in seconds
        """
        start_time = time.perf_counter()
        
        total_params = len(self.param_sources) if self.param_sources else len(self.param_dict)
        logging.info(f"[Inference Rank {self.rank}] Waiting to receive {total_params} parameters "
                    f"via RDMA...")
        
        received_count = 0
        total_bytes = 0
        
        # Receive based on param_sources if available
        if self.param_sources:
            for param_name, src_rank in self.param_sources.items():
                if param_name not in self.param_dict:
                    logging.warning(f"[Inference Rank {self.rank}] Unexpected parameter {param_name}")
                    continue
                
                param = self.param_dict[param_name]
                
                # Ensure parameter is on GPU
                if not param.is_cuda:
                    param = param.cuda()
                
                # Register memory region
                mr_id = self.register_memory_region(param.data)
                
                # Get connection ID
                if src_rank not in self.connections:
                    logging.error(f"[Inference Rank {self.rank}] No connection from rank {src_rank}")
                    continue
                
                conn_id = self.connections[src_rank]
                
                # Receive via RDMA
                ptr = param.data.data_ptr()
                size_bytes = param.numel() * param.element_size()
                
                ok = self.ep.recv(conn_id, mr_id, ptr, size_bytes)
                
                if not ok:
                    logging.error(f"[Inference Rank {self.rank}] RDMA recv failed for {param_name} "
                                f"from rank {src_rank}")
                else:
                    received_count += 1
                    total_bytes += size_bytes
        
        else:
            # Receive all parameters (assume ordered by state_dict)
            for param_name, param in self.param_dict.items():
                # Create receive buffer
                recv_tensor = torch.empty_like(param.data, device="cuda")
                
                # Register memory region
                mr_id = self.register_memory_region(recv_tensor)
                
                # Receive from any training rank (use first connection)
                if not self.connections:
                    logging.error(f"[Inference Rank {self.rank}] No connections available")
                    break
                
                conn_id = list(self.connections.values())[0]
                
                ptr = recv_tensor.data_ptr()
                size_bytes = recv_tensor.numel() * recv_tensor.element_size()
                
                ok = self.ep.recv(conn_id, mr_id, ptr, size_bytes)
                
                if ok:
                    # Copy received tensor to parameter
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
        """
        Run inference with the current model weights
        
        Args:
            input_ids: Input token IDs
            **kwargs: Additional arguments for generation
        
        Returns:
            Generated token IDs
        """
        self.model.eval()
        
        # Move input to correct device
        if input_ids.device != next(self.model.parameters()).device:
            input_ids = input_ids.to(next(self.model.parameters()).device)
        
        # Generate
        outputs = self.model.generate(
            input_ids=input_ids,
            max_length=kwargs.get('max_length', 100),
            do_sample=kwargs.get('do_sample', True),
            temperature=kwargs.get('temperature', 1.0),
            top_p=kwargs.get('top_p', 0.9),
        )
        
        return outputs
    
    def verify_weights_updated(self, previous_params: Dict[str, torch.Tensor]) -> bool:
        """
        Verify that weights were actually updated
        
        Args:
            previous_params: Dictionary of parameter tensors before update
        
        Returns:
            True if weights changed, False otherwise
        """
        changed = 0
        unchanged = 0
        
        for name, param in self.param_dict.items():
            if name in previous_params:
                if not torch.equal(param.data, previous_params[name]):
                    changed += 1
                else:
                    unchanged += 1
        
        logging.info(f"[Inference Rank {self.rank}] Weight update verification: "
                    f"{changed} changed, {unchanged} unchanged")
        
        return changed > 0


class BroadcasterBasedInferenceWorker(RDMAInferenceWorker):
    """
    Inference worker that follows the broadcaster pattern from your uploaded file
    Simpler connection model where broadcaster connects to all receivers
    """
    
    def receive_model_from_broadcaster(self, broadcaster_rank: int) -> float:
        """
        Receive entire model from broadcaster (rank 0 pattern)
        Similar to the uploaded gpu_transfer_wikitext2.py pattern
        
        Args:
            broadcaster_rank: Rank of the broadcaster
            
        Returns:
            Time taken to receive model
        """
        logging.info(f"[Inference Rank {self.rank}] Waiting for broadcaster connection...")
        
        # Accept connection from broadcaster
        ok, r_ip, r_gpu, conn_id = self.ep.accept()
        if not ok:
            raise RuntimeError("Failed to accept broadcaster connection")
        
        logging.info(f"[Inference Rank {self.rank}] Connected to broadcaster: "
                    f"IP={r_ip}, GPU={r_gpu}, conn_id={conn_id}")
        
        # Receive all model parameters
        start_time = time.perf_counter()
        
        state_dict = self.model.state_dict()
        total_tensors = len(list(state_dict.items()))
        total_bytes = 0
        
        logging.info(f"[Inference Rank {self.rank}] Receiving {total_tensors} tensors...")
        
        for idx, (name, param) in enumerate(state_dict.items(), 1):
            # Create receive buffer
            recv_tensor = torch.empty_like(param, device="cuda")
            size_bytes = recv_tensor.numel() * recv_tensor.element_size()
            ptr = recv_tensor.data_ptr()
            
            # Register memory region
            ok, mr_id = self.ep.reg(ptr, size_bytes)
            if not ok:
                raise RuntimeError(f"Failed to register tensor {name}")
            
            # Receive tensor
            ok = self.ep.recv(conn_id, mr_id, ptr, size_bytes)
            if not ok:
                raise RuntimeError(f"Receive failed for {name}")
            
            # Copy to model
            self.model.state_dict()[name].copy_(recv_tensor)
            total_bytes += size_bytes
            
            if idx % 20 == 0 or idx == total_tensors:
                progress_pct = (idx / total_tensors) * 100
                logging.info(f"[Inference Rank {self.rank}] Progress: {progress_pct:.1f}% "
                            f"({idx}/{total_tensors})")
        
        receive_time = time.perf_counter() - start_time
        bandwidth_mbps = (total_bytes / receive_time / 1e6)
        
        logging.info(f"[Inference Rank {self.rank}] Received all parameters in "
                    f"{receive_time:.2f}s ({bandwidth_mbps:.2f} MB/s)")
        
        return receive_time


# Example usage
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
    
    # Determine training ranks (example: ranks 0-15)
    training_ranks = list(range(0, 16))
    
    # Create RDMA worker
    worker = RDMAInferenceWorker(
        model=model,
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        training_ranks=training_ranks
    )
    
    logging.info(f"[Rank {rank}] RDMA Inference Worker initialized")
    
    # Clean up
    dist.destroy_process_group()