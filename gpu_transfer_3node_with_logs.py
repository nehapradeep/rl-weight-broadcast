from __future__ import annotations
import torch, time, os, sys
import torch.distributed as dist
import logging
from datetime import datetime

from transformers import GPT2LMHeadModel
from uccl import p2p

# Setup logging
def setup_logging(rank):
    """Setup logging with timestamps and rank info"""
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"{log_dir}/rank_{rank}_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | Rank %(rank)d | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Add rank to all log records
    old_factory = logging.getLogRecordFactory()
    def record_factory(*args, **kwargs):
        record = old_factory(*args, **kwargs)
        record.rank = rank
        return record
    logging.setLogRecordFactory(record_factory)
    
    logging.info(f"Logging initialized. Log file: {log_file}")
    return log_file


def broadcast_model(ep, conn_ids, model, rank):
    """Send model to multiple receivers with detailed logging"""
    state_dict = model.state_dict()
    total_tensors = len(list(state_dict.items()))
    total_size_mb = sum(t.numel() * t.element_size() for t in state_dict.values()) / 1e6
    
    logging.info("="*80)
    logging.info(f"BROADCAST START - Sending to {len(conn_ids)} receivers")
    logging.info(f"Total tensors: {total_tensors}")
    logging.info(f"Total size: {total_size_mb:.2f} MB")
    logging.info("="*80)
    
    broadcast_start = time.perf_counter()
    tensor_times = []
    
    for idx, (name, tensor) in enumerate(state_dict.items(), 1):
        tensor_start = time.perf_counter()
        
        if not tensor.is_cuda:
            tensor = tensor.cuda()
        
        size_bytes = tensor.numel() * tensor.element_size()
        size_mb = size_bytes / 1e6
        ptr = tensor.data_ptr()
        
        # Register memory
        logging.debug(f"[{idx}/{total_tensors}] Registering {name}")
        ok, mr_id = ep.reg(ptr, size_bytes)
        assert ok, f"Failed to register tensor {name}"
        
        # Send to all receivers
        for receiver_idx, conn_id in enumerate(conn_ids, 1):
            send_start = time.perf_counter()
            ok = ep.send(conn_id, mr_id, ptr, size_bytes)
            send_time = time.perf_counter() - send_start
            
            assert ok, f"Send failed for {name} to receiver {receiver_idx}"
            
            bandwidth_gbps = (size_bytes / 1e9) / send_time if send_time > 0 else 0
            logging.info(f"[{idx}/{total_tensors}] Sent {name} to Receiver {receiver_idx} | "
                        f"Size: {size_mb:.2f} MB | Time: {send_time:.3f}s | "
                        f"Bandwidth: {bandwidth_gbps:.2f} GB/s")
        
        tensor_time = time.perf_counter() - tensor_start
        tensor_times.append(tensor_time)
        
        # Progress update every 10 tensors
        if idx % 10 == 0:
            progress_pct = (idx / total_tensors) * 100
            elapsed = time.perf_counter() - broadcast_start
            avg_time = sum(tensor_times) / len(tensor_times)
            eta = avg_time * (total_tensors - idx)
            logging.info(f"Progress: {progress_pct:.1f}% ({idx}/{total_tensors}) | "
                        f"Elapsed: {elapsed:.1f}s | ETA: {eta:.1f}s")
    
    total_time = time.perf_counter() - broadcast_start
    avg_bandwidth = (total_size_mb / 1000) / total_time  # GB/s
    
    logging.info("="*80)
    logging.info(f"BROADCAST COMPLETE")
    logging.info(f"Total time: {total_time:.2f}s")
    logging.info(f"Average bandwidth: {avg_bandwidth:.2f} GB/s")
    logging.info(f"Average time per tensor: {sum(tensor_times)/len(tensor_times):.3f}s")
    logging.info("="*80)


def recv_model(ep, conn_id, model, rank):
    """Receive model from broadcaster with detailed logging"""
    state_dict = model.state_dict()
    total_tensors = len(list(state_dict.items()))
    total_size_mb = sum(t.numel() * t.element_size() for t in state_dict.values()) / 1e6
    
    logging.info("="*80)
    logging.info(f"RECEIVE START")
    logging.info(f"Total tensors: {total_tensors}")
    logging.info(f"Total size: {total_size_mb:.2f} MB")
    logging.info("="*80)
    
    recv_start = time.perf_counter()
    tensor_times = []
    
    for idx, (name, tensor) in enumerate(state_dict.items(), 1):
        tensor_start = time.perf_counter()
        
        recv_tensor = torch.empty_like(tensor, device="cuda")
        size_bytes = recv_tensor.numel() * recv_tensor.element_size()
        size_mb = size_bytes / 1e6
        ptr = recv_tensor.data_ptr()
        
        # Register memory
        logging.debug(f"[{idx}/{total_tensors}] Registering {name}")
        ok, mr_id = ep.reg(ptr, size_bytes)
        assert ok, f"Failed to register tensor {name}"
        
        # Receive tensor
        recv_time_start = time.perf_counter()
        ok = ep.recv(conn_id, mr_id, ptr, size_bytes)
        recv_time = time.perf_counter() - recv_time_start
        
        assert ok, f"Receive failed for {name}"
        
        model.state_dict()[name].copy_(recv_tensor)
        
        tensor_time = time.perf_counter() - tensor_start
        tensor_times.append(tensor_time)
        
        bandwidth_gbps = (size_bytes / 1e9) / recv_time if recv_time > 0 else 0
        logging.info(f"[{idx}/{total_tensors}] Received {name} | "
                    f"Size: {size_mb:.2f} MB | Time: {recv_time:.3f}s | "
                    f"Bandwidth: {bandwidth_gbps:.2f} GB/s")
        
        # Progress update every 10 tensors
        if idx % 10 == 0:
            progress_pct = (idx / total_tensors) * 100
            elapsed = time.perf_counter() - recv_start
            avg_time = sum(tensor_times) / len(tensor_times)
            eta = avg_time * (total_tensors - idx)
            logging.info(f"Progress: {progress_pct:.1f}% ({idx}/{total_tensors}) | "
                        f"Elapsed: {elapsed:.1f}s | ETA: {eta:.1f}s")
    
    total_time = time.perf_counter() - recv_start
    avg_bandwidth = (total_size_mb / 1000) / total_time  # GB/s
    
    logging.info("="*80)
    logging.info(f"RECEIVE COMPLETE")
    logging.info(f"Total time: {total_time:.2f}s")
    logging.info(f"Average bandwidth: {avg_bandwidth:.2f} GB/s")
    logging.info(f"Average time per tensor: {sum(tensor_times)/len(tensor_times):.3f}s")
    logging.info("="*80)


def main():
    dist.init_process_group(backend="gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    # Setup logging first
    log_file = setup_logging(rank)
    
    logging.info(f"Process started - Rank: {rank}, World size: {world_size}")
    assert world_size == 3, "Run with three ranks (1 broadcaster + 2 receivers)."

    local_gpu = rank
    torch.cuda.set_device(local_gpu)
    logging.info(f"CUDA device set to GPU {local_gpu}")

    # Initialize endpoint
    logging.info("Initializing P2P endpoint...")
    ep = p2p.Endpoint(local_gpu, 4)
    local_md = ep.get_metadata()
    logging.info(f"Local metadata obtained (size: {len(local_md)} bytes)")

    # Exchange metadata - all-to-all style
    logging.info("Starting metadata exchange...")
    all_metadata = [None] * world_size
    all_metadata[rank] = local_md
    
    metadata_start = time.perf_counter()
    for i in range(world_size):
        if i == rank:
            # Send my metadata to all others
            for j in range(world_size):
                if j != rank:
                    dist.send(torch.ByteTensor(list(local_md)), dst=j)
                    logging.debug(f"Sent metadata to rank {j}")
        else:
            # Receive metadata from rank i
            remote_md = torch.zeros(len(local_md), dtype=torch.uint8)
            dist.recv(remote_md, src=i)
            all_metadata[i] = bytes(remote_md.tolist())
            logging.debug(f"Received metadata from rank {i}")
    
    metadata_time = time.perf_counter() - metadata_start
    logging.info(f"Metadata exchange complete in {metadata_time:.2f}s")

    if rank == 0:
        # Broadcaster: connect to rank 1 and rank 2
        logging.info("="*80)
        logging.info("BROADCASTER MODE")
        logging.info("="*80)
        logging.info("Connecting to receivers...")
        conn_ids = []
        
        for receiver_rank in [1, 2]:
            connect_start = time.perf_counter()
            ip, port, r_gpu = p2p.Endpoint.parse_metadata(all_metadata[receiver_rank])
            logging.info(f"Parsed receiver {receiver_rank} metadata: IP={ip}, Port={port}, GPU={r_gpu}")
            
            ok, conn_id = ep.connect(ip, r_gpu, remote_port=port)
            connect_time = time.perf_counter() - connect_start
            
            assert ok, f"Connect failed to rank {receiver_rank}"
            conn_ids.append(conn_id)
            logging.info(f"Connected to receiver {receiver_rank} (conn_id={conn_id}) in {connect_time:.2f}s")

        logging.info("Loading model...")
        model_load_start = time.perf_counter()
        model = GPT2LMHeadModel.from_pretrained("gpt2").cuda()
        model_load_time = time.perf_counter() - model_load_start
        logging.info(f"Model loaded in {model_load_time:.2f}s")
        
        broadcast_model(ep, conn_ids, model, rank)

    else:
        # Receivers (rank 1 and 2): accept connection from rank 0
        logging.info("="*80)
        logging.info(f"RECEIVER MODE (Rank {rank})")
        logging.info("="*80)
        logging.info("Waiting for broadcaster connection...")
        
        accept_start = time.perf_counter()
        ok, r_ip, r_gpu, conn_id = ep.accept()
        accept_time = time.perf_counter() - accept_start
        
        assert ok, f"Accept failed"
        logging.info(f"Connected to broadcaster (IP={r_ip}, GPU={r_gpu}, conn_id={conn_id}) in {accept_time:.2f}s")

        logging.info("Loading model...")
        model_load_start = time.perf_counter()
        model = GPT2LMHeadModel.from_pretrained("gpt2").cuda()
        model_load_time = time.perf_counter() - model_load_start
        logging.info(f"Model loaded in {model_load_time:.2f}s")
        
        recv_model(ep, conn_id, model, rank)

    logging.info("Destroying process group...")
    dist.destroy_process_group()
    logging.info("Process complete. Exiting.")
    logging.info(f"Full log saved to: {log_file}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logging.warning("Interrupted by user")
        sys.exit(0)
    except Exception as e:
        logging.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)
