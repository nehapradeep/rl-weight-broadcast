# test_4node_8gpu.py
import os
import torch
import torch.distributed as dist
from uccl import p2p

dist.init_process_group(backend="gloo")
rank = dist.get_rank()
world_size = dist.get_world_size()
local_rank = int(os.environ.get("LOCAL_RANK", 0))

print(f"[Rank {rank}] Initialized, local_rank={local_rank}, world_size={world_size}", flush=True)

torch.cuda.set_device(local_rank)

# Create endpoint
ep = p2p.Endpoint(local_rank, 4)
local_metadata = ep.get_metadata()
ip, port, gpu = p2p.Endpoint.parse_metadata(local_metadata)
print(f"[Rank {rank}] Endpoint: IP={ip}, Port={port}, GPU={gpu}", flush=True)

# Pattern matching your setup:
# Node 0: ranks 0-7 (training)
# Node 1: ranks 8-15 (training)
# Node 2: ranks 16-23 (training)
# Node 3: ranks 24-31 (inference)
#
# Each training rank connects to ONE inference rank (round-robin)
# rank 0 -> rank 24, rank 1 -> rank 25, ... rank 7 -> rank 31
# rank 8 -> rank 24, rank 9 -> rank 25, ... (wraps around)

if rank < 24:
    # Training ranks (0-23) - will connect
    target_rank = 24 + (rank % 8)  # Map to inference ranks 24-31
    
    # Exchange metadata
    dist.send(torch.ByteTensor(list(local_metadata)), dst=target_rank)
    remote_tensor = torch.zeros(len(local_metadata), dtype=torch.uint8)
    dist.recv(remote_tensor, src=target_rank)
    remote_metadata = bytes(remote_tensor.tolist())
    
    r_ip, r_port, r_gpu = p2p.Endpoint.parse_metadata(remote_metadata)
    print(f"[Rank {rank}] Will connect to rank {target_rank}: IP={r_ip}, Port={r_port}", flush=True)
    
    print(f"[Rank {rank}] Connecting...", flush=True)
    ok, conn_id = ep.connect(r_ip, r_gpu, remote_port=r_port)
    print(f"[Rank {rank}] Connect result: ok={ok}", flush=True)

else:
    # Inference ranks (24-31) - will accept 3 connections each
    # rank 24 accepts from 0, 8, 16
    # rank 25 accepts from 1, 9, 17
    # etc.
    
    my_sources = [r for r in range(24) if (r % 8) == (rank - 24)]
    print(f"[Rank {rank}] Will accept from ranks: {my_sources}", flush=True)
    
    for source_rank in my_sources:
        # Exchange metadata
        remote_tensor = torch.zeros(len(local_metadata), dtype=torch.uint8)
        dist.recv(remote_tensor, src=source_rank)
        dist.send(torch.ByteTensor(list(local_metadata)), dst=source_rank)
        remote_metadata = bytes(remote_tensor.tolist())
        
        r_ip, r_port, r_gpu = p2p.Endpoint.parse_metadata(remote_metadata)
        print(f"[Rank {rank}] Metadata from rank {source_rank}: IP={r_ip}, Port={r_port}", flush=True)
        
        print(f"[Rank {rank}] Accepting from rank {source_rank}...", flush=True)
        ok, c_ip, c_gpu, conn_id = ep.accept()
        print(f"[Rank {rank}] Accept result: ok={ok}", flush=True)

dist.barrier()
print(f"[Rank {rank}] SUCCESS!", flush=True)
dist.destroy_process_group()