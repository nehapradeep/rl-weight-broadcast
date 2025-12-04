# test_2node_2gpu.py
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

# Simple pattern: rank 0 connects to rank 2, rank 1 connects to rank 3
# Node 0 has ranks 0,1 (training)
# Node 1 has ranks 2,3 (inference)

if rank < 2:
    # Training ranks (0, 1) - will connect
    target_rank = rank + 2  # 0->2, 1->3
    
    # Exchange metadata with target
    dist.send(torch.ByteTensor(list(local_metadata)), dst=target_rank)
    remote_tensor = torch.zeros(len(local_metadata), dtype=torch.uint8)
    dist.recv(remote_tensor, src=target_rank)
    remote_metadata = bytes(remote_tensor.tolist())
    
    r_ip, r_port, r_gpu = p2p.Endpoint.parse_metadata(remote_metadata)
    print(f"[Rank {rank}] Remote: IP={r_ip}, Port={r_port}, GPU={r_gpu}", flush=True)
    
    print(f"[Rank {rank}] Connecting...", flush=True)
    ok, conn_id = ep.connect(r_ip, r_gpu, remote_port=r_port)
    print(f"[Rank {rank}] Connect result: ok={ok}", flush=True)
else:
    # Inference ranks (2, 3) - will accept
    source_rank = rank - 2  # 2->0, 3->1
    
    # Exchange metadata with source
    remote_tensor = torch.zeros(len(local_metadata), dtype=torch.uint8)
    dist.recv(remote_tensor, src=source_rank)
    dist.send(torch.ByteTensor(list(local_metadata)), dst=source_rank)
    remote_metadata = bytes(remote_tensor.tolist())
    
    r_ip, r_port, r_gpu = p2p.Endpoint.parse_metadata(remote_metadata)
    print(f"[Rank {rank}] Remote: IP={r_ip}, Port={r_port}, GPU={r_gpu}", flush=True)
    
    print(f"[Rank {rank}] Accepting...", flush=True)
    ok, c_ip, c_gpu, conn_id = ep.accept()
    print(f"[Rank {rank}] Accept result: ok={ok}", flush=True)

dist.barrier()
print(f"[Rank {rank}] SUCCESS!", flush=True)
dist.destroy_process_group()