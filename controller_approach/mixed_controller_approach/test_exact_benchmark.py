# test_exact_benchmark.py
import os
import torch
import torch.distributed as dist
from uccl import p2p

# Initialize like benchmark
dist.init_process_group(backend="gloo")
rank = dist.get_rank()
world_size = dist.get_world_size()

print(f"[Rank {rank}] Initialized, world_size={world_size}")

local_gpu_idx = 0  # Like benchmark default
num_cpus = 4       # Like benchmark default

torch.cuda.set_device(f"cuda:{local_gpu_idx}")

# Create endpoint (exactly like benchmark)
ep = p2p.Endpoint(local_gpu_idx, num_cpus)
local_metadata = ep.get_metadata()

print(f"[Rank {rank}] Endpoint created")

# Exchange metadata (EXACTLY like benchmark lines 557-566)
if rank == 0:
    dist.send(torch.ByteTensor(list(local_metadata)), dst=1)
    remote_metadata_tensor = torch.zeros(len(local_metadata), dtype=torch.uint8)
    dist.recv(remote_metadata_tensor, src=1)
    remote_metadata = bytes(remote_metadata_tensor.tolist())
else:
    remote_metadata_tensor = torch.zeros(len(local_metadata), dtype=torch.uint8)
    dist.recv(remote_metadata_tensor, src=0)
    dist.send(torch.ByteTensor(list(local_metadata)), dst=0)
    remote_metadata = bytes(remote_metadata_tensor.tolist())

print(f"[Rank {rank}] Metadata exchanged")

# Parse remote metadata
ip, port, r_gpu = p2p.Endpoint.parse_metadata(remote_metadata)
print(f"[Rank {rank}] Remote: IP={ip}, Port={port}, GPU={r_gpu}")

# Connect/Accept (EXACTLY like benchmark _run_client/_run_server)
if rank == 0:
    # Client (like _run_client line 151-152)
    ok, conn_id = ep.connect(ip, r_gpu, remote_port=port)
    assert ok, "[Client] Failed to connect"
    print(f"[Client] Connected! conn_id={conn_id}")
else:
    # Server (like _run_server line 51-53)
    ok, r_ip, r_gpu, conn_id = ep.accept()
    assert ok, "[Server] Failed to accept"
    print(f"[Server] Accepted! conn_id={conn_id}")

print(f"[Rank {rank}] SUCCESS!")
dist.destroy_process_group()