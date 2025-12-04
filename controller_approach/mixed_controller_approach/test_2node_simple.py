# test_debug_stuck.py
import os
import torch
import torch.distributed as dist
from uccl import p2p

rank = int(os.environ.get("RANK", 0))
local_rank = int(os.environ.get("LOCAL_RANK", 0))

torch.cuda.set_device(local_rank)

print(f"[Rank {rank}] Step 1: Before init_process_group", flush=True)
dist.init_process_group(backend="gloo")
print(f"[Rank {rank}] Step 2: After init_process_group", flush=True)

print(f"[Rank {rank}] Step 3: Creating endpoint...", flush=True)
ep = p2p.Endpoint(local_rank, 16)
print(f"[Rank {rank}] Step 4: Endpoint created", flush=True)

local_metadata = ep.get_metadata()
ip, port, gpu = p2p.Endpoint.parse_metadata(local_metadata)
print(f"[Rank {rank}] Step 5: My endpoint: IP={ip}, Port={port}, GPU={gpu}", flush=True)

# Metadata exchange
print(f"[Rank {rank}] Step 6: Starting metadata exchange...", flush=True)
if rank == 0:
    print(f"[Rank 0] Step 6a: Sending metadata to rank 1...", flush=True)
    dist.send(torch.ByteTensor(list(local_metadata)), dst=1)
    print(f"[Rank 0] Step 6b: Sent, now receiving...", flush=True)
    remote_tensor = torch.zeros(len(local_metadata), dtype=torch.uint8)
    dist.recv(remote_tensor, src=1)
    print(f"[Rank 0] Step 6c: Received!", flush=True)
    remote_metadata = bytes(remote_tensor.tolist())
else:
    print(f"[Rank 1] Step 6a: Receiving metadata from rank 0...", flush=True)
    remote_tensor = torch.zeros(len(local_metadata), dtype=torch.uint8)
    dist.recv(remote_tensor, src=0)
    print(f"[Rank 1] Step 6b: Received, now sending...", flush=True)
    dist.send(torch.ByteTensor(list(local_metadata)), dst=0)
    print(f"[Rank 1] Step 6c: Sent!", flush=True)
    remote_metadata = bytes(remote_tensor.tolist())

r_ip, r_port, r_gpu = p2p.Endpoint.parse_metadata(remote_metadata)
print(f"[Rank {rank}] Step 7: Remote: IP={r_ip}, Port={r_port}, GPU={r_gpu}", flush=True)

# Connect/Accept
print(f"[Rank {rank}] Step 8: Starting connect/accept...", flush=True)
if rank == 0:
    print(f"[Rank 0] Step 8a: Calling ep.connect()...", flush=True)
    ok, conn_id = ep.connect(r_ip, r_gpu, remote_port=r_port)
    print(f"[Rank 0] Step 8b: Connect returned: ok={ok}", flush=True)
else:
    print(f"[Rank 1] Step 8a: Calling ep.accept()...", flush=True)
    ok, c_ip, c_gpu, conn_id = ep.accept()
    print(f"[Rank 1] Step 8b: Accept returned: ok={ok}", flush=True)

print(f"[Rank {rank}] Step 9: Done! ok={ok}", flush=True)
dist.destroy_process_group()