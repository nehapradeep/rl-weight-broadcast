# test_uccl_2ranks.py
import os
import torch
import torch.distributed as dist
from uccl import p2p

rank = int(os.environ.get("RANK", 0))
local_rank = int(os.environ.get("LOCAL_RANK", 0))
world_size = int(os.environ.get("WORLD_SIZE", 1))

# Only use rank 0 and rank 24
if rank not in [0, 24]:
    print(f"[Rank {rank}] Skipping, only testing ranks 0 and 24")
    dist.init_process_group(backend="gloo")
    dist.barrier()
    dist.barrier()
    dist.barrier()
    dist.destroy_process_group()
    exit(0)

torch.cuda.set_device(local_rank)
dist.init_process_group(backend="gloo")

print(f"[Rank {rank}] Creating endpoint on GPU {local_rank}...")
ep = p2p.Endpoint(local_rank, 16)
local_metadata = ep.get_metadata()

ip, port, gpu = p2p.Endpoint.parse_metadata(local_metadata)
print(f"[Rank {rank}] My endpoint: IP={ip}, Port={port}, GPU={gpu}")

# Exchange metadata (like benchmark)
if rank == 0:
    # Send first, then receive
    send_tensor = torch.ByteTensor(list(local_metadata))
    dist.send(send_tensor, dst=24)
    
    recv_tensor = torch.zeros(len(local_metadata), dtype=torch.uint8)
    dist.recv(recv_tensor, src=24)
    remote_metadata = bytes(recv_tensor.tolist())
else:  # rank == 24
    # Receive first, then send
    recv_tensor = torch.zeros(len(local_metadata), dtype=torch.uint8)
    dist.recv(recv_tensor, src=0)
    
    send_tensor = torch.ByteTensor(list(local_metadata))
    dist.send(send_tensor, dst=0)
    remote_metadata = bytes(recv_tensor.tolist())

r_ip, r_port, r_gpu = p2p.Endpoint.parse_metadata(remote_metadata)
print(f"[Rank {rank}] Remote endpoint: IP={r_ip}, Port={r_port}, GPU={r_gpu}")

# Connect/Accept (like benchmark)
if rank == 0:
    print(f"[Rank 0] Connecting to {r_ip}:{r_gpu} via port {r_port}...")
    ok, conn_id = ep.connect(r_ip, r_gpu, remote_port=r_port)
    if ok:
        print(f"[Rank 0] SUCCESS! conn_id={conn_id}")
    else:
        print(f"[Rank 0] FAILED to connect")
else:  # rank == 24
    print(f"[Rank 24] Accepting connection...")
    ok, r_ip, r_gpu, conn_id = ep.accept()
    if ok:
        print(f"[Rank 24] SUCCESS! Accepted from {r_ip}, conn_id={conn_id}")
    else:
        print(f"[Rank 24] FAILED to accept")

dist.barrier()
print(f"[Rank {rank}] Test complete!")
dist.destroy_process_group()