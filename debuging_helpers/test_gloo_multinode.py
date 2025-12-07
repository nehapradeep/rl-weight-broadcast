# test_gloo_multinode.py
import os
import torch
import torch.distributed as dist

rank = int(os.environ.get("RANK", 0))
world_size = int(os.environ.get("WORLD_SIZE", 1))

# Initialize with GLOO backend
dist.init_process_group(backend="gloo", init_method="env://")

print(f"[Rank {rank}] GLOO initialized")

# Test send/recv between nodes
if rank == 0:
    tensor = torch.ByteTensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    print(f"[Rank 0] Sending to rank 24...")
    dist.send(tensor, dst=24)
    print(f"[Rank 0] Sent!")
    
elif rank == 24:
    tensor = torch.zeros(10, dtype=torch.uint8)
    print(f"[Rank 24] Receiving from rank 0...")
    dist.recv(tensor, src=0)
    print(f"[Rank 24] Received: {tensor.tolist()}")

else:
    print(f"[Rank {rank}] Waiting at barrier...")

dist.barrier()
print(f"[Rank {rank}] Done!")
dist.destroy_process_group()