# test_simple.py
import os
import torch.distributed as dist

rank = int(os.environ.get("RANK", 0))
print(f"[Rank {rank}] Hello!", flush=True)

dist.init_process_group(backend="gloo")
print(f"[Rank {rank}] Done!", flush=True)

dist.destroy_process_group()
