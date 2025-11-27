import os
import time
import torch
import torch.distributed as dist

def main():
    dist.init_process_group(backend="nccl", init_method="env://")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    if rank == 0:
        print(f"[Rank {rank}] PG initialized with world_size={world_size}")
    time.sleep(1)

    # Simple NCCL test: all_reduce
    x = torch.tensor([rank], device=device, dtype=torch.float32)
    dist.all_reduce(x, op=dist.ReduceOp.SUM)

    if rank == 0:
        print(f"[Rank {rank}] all_reduce sum over ranks = {x.item()} (expected = {world_size * (world_size - 1) / 2})")

    dist.barrier()
    if rank == 0:
        print("[Rank 0] DDP hello world finished successfully.")

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
