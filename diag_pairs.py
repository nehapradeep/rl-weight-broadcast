# diag_pairs.py
import os
import torch
import torch.distributed as dist


def init_dist():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    gpus_per_node = int(os.environ.get("LOCAL_WORLD_SIZE", 1))

    # For AMD GPUs, this picks the right local GPU
    torch.cuda.set_device(local_rank)

    # Use gloo for CPU metadata; easier and totally fine here
    dist.init_process_group(backend="gloo")

    assert world_size % gpus_per_node == 0
    nnodes = world_size // gpus_per_node
    assert nnodes == 2, f"diag_pairs assumes exactly 2 nodes, got {nnodes}"

    node_rank = rank // gpus_per_node    # 0 or 1
    local_index = rank % gpus_per_node   # GPU index within node

    peer_node = 1 - node_rank
    peer_rank = peer_node * gpus_per_node + local_index

    return rank, world_size, local_rank, node_rank, gpus_per_node, peer_rank


def main():
    rank, world_size, local_rank, node_rank, gpn, peer_rank = init_dist()

    if rank == 0:
        print(f"[INFO] world_size={world_size}, gpus_per_node={gpn}")
        print("[INFO] pairing GPU i on node 0 <-> GPU i on node 1")

    dist.barrier()

    device = torch.device("cuda", local_rank)

    if node_rank == 0:
        # SENDER: send one int to peer
        x = torch.tensor([rank], dtype=torch.int32)  # CPU tensor is fine for gloo
        print(f"[R{rank}] sending {int(x.item())} to R{peer_rank}")
        dist.send(x, dst=peer_rank)

    else:
        # RECEIVER: recv one int from peer
        y = torch.zeros(1, dtype=torch.int32)
        print(f"[R{rank}] waiting to recv from R{peer_rank}")
        dist.recv(y, src=peer_rank)
        print(f"[R{rank}] received {int(y.item())} from R{peer_rank}")

    dist.barrier()
    if rank == 0:
        print("[INFO] diag_pairs.py finished OK")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
