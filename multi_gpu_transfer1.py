import os
import time
import json

import torch
import torch.distributed as dist

import uccl  # must be the ROCm+P2P-enabled UCCL wheel


COMMIT_BYTE = 0xC3  # arbitrary single-byte commit marker


# -------------------------
# Distributed setup helpers
# -------------------------

def init_dist():
    """
    Initialize torch.distributed with gloo backend (CPU tensors only)
    and compute:
      - rank, world_size
      - local_rank (GPU index within node)
      - node_rank (0 or 1, derived from rank)
      - gpus_per_node
      - peer_rank (the rank on the other node with same local_index)

    Assumes exactly 2 nodes.
    """
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    gpus_per_node = int(os.environ.get("LOCAL_WORLD_SIZE", 1))

    # Each process gets its own GPU
    torch.cuda.set_device(local_rank)

    # Metadata/control only => gloo is fine
    dist.init_process_group(backend="gloo")

    assert world_size % gpus_per_node == 0, (
        f"world_size={world_size} not divisible by gpus_per_node={gpus_per_node}"
    )
    nnodes = world_size // gpus_per_node
    assert nnodes == 2, f"This script assumes exactly 2 nodes, got nnodes={nnodes}"

    node_rank = rank // gpus_per_node      # 0 or 1
    local_index = rank % gpus_per_node     # GPU index within node

    peer_node = 1 - node_rank
    peer_rank = peer_node * gpus_per_node + local_index

    if rank == 0:
        print(
            f"[init] world_size={world_size}, gpus_per_node={gpus_per_node}, "
            f"nnodes={nnodes}"
        )
    print(
        f"[init] R{rank}: node_rank={node_rank}, local_index={local_index}, "
        f"peer_rank={peer_rank}"
    )

    return rank, world_size, local_rank, node_rank, gpus_per_node, peer_rank


# -------------------------
# Metadata send/recv (CPU)
# -------------------------

def send_bytes(dst_rank: int, payload: bytes):
    """
    Send arbitrary bytes to dst_rank via torch.distributed (gloo, CPU tensors).
    Protocol: first send length (int32), then payload bytes (uint8 array).
    """
    length = torch.tensor([len(payload)], dtype=torch.int32)
    dist.send(length, dst=dst_rank)

    if len(payload) == 0:
        return

    buf = torch.tensor(list(payload), dtype=torch.uint8)
    dist.send(buf, dst=dst_rank)


def recv_bytes(src_rank: int) -> bytes:
    """
    Receive arbitrary bytes from src_rank via torch.distributed (gloo, CPU).
    """
    length = torch.zeros(1, dtype=torch.int32)
    dist.recv(length, src=src_rank)
    n = int(length.item())
    if n == 0:
        return b""

    buf = torch.empty(n, dtype=torch.uint8)
    dist.recv(buf, src=src_rank)
    return bytes(buf.tolist())


# -------------------------
# UCCL helper
# -------------------------

def try_write(conn, local_ptr: int, remote_md, nbytes: int, remote_off: int):
    """
    Call uccl.write(...) with two likely signatures:

      (a) write(conn, local_ptr, remote_md, nbytes, remote_off)
      (b) write(conn, remote_md, local_ptr, nbytes, remote_off)

    If your build uses a different ordering, adjust here.
    """
    try:
        uccl.write(conn, int(local_ptr), remote_md, int(nbytes), int(remote_off))
        return
    except TypeError:
        uccl.write(conn, remote_md, int(local_ptr), int(nbytes), int(remote_off))


# -------------------------
# Main transfer logic
# -------------------------

def run_transfer(rank, local_rank, node_rank, gpus_per_node, peer_rank):
    """
    For each GPU pair (local_rank on node 0, same local_rank on node 1):

      - If node_rank == 1: act as RECEIVER
          * allocate GPU buffer
          * uccl.reg + get_metadata
          * advertise + accept
          * send {addr, md, bytes} to peer via gloo
          * wait for 1-byte COMMIT from sender
          * print a few bytes of the received data

      - If node_rank == 0: act as SENDER
          * recv {addr, md, bytes} from peer via gloo
          * uccl.parse_metadata + connect
          * allocate local GPU buffer, fill pattern
          * chunked uccl.write into remote buffer
          * send COMMIT byte via uccl.send
    """
    device = torch.device("cuda", local_rank)

    # Size of transfer per GPU (adjust as needed)
    SLAB_MB = 256
    SLAB_BYTES = SLAB_MB * (1 << 20)

    if node_rank == 1:
        # ---------------- RECEIVER NODE ----------------
        print(f"[R{rank}] acting as RECEIVER; peer_rank={peer_rank}")

        # 1) Allocate destination GPU buffer
        dst = torch.empty(SLAB_BYTES, dtype=torch.uint8, device=device)

        # 2) Register with UCCL and get metadata
        mr = uccl.reg(int(dst.data_ptr()), int(SLAB_BYTES))
        md = uccl.get_metadata(mr)

        # 3) Advertise address and send metadata to peer via gloo
        addr = uccl.advertise()

        info = {"addr": addr, "bytes": SLAB_BYTES}
        if isinstance(md, (bytes, bytearray)):
            info["md_type"] = "bytes"
            info["md"] = md.hex()
        else:
            info["md_type"] = "str"
            info["md"] = str(md)

        payload = json.dumps(info).encode("utf-8")
        print(f"[R{rank}] sending metadata ({len(payload)} bytes) to R{peer_rank}")
        send_bytes(peer_rank, payload)

        # 4) Accept UCCL connection
        conn = uccl.accept(addr)
        print(f"[R{rank}] accepted UCCL conn; waiting for COMMIT byte...")

        # 5) Wait for 1-byte COMMIT via UCCL (small two-sided message)
        flag = bytearray(1)
        n = uccl.recv(conn, flag, 1)
        print(f"[R{rank}] COMMIT recv returned n={n}, flag={list(flag)}")
        if n == 1 and flag[0] == COMMIT_BYTE:
            print(f"[R{rank}] COMMIT received; data now in dst buffer.")
        else:
            print(f"[R{rank}] unexpected COMMIT data, n={n}, flag={list(flag)}")

        # 6) Inspect first few bytes
        head = dst[:8].cpu().tolist()
        print(f"[R{rank}] dst[:8] = {head}")

    else:
        # ---------------- SENDER NODE ----------------
        print(f"[R{rank}] acting as SENDER; peer_rank={peer_rank}")

        # 1) Receive metadata from peer via gloo
        print(f"[R{rank}] waiting to recv metadata from R{peer_rank}")
        payload = recv_bytes(peer_rank)
        info = json.loads(payload.decode("utf-8"))

        addr = info["addr"]
        total_bytes = int(info["bytes"])

        if info.get("md_type") == "bytes":
            md = bytes.fromhex(info["md"])
        else:
            md = info["md"]

        print(f"[R{rank}] got addr='{addr}', total_bytes={total_bytes/(1<<20):.1f} MB")

        # 2) Parse remote metadata and connect via UCCL
        remote_mr = uccl.parse_metadata(md)
        conn = uccl.connect(addr)
        print(f"[R{rank}] UCCL connect() succeeded; starting transfer...")

        # 3) Allocate source GPU buffer and fill with a pattern
        src = torch.empty(total_bytes, dtype=torch.uint8, device=device)
        src.fill_(0xAB)

        base_ptr = int(src.data_ptr())
        CHUNK = 2 * (1 << 20)  # 2MB chunk size
        written = 0

        t0 = time.time()
        while written < total_bytes:
            n = min(CHUNK, total_bytes - written)
            try_write(conn, base_ptr + written, remote_mr, n, written)
            written += n
        torch.cuda.synchronize()
        t1 = time.time()

        gb = total_bytes / 1e9
        print(
            f"[R{rank}] sent {total_bytes / (1<<20):.1f} MB "
            f"in {t1 - t0:.3f} s "
            f"({gb/(t1 - t0):.2f} GB/s)"
        )

        # 4) Send 1-byte COMMIT via UCCL to let receiver swap / use buffer
        uccl.send(conn, bytes([COMMIT_BYTE]), 1)
        print(f"[R{rank}] COMMIT sent.")


# -------------------------
# Entrypoint
# -------------------------

def main():
    rank, world_size, local_rank, node_rank, gpus_per_node, peer_rank = init_dist()

    dist.barrier()
    if rank == 0:
        print("[INFO] Starting multi-GPU UCCL transfer round")
    dist.barrier()

    run_transfer(rank, local_rank, node_rank, gpus_per_node, peer_rank)

    dist.barrier()
    if rank == 0:
        print("[INFO] All transfers completed")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
