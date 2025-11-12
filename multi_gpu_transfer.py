import os
import time
import json

import torch
import torch.distributed as dist

import uccl  # must be the P2P-enabled uccl you built/installed


COMMIT_BYTE = 0xC3  # arbitrary marker for "transfer complete"


def init_dist():
    """
    Initialize torch.distributed and return:
    rank, world_size, local_rank, node_rank, gpus_per_node, peer_rank
    Assumes exactly 2 nodes.
    """
    # torchrun sets these env vars
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    node_rank = int(os.environ.get("NODE_RANK", 0))
    gpus_per_node = int(os.environ.get("LOCAL_WORLD_SIZE", 1))

    # sanity
    assert world_size == 2 * gpus_per_node, (
        f"Expected world_size = 2 * gpus_per_node, "
        f"got world_size={world_size}, gpus_per_node={gpus_per_node}"
    )

    torch.cuda.set_device(local_rank)

    # init NCCL/RCCL backend
    dist.init_process_group(backend="nccl")

    # pair GPU i on node 0 with GPU i on node 1
    peer_node = 1 - node_rank
    peer_rank = peer_node * gpus_per_node + local_rank

    return rank, world_size, local_rank, node_rank, gpus_per_node, peer_rank


def send_bytes(dst_rank: int, payload: bytes):
    """
    Send arbitrary bytes to dst_rank using torch.distributed.
    """
    device = torch.device("cuda")
    length = torch.tensor([len(payload)], dtype=torch.int32, device=device)
    dist.send(length.cpu(), dst=dst_rank)

    if len(payload) == 0:
        return

    buf = torch.frombuffer(bytearray(payload), dtype=torch.uint8)
    buf = buf.to(device)
    dist.send(buf.cpu(), dst=dst_rank)


def recv_bytes(src_rank: int) -> bytes:
    """
    Receive arbitrary bytes from src_rank using torch.distributed.
    """
    device = torch.device("cuda")
    length = torch.zeros(1, dtype=torch.int32, device=device)
    dist.recv(length.cpu(), src=src_rank)
    n = int(length.item())
    if n == 0:
        return b""

    buf = torch.empty(n, dtype=torch.uint8, device=device)
    dist.recv(buf.cpu(), src=src_rank)
    return bytes(buf.cpu().tolist())


def try_write(conn, local_ptr: int, remote_md, nbytes: int, remote_off: int):
    """
    Call uccl.write(...) with two likely signatures:

      a) write(conn, local_ptr, remote_md, nbytes, remote_off)
      b) write(conn, remote_md, local_ptr, nbytes, remote_off)

    Adjust here if your uccl.write signature is different (you can print it
    with inspect.signature(uccl.write)).
    """
    try:
        uccl.write(conn, int(local_ptr), remote_md, int(nbytes), int(remote_off))
        return
    except TypeError:
        # try alternate ordering
        uccl.write(conn, remote_md, int(local_ptr), int(nbytes), int(remote_off))


def run_transfer(rank, local_rank, node_rank, gpus_per_node, peer_rank):
    """
    For each GPU pair (local_rank on node_rank, peer_rank on other node):

    - node_rank == 1: acts as RECEIVER
      allocates GPU buffer, regs with uccl.reg, advertises addr, accepts conn,
      and waits for commit byte, then verifies.

    - node_rank == 0: acts as SENDER
      allocates GPU buffer, connects to addr, parses metadata, writes into
      remote GPU memory, sends commit.
    """
    device = torch.device("cuda", local_rank)

    # Size of transfer per GPU (adjust as needed)
    SLAB_MB = 256
    SLAB_BYTES = SLAB_MB * (1 << 20)

    if node_rank == 1:
        # ---------------- RECEIVER ----------------
        # 1) Allocate GPU buffer to receive into
        dst = torch.empty(SLAB_BYTES, dtype=torch.uint8, device=device)

        # 2) Register buffer and get metadata
        mr = uccl.reg(int(dst.data_ptr()), int(SLAB_BYTES))
        md = uccl.get_metadata(mr)

        # 3) Advertise address and accept connection
        addr = uccl.advertise()
        # send addr + metadata to sender via torch.distributed
        info = {"addr": addr, "bytes": SLAB_BYTES}
        # md may be bytes; make it JSON-safe by base16 if needed
        if isinstance(md, (bytes, bytearray)):
            info["md_type"] = "bytes"
            info["md"] = md.hex()
        else:
            info["md_type"] = "str"
            info["md"] = str(md)

        payload = json.dumps(info).encode("utf-8")
        send_bytes(peer_rank, payload)

        conn = uccl.accept(addr)
        print(f"[R{rank}] Receiver accepted conn; waiting for commit...")

        # 4) Wait for 1-byte COMMIT from sender
        flag = bytearray(1)
        n = uccl.recv(conn, flag, 1)
        if n == 1 and flag[0] == COMMIT_BYTE:
            print(f"[R{rank}] Commit received; data should now be in dst buffer.")
        else:
            print(f"[R{rank}] Unexpected recv (n={n}, flag={list(flag)})")

        # 5) Optional: verify pattern
        head = dst[:8].cpu().tolist()
        print(f"[R{rank}] dst[:8] = {head}")

    else:
        # ---------------- SENDER ----------------
        # 1) Receive addr + metadata from receiver
        payload = recv_bytes(peer_rank)
        info = json.loads(payload.decode("utf-8"))
        addr = info["addr"]
        total_bytes = int(info["bytes"])

        if info.get("md_type") == "bytes":
            md = bytes.fromhex(info["md"])
        else:
            md = info["md"]

        # 2) Parse remote metadata and connect
        remote_mr = uccl.parse_metadata(md)
        conn = uccl.connect(addr)

        # 3) Allocate local GPU buffer and fill with pattern
        src = torch.empty(total_bytes, dtype=torch.uint8, device=device)
        src.fill_(0xAB)

        base_ptr = int(src.data_ptr())
        CHUNK = 2 * (1 << 20)  # 2MB
        written = 0

        t0 = time.time()
        while written < total_bytes:
            n = min(CHUNK, total_bytes - written)
            try_write(conn, base_ptr + written, remote_mr, n, written)
            written += n
        torch.cuda.synchronize()
        t1 = time.time()

        print(
            f"[R{rank}] Sent {total_bytes / (1<<20):.1f} MB "
            f"in {t1-t0:.3f} s "
            f"({(total_bytes/1e9)/(t1-t0):.2f} GB/s)"
        )

        # 4) Send 1-byte COMMIT
        uccl.send(conn, bytes([COMMIT_BYTE]), 1)
        print(f"[R{rank}] Commit sent.")


def main():
    rank, world_size, local_rank, node_rank, gpus_per_node, peer_rank = init_dist()

    if rank == 0:
        print(
            f"[INFO] world_size={world_size}, gpus_per_node={gpus_per_node}; "
            "pairing GPU i on node 0 with GPU i on node 1."
        )
    dist.barrier()

    run_transfer(rank, local_rank, node_rank, gpus_per_node, peer_rank)

    dist.barrier()
    if rank == 0:
        print("[INFO] All transfers done.")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
