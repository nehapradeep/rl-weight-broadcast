import os
import time
import json

import torch
import torch.nn as nn
import torch.distributed as dist

import uccl  # UCCL P2P-enabled ROCm wheel must be installed


COMMIT_BYTE = 0xC3  # arbitrary commit marker


# ==========================
#  Model + packing utilities
# ==========================

class ToyPolicyNet(nn.Module):
    """
    Tiny stand-in for an RL policy network.
    Replace this with your actual RL model later.
    """
    def __init__(self, obs_dim=128, hidden_dim=256, action_dim=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, x):
        return self.net(x)


def build_layout_from_state_dict(state_dict):
    """
    Build a deterministic layout over the model's parameters.

    Assumes all tensors are float32 for simplicity.
    Returns:
      layout = {
        "items": [
           {"name": ..., "shape": [...], "numel": ...},
           ...
        ],
        "total_elems": int
      }
    """
    items = []
    total_elems = 0

    # Sort by key to guarantee consistent order
    for name, tensor in sorted(state_dict.items()):
        assert tensor.dtype == torch.float32, (
            f"Only float32 tensors supported in this demo; {name} is {tensor.dtype}"
        )
        numel = tensor.numel()
        items.append({
            "name": name,
            "shape": list(tensor.shape),
            "numel": numel,
        })
        total_elems += numel

    return {"items": items, "total_elems": total_elems}


def pack_state_dict_inplace(model, layout, flat_buffer):
    """
    Fill 'flat_buffer' (1D float32 GPU tensor) with model params
    according to 'layout'.

    flat_buffer must have size layout["total_elems"].
    """
    sd = model.state_dict()
    offset = 0
    for entry in layout["items"]:
        name = entry["name"]
        numel = entry["numel"]
        shape = entry["shape"]
        t = sd[name]
        assert t.dtype == torch.float32
        view = t.view(-1)
        flat_buffer[offset:offset + numel].copy_(view)
        offset += numel

    assert offset == layout["total_elems"]
    return flat_buffer


def unpack_flat_to_state_dict(flat_buffer, model, layout):
    """
    Copy data from 'flat_buffer' (1D float32 GPU tensor) back into model
    according to 'layout'.
    """
    sd = model.state_dict()
    offset = 0
    for entry in layout["items"]:
        name = entry["name"]
        numel = entry["numel"]
        shape = entry["shape"]

        target = sd[name]
        view = flat_buffer[offset:offset + numel].view(*shape)
        # ensure device/dtype match (they should be float32 on GPU)
        target.copy_(view.to(device=target.device, dtype=target.dtype))
        offset += numel

    assert offset == layout["total_elems"]


# ==========================
#   Distributed setup
# ==========================

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

    # Each process uses one GPU (ROCm-backed under torch.cuda)
    torch.cuda.set_device(local_rank)

    # We only use dist for metadata → gloo (CPU) is fine
    dist.init_process_group(backend="gloo")

    assert world_size % gpus_per_node == 0, \
        f"world_size={world_size} not divisible by gpus_per_node={gpus_per_node}"
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


# ==========================
#   Metadata send / recv
# ==========================

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


# ==========================
#        UCCL helper
# ==========================

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


# ==========================
#     Main transfer logic
# ==========================

def setup_model_and_layout(local_rank):
    """
    Build the same toy model on both nodes and compute a layout over its weights.
    Returns:
      model (on GPU), layout dict, total_bytes (for RDMA buffer).
    """
    device = torch.device("cuda", local_rank)
    model = ToyPolicyNet().to(device)

    # You can randomize per-rank to see changes moving across later
    for p in model.parameters():
        nn.init.uniform_(p, a=-0.1, b=0.1)

    layout = build_layout_from_state_dict(model.state_dict())
    total_elems = layout["total_elems"]
    elem_size = 4  # float32
    total_bytes = total_elems * elem_size

    if local_rank == 0:
        print(
            f"[layout] total_elems={total_elems}, "
            f"approx_size={total_bytes / (1 << 20):.2f} MB"
        )

    return model, layout, total_bytes


def run_transfer(rank, local_rank, node_rank, gpus_per_node, peer_rank,
                 model, layout, total_bytes):
    """
    For each GPU pair (local_rank on node 0, same local_rank on node 1):

      - If node_rank == 1: act as RECEIVER
          * allocate GPU weights buffer (float32) and register its byte view
          * advertise + accept UCCL endpoint
          * send {addr, md, bytes} to peer via gloo
          * per-step: wait for COMMIT, then unpack into local model

      - If node_rank == 0: act as SENDER
          * recv {addr, md, bytes} from peer via gloo
          * parse_metadata + connect
          * allocate a flat buffer for weights
          * per-step: pack model → flat buffer, RDMA write, send COMMIT
    """
    device = torch.device("cuda", local_rank)
    elem_size = 4  # float32

    # Sanity check: layout should be consistent on both nodes
    assert total_bytes == layout["total_elems"] * elem_size

    if node_rank == 1:
        # ================== RECEIVER NODE ==================
        print(f"[R{rank}] acting as RECEIVER; peer_rank={peer_rank}")

        # 1) Allocate weights buffer on GPU and register its byte view
        weights_buf = torch.empty(layout["total_elems"], dtype=torch.float32, device=device)
        weights_bytes = weights_buf.view(torch.uint8)
        nbytes = weights_bytes.numel()

        mr = uccl.reg(int(weights_bytes.data_ptr()), int(nbytes))
        md = uccl.get_metadata(mr)

        addr = uccl.advertise()

        info = {
            "addr": addr,
            "bytes": nbytes,
        }
        if isinstance(md, (bytes, bytearray)):
            info["md_type"] = "bytes"
            info["md"] = md.hex()
        else:
            info["md_type"] = "str"
            info["md"] = str(md)

        # 2) Send metadata to peer via gloo
        payload = json.dumps(info).encode("utf-8")
        print(f"[R{rank}] sending metadata ({len(payload)} bytes) to R{peer_rank}")
        send_bytes(peer_rank, payload)

        # 3) Accept UCCL connection
        conn = uccl.accept(addr)
        print(f"[R{rank}] accepted UCCL conn; entering steps loop")

        # 4) Multi-step loop: each step receives new weights
        NUM_STEPS = 3
        for step in range(NUM_STEPS):
            print(f"[R{rank}] [step {step}] waiting for COMMIT...")
            flag = bytearray(1)
            n = uccl.recv(conn, flag, 1)
            if not (n == 1 and flag[0] == COMMIT_BYTE):
                print(f"[R{rank}] [step {step}] unexpected COMMIT data n={n}, flag={list(flag)}")
                break

            # At this point, weights_bytes have been overwritten by sender
            # Copy them into our model
            unpack_flat_to_state_dict(weights_buf, model, layout)

            # Just print some diagnostics
            with torch.no_grad():
                first_param = next(model.parameters())
                mean_val = first_param.mean().item()
            print(f"[R{rank}] [step {step}] updated model; first_param.mean={mean_val:.6f}")

        print(f"[R{rank}] receiver done with steps.")

    else:
        # ================== SENDER NODE ==================
        print(f"[R{rank}] acting as SENDER; peer_rank={peer_rank}")

        # 1) Receive metadata from peer via gloo
        print(f"[R{rank}] waiting to recv metadata from R{peer_rank}")
        payload = recv_bytes(peer_rank)
        info = json.loads(payload.decode("utf-8"))

        addr = info["addr"]
        remote_nbytes = int(info["bytes"])

        if info.get("md_type") == "bytes":
            md = bytes.fromhex(info["md"])
        else:
            md = info["md"]

        print(
            f"[R{rank}] got addr='{addr}', "
            f"remote_nbytes={remote_nbytes / (1 << 20):.2f} MB"
        )

        # Remote buffer should match our expected size
        assert remote_nbytes == total_bytes, (
            f"remote_nbytes={remote_nbytes} vs local total_bytes={total_bytes}"
        )

        # 2) Parse remote metadata and connect via UCCL
        remote_mr = uccl.parse_metadata(md)
        conn = uccl.connect(addr)
        print(f"[R{rank}] UCCL connect() succeeded; entering steps loop")

        # 3) Allocate a reusable flat buffer for model weights on this GPU
        flat_buf = torch.empty(layout["total_elems"], dtype=torch.float32, device=device)
        flat_bytes = flat_buf.view(torch.uint8)
        assert flat_bytes.numel() == total_bytes

        base_ptr = int(flat_bytes.data_ptr())
        nbytes = flat_bytes.numel()
        CHUNK = 2 * (1 << 20)  # 2MB

        # 4) Multi-step loop: each step packs and sends new weights
        NUM_STEPS = 3
        for step in range(NUM_STEPS):
            # Simulate model update (e.g., training step)
            with torch.no_grad():
                for p in model.parameters():
                    p.add_(0.01)  # tiny shift so receiver sees change

            # Pack current model into flat_buf
            pack_state_dict_inplace(model, layout, flat_buf)

            written = 0
            t0 = time.time()
            while written < nbytes:
                n = min(CHUNK, nbytes - written)
                try_write(conn, base_ptr + written, remote_mr, n, written)
                written += n
            torch.cuda.synchronize()
            t1 = time.time()

            gb = nbytes / 1e9
            print(
                f"[R{rank}] [step {step}] sent {nbytes / (1 << 20):.2f} MB "
                f"in {t1 - t0:.3f} s "
                f"({gb / (t1 - t0):.2f} GB/s)"
            )

            # Send COMMIT byte so receiver knows weights are ready
            uccl.send(conn, bytes([COMMIT_BYTE]), 1)
            print(f"[R{rank}] [step {step}] COMMIT sent.")

        print(f"[R{rank}] sender done with steps.")


# ==========================
#          Main
# ==========================

def main():
    rank, world_size, local_rank, node_rank, gpus_per_node, peer_rank = init_dist()

    # Build identical model + layout on all ranks
    model, layout, total_bytes = setup_model_and_layout(local_rank)

    dist.barrier()
    if rank == 0:
        print("[INFO] Starting multi-GPU UCCL weight transfer demo")
    dist.barrier()

    run_transfer(rank, local_rank, node_rank, gpus_per_node, peer_rank,
                 model, layout, total_bytes)

    dist.barrier()
    if rank == 0:
        print("[INFO] All transfers completed")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
