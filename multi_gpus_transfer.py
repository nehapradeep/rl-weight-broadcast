import os
import json
import time

import torch
import torch.nn as nn
import torch.distributed as dist

from uccl import p2p  # this matches benchmark_uccl_write.py style


# ============================================================
#  Tiny model + flatten/unflatten helpers (weights as float32)
# ============================================================

class ToyPolicyNet(nn.Module):
    """
    Simple stand-in for an RL policy/value model.
    Replace this with your actual RL network later.
    """
    def __init__(self, obs_dim=128, hidden_dim=256, action_dim=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x):
        return self.net(x)


def build_layout_from_state_dict(state_dict):
    """
    Build deterministic layout over the model parameters.
    Assumes all tensors are float32.
    """
    items = []
    total_elems = 0

    # sort by key name for deterministic ordering
    for name, tensor in sorted(state_dict.items()):
        assert tensor.dtype == torch.float32, \
            f"{name} has dtype {tensor.dtype}, expected float32"
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
    Write model parameters into flat_buffer (1D float32 tensor on GPU).
    """
    sd = model.state_dict()
    offset = 0
    for entry in layout["items"]:
        name = entry["name"]
        numel = entry["numel"]

        t = sd[name]
        view = t.view(-1)
        flat_buffer[offset:offset + numel].copy_(view)
        offset += numel

    assert offset == layout["total_elems"]
    return flat_buffer


def unpack_flat_to_state_dict(flat_buffer, model, layout):
    """
    Copy from flat_buffer (1D float32 tensor) back into model parameters.
    """
    sd = model.state_dict()
    offset = 0
    for entry in layout["items"]:
        name = entry["name"]
        numel = entry["numel"]
        shape = entry["shape"]

        target = sd[name]
        view = flat_buffer[offset:offset + numel].view(*shape)
        target.copy_(view.to(device=target.device, dtype=target.dtype))
        offset += numel

    assert offset == layout["total_elems"]


# ============================================================
#  Distributed init + rank mapping (2 nodes × N GPUs)
# ============================================================

def init_dist():
    """
    Initialize torch.distributed with gloo and derive:
      - rank, world_size
      - local_rank (GPU index within node)
      - node_rank (0 or 1)
      - gpus_per_node
      - peer_rank (pair across nodes with same local_rank)
    """
    dist.init_process_group(backend="gloo", init_method="env://")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    gpus_per_node = int(os.environ.get("LOCAL_WORLD_SIZE", 1))

    # each process uses one GPU (ROCm-backed under torch.cuda)
    torch.cuda.set_device(local_rank)

    assert world_size % gpus_per_node == 0, \
        f"world_size={world_size} not divisible by gpus_per_node={gpus_per_node}"
    nnodes = world_size // gpus_per_node
    assert nnodes == 2, f"this script assumes exactly 2 nodes, got nnodes={nnodes}"

    node_rank = rank // gpus_per_node       # 0 or 1
    local_index = rank % gpus_per_node      # 0..gpus_per_node-1

    if node_rank == 0:
        peer_rank = rank + gpus_per_node    # 0<->4,1<->5,…
    else:
        peer_rank = rank - gpus_per_node

    if rank == 0:
        print(
            f"[init] world_size={world_size}, gpus_per_node={gpus_per_node}, "
            f"nnodes={nnodes}",
            flush=True,
        )
    print(
        f"[init] R{rank}: node_rank={node_rank}, local_index={local_index}, "
        f"peer_rank={peer_rank}",
        flush=True,
    )

    return rank, world_size, local_rank, node_rank, gpus_per_node, peer_rank


# ============================================================
#  Small helpers for sending bytes via torch.distributed (gloo)
# ============================================================

def send_bytes(dst_rank: int, payload: bytes):
    """
    Send arbitrary bytes to dst_rank via CPU tensors:
    [int64 length] [uint8 * length]
    """
    length = torch.tensor([len(payload)], dtype=torch.int64)
    dist.send(length, dst=dst_rank)

    if len(payload) == 0:
        return

    buf = torch.tensor(list(payload), dtype=torch.uint8)
    dist.send(buf, dst=dst_rank)


def recv_bytes(src_rank: int) -> bytes:
    """
    Receive arbitrary bytes from src_rank via CPU tensors.
    """
    length = torch.zeros(1, dtype=torch.int64)
    dist.recv(length, src=src_rank)
    n = int(length.item())
    if n == 0:
        return b""

    buf = torch.empty(n, dtype=torch.uint8)
    dist.recv(buf, src=src_rank)
    return bytes(buf.tolist())


# ============================================================
#  UCCL p2p endpoint setup (aligned w/ benchmark_uccl_write)
# ============================================================

def setup_endpoint(local_gpu_idx: int, num_cpus: int = 1):
    """
    Create a p2p.Endpoint bound to a given local GPU.
    """
    ep = p2p.Endpoint(local_gpu_idx, num_cpus)
    return ep


def setup_p2p_connection(ep, node_rank, peer_rank):
    """
    Establish a p2p connection between this rank and its peer across nodes.

    Pattern (server = node_rank == 1, client = node_rank == 0):

    - Server:
        local_md = ep.get_metadata()
        send_bytes(peer_rank, local_md)
        ok, r_ip, r_gpu, conn_id = ep.accept()

    - Client:
        remote_md = recv_bytes(peer_rank)
        ip, port, r_gpu = p2p.Endpoint.parse_metadata(remote_md)
        ok, conn_id = ep.connect(ip, r_gpu, remote_port=port)
    """
    rank = dist.get_rank()

    if node_rank == 1:
        # Server side: advertises its listening address to the client
        local_md = ep.get_metadata()
        print(f"[R{rank}] sending endpoint metadata to R{peer_rank}", flush=True)
        send_bytes(peer_rank, local_md)

        print(f"[R{rank}] waiting for ep.accept()", flush=True)
        # signature from benchmark style: ok, ip, r_gpu, conn_id = ep.accept()
        ok, r_ip, r_gpu, conn_id = ep.accept()
        assert ok, f"[R{rank}] ep.accept() failed"
        print(
            f"[R{rank}] accept() OK from {r_ip} gpu={r_gpu}, conn_id={conn_id}",
            flush=True,
        )
    else:
        # Client side: receives server metadata and connects
        print(f"[R{rank}] waiting to recv endpoint metadata from R{peer_rank}", flush=True)
        remote_md = recv_bytes(peer_rank)

        # In the benchmark, parse_metadata is typically a classmethod
        # Endpoint.parse_metadata(remote_md) -> (ip, port, r_gpu)
        ip, port, r_gpu = p2p.Endpoint.parse_metadata(remote_md)

        print(f"[R{rank}] connecting to {ip}:{port} gpu={r_gpu}", flush=True)
        ok, conn_id = ep.connect(ip, r_gpu, remote_port=port)
        assert ok, f"[R{rank}] ep.connect() failed"
        print(f"[R{rank}] connect() OK, conn_id={conn_id}", flush=True)

    return conn_id


# ============================================================
#  Main one-shot weight transfer per GPU pair
# ============================================================

def run_transfer(rank, local_rank, node_rank, peer_rank):
    """
    For each GPU pair (local_rank on node 0, same local_rank on node 1):

    - Build same ToyPolicyNet on both sides.
    - Compute flat layout over weights.
    - Use UCCL p2p.Endpoint to:
        * Server (node_rank=1): allocate GPU weight buffer, reg+advertise,
          send fifo_blob to client.
        * Client (node_rank=0): pack model -> flat buffer, reg it, recv fifo_blob,
          call write_async(conn_id, mr_id, ptr, size, fifo_blob).
    - On server, unpack received flat buffer into its model.
    """
    device = torch.device("cuda", local_rank)

    # 1) Build model + layout on all ranks
    model = ToyPolicyNet().to(device)

    # initialize differently per rank so we can see it change on receiver
    torch.manual_seed(1234 + rank)
    with torch.no_grad():
        for p in model.parameters():
            nn.init.uniform_(p, a=-0.5, b=0.5)

    layout = build_layout_from_state_dict(model.state_dict())
    total_elems = layout["total_elems"]
    elem_size = 4  # float32
    total_bytes = total_elems * elem_size

    if rank == 0:
        print(
            f"[layout] total_elems={total_elems}, "
            f"approx_size={total_bytes / (1 << 20):.3f} MB",
            flush=True,
        )

    # 2) Create an Endpoint bound to this GPU
    ep = setup_endpoint(local_gpu_idx=local_rank, num_cpus=1)

    # 3) Setup connection with peer via UCCL p2p
    conn_id = setup_p2p_connection(ep, node_rank, peer_rank)

    # 4) Role split: node_rank=1 is the "receiver", node_rank=0 is "sender"
    if node_rank == 1:
        # ============== RECEIVER NODE ==============
        print(f"[R{rank}] acting as RECEIVER for peer R{peer_rank}", flush=True)

        # Allocate flat weight buffer on GPU and register its *byte* region
        weights_buf = torch.empty(total_elems, dtype=torch.float32, device=device)
        weights_bytes = weights_buf.view(torch.uint8)
        nbytes = weights_bytes.numel()

        ptr = int(weights_bytes.data_ptr())
        ok, mr_id = ep.reg(ptr, nbytes)
        assert ok, f"[R{rank}] ep.reg() failed"
        print(f"[R{rank}] reg OK: mr_id={mr_id}, nbytes={nbytes}", flush=True)

        # Advertise this memory to peer: get fifo_blob describing remote location
        ok, fifo_blob = ep.advertise(conn_id, mr_id, ptr, nbytes)
        # If your benchmark uses advertisev(...) for vectors, adapt here.
        assert ok, f"[R{rank}] ep.advertise() failed"
        assert isinstance(fifo_blob, (bytes, bytearray)), \
            f"fifo_blob must be bytes, got {type(fifo_blob)}"
        print(
            f"[R{rank}] advertise OK, fifo_blob_len={len(fifo_blob)}",
            flush=True,
        )

        # Send fifo_blob to peer over torch.distributed
        send_bytes(peer_rank, fifo_blob)
        print(f"[R{rank}] sent fifo_blob to R{peer_rank}", flush=True)

        # Wait for a small "done" flag over dist so we know transfer is complete
        done_flag = torch.zeros(1, dtype=torch.int32)
        dist.recv(done_flag, src=peer_rank)
        print(f"[R{rank}] done_flag={int(done_flag.item())}", flush=True)

        # Now weights_buf has the new params; unpack into model
        unpack_flat_to_state_dict(weights_buf, model, layout)
        with torch.no_grad():
            first_param_mean = next(model.parameters()).mean().item()
        print(
            f"[R{rank}] received weights, first_param.mean={first_param_mean:.6f}",
            flush=True,
        )

    else:
        # ============== SENDER NODE ==============
        print(f"[R{rank}] acting as SENDER for peer R{peer_rank}", flush=True)

        # Pack current model into a flat float32 buffer on GPU
        flat_buf = torch.empty(total_elems, dtype=torch.float32, device=device)
        pack_state_dict_inplace(model, layout, flat_buf)

        # Register its byte region with UCCL
        flat_bytes = flat_buf.view(torch.uint8)
        nbytes = flat_bytes.numel()
        ptr = int(flat_bytes.data_ptr())

        ok, mr_id = ep.reg(ptr, nbytes)
        assert ok, f"[R{rank}] ep.reg() failed"
        print(f"[R{rank}] reg OK: mr_id={mr_id}, nbytes={nbytes}", flush=True)

        # Receive fifo_blob from peer describing the remote memory
        fifo_blob = recv_bytes(peer_rank)
        print(
            f"[R{rank}] got fifo_blob from R{peer_rank}, len={len(fifo_blob)}",
            flush=True,
        )

        # Perform the actual RDMA write via write_async
        t0 = time.time()
        ok, transfer_id = ep.write_async(conn_id, mr_id, ptr, nbytes, fifo_blob)
        assert ok, f"[R{rank}] ep.write_async() failed"

        # Poll until done
        done = False
        while not done:
            ok, done = ep.poll_async(transfer_id)
            assert ok, f"[R{rank}] ep.poll_async() failed"
        torch.cuda.synchronize()
        t1 = time.time()

        gb = nbytes / 1e9
        print(
            f"[R{rank}] wrote {nbytes / (1 << 20):.3f} MB "
            f"in {t1 - t0:.3f} s "
            f"({gb / (t1 - t0):.2f} GB/s)",
            flush=True,
        )

        # Simple "done" flag over dist so receiver knows to unpack
        done_flag = torch.tensor([1], dtype=torch.int32)
        dist.send(done_flag, dst=peer_rank)
        print(f"[R{rank}] sent done_flag to R{peer_rank}", flush=True)


# ============================================================
#  Main
# ============================================================

def main():
    rank, world_size, local_rank, node_rank, gpus_per_node, peer_rank = init_dist()

    dist.barrier()
    if rank == 0:
        print("[INFO] Starting multi-GPU UCCL weight transfer (one shot)", flush=True)
    dist.barrier()

    run_transfer(rank, local_rank, node_rank, peer_rank)

    dist.barrier()
    if rank == 0:
        print("[INFO] All transfers completed", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
