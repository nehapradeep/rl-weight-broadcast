from __future__ import annotations
import torch, time, os, sys
import torch.distributed as dist

from transformers import GPT2LMHeadModel
from uccl import p2p

def broadcast_model(ep, conn_ids, model):
    """Send model to multiple receivers"""
    print(f"[Broadcaster] Sending {len(list(model.state_dict().items()))} tensors to {len(conn_ids)} receivers...")
    
    for name, tensor in model.state_dict().items():
        if not tensor.is_cuda:
            tensor = tensor.cuda()
        size_bytes = tensor.numel() * tensor.element_size()
        ptr = tensor.data_ptr()
        
        # Register memory once
        ok, mr_id = ep.reg(ptr, size_bytes)
        assert ok, f"[Broadcaster] Failed to register tensor {name}"
        
        # Send to all receivers
        for i, conn_id in enumerate(conn_ids):
            ok = ep.send(conn_id, mr_id, ptr, size_bytes)
            assert ok, f"[Broadcaster] Send failed for {name} to receiver {i+1}"
            print(f"[Broadcaster] Sent {name} to receiver {i+1} ({size_bytes/1e6:.2f} MB)")
    
    print("[Broadcaster] Model broadcast complete.")


def recv_model(ep, conn_id, model):
    """Receive model from broadcaster"""
    rank = dist.get_rank()
    print(f"[Receiver {rank}] Receiving {len(list(model.state_dict().items()))} tensors...")
    
    for name, tensor in model.state_dict().items():
        recv_tensor = torch.empty_like(tensor, device="cuda")
        size_bytes = recv_tensor.numel() * recv_tensor.element_size()
        ptr = recv_tensor.data_ptr()
        
        ok, mr_id = ep.reg(ptr, size_bytes)
        assert ok, f"[Receiver {rank}] Failed to register tensor {name}"
        
        ok = ep.recv(conn_id, mr_id, ptr, size_bytes)
        assert ok, f"[Receiver {rank}] Receive failed for {name}"
        
        model.state_dict()[name].copy_(recv_tensor)
        print(f"[Receiver {rank}] Received {name} ({size_bytes/1e6:.2f} MB)")
    
    print(f"[Receiver {rank}] Model transfer complete.")


def main():
    dist.init_process_group(backend="gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    assert world_size == 3, "Run with three ranks (1 broadcaster + 2 receivers)."

    local_gpu = rank
    torch.cuda.set_device(local_gpu)

    ep = p2p.Endpoint(local_gpu, 4)
    local_md = ep.get_metadata()

    # Exchange metadata - all-to-all style
    # Each rank needs to know about rank 0's metadata
    all_metadata = [None] * world_size
    all_metadata[rank] = local_md
    
    # Gather all metadata to all ranks
    for i in range(world_size):
        if i == rank:
            # Send my metadata to all others
            for j in range(world_size):
                if j != rank:
                    dist.send(torch.ByteTensor(list(local_md)), dst=j)
        else:
            # Receive metadata from rank i
            remote_md = torch.zeros(len(local_md), dtype=torch.uint8)
            dist.recv(remote_md, src=i)
            all_metadata[i] = bytes(remote_md.tolist())

    if rank == 0:
        # Broadcaster: connect to rank 1 and rank 2
        print("[Broadcaster] Connecting to receivers...")
        conn_ids = []
        
        for receiver_rank in [1, 2]:
            ip, port, r_gpu = p2p.Endpoint.parse_metadata(all_metadata[receiver_rank])
            ok, conn_id = ep.connect(ip, r_gpu, remote_port=port)
            assert ok, f"[Broadcaster] connect failed to rank {receiver_rank}"
            conn_ids.append(conn_id)
            print(f"[Broadcaster] Connected to receiver {receiver_rank}")

        model = GPT2LMHeadModel.from_pretrained("gpt2").cuda()
        start = time.perf_counter()
        broadcast_model(ep, conn_ids, model)
        print(f"[Broadcaster] Transfer finished in {time.perf_counter()-start:.2f}s")

    else:
        # Receivers (rank 1 and 2): accept connection from rank 0
        print(f"[Receiver {rank}] Waiting for broadcaster connection...")
        ok, r_ip, r_gpu, conn_id = ep.accept()
        assert ok, f"[Receiver {rank}] accept failed"
        print(f"[Receiver {rank}] Connected to broadcaster")

        model = GPT2LMHeadModel.from_pretrained("gpt2").cuda()
        start = time.perf_counter()
        recv_model(ep, conn_id, model)
        print(f"[Receiver {rank}] Transfer finished in {time.perf_counter()-start:.2f}s")

    dist.destroy_process_group()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
