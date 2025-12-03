from __future__ import annotations
import torch, time, os, sys
import torch.distributed as dist

from transformers import GPT2LMHeadModel
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen1.5-0.5B"
from uccl import p2p

def send_model(ep, conn_id, model):
    print(f"[Client] Sending {len(list(model.state_dict().items()))} tensors...")
    for name, tensor in model.state_dict().items():
        if not tensor.is_cuda:
            tensor = tensor.cuda()
        size_bytes = tensor.numel() * tensor.element_size()
        ptr = tensor.data_ptr()
        ok, mr_id = ep.reg(ptr, size_bytes)
        assert ok, f"[Client] Failed to register tensor {name}"
        ok = ep.send(conn_id, mr_id, ptr, size_bytes)
        assert ok, f"[Client] Send failed for {name}"
        print(f"[Client] Sent {name} ({size_bytes/1e6:.2f} MB)")
    print("[Client] Model transfer complete.")


def recv_model(ep, conn_id, model):
    print(f"[Server] Receiving {len(list(model.state_dict().items()))} tensors...")
    for name, tensor in model.state_dict().items():
        recv_tensor = torch.empty_like(tensor, device="cuda")
        size_bytes = recv_tensor.numel() * recv_tensor.element_size()
        ptr = recv_tensor.data_ptr()
        ok, mr_id = ep.reg(ptr, size_bytes)
        assert ok, f"[Server] Failed to register tensor {name}"
        ok = ep.recv(conn_id, mr_id, ptr, size_bytes)
        assert ok, f"[Server] Receive failed for {name}"
        model.state_dict()[name].copy_(recv_tensor)
        print(f"[Server] Received {name} ({size_bytes/1e6:.2f} MB)")
    print("[Server] Model transfer complete.")


def main():
    dist.init_process_group(backend="gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    assert world_size == 2, "Run with two ranks (client/server)."

    local_gpu = rank
    torch.cuda.set_device(local_gpu)

    ep = p2p.Endpoint(local_gpu, 4)
    local_md = ep.get_metadata()

    # exchange metadata
    if rank == 0:
        dist.send(torch.ByteTensor(list(local_md)), dst=1)
        remote_md = torch.zeros(len(local_md), dtype=torch.uint8)
        dist.recv(remote_md, src=1)
    else:
        remote_md = torch.zeros(len(local_md), dtype=torch.uint8)
        dist.recv(remote_md, src=0)
        dist.send(torch.ByteTensor(list(local_md)), dst=0)
    remote_metadata = bytes(remote_md.tolist())

    if rank == 0:
        ip, port, r_gpu = p2p.Endpoint.parse_metadata(remote_metadata)
        ok, conn_id = ep.connect(ip, r_gpu, remote_port=port)
        assert ok, "[Client] connect failed"

        #model = GPT2LMHeadModel.from_pretrained("gpt2").cuda()
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16).cuda()
        
        start = time.perf_counter()
        send_model(ep, conn_id, model)
        print(f"[Client] Transfer finished in {time.perf_counter()-start:.2f}s")

    else:
        ok, r_ip, r_gpu, conn_id = ep.accept()
        assert ok, "[Server] accept failed"

        model = GPT2LMHeadModel.from_pretrained("gpt2").cuda()
        start = time.perf_counter()
        recv_model(ep, conn_id, model)
        print(f"[Server] Transfer finished in {time.perf_counter()-start:.2f}s")

    dist.destroy_process_group()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)

