import os
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from uccl import p2p
import logging
from datetime import datetime


def setup_logging():
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"uccl_inference_{ts}.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )
    logging.info(f"Logging to {log_file}")
    return log_file


def recv_model(ep: p2p.Endpoint, conn_id, model: GPT2LMHeadModel):
    """
    Receive model weights over UCCL from the trainer.
    Mirrors the pattern from gpu_transfer_wikitext2.py:

      - state_dict = model.state_dict()
      - for each tensor:
          recv_tensor = torch.empty_like(tensor, device='cuda')
          size_bytes = recv_tensor.numel() * recv_tensor.element_size()
          ptr = recv_tensor.data_ptr()
          ok, mr_id = ep.reg(ptr, size_bytes)
          ok = ep.recv(conn_id, mr_id, ptr, size_bytes)
          model.state_dict()[name].copy_(recv_tensor)
    """
    state_dict = model.state_dict()
    total_tensors = len(list(state_dict.items()))
    total_size_mb = sum(t.numel() * t.element_size() for t in state_dict.values()) / 1e6

    logging.info("=" * 80)
    logging.info("UCCL RECEIVE START")
    logging.info(f"Total tensors: {total_tensors}")
    logging.info(f"Total size  : {total_size_mb:.2f} MB")
    logging.info("=" * 80)

    recv_start = torch.cuda.Event(enable_timing=True)
    recv_end = torch.cuda.Event(enable_timing=True)
    recv_start.record()

    for idx, (name, tensor) in enumerate(state_dict.items(), 1):
        recv_tensor = torch.empty_like(tensor, device="cuda")
        size_bytes = recv_tensor.numel() * recv_tensor.element_size()
        ptr = recv_tensor.data_ptr()

        ok, mr_id = ep.reg(ptr, size_bytes)
        assert ok, f"Failed to register tensor {name}"

        ok = ep.recv(conn_id, mr_id, ptr, size_bytes)
        assert ok, f"Receive failed for tensor {name}"

        model.state_dict()[name].copy_(recv_tensor)

        if idx % 20 == 0 or idx == total_tensors:
            pct = (idx / total_tensors) * 100
            logging.info(f"Progress: {pct:.1f}% ({idx}/{total_tensors})")

    recv_end.record()
    torch.cuda.synchronize()
    ms = recv_start.elapsed_time(recv_end)  # milliseconds
    total_time = ms / 1000.0
    avg_bandwidth = (total_size_mb / 1000.0) / total_time  # GB/s

    logging.info("=" * 80)
    logging.info("UCCL RECEIVE COMPLETE")
    logging.info(f"Total time: {total_time:.2f}s")
    logging.info(f"Average bandwidth: {avg_bandwidth:.2f} GB/s")
    logging.info("=" * 80)


def main():
    setup_logging()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    local_gpu = 0  # choose whichever GPU you want
    torch.cuda.set_device(local_gpu)
    logging.info(f"Using CUDA device {local_gpu} for inference")

    # Read trainer metadata (written by broadcast_trained_model_uccl)
    metadata_path = "uccl_trainer_metadata.bin"
    assert os.path.exists(metadata_path), f"{metadata_path} not found"
    with open(metadata_path, "rb") as f:
        trainer_md = f.read()

    ip, port, r_gpu = p2p.Endpoint.parse_metadata(trainer_md)
    logging.info(f"Trainer metadata parsed: ip={ip}, port={port}, remote_gpu={r_gpu}")

    # Create our endpoint and connect to trainer
    ep = p2p.Endpoint(local_gpu, 4)
    ok, conn_id = ep.connect(ip, r_gpu, remote_port=port)
    assert ok, "UCCL connect() to trainer failed"
    logging.info(f"UCCL connect() successful, conn_id={conn_id}")

    # Build same base GPT-2 model architecture
    logging.info("Loading base GPT-2 model + tokenizer on inference node...")
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = 1024
    logging.info("Base GPT-2 loaded")

    # Receive weights from trainer via UCCL
    recv_model(ep, conn_id, model)
    model.eval()

    # Quick generation sanity check
    prompt = "The meaning of life is"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=40,
            do_sample=True,
            top_k=50,
            top_p=0.95,
        )

    generated = tokenizer.decode(out_ids[0], skip_special_tokens=True)
    logging.info("=== UCCL INFERENCE CHECK ===")
    logging.info(f"Prompt : {prompt!r}")
    logging.info(f"Output : {generated!r}")
    logging.info("============================")


if __name__ == "__main__":
    main()
