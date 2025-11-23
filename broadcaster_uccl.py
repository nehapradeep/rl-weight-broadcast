# broadcaster_uccl.py
from __future__ import annotations
import os
import sys
import time
import logging
from datetime import datetime

import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel

from uccl import p2p


def setup_logging() -> str:
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"{log_dir}/broadcaster_{timestamp}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | Broadcaster | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout),
        ],
    )

    logging.info(f"Logging initialized. Log file: {log_file}")
    return log_file


def broadcast_model_uccl(ep: p2p.Endpoint, conn_ids, model: nn.Module):
    """Send model parameters to one or more receivers via UCCl."""
    state_dict = model.state_dict()
    total_tensors = len(list(state_dict.items()))
    total_size_mb = sum(t.numel() * t.element_size() for t in state_dict.values()) / 1e6

    logging.info("=" * 80)
    logging.info(f"UCCl BROADCAST START - Sending to {len(conn_ids)} receiver(s)")
    logging.info(f"Total tensors: {total_tensors}")
    logging.info(f"Total size: {total_size_mb:.2f} MB")
    logging.info("=" * 80)

    t0 = time.perf_counter()

    for idx, (name, tensor) in enumerate(state_dict.items(), 1):
        if not tensor.is_cuda:
            tensor = tensor.cuda()

        size_bytes = tensor.numel() * tensor.element_size()
        ptr = tensor.data_ptr()

        ok, mr_id = ep.reg(ptr, size_bytes)
        assert ok, f"UCCl: failed to register tensor {name}"

        for receiver_idx, conn_id in enumerate(conn_ids, 1):
            ok = ep.send(conn_id, mr_id, ptr, size_bytes)
            assert ok, f"UCCl: send failed for {name} to receiver {receiver_idx}"

        if idx % 20 == 0 or idx == total_tensors:
            pct = (idx / total_tensors) * 100
            logging.info(f"UCCl progress: {pct:.1f}% ({idx}/{total_tensors})")

    total_time = time.perf_counter() - t0
    bw_gbps = (total_size_mb / 1000.0) / total_time if total_time > 0 else 0.0

    logging.info("=" * 80)
    logging.info("UCCl BROADCAST COMPLETE")
    logging.info(f"Total time: {total_time:.2f}s")
    logging.info(f"Average bandwidth: {bw_gbps:.2f} GB/s")
    logging.info("=" * 80)


def wait_for_metadata(metadata_path: str, poll_interval: float = 1.0) -> bytes:
    logging.info(f"Waiting for UCCl metadata file from trainer at: {metadata_path}")
    while True:
        try:
            if os.path.exists(metadata_path) and os.path.getsize(metadata_path) > 0:
                with open(metadata_path, "rb") as f:
                    data = f.read()
                logging.info("Successfully read UCCl metadata from trainer.")
                return data
        except Exception as e:
            logging.warning(f"Error reading metadata file: {e}")
        time.sleep(poll_interval)


def main():
    log_file = setup_logging()

    # ------------------------------------------------------------------
    # Config
    # ------------------------------------------------------------------
    # Path that trainer rank 0 will write its UCCl metadata to.
    metadata_path = os.environ.get("UCCL_MD_PATH", "uccl_root_md.bin")

    # Which GPU broadcaster uses (single GPU)
    local_gpu_idx = int(os.environ.get("BROADCASTER_GPU", "0"))

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for broadcaster.")

    torch.cuda.set_device(local_gpu_idx)
    device = torch.device("cuda", local_gpu_idx)
    logging.info(f"Broadcaster using CUDA device index {local_gpu_idx}")

    # ------------------------------------------------------------------
    # Initialize UCCl endpoint on this GPU
    # ------------------------------------------------------------------
    logging.info("Initializing UCCl endpoint on broadcaster...")
    ep = p2p.Endpoint(local_gpu_idx, 4)
    local_md = ep.get_metadata()
    logging.info(f"Broadcaster UCCl local metadata size: {len(local_md)} bytes")

    # ------------------------------------------------------------------
    # Get trainer's UCCl metadata from shared file
    # ------------------------------------------------------------------
    root_md_bytes = wait_for_metadata(metadata_path)
    ip, port, r_gpu = p2p.Endpoint.parse_metadata(root_md_bytes)
    logging.info(
        f"Connecting to trainer root via UCCl: ip={ip}, port={port}, remote_gpu={r_gpu}"
    )

    ok, conn_id = ep.connect(ip, r_gpu, remote_port=port)
    assert ok, "UCCl connect failed to trainer root"
    logging.info(f"Broadcaster connected to trainer root (conn_id={conn_id})")

    # ------------------------------------------------------------------
    # Load model and broadcast weights
    # ------------------------------------------------------------------
    logging.info("Loading GPT-2 model on broadcaster...")
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    logging.info("Model loaded on broadcaster")

    broadcast_model_uccl(ep, [conn_id], model)
    logging.info("Broadcaster done sending weights. Exiting.")
    logging.info(f"Log file: {log_file}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logging.warning("Broadcaster interrupted by user")
        sys.exit(0)
    except Exception as e:
        logging.error(f"Broadcaster fatal error: {e}", exc_info=True)
        sys.exit(1)
