from __future__ import annotations
import torch, time, os, sys
import torch.distributed as dist
import logging
from datetime import datetime

from transformers import GPT2LMHeadModel, GPT2Tokenizer
from uccl import p2p

# Setup logging
def setup_logging(rank):
    """Setup logging with timestamps and rank info"""
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"{log_dir}/rank_{rank}_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | Rank %(rank)d | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Add rank to all log records
    old_factory = logging.getLogRecordFactory()
    def record_factory(*args, **kwargs):
        record = old_factory(*args, **kwargs)
        record.rank = rank
        return record
    logging.setLogRecordFactory(record_factory)
    
    logging.info(f"Logging initialized. Log file: {log_file}")
    return log_file


def broadcast_model(ep, conn_ids, model, rank):
    """Send model to multiple receivers with detailed logging"""
    state_dict = model.state_dict()
    total_tensors = len(list(state_dict.items()))
    total_size_mb = sum(t.numel() * t.element_size() for t in state_dict.values()) / 1e6
    
    logging.info("="*80)
    logging.info(f"BROADCAST START - Sending to {len(conn_ids)} receivers")
    logging.info(f"Total tensors: {total_tensors}")
    logging.info(f"Total size: {total_size_mb:.2f} MB")
    logging.info("="*80)
    
    broadcast_start = time.perf_counter()
    
    for idx, (name, tensor) in enumerate(state_dict.items(), 1):
        if not tensor.is_cuda:
            tensor = tensor.cuda()
        
        size_bytes = tensor.numel() * tensor.element_size()
        size_mb = size_bytes / 1e6
        ptr = tensor.data_ptr()
        
        # Register memory
        ok, mr_id = ep.reg(ptr, size_bytes)
        assert ok, f"Failed to register tensor {name}"
        
        # Send to all receivers
        for receiver_idx, conn_id in enumerate(conn_ids, 1):
            ok = ep.send(conn_id, mr_id, ptr, size_bytes)
            assert ok, f"Send failed for {name} to receiver {receiver_idx}"
        
        if idx % 20 == 0 or idx == total_tensors:
            progress_pct = (idx / total_tensors) * 100
            logging.info(f"Progress: {progress_pct:.1f}% ({idx}/{total_tensors})")
    
    total_time = time.perf_counter() - broadcast_start
    avg_bandwidth = (total_size_mb / 1000) / total_time  # GB/s
    
    logging.info("="*80)
    logging.info(f"BROADCAST COMPLETE")
    logging.info(f"Total time: {total_time:.2f}s")
    logging.info(f"Average bandwidth: {avg_bandwidth:.2f} GB/s")
    logging.info("="*80)


def recv_model(ep, conn_id, model, rank):
    """Receive model from broadcaster with detailed logging"""
    state_dict = model.state_dict()
    total_tensors = len(list(state_dict.items()))
    total_size_mb = sum(t.numel() * t.element_size() for t in state_dict.values()) / 1e6
    
    logging.info("="*80)
    logging.info(f"RECEIVE START")
    logging.info(f"Total tensors: {total_tensors}")
    logging.info(f"Total size: {total_size_mb:.2f} MB")
    logging.info("="*80)
    
    recv_start = time.perf_counter()
    
    for idx, (name, tensor) in enumerate(state_dict.items(), 1):
        recv_tensor = torch.empty_like(tensor, device="cuda")
        size_bytes = recv_tensor.numel() * recv_tensor.element_size()
        ptr = recv_tensor.data_ptr()
        
        # Register memory
        ok, mr_id = ep.reg(ptr, size_bytes)
        assert ok, f"Failed to register tensor {name}"
        
        # Receive tensor
        ok = ep.recv(conn_id, mr_id, ptr, size_bytes)
        assert ok, f"Receive failed for {name}"
        
        model.state_dict()[name].copy_(recv_tensor)
        
        if idx % 20 == 0 or idx == total_tensors:
            progress_pct = (idx / total_tensors) * 100
            logging.info(f"Progress: {progress_pct:.1f}% ({idx}/{total_tensors})")
    
    total_time = time.perf_counter() - recv_start
    avg_bandwidth = (total_size_mb / 1000) / total_time  # GB/s
    
    logging.info("="*80)
    logging.info(f"RECEIVE COMPLETE")
    logging.info(f"Total time: {total_time:.2f}s")
    logging.info(f"Average bandwidth: {avg_bandwidth:.2f} GB/s")
    logging.info("="*80)


def run_training(model, tokenizer, num_steps=10):
    """Training loop for rank 1"""
    logging.info("="*80)
    logging.info("TRAINING NODE - Starting training")
    logging.info("="*80)
    
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    
    # Sample training data
    training_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is transforming the world.",
        "Deep learning models require significant computational resources.",
        "Natural language processing enables computers to understand human language.",
        "Artificial intelligence is the future of technology.",
    ]
    
    logging.info(f"Training on {len(training_texts)} samples for {num_steps} steps")
    
    total_train_start = time.perf_counter()
    
    for step in range(num_steps):
        step_start = time.perf_counter()
        
        # Randomly select a training text
        text = training_texts[step % len(training_texts)]
        
        # Tokenize
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
        input_ids = inputs["input_ids"].cuda()
        attention_mask = inputs["attention_mask"].cuda()
        
        # Forward pass
        outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
        loss = outputs.loss
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        step_time = time.perf_counter() - step_start
        
        logging.info(f"Step {step+1}/{num_steps} | Loss: {loss.item():.4f} | "
                    f"Time: {step_time:.3f}s | Text: '{text[:50]}...'")
        
        # Log gradient norms every 5 steps
        if (step + 1) % 5 == 0:
            total_norm = 0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            logging.info(f"  -> Gradient norm: {total_norm:.4f}")
    
    total_train_time = time.perf_counter() - total_train_start
    avg_step_time = total_train_time / num_steps
    
    logging.info("="*80)
    logging.info("TRAINING COMPLETE")
    logging.info(f"Total training time: {total_train_time:.2f}s")
    logging.info(f"Average time per step: {avg_step_time:.3f}s")
    logging.info(f"Steps per second: {num_steps/total_train_time:.2f}")
    logging.info("="*80)
    
    # Save checkpoint
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = f"{checkpoint_dir}/model_rank1_trained.pt"
    torch.save(model.state_dict(), checkpoint_path)
    logging.info(f"Model checkpoint saved to: {checkpoint_path}")


def run_inference(model, tokenizer, num_samples=5):
    """Inference loop for rank 2"""
    logging.info("="*80)
    logging.info("INFERENCE NODE - Starting inference")
    logging.info("="*80)
    
    model.eval()
    
    # Sample prompts for inference
    prompts = [
        "Once upon a time",
        "The future of artificial intelligence",
        "In a world where technology",
        "Scientists have discovered",
        "The most important thing in life",
    ]
    
    logging.info(f"Running inference on {num_samples} prompts")
    
    total_inference_start = time.perf_counter()
    
    with torch.no_grad():
        for idx, prompt in enumerate(prompts[:num_samples], 1):
            inference_start = time.perf_counter()
            
            logging.info(f"\n--- Sample {idx}/{num_samples} ---")
            logging.info(f"Prompt: '{prompt}'")
            
            # Tokenize input
            inputs = tokenizer(prompt, return_tensors="pt")
            input_ids = inputs["input_ids"].cuda()
            
            # Generate text
            generation_start = time.perf_counter()
            output_ids = model.generate(
                input_ids,
                max_length=50,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                pad_token_id=tokenizer.eos_token_id
            )
            generation_time = time.perf_counter() - generation_start
            
            # Decode output
            generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            
            inference_time = time.perf_counter() - inference_start
            tokens_generated = output_ids.shape[1] - input_ids.shape[1]
            tokens_per_sec = tokens_generated / generation_time if generation_time > 0 else 0
            
            logging.info(f"Generated: '{generated_text}'")
            logging.info(f"Inference time: {inference_time:.3f}s | "
                        f"Generation time: {generation_time:.3f}s | "
                        f"Tokens generated: {tokens_generated} | "
                        f"Tokens/sec: {tokens_per_sec:.1f}")
    
    total_inference_time = time.perf_counter() - total_inference_start
    avg_inference_time = total_inference_time / num_samples
    
    logging.info("\n" + "="*80)
    logging.info("INFERENCE COMPLETE")
    logging.info(f"Total inference time: {total_inference_time:.2f}s")
    logging.info(f"Average time per sample: {avg_inference_time:.3f}s")
    logging.info(f"Samples per second: {num_samples/total_inference_time:.2f}")
    logging.info("="*80)


def main():
    dist.init_process_group(backend="gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    # Setup logging first
    log_file = setup_logging(rank)
    
    logging.info(f"Process started - Rank: {rank}, World size: {world_size}")
    assert world_size == 3, "Run with three ranks (1 broadcaster + 1 training + 1 inference)."

    local_gpu = rank
    torch.cuda.set_device(local_gpu)
    logging.info(f"CUDA device set to GPU {local_gpu}")

    # Initialize endpoint
    logging.info("Initializing P2P endpoint...")
    ep = p2p.Endpoint(local_gpu, 4)
    local_md = ep.get_metadata()
    logging.info(f"Local metadata obtained (size: {len(local_md)} bytes)")

    # Exchange metadata - all-to-all style
    logging.info("Starting metadata exchange...")
    all_metadata = [None] * world_size
    all_metadata[rank] = local_md
    
    metadata_start = time.perf_counter()
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
    
    metadata_time = time.perf_counter() - metadata_start
    logging.info(f"Metadata exchange complete in {metadata_time:.2f}s")

    if rank == 0:
        # Broadcaster: connect to rank 1 and rank 2
        logging.info("="*80)
        logging.info("BROADCASTER MODE")
        logging.info("="*80)
        logging.info("Connecting to receivers...")
        conn_ids = []
        
        for receiver_rank in [1, 2]:
            ip, port, r_gpu = p2p.Endpoint.parse_metadata(all_metadata[receiver_rank])
            logging.info(f"Connecting to rank {receiver_rank}: IP={ip}, Port={port}, GPU={r_gpu}")
            
            ok, conn_id = ep.connect(ip, r_gpu, remote_port=port)
            assert ok, f"Connect failed to rank {receiver_rank}"
            conn_ids.append(conn_id)
            
            node_type = "Training Node" if receiver_rank == 1 else "Inference Node"
            logging.info(f"Connected to {node_type} (rank {receiver_rank}, conn_id={conn_id})")

        logging.info("Loading model...")
        model = GPT2LMHeadModel.from_pretrained("gpt2").cuda()
        logging.info("Model loaded")
        
        broadcast_model(ep, conn_ids, model, rank)
        
        logging.info("\nBroadcast complete. Training node (rank 1) and Inference node (rank 2) "
                    "are now processing...")

    elif rank == 1:
        # Training node
        logging.info("="*80)
        logging.info("TRAINING NODE (Rank 1)")
        logging.info("="*80)
        logging.info("Waiting for broadcaster connection...")
        
        ok, r_ip, r_gpu, conn_id = ep.accept()
        assert ok, "Accept failed"
        logging.info(f"Connected to broadcaster")

        logging.info("Loading model and tokenizer...")
        model = GPT2LMHeadModel.from_pretrained("gpt2").cuda()
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        logging.info("Model and tokenizer loaded")
        
        recv_model(ep, conn_id, model, rank)
        
        # Start training
        run_training(model, tokenizer, num_steps=10)

    else:  # rank == 2
        # Inference node
        logging.info("="*80)
        logging.info("INFERENCE NODE (Rank 2)")
        logging.info("="*80)
        logging.info("Waiting for broadcaster connection...")
        
        ok, r_ip, r_gpu, conn_id = ep.accept()
        assert ok, "Accept failed"
        logging.info(f"Connected to broadcaster")

        logging.info("Loading model and tokenizer...")
        model = GPT2LMHeadModel.from_pretrained("gpt2").cuda()
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        logging.info("Model and tokenizer loaded")
        
        recv_model(ep, conn_id, model, rank)
        
        # Start inference
        run_inference(model, tokenizer, num_samples=5)

    logging.info("Destroying process group...")
    dist.destroy_process_group()
    logging.info("Process complete. Exiting.")
    logging.info(f"Full log saved to: {log_file}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logging.warning("Interrupted by user")
        sys.exit(0)
    except Exception as e:
        logging.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)
