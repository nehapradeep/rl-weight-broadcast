from __future__ import annotations
import torch, time, os, sys
import torch.distributed as dist
import logging
from datetime import datetime
import math

from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader
from uccl import p2p
import torch.nn as nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers.models.gpt2.modeling_gpt2 import GPT2Block
from torch.utils.data.distributed import DistributedSampler

# ---------------------------
# Logging
# ---------------------------
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


# ---------------------------
# RDMA send/recv helpers
# ---------------------------
def broadcast_model(ep, conn_ids, model, rank):
    """Send model to multiple receivers with detailed logging"""
    state_dict = model.state_dict()
    items = list(state_dict.items())
    total_tensors = len(items)
    total_size_mb = sum(t.numel() * t.element_size() for _, t in items) / 1e6
    
    logging.info("="*80)
    logging.info(f"BROADCAST START - Sending to {len(conn_ids)} receivers")
    logging.info(f"Total tensors: {total_tensors}")
    logging.info(f"Total size: {total_size_mb:.2f} MB")
    logging.info("="*80)
    
    broadcast_start = time.perf_counter()
    
    for idx, (name, tensor) in enumerate(items, 1):
        # ensure tensor on GPU for RDMA
        if not tensor.is_cuda:
            tensor = tensor.cuda()
        
        size_bytes = tensor.numel() * tensor.element_size()
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
    avg_bandwidth = (total_size_mb / 1000) / total_time  # GB/s approximation
    
    logging.info("="*80)
    logging.info(f"BROADCAST COMPLETE")
    logging.info(f"Total time: {total_time:.2f}s")
    logging.info(f"Average bandwidth: {avg_bandwidth:.2f} GB/s")
    logging.info("="*80)


def recv_model(ep, conn_id, model, rank):
    """Receive model from broadcaster with detailed logging"""
    state_dict = model.state_dict()
    items = list(state_dict.items())
    total_tensors = len(items)
    total_size_mb = sum(t.numel() * t.element_size() for _, t in items) / 1e6
    
    logging.info("="*80)
    logging.info(f"RECEIVE START")
    logging.info(f"Total tensors: {total_tensors}")
    logging.info(f"Total size: {total_size_mb:.2f} MB")
    logging.info("="*80)
    
    recv_start = time.perf_counter()
    
    for idx, (name, tensor) in enumerate(items, 1):
        # allocate recv tensor on GPU
        recv_tensor = torch.empty_like(tensor, device="cuda")
        size_bytes = recv_tensor.numel() * recv_tensor.element_size()
        ptr = recv_tensor.data_ptr()
        
        # Register memory
        ok, mr_id = ep.reg(ptr, size_bytes)
        assert ok, f"Failed to register tensor {name}"
        
        # Receive tensor
        ok = ep.recv(conn_id, mr_id, ptr, size_bytes)
        assert ok, f"Receive failed for {name}"
        
        # Copy into model parameter / buffer (assumes model on GPU)
        target = model.state_dict()[name]
        # target may be on cuda already if model.cuda() was called
        if not target.is_cuda:
            # move model param/buffer to cuda if needed (should normally be already)
            target = target.cuda()
        # use in-place copy
        model.state_dict()[name].copy_(recv_tensor)
        
        if idx % 20 == 0 or idx == total_tensors:
            progress_pct = (idx / total_tensors) * 100
            logging.info(f"Progress: {progress_pct:.1f}% ({idx}/{total_tensors})")
    
    total_time = time.perf_counter() - recv_start
    avg_bandwidth = (total_size_mb / 1000) / total_time  # GB/s approx
    
    logging.info("="*80)
    logging.info(f"RECEIVE COMPLETE")
    logging.info(f"Total time: {total_time:.2f}s")
    logging.info(f"Average bandwidth: {avg_bandwidth:.2f} GB/s")
    logging.info("="*80)


# ---------------------------
# Dataset preparation
# ---------------------------
def prepare_dataset(tokenizer, max_length=128, num_samples=200):
    """Load and prepare WikiText-2 dataset"""
    logging.info("="*80)
    logging.info("LOADING DATASET")
    logging.info("="*80)
    logging.info("Loading WikiText-2 dataset from Hugging Face...")
    
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    logging.info(f"Dataset loaded: {len(dataset)} total examples")
    
    dataset = dataset.filter(lambda x: len(x["text"].strip()) > 0)
    logging.info(f"After filtering empty texts: {len(dataset)} examples")
    
    if len(dataset) > num_samples:
        dataset = dataset.select(range(num_samples))
        logging.info(f"Using subset: {num_samples} examples")
    
    logging.info("Tokenizing dataset...")
    
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length
        )
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names
    )
    
    tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])
    
    logging.info(f"Tokenization complete")
    logging.info(f"  - Examples: {len(tokenized_dataset)}")
    logging.info(f"  - Max length: {max_length} tokens")
    logging.info("="*80)
    
    return tokenized_dataset


# ---------------------------
# Training (rank 1..N)
# ---------------------------
def run_training(model, tokenizer, num_epochs=2, batch_size=4, lr=5e-5):
    """Training loop for trainer ranks"""
    logging.info("="*80)
    logging.info("TRAINING NODE - Starting training on WikiText-2")
    logging.info(f"Visible GPUs on this node: {torch.cuda.device_count()}")
    logging.info(f"Model class: {model.__class__.__name__}")
    logging.info("="*80)
    
    train_dataset = prepare_dataset(tokenizer, max_length=128, num_samples=200)
    # Use distributed sampler with explicit replicas/rank
    sampler = DistributedSampler(train_dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank(), shuffle=True)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler
    )
    
    total_steps = len(train_dataloader) * num_epochs
    logging.info(f"Training configuration:")
    logging.info(f"  - Examples: {len(train_dataset)}")
    logging.info(f"  - Batch size: {batch_size}")
    logging.info(f"  - Epochs: {num_epochs}")
    logging.info(f"  - Steps per epoch: {len(train_dataloader)}")
    logging.info(f"  - Total steps: {total_steps}")
    logging.info(f"  - Learning rate: {lr}")
    logging.info("="*80)
    
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # Proper AMP usage
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    total_train_start = time.perf_counter()
    global_step = 0
    epoch_losses = []
    
    for epoch in range(num_epochs):
        epoch_start = time.perf_counter()
        epoch_loss = 0.0
        epoch_perplexity = 0.0
        
        logging.info(f"\n{'='*80}")
        logging.info(f"EPOCH {epoch + 1}/{num_epochs}")
        logging.info(f"{'='*80}")
        
        sampler.set_epoch(epoch)  # set epoch for DistributedSampler
        
        for batch_idx, batch in enumerate(train_dataloader):
            step_start = time.perf_counter()
            global_step += 1
            
            # Move batch to GPU
            input_ids = batch["input_ids"].cuda(non_blocking=True)
            attention_mask = batch["attention_mask"].cuda(non_blocking=True)
            
            # Forward + backward with autocast
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=input_ids
                )
                loss = outputs.loss
            
            # reduce if per-GPU vector
            if isinstance(loss, torch.Tensor) and loss.dim() > 0:
                loss_scalar = loss.mean()
            else:
                loss_scalar = loss

            scaler.scale(loss_scalar).backward()
            scaler.step(optimizer)
            scaler.update()
            
            step_time = time.perf_counter() - step_start
            
            loss_value = loss_scalar.item()
            perplexity = math.exp(loss_value) if loss_value < 100 else float('inf')
            
            epoch_loss += loss_value
            epoch_perplexity += perplexity
            
            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(train_dataloader):
                logging.info(
                    f"Step {global_step}/{total_steps} "
                    f"(Epoch {epoch+1}, Batch {batch_idx+1}/{len(train_dataloader)}) | "
                    f"Loss: {loss_value:.4f} | Perplexity: {perplexity:.2f} | "
                    f"Time: {step_time:.3f}s"
                )
            
            if global_step % 20 == 0:
                total_norm_sq = 0.0
                for p in model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm_sq += param_norm.item() ** 2
                total_norm = total_norm_sq ** 0.5
                logging.info(f"  -> Gradient norm: {total_norm:.4f}")
        
        epoch_time = time.perf_counter() - epoch_start
        avg_epoch_loss = epoch_loss / max(1, len(train_dataloader))
        avg_epoch_perplexity = epoch_perplexity / max(1, len(train_dataloader))
        epoch_losses.append(avg_epoch_loss)
        
        logging.info(f"\n{'='*80}")
        logging.info(f"EPOCH {epoch + 1} SUMMARY")
        logging.info(f"Average Loss: {avg_epoch_loss:.4f}")
        logging.info(f"Average Perplexity: {avg_epoch_perplexity:.2f}")
        logging.info(f"Epoch Time: {epoch_time:.2f}s")
        logging.info(f"{'='*80}\n")
    
    total_train_time = time.perf_counter() - total_train_start
    avg_step_time = total_train_time / max(1, total_steps)
    
    logging.info("="*80)
    logging.info("TRAINING COMPLETE")
    logging.info(f"Total training time: {total_train_time:.2f}s")
    logging.info(f"Total steps: {total_steps}")
    logging.info(f"Average time per step: {avg_step_time:.3f}s")
    logging.info(f"Steps per second: {total_steps/total_train_time:.2f}")
    logging.info(f"Final loss: {epoch_losses[-1]:.4f}")
    logging.info(f"Loss improvement: {epoch_losses[0] - epoch_losses[-1]:.4f}")
    logging.info("="*80)
    
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Save model state depending on FSDP
    save_model = model
    if isinstance(save_model, FSDP):
        # FSDP: use state_dict() (it may return sharded; user can use FSDP.full_state_dict if available)
        try:
            sd = save_model.state_dict()
        except Exception:
            sd = save_model.module.state_dict()
    else:
        sd = save_model.state_dict()
    
    checkpoint_path = f"{checkpoint_dir}/model_rank{dist.get_rank()}_wikitext2_trained.pt"
    torch.save({
        'model_state_dict': sd,
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch_losses': epoch_losses,
        'total_steps': total_steps,
        'final_loss': epoch_losses[-1],
    }, checkpoint_path)
    logging.info(f"Model checkpoint saved to: {checkpoint_path}")


# ---------------------------
# Inference (rank last)
# ---------------------------
def run_inference(model, tokenizer, num_samples=5):
    logging.info("="*80)
    logging.info("INFERENCE NODE - Starting inference")
    logging.info("="*80)
    
    model.eval()
    prompts = [
        "Once upon a time",
        "The future of artificial intelligence",
        "In a world where technology",
        "Scientists have discovered",
        "The most important thing in life",
        "Deep learning models",
        "Natural language processing",
        "Machine learning algorithms",
    ]
    
    logging.info(f"Running inference on {num_samples} prompts")
    logging.info("Generation settings: max_length=50, temperature=0.7, top_k=50, top_p=0.95")
    
    total_inference_start = time.perf_counter()
    total_tokens = 0
    
    with torch.no_grad():
        for idx, prompt in enumerate(prompts[:num_samples], 1):
            inference_start = time.perf_counter()
            
            logging.info(f"\n{'='*60}")
            logging.info(f"SAMPLE {idx}/{num_samples}")
            logging.info(f"{'='*60}")
            logging.info(f"Prompt: '{prompt}'")
            
            inputs = tokenizer(prompt, return_tensors="pt")
            input_ids = inputs["input_ids"].cuda()
            
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
            
            generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            
            inference_time = time.perf_counter() - inference_start
            tokens_generated = output_ids.shape[1] - input_ids.shape[1]
            total_tokens += tokens_generated
            tokens_per_sec = tokens_generated / generation_time if generation_time > 0 else 0
            
            logging.info(f"Output: '{generated_text}'")
            logging.info(f"{'='*60}")
            logging.info(f"Metrics:")
            logging.info(f"  - Total inference time: {inference_time:.3f}s")
            logging.info(f"  - Generation time: {generation_time:.3f}s")
            logging.info(f"  - Tokens generated: {tokens_generated}")
            logging.info(f"  - Tokens/sec: {tokens_per_sec:.1f}")
            logging.info(f"{'='*60}")
    
    total_inference_time = time.perf_counter() - total_inference_start
    avg_inference_time = total_inference_time / max(1, num_samples)
    avg_tokens_per_sec = total_tokens / max(1e-9, total_inference_time)
    
    logging.info("\n" + "="*80)
    logging.info("INFERENCE COMPLETE")
    logging.info(f"Total inference time: {total_inference_time:.2f}s")
    logging.info(f"Average time per sample: {avg_inference_time:.3f}s")
    logging.info(f"Samples per second: {num_samples/total_inference_time:.2f}")
    logging.info(f"Total tokens generated: {total_tokens}")
    logging.info(f"Average tokens/sec: {avg_tokens_per_sec:.1f}")
    logging.info("="*80)


# ---------------------------
# Main (all ranks)
# ---------------------------
def main():
    # init from environment (torchrun)
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    log_file = setup_logging(rank)
    logging.info(f"Process started - Rank: {rank}, World size: {world_size}")
    assert world_size >= 3, "Need at least 3 ranks: broadcaster + trainer(s) + inference"

    # Setup CUDA device and local_gpu
    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        torch.cuda.set_device(local_rank)
        local_gpu = torch.cuda.current_device()         # <-- FIX: define local_gpu
        logging.info(f"CUDA device set to GPU {local_gpu}")
    else:
        local_gpu = None
        logging.info("CUDA not available. (RDMA will not work.)")
    
    logging.info("Initializing P2P endpoint...")
    ep = p2p.Endpoint(local_gpu, 4)
    local_md = ep.get_metadata()
    logging.info(f"Local metadata obtained (size: {len(local_md)} bytes)")
    
    # Exchange metadata robustly using all_gather_object
    logging.info("Starting metadata exchange (all_gather_object)...")
    all_metadata = [None] * world_size
    dist.all_gather_object(all_metadata, local_md)
    logging.info("Metadata exchange complete")
    
    if rank == 0:
        # Broadcaster
        logging.info("="*80)
        logging.info("BROADCASTER MODE")
        logging.info("="*80)
        logging.info("Connecting to receivers...")
        conn_ids = []
        
        # connect to all non-zero ranks except possibly extra ones
        receiver_ranks = [r for r in range(1, world_size)]  # all other ranks
        for receiver_rank in receiver_ranks:
            # parse remote metadata
            ip, port, r_gpu = p2p.Endpoint.parse_metadata(all_metadata[receiver_rank])
            logging.info(f"Connecting to rank {receiver_rank}: IP={ip}, Port={port}, GPU={r_gpu}")
            
            ok, conn_id = ep.connect(ip, r_gpu, remote_port=port)
            assert ok, f"Connect failed to rank {receiver_rank}"
            conn_ids.append((receiver_rank, conn_id))
            node_type = "Training Node" if receiver_rank != world_size - 1 else "Inference Node"
            logging.info(f"Connected to {node_type} (rank {receiver_rank}, conn_id={conn_id})")
        
        logging.info("Loading model on broadcaster...")
        model = GPT2LMHeadModel.from_pretrained("gpt2").cuda()
        logging.info("Model loaded")
        
        # pass just list of conn_ids ints to broadcast_model
        broadcast_model(ep, [c for _, c in conn_ids], model, rank)
        
        logging.info("\nBroadcast complete. Other nodes should now process...")
    
    elif 1 <= rank < world_size - 1:
        # Trainer(s)
        logging.info("="*80)
        logging.info(f"TRAINING NODE (Rank {rank})")
        logging.info("="*80)
        logging.info("Waiting for broadcaster connection...")
        
        ok, r_ip, r_gpu, conn_id = ep.accept()
        assert ok, "Accept failed"
        logging.info("Connected to broadcaster")
        
        logging.info("Loading model and tokenizer...")
        base_model = GPT2LMHeadModel.from_pretrained("gpt2").cuda()
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token  # use eos token as pad
        logging.info("Model and tokenizer loaded")
        
        recv_model(ep, conn_id, base_model, rank)
        # FSDP auto wrap policy for GPT-2 transformer blocks
        auto_wrap_policy = transformer_auto_wrap_policy({GPT2Block})
        logging.info("Wrapping model with FSDP across nodes...")
        model = FSDP(
            base_model,
            auto_wrap_policy=auto_wrap_policy,
            device_id=torch.cuda.current_device()
        )
        
        run_training(model, tokenizer, num_epochs=2, batch_size=4, lr=5e-5)
    
    elif rank == world_size - 1:
        # Inference node (last rank)
        logging.info("="*80)
        logging.info("INFERENCE NODE")
        logging.info("="*80)
        logging.info("Waiting for broadcaster connection...")
        
        ok, r_ip, r_gpu, conn_id = ep.accept()
        assert ok, "Accept failed"
        logging.info("Connected to broadcaster")
        
        logging.info("Loading model and tokenizer...")
        model = GPT2LMHeadModel.from_pretrained("gpt2").cuda()
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        logging.info("Model and tokenizer loaded")
        
        recv_model(ep, conn_id, model, rank)
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
