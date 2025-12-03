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
    
    # Keep references to temporary GPU tensors to prevent GC before send completes
    temp_tensors = [] 

    for idx, (name, tensor) in enumerate(items, 1):
        # ensure tensor on GPU for RDMA
        if not tensor.is_cuda:
            tensor = tensor.cuda()
            temp_tensors.append(tensor) # Keep reference
        
        # Ensure contiguous memory for RDMA
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()
            temp_tensors.append(tensor)

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
    avg_bandwidth = (total_size_mb / 1000) / total_time if total_time > 0 else 0
    
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
        if not recv_tensor.is_contiguous():
            recv_tensor = recv_tensor.contiguous()

        size_bytes = recv_tensor.numel() * recv_tensor.element_size()
        ptr = recv_tensor.data_ptr()
        
        # Register memory
        ok, mr_id = ep.reg(ptr, size_bytes)
        assert ok, f"Failed to register tensor {name}"
        
        # Receive tensor
        ok = ep.recv(conn_id, mr_id, ptr, size_bytes)
        assert ok, f"Receive failed for {name}"
        
        # Copy into model parameter / buffer
        with torch.no_grad():
            if name in model.state_dict():
                model.state_dict()[name].copy_(recv_tensor)
            else:
                logging.warning(f"Key {name} not found in local model, skipping.")
        
        if idx % 20 == 0 or idx == total_tensors:
            progress_pct = (idx / total_tensors) * 100
            logging.info(f"Progress: {progress_pct:.1f}% ({idx}/{total_tensors})")
    
    total_time = time.perf_counter() - recv_start
    avg_bandwidth = (total_size_mb / 1000) / total_time if total_time > 0 else 0
    
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
    logging.info("Loading WikiText-2 dataset from Hugging Face...")
    
    # Disable HF caching logs for cleaner output
    logging.getLogger("datasets").setLevel(logging.ERROR)
    
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    dataset = dataset.filter(lambda x: len(x["text"].strip()) > 0)
    
    if len(dataset) > num_samples:
        dataset = dataset.select(range(num_samples))
    
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
    return tokenized_dataset


# ---------------------------
# Training (rank 1..N)
# ---------------------------
def run_training(model, tokenizer, train_group, trainer_ranks, num_epochs=2, batch_size=4, lr=5e-5):
    """Training loop for trainer ranks"""
    logging.info("="*80)
    logging.info("TRAINING NODE - Starting training on WikiText-2")
    logging.info(f"Model class: {model.__class__.__name__}")
    logging.info("="*80)
    
    train_dataset = prepare_dataset(tokenizer, max_length=128, num_samples=200)
    
    # FIX: Calculate local rank relative to the trainer group, NOT global world size
    # If we use global world size, trainers will skip data indices meant for Rank 0 and Rank N
    global_rank = dist.get_rank()
    rank_in_group = trainer_ranks.index(global_rank)
    num_trainers = len(trainer_ranks)

    sampler = DistributedSampler(
        train_dataset, 
        num_replicas=num_trainers, 
        rank=rank_in_group, 
        shuffle=True
    )
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler
    )
    
    total_steps = len(train_dataloader) * num_epochs
    
    # Ensure model is in train mode
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    total_train_start = time.perf_counter()
    global_step = 0
    epoch_losses = []
    
    for epoch in range(num_epochs):
        epoch_start = time.perf_counter()
        epoch_loss = 0.0
        
        logging.info(f"EPOCH {epoch + 1}/{num_epochs}")
        sampler.set_epoch(epoch) 
        
        for batch_idx, batch in enumerate(train_dataloader):
            step_start = time.perf_counter()
            global_step += 1
            
            input_ids = batch["input_ids"].cuda(non_blocking=True)
            attention_mask = batch["attention_mask"].cuda(non_blocking=True)
            
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=input_ids
                )
                loss = outputs.loss
            
            # Handle FSDP loss which might be sharded or 1D
            loss_scalar = loss.mean() 

            scaler.scale(loss_scalar).backward()
            scaler.step(optimizer)
            scaler.update()
            
            loss_value = loss_scalar.item()
            epoch_loss += loss_value
            
            if (batch_idx + 1) % 10 == 0:
                perplexity = math.exp(loss_value) if loss_value < 100 else float('inf')
                logging.info(
                    f"Step {global_step}/{total_steps} | "
                    f"Loss: {loss_value:.4f} | PPL: {perplexity:.2f}"
                )
        
        avg_epoch_loss = epoch_loss / max(1, len(train_dataloader))
        epoch_losses.append(avg_epoch_loss)
        logging.info(f"End of Epoch {epoch+1} - Avg Loss: {avg_epoch_loss:.4f}")

    # ---------------------------
    # FSDP Saving Mechanism
    # ---------------------------
    # Only one rank per FSDP group (usually rank 0 of the group) should coordinate the save, 
    # but we need to set barrier so all finish.
    dist.barrier(group=train_group)
    
    # For simplicity in this script, we are saving the local shard (state_dict) 
    # To save full model, we need FullStateDictConfig.
    # Here we just save the state_dict as requested, but handle the FSDP access carefully.
    try:
        if isinstance(model, FSDP):
            # To get the full state dict, we need a context manager
            # Note: This gathers weights to CPU on rank 0 of the group
            with FSDP.state_dict_type(model, torch.distributed.fsdp.StateDictType.FULL_STATE_DICT):
                sd = model.state_dict()
        else:
            sd = model.state_dict()
            
        # Only the first trainer saves the checkpoint to avoid file corruption
        if rank_in_group == 0:
            checkpoint_dir = "checkpoints"
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_path = f"{checkpoint_dir}/wikitext2_fsdp_model.pt"
            
            torch.save({
                'model_state_dict': sd,
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch_losses': epoch_losses,
            }, checkpoint_path)
            logging.info(f"Model checkpoint saved to: {checkpoint_path}")
            
    except Exception as e:
        logging.error(f"Error saving checkpoint: {e}")

    logging.info("Training Complete.")


# ---------------------------
# Inference (rank last)
# ---------------------------
def run_inference(model, tokenizer, num_samples=5):
    logging.info("="*80)
    logging.info("INFERENCE NODE - Starting inference")
    logging.info("="*80)
    
    model.eval()
    prompts = ["The future of AI is"]
    
    total_tokens = 0
    start_time = time.perf_counter()
    
    with torch.no_grad():
        for i in range(num_samples):
            prompt = prompts[i % len(prompts)]
            inputs = tokenizer(prompt, return_tensors="pt")
            input_ids = inputs["input_ids"].cuda()
            
            output_ids = model.generate(
                input_ids,
                max_length=50,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
            
            generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            logging.info(f"Sample {i+1}: {generated_text}")

    logging.info(f"Inference done in {time.perf_counter() - start_time:.2f}s")


# ---------------------------
# Main (all ranks)
# ---------------------------
def main():
    # init from environment (torchrun)
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    log_file = setup_logging(rank)
    
    # FIX: Ensure we have enough ranks
    assert world_size >= 3, "Need at least 3 ranks: 1 Broadcaster, 1+ Trainers, 1 Inference"

    # FIX: Setup CUDA device correctly before any P2P / Torch ops
    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        local_gpu = torch.cuda.current_device()
        logging.info(f"CUDA device set to GPU {local_gpu}")
    else:
        logging.error("CUDA required")
        sys.exit(1)
    
    # ---------------------------
    # Define Process Groups
    # ---------------------------
    # We have 3 roles:
    # Rank 0: Broadcaster
    # Rank 1 to N-2: Trainers
    # Rank N-1: Inference
    
    trainer_ranks = list(range(1, world_size - 1))
    # Create a specific process group for trainers. 
    # FSDP MUST run on this group, otherwise it waits for Rank 0/N-1 (deadlock).
    logging.info(f"Initializing Trainer Process Group for ranks: {trainer_ranks}")
    trainer_group = dist.new_group(ranks=trainer_ranks)

    # P2P Setup
    ep = p2p.Endpoint(local_gpu, 4)
    local_md = ep.get_metadata()
    
    # Exchange metadata
    all_metadata = [None] * world_size
    dist.all_gather_object(all_metadata, local_md)
    
    # ---------------------------
    # Logic Branching
    # ---------------------------
    if rank == 0:
        # Broadcaster
        logging.info("BROADCASTER MODE")
        conn_ids = []
        
        receiver_ranks = [r for r in range(1, world_size)] 
        for receiver_rank in receiver_ranks:
            ip, port, r_gpu = p2p.Endpoint.parse_metadata(all_metadata[receiver_rank])
            ok, conn_id = ep.connect(ip, r_gpu, remote_port=port)
            if ok:
                conn_ids.append((receiver_rank, conn_id))
        
        model = GPT2LMHeadModel.from_pretrained("gpt2").cuda()
        # Pass list of conn_ids only
        broadcast_model(ep, [c for _, c in conn_ids], model, rank)
        
        logging.info("Broadcast complete. Waiting for others to finish...")
        # Broadcaster waits for everyone before destroying group
        dist.barrier() 
    
    elif rank in trainer_ranks:
        # Trainer(s)
        logging.info(f"TRAINING NODE (Rank {rank})")
        
        # Accept connection from broadcaster
        ok, r_ip, r_gpu, conn_id = ep.accept()
        assert ok, "Accept failed"
        
        base_model = GPT2LMHeadModel.from_pretrained("gpt2").cuda()
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        
        # Receive weights via RDMA
        recv_model(ep, conn_id, base_model, rank)
        
        # FIX: Initialize FSDP with the specific trainer_group
        # If we don't pass process_group, it uses global group -> deadlock
        auto_wrap_policy = transformer_auto_wrap_policy({GPT2Block})
        
        model = FSDP(
            base_model,
            auto_wrap_policy=auto_wrap_policy,
            process_group=trainer_group,  # CRITICAL FIX
            device_id=local_gpu
        )
        
        run_training(
            model, tokenizer, 
            train_group=trainer_group, 
            trainer_ranks=trainer_ranks
        )
        
        logging.info("Training node finished.")
        dist.barrier()
    
    elif rank == world_size - 1:
        # Inference node
        logging.info("INFERENCE NODE")
        
        ok, r_ip, r_gpu, conn_id = ep.accept()
        assert ok, "Accept failed"
        
        model = GPT2LMHeadModel.from_pretrained("gpt2").cuda()
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        
        recv_model(ep, conn_id, model, rank)
        run_inference(model, tokenizer, num_samples=5)
        
        logging.info("Inference node finished.")
        dist.barrier()
    
    dist.destroy_process_group()
    logging.info("Process complete.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
    except Exception as e:
        logging.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)