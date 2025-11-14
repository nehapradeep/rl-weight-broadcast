from __future__ import annotations
import torch, time, os, sys
import torch.distributed as dist
import logging
from datetime import datetime
import math

from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
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
    avg_bandwidth = (total_size_mb / 1000) / total_time
    
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
        
        ok, mr_id = ep.reg(ptr, size_bytes)
        assert ok, f"Failed to register tensor {name}"
        
        ok = ep.recv(conn_id, mr_id, ptr, size_bytes)
        assert ok, f"Receive failed for {name}"
        
        model.state_dict()[name].copy_(recv_tensor)
        
        if idx % 20 == 0 or idx == total_tensors:
            progress_pct = (idx / total_tensors) * 100
            logging.info(f"Progress: {progress_pct:.1f}% ({idx}/{total_tensors})")
    
    total_time = time.perf_counter() - recv_start
    avg_bandwidth = (total_size_mb / 1000) / total_time
    
    logging.info("="*80)
    logging.info(f"RECEIVE COMPLETE")
    logging.info(f"Total time: {total_time:.2f}s")
    logging.info(f"Average bandwidth: {avg_bandwidth:.2f} GB/s")
    logging.info("="*80)


def prepare_dataset(tokenizer, max_length=128, num_train_samples=400, num_val_samples=100):
    """Load and prepare WikiText-2 dataset with train/validation split"""
    logging.info("="*80)
    logging.info("LOADING DATASET")
    logging.info("="*80)
    logging.info("Loading WikiText-2 dataset from Hugging Face...")
    
    # Load train and validation splits
    train_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    val_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
    
    logging.info(f"Train dataset loaded: {len(train_dataset)} total examples")
    logging.info(f"Validation dataset loaded: {len(val_dataset)} total examples")
    
    # Filter out empty texts
    train_dataset = train_dataset.filter(lambda x: len(x["text"].strip()) > 0)
    val_dataset = val_dataset.filter(lambda x: len(x["text"].strip()) > 0)
    
    logging.info(f"After filtering - Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    # Take subsets
    if len(train_dataset) > num_train_samples:
        train_dataset = train_dataset.select(range(num_train_samples))
        logging.info(f"Using train subset: {num_train_samples} examples")
    
    if len(val_dataset) > num_val_samples:
        val_dataset = val_dataset.select(range(num_val_samples))
        logging.info(f"Using validation subset: {num_val_samples} examples")
    
    logging.info("Tokenizing datasets...")
    
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length
        )
    
    train_tokenized = train_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=train_dataset.column_names
    )
    
    val_tokenized = val_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=val_dataset.column_names
    )
    
    # Set format to torch tensors
    train_tokenized.set_format(type='torch', columns=['input_ids', 'attention_mask'])
    val_tokenized.set_format(type='torch', columns=['input_ids', 'attention_mask'])
    
    logging.info(f"Tokenization complete")
    logging.info(f"Training set: {len(train_tokenized)} examples")
    logging.info(f"Validation set: {len(val_tokenized)} examples")
    logging.info(f"Max length: {max_length} tokens")
    logging.info("="*80)
    
    return train_tokenized, val_tokenized


def evaluate_model(model, val_dataloader, training_group, training_rank):
    """Evaluate model on validation set"""
    model.eval()
    total_loss = 0.0
    total_batches = 0
    
    logging.info("Running validation...")
    
    with torch.no_grad():
        for batch in val_dataloader:
            input_ids = batch["input_ids"].cuda()
            attention_mask = batch["attention_mask"].cuda()
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids
            )
            total_loss += outputs.loss.item()
            total_batches += 1
    
    avg_loss = total_loss / total_batches if total_batches > 0 else 0
    perplexity = math.exp(avg_loss) if avg_loss < 100 else float('inf')
    
    # Synchronize validation metrics across training nodes
    loss_tensor = torch.tensor([avg_loss], device='cuda')
    dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG, group=training_group)
    global_avg_loss = loss_tensor.item()
    global_perplexity = math.exp(global_avg_loss) if global_avg_loss < 100 else float('inf')
    
    logging.info(f"Validation - Local Loss: {avg_loss:.4f}, Perplexity: {perplexity:.2f}")
    logging.info(f"Validation - Global Loss: {global_avg_loss:.4f}, Perplexity: {global_perplexity:.2f}")
    
    model.train()
    return global_avg_loss, global_perplexity


def test_generation(model, tokenizer, training_rank):
    """Test text generation to verify model is learning"""
    model.eval()
    
    test_prompts = [
        "The",
        "In the",
        "Machine learning"
    ]
    
    logging.info("\n" + "="*80)
    logging.info(f"TEXT GENERATION TEST (Node {training_rank + 1})")
    logging.info("="*80)
    
    with torch.no_grad():
        for prompt in test_prompts:
            inputs = tokenizer(prompt, return_tensors="pt")
            input_ids = inputs["input_ids"].cuda()
            
            output_ids = model.module.generate(  # .module to access underlying model
                input_ids,
                max_length=30,
                num_return_sequences=1,
                temperature=0.8,
                do_sample=True,
                top_k=50,
                pad_token_id=tokenizer.eos_token_id
            )
            
            generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            logging.info(f"Prompt: '{prompt}' → Generated: '{generated_text}'")
    
    logging.info("="*80 + "\n")
    model.train()


def run_distributed_training(model, tokenizer, rank, world_size, training_group, num_epochs=2, batch_size=4, lr=5e-5):
    """Distributed training across multiple nodes with validation"""
    training_rank = rank - 1  # Training ranks are 1 and 2, so training_rank is 0 and 1
    
    logging.info("="*80)
    logging.info(f"DISTRIBUTED TRAINING NODE {training_rank + 1}/2")
    logging.info("="*80)
    
    # Prepare datasets (train and validation)
    train_dataset, val_dataset = prepare_dataset(tokenizer, max_length=128, 
                                                  num_train_samples=400, num_val_samples=100)
    
    # Create DistributedSampler for training data
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=2,
        rank=training_rank,
        shuffle=True
    )
    
    # Create DataLoader for training
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler
    )
    
    # Create DataLoader for validation (no distributed sampler needed)
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    
    # Wrap model with DistributedDataParallel
    model = DDP(model, device_ids=[rank], process_group=training_group)
    
    total_steps = len(train_dataloader) * num_epochs
    total_dataset_size = len(train_dataset)
    local_dataset_size = len(train_dataloader.dataset) // 2
    
    logging.info(f"Training configuration:")
    logging.info(f"  - Dataset: WikiText-2")
    logging.info(f"  - Total train examples: {total_dataset_size}")
    logging.info(f"  - Train examples on this node: ~{local_dataset_size}")
    logging.info(f"  - Validation examples: {len(val_dataset)}")
    logging.info(f"  - Training nodes: 2 (Rank 1 & 2)")
    logging.info(f"  - This training rank: {training_rank}")
    logging.info(f"  - Batch size per node: {batch_size}")
    logging.info(f"  - Effective batch size: {batch_size * 2}")
    logging.info(f"  - Epochs: {num_epochs}")
    logging.info(f"  - Steps per epoch: {len(train_dataloader)}")
    logging.info(f"  - Total steps: {total_steps}")
    logging.info(f"  - Learning rate: {lr}")
    logging.info(f"  - Gradient synchronization: Enabled (DDP)")
    logging.info(f"  - Loss function: Cross-Entropy (Causal Language Modeling)")
    logging.info("="*80)
    
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    # ========== EVALUATE BEFORE TRAINING ==========
    logging.info("\n" + "="*80)
    logging.info("EVALUATION BEFORE TRAINING")
    logging.info("="*80)
    initial_val_loss, initial_val_perplexity = evaluate_model(model, val_dataloader, 
                                                               training_group, training_rank)
    logging.info(f"Initial Validation Loss: {initial_val_loss:.4f}")
    logging.info(f"Initial Validation Perplexity: {initial_val_perplexity:.2f}")
    logging.info("="*80 + "\n")
    
    # Test generation before training
    if training_rank == 0:
        logging.info("Text generation BEFORE training:")
        test_generation(model, tokenizer, training_rank)
    
    # ========== START TRAINING ==========
    logging.info("\n" + "="*80)
    logging.info("STARTING DISTRIBUTED TRAINING")
    logging.info("="*80)
    
    # Synchronize start time across nodes
    dist.barrier()
    total_train_start = time.perf_counter()
    
    global_step = 0
    epoch_losses = []
    epoch_val_losses = []
    
    for epoch in range(num_epochs):
        epoch_start = time.perf_counter()
        epoch_loss = 0.0
        epoch_perplexity = 0.0
        
        # Set epoch for sampler (important for proper shuffling)
        train_sampler.set_epoch(epoch)
        
        logging.info(f"\n{'='*80}")
        logging.info(f"EPOCH {epoch + 1}/{num_epochs} (Training Node {training_rank + 1})")
        logging.info(f"{'='*80}")
        
        for batch_idx, batch in enumerate(train_dataloader):
            step_start = time.perf_counter()
            global_step += 1
            
            input_ids = batch["input_ids"].cuda()
            attention_mask = batch["attention_mask"].cuda()
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids
            )
            loss = outputs.loss
            
            # Backward pass (gradients automatically synchronized via DDP)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            step_time = time.perf_counter() - step_start
            perplexity = math.exp(loss.item()) if loss.item() < 100 else float('inf')
            
            epoch_loss += loss.item()
            epoch_perplexity += perplexity
            
            # Log every 10 steps
            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(train_dataloader):
                logging.info(
                    f"Node {training_rank + 1} | Step {global_step}/{total_steps} "
                    f"(Epoch {epoch+1}, Batch {batch_idx+1}/{len(train_dataloader)}) | "
                    f"Loss: {loss.item():.4f} | Perplexity: {perplexity:.2f} | Time: {step_time:.3f}s"
                )
            
            # Log gradient norms every 20 steps
            if global_step % 20 == 0:
                total_norm = 0
                for p in model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5
                logging.info(f"  -> Gradient norm: {total_norm:.4f}")
        
        # Epoch training summary
        epoch_time = time.perf_counter() - epoch_start
        avg_epoch_loss = epoch_loss / len(train_dataloader)
        avg_epoch_perplexity = epoch_perplexity / len(train_dataloader)
        epoch_losses.append(avg_epoch_loss)
        
        # Synchronize training metrics across nodes
        loss_tensor = torch.tensor([avg_epoch_loss], device='cuda')
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG, group=training_group)
        global_avg_loss = loss_tensor.item()
        
        logging.info(f"\n{'='*80}")
        logging.info(f"EPOCH {epoch + 1} TRAINING SUMMARY (Node {training_rank + 1})")
        logging.info(f"Local Average Training Loss: {avg_epoch_loss:.4f}")
        logging.info(f"Global Average Training Loss: {global_avg_loss:.4f}")
        logging.info(f"Average Training Perplexity: {avg_epoch_perplexity:.2f}")
        logging.info(f"Epoch Time: {epoch_time:.2f}s")
        
        # ========== VALIDATION AFTER EPOCH ==========
        logging.info(f"\nRunning validation after epoch {epoch + 1}...")
        val_loss, val_perplexity = evaluate_model(model, val_dataloader, 
                                                   training_group, training_rank)
        epoch_val_losses.append(val_loss)
        
        logging.info(f"Validation Loss: {val_loss:.4f}")
        logging.info(f"Validation Perplexity: {val_perplexity:.2f}")
        logging.info(f"{'='*80}\n")
    
    # Synchronize end time across nodes
    dist.barrier()
    total_train_time = time.perf_counter() - total_train_start
    
    # ========== FINAL EVALUATION ==========
    logging.info("\n" + "="*80)
    logging.info("FINAL EVALUATION AFTER TRAINING")
    logging.info("="*80)
    final_val_loss, final_val_perplexity = evaluate_model(model, val_dataloader,
                                                           training_group, training_rank)
    
    # Test generation after training
    if training_rank == 0:
        logging.info("\nText generation AFTER training:")
        test_generation(model, tokenizer, training_rank)
    
    # ========== TRAINING COMPLETE SUMMARY ==========
    avg_step_time = total_train_time / total_steps
    train_loss_improvement = epoch_losses[0] - epoch_losses[-1]
    val_loss_improvement = initial_val_loss - final_val_loss
    val_perplexity_improvement = initial_val_perplexity - final_val_perplexity
    
    logging.info("\n" + "="*80)
    logging.info(f"DISTRIBUTED TRAINING COMPLETE (Node {training_rank + 1})")
    logging.info("="*80)
    logging.info(f"Dataset: WikiText-2 (400 train samples, 100 val samples)")
    logging.info(f"TOTAL TRAINING TIME: {total_train_time:.2f}s")
    logging.info(f"Total steps on this node: {total_steps}")
    logging.info(f"Average time per step: {avg_step_time:.3f}s")
    logging.info(f"Steps per second: {total_steps/total_train_time:.2f}")
    logging.info("")
    logging.info("TRAINING METRICS:")
    logging.info(f"  Initial training loss: {epoch_losses[0]:.4f}")
    logging.info(f"  Final training loss: {epoch_losses[-1]:.4f}")
    logging.info(f"  Training loss improvement: {train_loss_improvement:.4f} ({'✓ Improved' if train_loss_improvement > 0 else '✗ No improvement'})")
    logging.info("")
    logging.info("VALIDATION METRICS:")
    logging.info(f"  Initial validation loss: {initial_val_loss:.4f}")
    logging.info(f"  Final validation loss: {final_val_loss:.4f}")
    logging.info(f"  Validation loss improvement: {val_loss_improvement:.4f} ({'✓ Improved' if val_loss_improvement > 0 else '✗ No improvement'})")
    logging.info(f"  Initial validation perplexity: {initial_val_perplexity:.2f}")
    logging.info(f"  Final validation perplexity: {final_val_perplexity:.2f}")
    logging.info(f"  Perplexity improvement: {val_perplexity_improvement:.2f} ({'✓ Lower is better' if val_perplexity_improvement > 0 else '✗ Higher'})")
    logging.info("")
    if val_loss_improvement > 0:
        logging.info("✓ TRAINING IS WORKING CORRECTLY - Model is learning!")
    else:
        logging.info("⚠ WARNING - Model may not be learning properly. Check hyperparameters.")
    logging.info("="*80)
    
    # Only rank 1 saves checkpoint
    if training_rank == 0:
        checkpoint_dir = "checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = f"{checkpoint_dir}/model_distributed_trained.pt"
        torch.save({
            'model_state_dict': model.module.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch_losses': epoch_losses,
            'epoch_val_losses': epoch_val_losses,
            'initial_val_loss': initial_val_loss,
            'final_val_loss': final_val_loss,
            'total_training_time': total_train_time,
            'total_steps': total_steps,
        }, checkpoint_path)
        logging.info(f"Model checkpoint saved to: {checkpoint_path}")


def main():
    dist.init_process_group(backend="gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    log_file = setup_logging(rank)
    
    logging.info(f"Process started - Rank: {rank}, World size: {world_size}")
    assert world_size == 3, "Run with 3 ranks (1 broadcaster + 2 distributed training)."

    local_gpu = rank
    torch.cuda.set_device(local_gpu)
    logging.info(f"CUDA device set to GPU {local_gpu}")

    # Create process group for training nodes (ranks 1 and 2)
    training_ranks = [1, 2]
    training_group = dist.new_group(training_ranks)
    
    # Initialize endpoint
    logging.info("Initializing P2P endpoint...")
    ep = p2p.Endpoint(local_gpu, 4)
    local_md = ep.get_metadata()

    # Exchange metadata
    logging.info("Starting metadata exchange...")
    all_metadata = [None] * world_size
    all_metadata[rank] = local_md
    
    for i in range(world_size):
        if i == rank:
            for j in range(world_size):
                if j != rank:
                    dist.send(torch.ByteTensor(list(local_md)), dst=j)
        else:
            remote_md = torch.zeros(len(local_md), dtype=torch.uint8)
            dist.recv(remote_md, src=i)
            all_metadata[i] = bytes(remote_md.tolist())
    
    logging.info(f"Metadata exchange complete")

    if rank == 0:
        # Broadcaster
        logging.info("="*80)
        logging.info("BROADCASTER MODE")
        logging.info("="*80)
        logging.info("Broadcasting to 2 training nodes for distributed training")
        
        conn_ids = []
        for receiver_rank in [1, 2]:
            ip, port, r_gpu = p2p.Endpoint.parse_metadata(all_metadata[receiver_rank])
            ok, conn_id = ep.connect(ip, r_gpu, remote_port=port)
            assert ok, f"Connect failed to rank {receiver_rank}"
            conn_ids.append(conn_id)
            
            logging.info(f"Connected to Training Node {receiver_rank} (rank {receiver_rank})")

        logging.info("Loading model...")
        model = GPT2LMHeadModel.from_pretrained("gpt2").cuda()
        
        broadcast_model(ep, conn_ids, model, rank)
        
        logging.info("\nBroadcast complete. Both training nodes will now train in parallel "
                    "with gradient synchronization.")

    else:  # rank in [1, 2]
        # Training nodes (distributed training)
        training_rank = rank - 1
        logging.info("="*80)
        logging.info(f"DISTRIBUTED TRAINING NODE {training_rank + 1}/2 (Rank {rank})")
        logging.info("="*80)
        
        ok, r_ip, r_gpu, conn_id = ep.accept()
        assert ok, "Accept failed"
        logging.info(f"Connected to broadcaster")

        model = GPT2LMHeadModel.from_pretrained("gpt2").cuda()
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        
        recv_model(ep, conn_id, model, rank)
        
        # Run distributed training
        run_distributed_training(model, tokenizer, rank, world_size, training_group, 
                                num_epochs=2, batch_size=4, lr=5e-5)

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
