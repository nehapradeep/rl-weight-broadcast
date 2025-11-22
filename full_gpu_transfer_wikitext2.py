from __future__ import annotations
import torch, time, os, sys
import torch.distributed as dist
import logging
from datetime import datetime
import math
from dataclasses import dataclass
from typing import Dict, Tuple, Any

from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader
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


def prepare_dataset(tokenizer, max_length=128, num_samples=200):
    """Load and prepare WikiText-2 dataset"""
    logging.info("="*80)
    logging.info("LOADING DATASET")
    logging.info("="*80)
    logging.info("Loading WikiText-2 dataset from Hugging Face...")
    
    # Load WikiText-2 (small dataset, good for language modeling)
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    logging.info(f"Dataset loaded: {len(dataset)} total examples")
    
    # Filter out empty texts
    dataset = dataset.filter(lambda x: len(x["text"].strip()) > 0)
    logging.info(f"After filtering empty texts: {len(dataset)} examples")
    
    # Take a subset for faster training
    if len(dataset) > num_samples:
        dataset = dataset.select(range(num_samples))
        logging.info(f"Using subset: {num_samples} examples")
    
    # Tokenize the dataset
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
    
    # Set format to torch tensors
    tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])
    
    logging.info(f"Tokenization complete")
    logging.info(f"Dataset info:")
    logging.info(f"  - Examples: {len(tokenized_dataset)}")
    logging.info(f"  - Max length: {max_length} tokens")
    logging.info("="*80)
    
    return tokenized_dataset


def run_training(model, tokenizer, num_epochs=2, batch_size=4, lr=5e-5):
    """Training loop for rank 1 with real dataset"""
    logging.info("="*80)
    logging.info("TRAINING NODE - Starting training on WikiText-2")
    logging.info("="*80)
    
    # Prepare dataset
    train_dataset = prepare_dataset(tokenizer, max_length=128, num_samples=200)
    
    # Create DataLoader
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    
    total_steps = len(train_dataloader) * num_epochs
    logging.info(f"Training configuration:")
    logging.info(f"  - Dataset: WikiText-2")
    logging.info(f"  - Examples: {len(train_dataset)}")
    logging.info(f"  - Batch size: {batch_size}")
    logging.info(f"  - Epochs: {num_epochs}")
    logging.info(f"  - Steps per epoch: {len(train_dataloader)}")
    logging.info(f"  - Total steps: {total_steps}")
    logging.info(f"  - Learning rate: {lr}")
    logging.info(f"  - Loss function: Cross-Entropy (Causal Language Modeling)")
    logging.info("="*80)
    
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
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
        
        for batch_idx, batch in enumerate(train_dataloader):
            step_start = time.perf_counter()
            global_step += 1
            
            # Move batch to GPU
            input_ids = batch["input_ids"].cuda()
            attention_mask = batch["attention_mask"].cuda()
            
            # Forward pass (labels = input_ids for causal language modeling)
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids
            )
            loss = outputs.loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            step_time = time.perf_counter() - step_start
            
            # Calculate perplexity
            perplexity = math.exp(loss.item()) if loss.item() < 100 else float('inf')
            
            # Track metrics
            epoch_loss += loss.item()
            epoch_perplexity += perplexity
            
            # Log every 10 steps or last step
            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(train_dataloader):
                logging.info(
                    f"Step {global_step}/{total_steps} (Epoch {epoch+1}, Batch {batch_idx+1}/{len(train_dataloader)}) | "
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
        
        # Epoch summary
        epoch_time = time.perf_counter() - epoch_start
        avg_epoch_loss = epoch_loss / len(train_dataloader)
        avg_epoch_perplexity = epoch_perplexity / len(train_dataloader)
        epoch_losses.append(avg_epoch_loss)
        
        logging.info(f"\n{'='*80}")
        logging.info(f"EPOCH {epoch + 1} SUMMARY")
        logging.info(f"Average Loss: {avg_epoch_loss:.4f}")
        logging.info(f"Average Perplexity: {avg_epoch_perplexity:.2f}")
        logging.info(f"Epoch Time: {epoch_time:.2f}s")
        logging.info(f"{'='*80}\n")
    
    total_train_time = time.perf_counter() - total_train_start
    avg_step_time = total_train_time / total_steps
    
    logging.info("="*80)
    logging.info("TRAINING COMPLETE")
    logging.info(f"Dataset: WikiText-2 (200 samples)")
    logging.info(f"Total training time: {total_train_time:.2f}s")
    logging.info(f"Total steps: {total_steps}")
    logging.info(f"Average time per step: {avg_step_time:.3f}s")
    logging.info(f"Steps per second: {total_steps/total_train_time:.2f}")
    logging.info(f"Final loss: {epoch_losses[-1]:.4f}")
    logging.info(f"Loss improvement: {epoch_losses[0] - epoch_losses[-1]:.4f}")
    logging.info("="*80)
    
    # Save checkpoint
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = f"{checkpoint_dir}/model_rank1_wikitext2_trained.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch_losses': epoch_losses,
        'total_steps': total_steps,
        'final_loss': epoch_losses[-1],
    }, checkpoint_path)
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
        "Deep learning models",
        "Natural language processing",
        "Machine learning algorithms",
    ]
    
    logging.info(f"Running inference on {num_samples} prompts")
    logging.info(f"Generation settings: max_length=50, temperature=0.7, top_k=50, top_p=0.95")
    
    total_inference_start = time.perf_counter()
    total_tokens = 0
    
    with torch.no_grad():
        for idx, prompt in enumerate(prompts[:num_samples], 1):
            inference_start = time.perf_counter()
            
            logging.info(f"\n{'='*60}")
            logging.info(f"SAMPLE {idx}/{num_samples}")
            logging.info(f"{'='*60}")
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
    avg_inference_time = total_inference_time / num_samples
    avg_tokens_per_sec = total_tokens / total_inference_time
    
    logging.info("\n" + "="*80)
    logging.info("INFERENCE COMPLETE")
    logging.info(f"Total inference time: {total_inference_time:.2f}s")
    logging.info(f"Average time per sample: {avg_inference_time:.3f}s")
    logging.info(f"Samples per second: {num_samples/total_inference_time:.2f}")
    logging.info(f"Total tokens generated: {total_tokens}")
    logging.info(f"Average tokens/sec: {avg_tokens_per_sec:.1f}")
    logging.info("="*80)

def compute_weight_transfer_schedule(
    trainers: list[ParametersMetadata],
    rollouts: list[ParametersMetadata],
    trainer_named_parameters: dict[str, ParameterMetadata],
    name_mappings: list[WeightNameMapping],
) -> WeightTransferSchedule:
    mesh_set: set[Mesh] = ...
    mesh_groups = find_disjoint_mesh_groups(mesh_set)
    trainer_tables = [WeightTransferRoutingTable([]) for _ in trainers]
    for mesh_group in mesh_groups:
        trainer_groups = generate_table_for_mesh_group(trainers, rollouts, name_mappings_with_mesh, mesh_group)
        for table, group in zip(trainer_tables, trainer_groups):
            table.groups.append(group)
    return WeightTransferSchedule(trainer_tables)


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

        # Controller side (rank 0) – after model is loaded
        logging.info("Model loaded")
        # 1) Gather metadata from all participant nodes (trainers + rollouts)
        all_metadata = {rank: local_md}
        for peer in range(world_size):
            if peer == rank:
                continue
            logging.info(f"Requesting metadata from rank {peer}")
            # Assuming you send a request or use dist.send/recv
            dist.send(torch.ByteTensor(list(local_md)), dst=peer)
            remote_md_tensor = torch.zeros(len(local_md), dtype=torch.uint8)
            dist.recv(remote_md_tensor, src=peer)
            all_metadata[peer] = bytes(remote_md_tensor.tolist())
        logging.info(f"Received metadata from all peers: {all_metadata.keys()}")

        # 2) Compute routing schedule using your match & routing logic
        # (You need to implement compute_weight_transfer_schedule, grouping, etc.)
        schedule = compute_weight_transfer_schedule(
            trainer_named_parameters = trainer_params,   # e.g., metadata from trainers
            rollout_named_parameters = rollout_params,   # e.g., metadata from rollouts
            name_mappings = name_mappings                 # how trainer names map to rollout names
        )
        logging.info("Routing schedule computed")

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
        
        # Start training with WikiText-2 dataset
        run_training(model, tokenizer, num_epochs=2, batch_size=4, lr=5e-5)

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
