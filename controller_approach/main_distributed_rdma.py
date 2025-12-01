"""
Main Distributed Training Script with RDMA Weight Transfers
Uses UCCL P2P for efficient GPU-to-GPU weight transfers

Setup for 4 nodes × 8 AMD GPUs:
- Node 0 (ranks 0-7): Controller + Training
- Node 1 (ranks 8-15): Training  
- Node 2 (ranks 16-23): Controller (runs routing computation)
- Node 3 (ranks 24-31): Inference

Usage:
    # On each node, run:
    torchrun --nproc_per_node=8 \
             --nnodes=4 \
             --node_rank=$NODE_RANK \
             --master_addr=$MASTER_ADDR \
             --master_port=29500 \
             main_distributed_rdma.py
"""

import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import GPT2LMHeadModel, GPT2Config, GPT2Tokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader, DistributedSampler
import argparse
import logging
from datetime import datetime

from weight_transfer_controller import (
    WeightTransferController,
    simulate_parameter_distribution
)
from training_worker_rdma import OptimizedRDMATrainingWorker
from inference_worker_rdma import RDMAInferenceWorker


def setup_logging(rank: int):
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
            logging.StreamHandler()
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


def setup_distributed():
    """Initialize distributed environment"""
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    # Initialize process group with gloo for metadata exchange
    dist.init_process_group(
        backend="gloo",  # Use gloo for CPU metadata exchange
        init_method="env://",
        world_size=world_size,
        rank=rank
    )
    
    # Set device
    torch.cuda.set_device(local_rank)
    
    return rank, local_rank, world_size


def cleanup_distributed():
    """Clean up distributed environment"""
    dist.destroy_process_group()


def get_role(rank: int, world_size: int):
    """
    Determine the role of this rank
    
    Setup:
    - Ranks 0-15: Training (nodes 0-1)
    - Rank 16: Controller (node 2) 
    - Ranks 24-31: Inference (node 3)
    """
    if rank < 16:
        return "training"
    elif rank == 16:
        return "controller"
    elif rank >= 24 and rank < 32:
        return "inference"
    else:
        return "idle"  # Ranks 17-23 are idle


def prepare_model_and_data(rank: int, local_rank: int, world_size: int):
    """Prepare model and dataset"""
    logging.info(f"[Rank {rank}] Loading GPT-2 model...")
    config = GPT2Config.from_pretrained("gpt2")
    model = GPT2LMHeadModel(config)
    
    # Move model to GPU
    device = torch.device(f"cuda:{local_rank}")
    model = model.to(device)
    
    # Wrap with DDP for training ranks
    role = get_role(rank, world_size)
    if role == "training":
        model = DDP(model, device_ids=[local_rank])
        logging.info(f"[Rank {rank}] Model wrapped with DDP")
    
    # Load tokenizer and dataset
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load wikitext-2 dataset
    logging.info(f"[Rank {rank}] Loading wikitext-2 dataset...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    
    # Filter empty texts
    dataset = dataset.filter(lambda x: len(x["text"].strip()) > 0)
    
    # Take subset
    if len(dataset) > 200:
        dataset = dataset.select(range(200))
    
    # Tokenize
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=128,
            padding="max_length"
        )
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names
    )
    
    tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])
    
    # Create distributed sampler for training
    if role == "training":
        sampler = DistributedSampler(
            tokenized_dataset,
            num_replicas=16,  # Only training ranks
            rank=rank if rank < 16 else 0,
            shuffle=True
        )
        dataloader = DataLoader(
            tokenized_dataset,
            batch_size=4,
            sampler=sampler
        )
    else:
        dataloader = None
    
    return model, tokenizer, dataloader, device


def run_controller(rank: int, world_size: int, model: torch.nn.Module):
    """Run controller to compute routing tables"""
    logging.info(f"\n{'='*60}")
    logging.info(f"[Rank {rank}] CONTROLLER: Computing routing tables")
    logging.info(f"{'='*60}\n")
    
    # Define training and inference ranks
    training_ranks = list(range(0, 16))  # Nodes 0-1
    inference_ranks = list(range(24, 32))  # Node 3
    
    # Get the base model (unwrap DDP if needed)
    base_model = model.module if hasattr(model, 'module') else model
    
    # Simulate parameter distribution
    trainer_params, inference_params = simulate_parameter_distribution(
        base_model,
        training_ranks,
        inference_ranks,
        use_data_parallel=True
    )
    
    # Create controller
    controller = WeightTransferController(
        training_ranks=training_ranks,
        inference_ranks=inference_ranks,
        world_size=world_size
    )
    
    # Compute routing tables
    routing_tables = controller.compute_routing_tables(
        trainer_params,
        inference_params
    )
    
    # Print routing details
    controller.print_routing_details(routing_tables)
    
    return routing_tables


def run_training_rdma(
    rank: int,
    local_rank: int,
    world_size: int,
    model: torch.nn.Module,
    dataloader: DataLoader,
    routing_table,
    num_epochs: int = 1
):
    """Run training loop with RDMA weight updates"""
    logging.info(f"\n{'='*60}")
    logging.info(f"[Rank {rank}] TRAINING with RDMA: Starting training loop")
    logging.info(f"{'='*60}\n")
    
    # Create RDMA worker
    base_model = model.module if hasattr(model, 'module') else model
    rdma_worker = OptimizedRDMATrainingWorker(
        model=base_model,
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        num_max_connections=16
    )
    
    # Exchange metadata with all ranks
    logging.info(f"[Rank {rank}] Exchanging RDMA metadata with all ranks...")
    all_ranks = list(range(world_size))
    all_metadata = rdma_worker.exchange_metadata(all_ranks)
    logging.info(f"[Rank {rank}] Metadata exchange complete")
    
    # Setup connections to inference ranks
    # Training ranks will connect in deterministic order (0, 1, 2, 3, ...)
    inference_ranks = list(range(24, 32))
    logging.info(f"[Rank {rank}] Setting up RDMA connections to inference ranks {inference_ranks}...")
    rdma_worker.setup_connections(inference_ranks, all_metadata)
    logging.info(f"[Rank {rank}] RDMA connections established")
    
    # Set routing table (this would come from controller in real impl)
    if routing_table:
        rdma_worker.set_routing_table(routing_table)
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    
    # Training loop
    model.train()
    global_step = 0
    
    for epoch in range(num_epochs):
        logging.info(f"[Rank {rank}] Epoch {epoch+1}/{num_epochs}")
        
        dataloader.sampler.set_epoch(epoch)
        
        for step, batch in enumerate(dataloader):
            # Move batch to device
            input_ids = batch["input_ids"].to(f"cuda:{local_rank}")
            attention_mask = batch["attention_mask"].to(f"cuda:{local_rank}")
            
            # Forward pass
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
            
            global_step += 1
            
            # Log
            if step % 10 == 0:
                logging.info(f"[Rank {rank}] Step {global_step}, Loss: {loss.item():.4f}")
            
            # Periodic RDMA weight update (every 100 steps)
            if global_step % 100 == 0:
                logging.info(f"\n[Rank {rank}] === RDMA WEIGHT UPDATE at step {global_step} ===")
                
                # Synchronize all training ranks before weight transfer
                logging.info(f"[Rank {rank}] Synchronizing before RDMA transfer...")
                dist.barrier()
                
                # Transfer weights via RDMA
                try:
                    logging.info(f"[Rank {rank}] Starting RDMA weight transfer...")
                    transfer_time = rdma_worker.transfer_weights_pipelined()
                    logging.info(f"[Rank {rank}] RDMA weight update completed in "
                                f"{transfer_time:.2f}s")
                except Exception as e:
                    logging.error(f"[Rank {rank}] RDMA weight update failed: {e}", exc_info=True)
                
                # Synchronize after transfer
                logging.info(f"[Rank {rank}] Synchronizing after RDMA transfer...")
                dist.barrier()
                logging.info(f"[Rank {rank}] === RDMA WEIGHT UPDATE COMPLETE ===\n")
            
            # Limit steps for demo
            if global_step >= 50:
                break
        
        if global_step >= 50:
            break
    
    logging.info(f"[Rank {rank}] Training completed")


def run_inference_rdma(
    rank: int,
    local_rank: int,
    world_size: int,
    model: torch.nn.Module,
    tokenizer
):
    """Run inference loop with RDMA weight updates"""
    logging.info(f"\n{'='*60}")
    logging.info(f"[Rank {rank}] INFERENCE with RDMA: Starting inference engine")
    logging.info(f"{'='*60}\n")
    
    # Define training ranks to receive from (in order)
    training_ranks = list(range(0, 16))
    
    # Create RDMA inference worker
    rdma_worker = RDMAInferenceWorker(
        model=model,
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        training_ranks=training_ranks,
        num_max_connections=16
    )
    
    # Exchange metadata with all ranks
    logging.info(f"[Rank {rank}] Exchanging RDMA metadata with all ranks...")
    all_ranks = list(range(world_size))
    all_metadata = rdma_worker.exchange_metadata(all_ranks)
    logging.info(f"[Rank {rank}] Metadata exchange complete")
    
    # Accept connections from training ranks in deterministic order
    logging.info(f"[Rank {rank}] Setting up RDMA connections from training ranks {training_ranks}...")
    rdma_worker.setup_connections(training_ranks=training_ranks)
    logging.info(f"[Rank {rank}] RDMA connections established")
    
    # Prepare test prompts
    test_prompts = [
        "The quick brown fox",
        "Once upon a time",
        "In a galaxy far far away",
    ]
    
    inputs = tokenizer(
        test_prompts,
        return_tensors="pt",
        padding=True,
        truncation=True
    )
    input_ids = inputs["input_ids"].to(f"cuda:{local_rank}")
    
    logging.info(f"[Rank {rank}] Waiting for RDMA weight updates and running inference...")
    
    # Simulation: wait for weight updates at same intervals as training
    inference_step = 0
    while inference_step < 50:
        # Check if it's time for weight update (every 100 steps aligned with training)
        if inference_step % 100 == 0 and inference_step > 0:
            logging.info(f"\n[Rank {rank}] === RECEIVING RDMA WEIGHT UPDATE ===")
            
            # Synchronize before receiving
            logging.info(f"[Rank {rank}] Synchronizing before RDMA receive...")
            dist.barrier()
            
            # Receive weights via RDMA
            try:
                logging.info(f"[Rank {rank}] Starting RDMA weight receive...")
                receive_time = rdma_worker.receive_weights()
                logging.info(f"[Rank {rank}] RDMA weight update received in "
                            f"{receive_time:.2f}s")
            except Exception as e:
                logging.error(f"[Rank {rank}] RDMA weight receive failed: {e}", exc_info=True)
            
            # Synchronize after receiving
            logging.info(f"[Rank {rank}] Synchronizing after RDMA receive...")
            dist.barrier()
            logging.info(f"[Rank {rank}] === RDMA WEIGHT UPDATE COMPLETE ===\n")
        
        # Run inference periodically
        if inference_step % 20 == 0:
            logging.info(f"[Rank {rank}] Running inference at step {inference_step}...")
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=input_ids[:1],
                    max_length=50,
                    do_sample=True,
                    temperature=0.8
                )
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            logging.info(f"[Rank {rank}] Generated: {generated_text[:100]}...")
        
        inference_step += 1
        
        import time
        time.sleep(0.1)
    
    logging.info(f"[Rank {rank}] Inference completed")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_epochs", type=int, default=1)
    args = parser.parse_args()
    
    # Setup distributed
    rank, local_rank, world_size = setup_distributed()
    
    # Setup logging
    log_file = setup_logging(rank)
    
    logging.info(f"\n{'='*60}")
    logging.info(f"Rank {rank}/{world_size} (Local Rank {local_rank})")
    logging.info(f"{'='*60}\n")
    
    # Determine role
    role = get_role(rank, world_size)
    logging.info(f"[Rank {rank}] Role: {role.upper()}")
    
    if role == "idle":
        logging.info(f"[Rank {rank}] Idle rank, exiting...")
        cleanup_distributed()
        return
    
    # Prepare model and data
    model, tokenizer, dataloader, device = prepare_model_and_data(
        rank, local_rank, world_size
    )
    
    try:
        # Run controller on rank 16
        routing_tables = None
        if rank == 16:
            routing_tables = run_controller(rank, world_size, model)
        
        # Synchronize all ranks
        dist.barrier()
        
        # Run appropriate role
        if role == "training":
            # Each training rank gets its routing table
            # In practice, controller would broadcast these
            routing_table = routing_tables[rank] if routing_tables else None
            run_training_rdma(rank, local_rank, world_size, model, dataloader, 
                            routing_table, args.num_epochs)
        
        elif role == "inference":
            run_inference_rdma(rank, local_rank, world_size, model, tokenizer)
        
        elif role == "controller":
            # Controller has done its job, can idle or participate in training
            logging.info(f"[Rank {rank}] Controller finished, idling...")
        
        # Final synchronization
        dist.barrier()
        
        if rank == 0:
            logging.info(f"\n{'='*60}")
            logging.info("ALL RANKS COMPLETED SUCCESSFULLY")
            logging.info(f"{'='*60}\n")
    
    except Exception as e:
        logging.error(f"[Rank {rank}] Error: {e}", exc_info=True)
        raise
    
    finally:
        # Cleanup
        cleanup_distributed()
        logging.info(f"[Rank {rank}] Full log saved to: {log_file}")


if __name__ == "__main__":
    main()
