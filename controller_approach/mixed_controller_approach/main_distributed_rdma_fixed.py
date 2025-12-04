"""
Main Distributed Training Script with RDMA Weight Transfers - FIXED
Uses UCCL P2P with GLOO-style metadata exchange (like benchmark_uccl.py)

KEY FIX: 
- Uses GLOO backend for metadata exchange (CPU tensors, like benchmark_uccl.py)
- Uses NCCL for DDP gradient sync

Setup for 4 nodes × 8 AMD GPUs:
- Node 0 (ranks 0-7): Training (rank 0 also runs controller)
- Node 1 (ranks 8-15): Training
- Node 2 (ranks 16-23): Training
- Node 3 (ranks 24-31): Inference

Total: 24 training GPUs, 8 inference GPUs

Usage:
    torchrun --nproc_per_node=8 \
             --nnodes=4 \
             --node_rank=$NODE_RANK \
             --master_addr=$MASTER_ADDR \
             --master_port=29501 \
             main_distributed_rdma_fixed.py
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
import pickle
import time

# Global process groups
TRAIN_RANKS = None
TRAIN_GROUP = None   # NCCL group for DDP
GLOO_GROUP = None    # GLOO group for metadata exchange (like benchmark_uccl.py)

from weight_transfer_controller import (
    WeightTransferController,
    simulate_parameter_distribution
)
from training_worker_rdma_fixed import OptimizedRDMATrainingWorker
from inference_worker_rdma_fixed import RDMAInferenceWorker


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
    
    old_factory = logging.getLogRecordFactory()
    def record_factory(*args, **kwargs):
        record = old_factory(*args, **kwargs)
        record.rank = rank
        return record
    logging.setLogRecordFactory(record_factory)
    
    logging.info(f"Logging initialized. Log file: {log_file}")
    return log_file


def setup_distributed():
    """
    Initialize distributed environment with:
    - NCCL backend for DDP (GPU gradient sync)
    - GLOO backend for metadata exchange (CPU tensors, like benchmark_uccl.py)
    """
    global TRAIN_RANKS, TRAIN_GROUP, GLOO_GROUP

    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    # Initialize NCCL as default backend (for DDP gradient sync)
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=world_size,
        rank=rank,
    )

    torch.cuda.set_device(local_rank)

    # ===== TRAINING SUBGROUP (NCCL): Ranks 0-23 for DDP gradient sync =====
    TRAIN_RANKS = list(range(0, 24))
    pg_train = dist.new_group(ranks=TRAIN_RANKS, backend="nccl")

    if rank in TRAIN_RANKS:
        TRAIN_GROUP = pg_train
        logging.info(f"[Rank {rank}] Joined TRAIN_GROUP (NCCL) with 24 ranks")
    else:
        TRAIN_GROUP = None

    # ===== GLOO GROUP: For metadata exchange (CPU tensors) =====
    # This is the KEY FIX - benchmark_uccl.py uses GLOO for metadata exchange
    # NCCL doesn't support CPU tensors, so metadata exchange was failing
    GLOO_GROUP = dist.new_group(ranks=list(range(world_size)), backend="gloo")
    logging.info(f"[Rank {rank}] Created GLOO_GROUP for metadata exchange (like benchmark_uccl.py)")

    logging.info(f"[Rank {rank}] Distributed setup: NCCL for DDP, GLOO for metadata")

    return rank, local_rank, world_size


def prepare_model_and_data(rank: int, local_rank: int, world_size: int):
    """Prepare model and dataset"""
    global TRAIN_RANKS, TRAIN_GROUP

    logging.info(f"[Rank {rank}] Loading GPT-2 model...")

    config = GPT2Config.from_pretrained("gpt2")
    model = GPT2LMHeadModel(config)

    device = torch.device(f"cuda:{local_rank}")
    model = model.to(device)

    role = get_role(rank, world_size)

    # Wrap with DDP for training ranks (uses NCCL)
    if role == "training":
        if TRAIN_GROUP is None or TRAIN_RANKS is None:
            raise RuntimeError(f"[Rank {rank}] TRAIN_GROUP not initialized")

        logging.info(f"[Rank {rank}] Wrapping model with DDP using TRAIN_GROUP (NCCL)")

        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            process_group=TRAIN_GROUP,
        )
    else:
        logging.info(f"[Rank {rank}] Role={role}, model NOT wrapped with DDP")

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    logging.info(f"[Rank {rank}] Loading wikitext-2 dataset...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    dataset = dataset.filter(lambda x: len(x["text"].strip()) > 0)

    if len(dataset) > 200:
        dataset = dataset.select(range(200))

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=128,
            padding="max_length",
        )

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
    )

    tokenized_dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask"],
    )

    if role == "training":
        train_world_size = len(TRAIN_RANKS)
        train_rank = TRAIN_RANKS.index(rank)

        logging.info(f"[Rank {rank}] TRAIN subgroup rank={train_rank}/{train_world_size}")

        sampler = DistributedSampler(
            tokenized_dataset,
            num_replicas=train_world_size,
            rank=train_rank,
            shuffle=True,
        )

        dataloader = DataLoader(
            tokenized_dataset,
            batch_size=4,
            sampler=sampler,
        )
    else:
        dataloader = None

    return model, tokenizer, dataloader, device


def cleanup_distributed():
    """Clean up distributed environment"""
    dist.destroy_process_group()


def get_role(rank: int, world_size: int):
    """Determine the role of this rank"""
    if rank < 24:
        return "training"
    elif 24 <= rank < 32:
        return "inference"
    else:
        return "idle"


def run_controller(rank: int, world_size: int, model: torch.nn.Module):
    """Controller logic (runs on rank 0)"""
    logging.info(f"\n{'='*60}")
    logging.info(f"[Rank {rank}] CONTROLLER: Computing routing tables")
    logging.info(f"{'='*60}\n")
    
    training_ranks = list(range(0, 24))
    inference_ranks = list(range(24, 32))
    
    logging.info(f"[Rank {rank}] Training ranks: {training_ranks}")
    logging.info(f"[Rank {rank}] Inference ranks: {inference_ranks}")
    
    trainer_params, inference_params = simulate_parameter_distribution(
        model, training_ranks, inference_ranks, use_data_parallel=True
    )
    
    controller = WeightTransferController(
        training_ranks=training_ranks,
        inference_ranks=inference_ranks,
        world_size=world_size
    )
    
    routing_tables = controller.compute_routing_tables(
        trainer_params, inference_params
    )
    
    controller.print_routing_details(routing_tables)
    
    logging.info(f"\n[Rank {rank}] Controller computation complete")
    
    return routing_tables


def broadcast_routing_table(routing_tables: dict, rank: int, training_ranks: list):
    """Broadcast routing tables from rank 0 to all training ranks"""
    if rank == 0:
        logging.info(f"[Rank {rank}] Broadcasting routing tables to training ranks...")
        
        for target_rank in training_ranks:
            if target_rank == 0:
                continue
            
            routing_table = routing_tables.get(target_rank, None)
            data = pickle.dumps(routing_table)
            
            size_tensor = torch.tensor([len(data)], dtype=torch.int64).cuda()
            dist.send(size_tensor, dst=target_rank)
            
            data_tensor = torch.ByteTensor(list(data)).cuda()
            dist.send(data_tensor, dst=target_rank)
            
            logging.info(f"[Rank {rank}] Sent routing table to rank {target_rank}")
        
        return routing_tables.get(0, None)
    
    else:
        logging.info(f"[Rank {rank}] Receiving routing table from rank 0...")
        
        size_tensor = torch.zeros(1, dtype=torch.int64).cuda()
        dist.recv(size_tensor, src=0)
        data_size = size_tensor.item()
        
        data_tensor = torch.zeros(data_size, dtype=torch.uint8).cuda()
        dist.recv(data_tensor, src=0)
        
        data = bytes(data_tensor.cpu().tolist())
        routing_table = pickle.loads(data)
        
        logging.info(f"[Rank {rank}] Received routing table")
        
        return routing_table


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
    global GLOO_GROUP
    
    logging.info(f"[CHECKPOINT T1] Rank {rank} - ENTERED run_training_rdma")
    
    logging.info(f"\n{'='*60}")
    logging.info(f"[Rank {rank}] TRAINING with RDMA (using GLOO for metadata)")
    logging.info(f"{'='*60}\n")
    
    # Create RDMA training worker
    logging.info(f"[CHECKPOINT T2] Rank {rank} - Creating RDMA worker")
    rdma_worker = OptimizedRDMATrainingWorker(
        model=model,
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        num_max_connections=16
    )
    logging.info(f"[CHECKPOINT T3] Rank {rank} - RDMA worker created")
    
    # Set routing table
    if routing_table:
        rdma_worker.set_routing_table(routing_table)
        logging.info(f"[Rank {rank}] Routing table set with {len(routing_table.transfers)} transfers")
    else:
        logging.warning(f"[Rank {rank}] No routing table received!")
    
    # Wait for all endpoints to initialize
    logging.info(f"[Rank {rank}] Waiting 5s for all endpoints to initialize...")
    time.sleep(5)
    
    # Exchange metadata using GLOO (like benchmark_uccl.py)
    training_ranks = list(range(0, 24))
    inference_ranks = list(range(24, 32))
    all_ranks = training_ranks + inference_ranks
    
    logging.info(f"[CHECKPOINT T4] Rank {rank} - Starting metadata exchange (GLOO)")
    all_metadata = rdma_worker.exchange_metadata(all_ranks, gloo_group=GLOO_GROUP)
    logging.info(f"[CHECKPOINT T5] Rank {rank} - Metadata exchange complete")
    
    dist.barrier()
    logging.info(f"[CHECKPOINT T6] Rank {rank} - After metadata barrier")
    
    # Wait for inference ranks to signal ready
    logging.info(f"[Rank {rank}] Waiting for inference ranks to be ready...")
    dist.barrier()
    logging.info(f"[Rank {rank}] Inference ranks ready, proceeding to connect")
    
    # Setup RDMA connections to inference ranks
    logging.info(f"[CHECKPOINT T7] Rank {rank} - Setting up RDMA connections")
    rdma_worker.setup_connections(inference_ranks, all_metadata)
    logging.info(f"[CHECKPOINT T8] Rank {rank} - RDMA connections established")
    
    dist.barrier()
    logging.info(f"[CHECKPOINT T9] Rank {rank} - All ranks completed connection setup")
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    
    # Training loop
    model.train()
    global_step = 0
    
    logging.info(f"[CHECKPOINT T10] Rank {rank} - Starting training loop")
    
    for epoch in range(num_epochs):
        logging.info(f"[Rank {rank}] Epoch {epoch+1}/{num_epochs}")
        
        dataloader.sampler.set_epoch(epoch)
        
        for step, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(f"cuda:{local_rank}")
            attention_mask = batch["attention_mask"].to(f"cuda:{local_rank}")
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids
            )
            loss = outputs.loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            global_step += 1
            
            if step % 10 == 0:
                logging.info(f"[Rank {rank}] Step {global_step}, Loss: {loss.item():.4f}")
            
            # Periodic RDMA weight update
            if global_step % 100 == 0:
                logging.info(f"\n[Rank {rank}] === RDMA WEIGHT UPDATE at step {global_step} ===")
                
                torch.cuda.synchronize()
                dist.barrier(group=TRAIN_GROUP)
                
                try:
                    logging.info(f"[Rank {rank}] Starting RDMA weight transfer...")
                    transfer_time = rdma_worker.transfer_weights_pipelined()
                    logging.info(f"[Rank {rank}] RDMA completed in {transfer_time:.2f}s")
                except Exception as e:
                    logging.error(f"[Rank {rank}] RDMA failed: {e}", exc_info=True)
                
                torch.cuda.synchronize()
                dist.barrier(group=TRAIN_GROUP)
                logging.info(f"[Rank {rank}] === RDMA WEIGHT UPDATE COMPLETE ===\n")
            
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
    global GLOO_GROUP
    
    logging.info(f"[CHECKPOINT I1] Rank {rank} - ENTERED run_inference_rdma")
    
    logging.info(f"\n{'='*60}")
    logging.info(f"[Rank {rank}] INFERENCE with RDMA (using GLOO for metadata)")
    logging.info(f"{'='*60}\n")
    
    training_ranks = list(range(0, 24))
    
    # Create RDMA inference worker
    logging.info(f"[CHECKPOINT I2] Rank {rank} - Creating RDMA worker")
    rdma_worker = RDMAInferenceWorker(
        model=model,
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        training_ranks=training_ranks,
        num_max_connections=24
    )
    logging.info(f"[CHECKPOINT I3] Rank {rank} - RDMA worker created")
    
    # Wait for all endpoints to initialize
    logging.info(f"[Rank {rank}] Waiting 5s for all endpoints to initialize...")
    time.sleep(5)
    
    # Exchange metadata using GLOO (like benchmark_uccl.py)
    inference_ranks = list(range(24, 32))
    all_ranks = training_ranks + inference_ranks
    
    logging.info(f"[CHECKPOINT I4] Rank {rank} - Starting metadata exchange (GLOO)")
    all_metadata = rdma_worker.exchange_metadata(all_ranks, gloo_group=GLOO_GROUP)
    logging.info(f"[CHECKPOINT I5] Rank {rank} - Metadata exchange complete")
    
    dist.barrier()
    logging.info(f"[CHECKPOINT I6] Rank {rank} - After metadata barrier")
    
    # Signal to training ranks that we're ready
    logging.info(f"[Rank {rank}] Signaling ready to accept connections")
    dist.barrier()
    logging.info(f"[Rank {rank}] All ranks synchronized")
    
    # Accept connections from training ranks
    logging.info(f"[CHECKPOINT I7] Rank {rank} - Accepting RDMA connections")
    rdma_worker.setup_connections(training_ranks=training_ranks)
    logging.info(f"[CHECKPOINT I8] Rank {rank} - All connections accepted")
    
    dist.barrier()
    logging.info(f"[CHECKPOINT I9] Rank {rank} - All ranks completed connection setup")
    
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
    
    logging.info(f"[CHECKPOINT I10] Rank {rank} - Starting inference loop")
    
    inference_step = 0
    while inference_step < 50:
        if inference_step % 100 == 0 and inference_step > 0:
            logging.info(f"\n[Rank {rank}] === RECEIVING RDMA WEIGHT UPDATE ===")
            
            torch.cuda.synchronize()
            dist.barrier()
            
            try:
                logging.info(f"[Rank {rank}] Starting RDMA weight receive...")
                receive_time = rdma_worker.receive_weights()
                logging.info(f"[Rank {rank}] RDMA received in {receive_time:.2f}s")
            except Exception as e:
                logging.error(f"[Rank {rank}] RDMA receive failed: {e}", exc_info=True)
            
            torch.cuda.synchronize()
            dist.barrier()
            logging.info(f"[Rank {rank}] === RDMA WEIGHT UPDATE COMPLETE ===\n")
        
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
        # Run controller on rank 0 ONLY
        routing_tables = {}
        if rank == 0:
            logging.info(f"[CHECKPOINT C1] Rank {rank} - Starting controller")
            routing_tables = run_controller(rank, world_size, model)
            logging.info(f"[CHECKPOINT C2] Rank {rank} - Controller finished")
        
        # Synchronize all ranks after controller
        logging.info(f"[CHECKPOINT] Rank {rank} - Waiting at controller barrier...")
        dist.barrier()
        logging.info(f"[CHECKPOINT] Rank {rank} - Passed controller barrier")
        
        # Broadcast routing tables to all training ranks
        if role == "training":
            logging.info(f"[CHECKPOINT] Rank {rank} - Getting routing table...")
            training_ranks = list(range(0, 24))
            routing_table = broadcast_routing_table(routing_tables, rank, training_ranks)
            logging.info(f"[CHECKPOINT] Rank {rank} - Routing table received")
            
            logging.info(f"[CHECKPOINT] Rank {rank} - ABOUT TO ENTER run_training_rdma")
            run_training_rdma(rank, local_rank, world_size, model, dataloader, 
                            routing_table, args.num_epochs)
            logging.info(f"[CHECKPOINT] Rank {rank} - EXITED run_training_rdma")
        
        elif role == "inference":
            logging.info(f"[CHECKPOINT] Rank {rank} - ABOUT TO ENTER run_inference_rdma")
            run_inference_rdma(rank, local_rank, world_size, model, tokenizer)
            logging.info(f"[CHECKPOINT] Rank {rank} - EXITED run_inference_rdma")
        
        # Final synchronization
        logging.info(f"[CHECKPOINT] Rank {rank} - Waiting at final barrier...")
        dist.barrier()
        logging.info(f"[CHECKPOINT] Rank {rank} - Passed final barrier")
        
        if rank == 0:
            logging.info(f"\n{'='*60}")
            logging.info("ALL RANKS COMPLETED SUCCESSFULLY")
            logging.info(f"{'='*60}\n")
    
    except Exception as e:
        logging.error(f"[Rank {rank}] Error: {e}", exc_info=True)
        raise
    
    finally:
        cleanup_distributed()
        logging.info(f"[Rank {rank}] Full log saved to: {log_file}")


if __name__ == "__main__":
    main()