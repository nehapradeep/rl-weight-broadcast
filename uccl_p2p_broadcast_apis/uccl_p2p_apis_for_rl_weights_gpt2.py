# main_rdma_configurable.py
import os
import sys
import argparse
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import GPT2LMHeadModel, GPT2Config, GPT2Tokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader, DistributedSampler
from uccl import p2p
import time
import math

# ============ PARSE ARGUMENTS ============
parser = argparse.ArgumentParser(description='Configurable RDMA Training + Inference')
parser.add_argument('--num_training', type=int, default=8, help='Number of training GPUs')
parser.add_argument('--num_inference', type=int, default=8, help='Number of inference GPUs')
parser.add_argument('--gpus_per_node', type=int, default=8, help='GPUs per node')
parser.add_argument('--num_shards', type=int, default=8, help='Number of model shards')
parser.add_argument('--epochs', type=int, default=2, help='Training epochs')
args = parser.parse_args()

# ============ CONFIGURATION ============
NUM_TRAINING_RANKS = args.num_training
NUM_INFERENCE_RANKS = args.num_inference
GPUS_PER_NODE = args.gpus_per_node
NUM_SHARDS = min(args.num_shards, NUM_TRAINING_RANKS, NUM_INFERENCE_RANKS)
NUM_EPOCHS = args.epochs

rank = int(os.environ.get("RANK", 0))
local_rank = int(os.environ.get("LOCAL_RANK", 0))
world_size = int(os.environ.get("WORLD_SIZE", NUM_TRAINING_RANKS + NUM_INFERENCE_RANKS))

TRAIN_RANKS = list(range(0, NUM_TRAINING_RANKS))
INFERENCE_RANKS = list(range(NUM_TRAINING_RANKS, NUM_TRAINING_RANKS + NUM_INFERENCE_RANKS))

role = "training" if rank in TRAIN_RANKS else "inference"

def log(msg):
    print(f"[Rank {rank}] {msg}", flush=True)
    sys.stdout.flush()

# ============ PRINT TOPOLOGY ============
if rank == 0:
    total_nodes = (NUM_TRAINING_RANKS + NUM_INFERENCE_RANKS) // GPUS_PER_NODE
    training_nodes = NUM_TRAINING_RANKS // GPUS_PER_NODE
    inference_nodes = NUM_INFERENCE_RANKS // GPUS_PER_NODE
    
    log("")
    log("=" * 70)
    log("                    CLUSTER CONFIGURATION")
    log("=" * 70)
    log(f"  Total GPUs:          {world_size}")
    log(f"  Training GPUs:       {NUM_TRAINING_RANKS} (Ranks 0-{NUM_TRAINING_RANKS-1})")
    log(f"  Inference GPUs:      {NUM_INFERENCE_RANKS} (Ranks {NUM_TRAINING_RANKS}-{NUM_TRAINING_RANKS+NUM_INFERENCE_RANKS-1})")
    log(f"  GPUs per Node:       {GPUS_PER_NODE}")
    log(f"  Total Nodes:         {total_nodes}")
    log(f"  Training Nodes:      {training_nodes}")
    log(f"  Inference Nodes:     {inference_nodes}")
    log(f"  Model Shards:        {NUM_SHARDS}")
    log(f"  Active Senders:      {NUM_SHARDS} (Ranks 0-{NUM_SHARDS-1})")
    if NUM_TRAINING_RANKS > NUM_SHARDS:
        log(f"  Idle Training:       {NUM_TRAINING_RANKS - NUM_SHARDS} (Ranks {NUM_SHARDS}-{NUM_TRAINING_RANKS-1})")
    log("")
    log("  RDMA Pattern:")
    log(f"    - {NUM_SHARDS} shards sent to {NUM_INFERENCE_RANKS} receivers")
    log(f"    - {NUM_SHARDS * NUM_INFERENCE_RANKS} parallel RDMA transfers")
    log("=" * 70)
    log("")

# ============ STEP 1: GLOO INIT ============
log("Step 1: GLOO init")
dist.init_process_group(backend="gloo")
torch.cuda.set_device(local_rank)

# ============ STEP 2: LOAD MODEL ============
log("Step 2: Load model")
config = GPT2Config.from_pretrained("gpt2")
model = GPT2LMHeadModel(config).to(f"cuda:{local_rank}")

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

model_size_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
model_size_mb = model_size_bytes / 1e6

if rank == 0:
    log(f"Model size: {model_size_mb:.2f} MB ({model_size_mb/1e3:.2f} GB)")

# ============ STEP 3: DDP FOR TRAINING ============
log("Step 3: DDP setup")
TRAIN_GROUP = None

if rank in TRAIN_RANKS:
    TRAIN_GROUP = dist.new_group(ranks=TRAIN_RANKS, backend="nccl")
    model = DDP(model, device_ids=[local_rank], process_group=TRAIN_GROUP)
    log("DDP wrapped")
else:
    log("Inference rank - no DDP")

dist.barrier()

# ============ SHARD MODEL PARAMETERS ============
def get_param_shards(model, num_shards):
    """Split model parameters into shards"""
    if isinstance(model, DDP):
        params = list(model.module.named_parameters())
    else:
        params = list(model.named_parameters())
    
    total_size = sum(p.numel() * p.element_size() for _, p in params)
    shard_size_target = total_size // num_shards
    
    shards = [[] for _ in range(num_shards)]
    shard_sizes = [0] * num_shards
    current_shard = 0
    current_size = 0
    
    for name, param in params:
        param_size = param.numel() * param.element_size()
        shards[current_shard].append((name, param))
        shard_sizes[current_shard] += param_size
        current_size += param_size
        
        if current_size >= shard_size_target and current_shard < num_shards - 1:
            current_shard += 1
            current_size = 0
    
    return shards, shard_sizes

# ============ STEP 4: RDMA SETUP ============
log("Step 4: RDMA setup")
ep = p2p.Endpoint(local_rank, 4)
local_metadata = ep.get_metadata()
ip, port, gpu = p2p.Endpoint.parse_metadata(local_metadata)
log(f"Endpoint: IP={ip}, Port={port}, GPU={gpu}")

connections = {}

if rank < NUM_SHARDS:
    # Sender ranks: connect to all inference ranks
    for inf_rank in INFERENCE_RANKS:
        dist.send(torch.ByteTensor(list(local_metadata)), dst=inf_rank)
        remote_tensor = torch.zeros(len(local_metadata), dtype=torch.uint8)
        dist.recv(remote_tensor, src=inf_rank)
        remote_metadata = bytes(remote_tensor.tolist())
        
        r_ip, r_port, r_gpu = p2p.Endpoint.parse_metadata(remote_metadata)
        ok, conn_id = ep.connect(r_ip, r_gpu, remote_port=r_port)
        if ok:
            connections[inf_rank] = conn_id
    log(f"Connected to {len(connections)} inference ranks")

elif rank in INFERENCE_RANKS:
    # Inference ranks: accept from sender ranks (0 to NUM_SHARDS-1)
    for src_rank in range(NUM_SHARDS):
        remote_tensor = torch.zeros(len(local_metadata), dtype=torch.uint8)
        dist.recv(remote_tensor, src=src_rank)
        dist.send(torch.ByteTensor(list(local_metadata)), dst=src_rank)
        
        ok, c_ip, c_gpu, conn_id = ep.accept()
        if ok:
            connections[src_rank] = conn_id
    log(f"Accepted from {len(connections)} sender ranks")

else:
    log("No RDMA connections (idle during transfer)")

dist.barrier()

# ============ STEP 5: PREPARE DATA ============
dataloader = None
if role == "training":
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    dataset = dataset.filter(lambda x: len(x["text"].strip()) > 0)
    dataset = dataset.select(range(min(500, len(dataset))))
    tokenized = dataset.map(
        lambda x: tokenizer(x["text"], truncation=True, max_length=128, padding="max_length"),
        batched=True, remove_columns=dataset.column_names
    )
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask"])
    sampler = DistributedSampler(tokenized, num_replicas=NUM_TRAINING_RANKS, rank=TRAIN_RANKS.index(rank))
    dataloader = DataLoader(tokenized, batch_size=4, sampler=sampler)

dist.barrier()

# ============ STEP 6: TRAINING ============
total_training_time = 0
avg_loss = 0
perplexity = 0

if role == "training":
    if rank == 0:
        log("")
        log("=" * 70)
        log("                       PHASE 1: TRAINING")
        log("=" * 70)
        log(f"Training with {NUM_TRAINING_RANKS} GPUs, {NUM_EPOCHS} epochs")
        log("")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    model.train()
    training_start = time.perf_counter()
    
    for epoch in range(NUM_EPOCHS):
        epoch_start = time.perf_counter()
        dataloader.sampler.set_epoch(epoch)
        epoch_loss = 0
        epoch_steps = 0
        epoch_tokens = 0
        
        for batch in dataloader:
            input_ids = batch["input_ids"].to(f"cuda:{local_rank}")
            attention_mask = batch["attention_mask"].to(f"cuda:{local_rank}")
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs.loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_steps += 1
            epoch_tokens += attention_mask.sum().item()
        
        epoch_time = time.perf_counter() - epoch_start
        avg_loss = epoch_loss / epoch_steps if epoch_steps > 0 else 0
        perplexity = math.exp(avg_loss) if avg_loss < 20 else float('inf')
        tokens_per_sec = epoch_tokens / epoch_time if epoch_time > 0 else 0
        
        if rank == 0:
            log(f"Epoch {epoch+1}/{NUM_EPOCHS}: Loss={avg_loss:.4f}, PPL={perplexity:.2f}, "
                f"Tok/s={tokens_per_sec:.0f}, Time={epoch_time:.2f}s")
    
    total_training_time = time.perf_counter() - training_start
    
    if rank == 0:
        log("")
        log(f"Training Complete: {total_training_time:.2f}s, Final PPL={perplexity:.2f}")
else:
    log("Waiting for training...")

dist.barrier()

# ============ STEP 7: SHARDED RDMA WEIGHT TRANSFER ============
if rank == 0:
    log("")
    log("=" * 70)
    log("                PHASE 2: RDMA WEIGHT TRANSFER")
    log("=" * 70)

shards, shard_sizes = get_param_shards(model, num_shards=NUM_SHARDS)

if rank == 0:
    log("")
    log("Shard Distribution:")
    log("-" * 50)
    for i, size in enumerate(shard_sizes):
        log(f"  Shard {i} (Rank {i}): {size/1e6:6.2f} MB")
    log("-" * 50)
    log(f"  Total: {sum(shard_sizes)/1e6:.2f} MB")
    log("")

transfer_start = time.perf_counter()

if rank < NUM_SHARDS:
    # Sender: send my shard to all inference ranks
    my_shard = shards[rank]
    
    registered_mrs = {}
    for name, param in my_shard:
        tensor = param.data.contiguous()
        ok, mr_id = ep.reg(tensor.data_ptr(), tensor.numel() * tensor.element_size())
        if ok:
            registered_mrs[name] = (mr_id, tensor.data_ptr(), tensor.numel() * tensor.element_size())
    
    torch.cuda.synchronize()
    dist.barrier()
    
    t0 = time.perf_counter()
    total_bytes = 0
    
    for inf_rank in INFERENCE_RANKS:
        conn_id = connections[inf_rank]
        for name, (mr_id, ptr, size) in registered_mrs.items():
            ok = ep.send(conn_id, mr_id, ptr, size)
            if ok:
                total_bytes += size
    
    duration = time.perf_counter() - t0
    bw = (total_bytes / 1e9) / duration if duration > 0 else 0
    
    log(f"UCCL Broadcast Complete. Time: {duration:.4f}s | BW: {bw:.2f} GB/s | "
        f"Sent: {total_bytes/1e6:.1f} MB to {NUM_INFERENCE_RANKS} receivers")

elif rank in INFERENCE_RANKS:
    # Receiver: receive all shards
    registered_mrs = {}
    for name, param in model.named_parameters():
        tensor = param.data.contiguous()
        ok, mr_id = ep.reg(tensor.data_ptr(), tensor.numel() * tensor.element_size())
        if ok:
            registered_mrs[name] = (mr_id, tensor.data_ptr(), tensor.numel() * tensor.element_size())
    
    torch.cuda.synchronize()
    dist.barrier()
    
    t0 = time.perf_counter()
    total_bytes = 0
    
    for src_rank in range(NUM_SHARDS):
        conn_id = connections[src_rank]
        src_shard = shards[src_rank]
        
        for name, _ in src_shard:
            if name in registered_mrs:
                mr_id, ptr, size = registered_mrs[name]
                ok = ep.recv(conn_id, mr_id, ptr, size)
                if ok:
                    total_bytes += size
    
    duration = time.perf_counter() - t0
    bw = (total_bytes / 1e9) / duration if duration > 0 else 0
    
    log(f"UCCL Receive Complete. Updated Model. Time: {duration:.4f}s | BW: {bw:.2f} GB/s | "
        f"Received: {total_bytes/1e6:.1f} MB from {NUM_SHARDS} senders")

else:
    torch.cuda.synchronize()
    dist.barrier()

dist.barrier()
transfer_time = time.perf_counter() - transfer_start

if rank == 0:
    total_data_mb = model_size_mb * NUM_INFERENCE_RANKS
    aggregate_bw = (total_data_mb / 1e3) / transfer_time if transfer_time > 0 else 0
    
    log("")
    log("Transfer Summary:")
    log("-" * 50)
    log(f"  Wall Clock Time:     {transfer_time:.4f}s")
    log(f"  Total Data Moved:    {total_data_mb:.2f} MB")
    log(f"  Aggregate BW:        {aggregate_bw:.2f} GB/s ({aggregate_bw*8:.2f} Gbps)")
    log("-" * 50)

# ============ STEP 8: INFERENCE ============
if role == "inference":
    if rank == INFERENCE_RANKS[0]:
        log("")
        log("=" * 70)
        log("                      PHASE 3: INFERENCE")
        log("=" * 70)
    
    model.eval()
    prompts = ["The meaning of life is", "Artificial intelligence will"]
    
    inference_start = time.perf_counter()
    total_tokens = 0
    
    for i, prompt in enumerate(prompts):
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(f"cuda:{local_rank}")
        input_len = input_ids.shape[1]
        
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids, max_length=60, do_sample=True,
                temperature=0.8, pad_token_id=tokenizer.eos_token_id
            )
        
        tokens = outputs.shape[1] - input_len
        total_tokens += tokens
        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        if rank == INFERENCE_RANKS[0]:
            log(f"Prompt: \"{prompt}\"")
            log(f"Output: {text}")
            log("-" * 40)
    
    inference_time = time.perf_counter() - inference_start
    
    if rank == INFERENCE_RANKS[0]:
        log(f"Inference Time: {inference_time:.2f}s | Throughput: {total_tokens/inference_time:.1f} tok/s")

dist.barrier()

# ============ FINAL SUMMARY ============
if rank == 0:
    log("")
    log("=" * 70)
    log("                        FINAL SUMMARY")
    log("=" * 70)
    log(f"  Training:    {total_training_time:.2f}s | PPL: {perplexity:.2f}")
    log(f"  Transfer:    {transfer_time:.4f}s | {aggregate_bw:.2f} GB/s")
    log(f"  Config:      {NUM_TRAINING_RANKS} train + {NUM_INFERENCE_RANKS} infer GPUs")
    log("=" * 70)

dist.destroy_process_group()