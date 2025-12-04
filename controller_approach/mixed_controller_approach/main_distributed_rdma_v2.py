# main_rdma_v4_simple.py
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import GPT2LMHeadModel, GPT2Config, GPT2Tokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader, DistributedSampler
from uccl import p2p
import pickle
import time
import math
from dataclasses import dataclass, field
from typing import List

rank = int(os.environ.get("RANK", 0))
local_rank = int(os.environ.get("LOCAL_RANK", 0))

def log(msg):
    print(f"[Rank {rank}] {msg}", flush=True)

@dataclass
class WeightTransferEntry:
    param_name: str
    src_rank: int
    dst_rank: int
    size_bytes: int

@dataclass 
class RoutingTable:
    training_rank: int
    transfers: List[WeightTransferEntry] = field(default_factory=list)
    total_bytes: int = 0

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

# ============ STEP 3: DDP FOR TRAINING ============
log("Step 3: DDP setup")
TRAIN_RANKS = list(range(0, 24))
TRAIN_GROUP = None
role = "training" if rank < 24 else "inference"

if rank in TRAIN_RANKS:
    TRAIN_GROUP = dist.new_group(ranks=TRAIN_RANKS, backend="nccl")
    model = DDP(model, device_ids=[local_rank], process_group=TRAIN_GROUP)
    log("DDP wrapped")

dist.barrier()

# ============ STEP 4: COMPUTE ROUTING TABLES ============
log("Step 4: Compute routing tables")
routing_tables = {}

if rank == 0:
    if isinstance(model, DDP):
        params = list(model.module.named_parameters())
    else:
        params = list(model.named_parameters())
    
    for train_rank in range(8):
        rt = RoutingTable(training_rank=train_rank)
        target_inf = 24 + train_rank
        
        for name, param in params:
            size_bytes = param.numel() * param.element_size()
            entry = WeightTransferEntry(
                param_name=name,
                src_rank=train_rank,
                dst_rank=target_inf,
                size_bytes=size_bytes
            )
            rt.transfers.append(entry)
            rt.total_bytes += size_bytes
        
        routing_tables[train_rank] = rt
    
    # Print routing table
    log("")
    log("=" * 70)
    log("ROUTING TABLE")
    log("=" * 70)
    for train_rank in range(8):
        rt = routing_tables[train_rank]
        log(f"Rank {train_rank} -> Rank {24+train_rank}: {len(rt.transfers)} params, {rt.total_bytes/1e6:.2f} MB")
    log("=" * 70)

dist.barrier()

# ============ STEP 5: BROADCAST ROUTING ============
log("Step 5: Broadcast routing tables")
routing_table = None

if role == "training" and rank < 8:
    if rank == 0:
        for t in range(1, 8):
            data = pickle.dumps(routing_tables[t])
            dist.send(torch.tensor([len(data)], dtype=torch.int64), dst=t)
            dist.send(torch.ByteTensor(list(data)), dst=t)
        routing_table = routing_tables[0]
    else:
        size = torch.zeros(1, dtype=torch.int64)
        dist.recv(size, src=0)
        data = torch.zeros(int(size.item()), dtype=torch.uint8)
        dist.recv(data, src=0)
        routing_table = pickle.loads(bytes(data.tolist()))

dist.barrier()

# ============ STEP 6: RDMA SETUP ============
log("Step 6: RDMA setup")
ep = p2p.Endpoint(local_rank, 4)
local_metadata = ep.get_metadata()
ip, port, gpu = p2p.Endpoint.parse_metadata(local_metadata)
log(f"Endpoint: IP={ip}, Port={port}, GPU={gpu}")

connections = {}

if rank < 8:
    target_rank = 24 + rank
    dist.send(torch.ByteTensor(list(local_metadata)), dst=target_rank)
    remote_tensor = torch.zeros(len(local_metadata), dtype=torch.uint8)
    dist.recv(remote_tensor, src=target_rank)
    remote_metadata = bytes(remote_tensor.tolist())
    r_ip, r_port, r_gpu = p2p.Endpoint.parse_metadata(remote_metadata)
    ok, conn_id = ep.connect(r_ip, r_gpu, remote_port=r_port)
    log(f"Connected to rank {target_rank}: {ok}")
    if ok:
        connections[target_rank] = conn_id

elif rank >= 24:
    source_rank = rank - 24
    remote_tensor = torch.zeros(len(local_metadata), dtype=torch.uint8)
    dist.recv(remote_tensor, src=source_rank)
    dist.send(torch.ByteTensor(list(local_metadata)), dst=source_rank)
    ok, c_ip, c_gpu, conn_id = ep.accept()
    log(f"Accepted from rank {source_rank}: {ok}")
    if ok:
        connections[source_rank] = conn_id

dist.barrier()

# ============ STEP 7: PREPARE DATA ============
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
    sampler = DistributedSampler(tokenized, num_replicas=24, rank=TRAIN_RANKS.index(rank))
    dataloader = DataLoader(tokenized, batch_size=4, sampler=sampler)

dist.barrier()

# ============ STEP 8: TRAINING ============
total_training_time = 0
avg_loss = 0
perplexity = 0

if role == "training":
    if rank == 0:
        log("")
        log("=" * 70)
        log("PHASE 1: TRAINING")
        log("=" * 70)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    model.train()
    num_epochs = 2
    training_start = time.perf_counter()
    
    for epoch in range(num_epochs):
        epoch_start = time.perf_counter()
        dataloader.sampler.set_epoch(epoch)
        epoch_loss = 0
        epoch_steps = 0
        
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
        
        epoch_time = time.perf_counter() - epoch_start
        avg_loss = epoch_loss / epoch_steps if epoch_steps > 0 else 0
        perplexity = math.exp(avg_loss) if avg_loss < 20 else float('inf')
        
        if rank == 0:
            log(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Perplexity={perplexity:.2f}, Time={epoch_time:.2f}s")
    
    total_training_time = time.perf_counter() - training_start
    
    if rank == 0:
        log(f"Training Complete: {total_training_time:.2f}s, Final Perplexity={perplexity:.2f}")

dist.barrier()

# ============ STEP 9: RDMA WEIGHT TRANSFER ============
if rank == 0:
    log("")
    log("=" * 70)
    log("PHASE 2: RDMA WEIGHT TRANSFER")
    log("=" * 70)

transfer_start = time.perf_counter()

if role == "training" and rank < 8:
    target_rank = 24 + rank
    conn_id = connections[target_rank]
    
    if isinstance(model, DDP):
        param_dict = {n: p for n, p in model.module.named_parameters()}
    else:
        param_dict = {n: p for n, p in model.named_parameters()}
    
    registered_mrs = {}
    for name, param in param_dict.items():
        tensor = param.data.contiguous()
        ok, mr_id = ep.reg(tensor.data_ptr(), tensor.numel() * tensor.element_size())
        if ok:
            registered_mrs[name] = (mr_id, tensor.data_ptr(), tensor.numel() * tensor.element_size())
    
    torch.cuda.synchronize()
    dist.barrier()
    
    send_start = time.perf_counter()
    total_bytes = 0
    for entry in routing_table.transfers:
        if entry.param_name in registered_mrs:
            mr_id, ptr, size = registered_mrs[entry.param_name]
            ok = ep.send(conn_id, mr_id, ptr, size)
            if ok:
                total_bytes += size
    
    send_time = time.perf_counter() - send_start
    gbps = (total_bytes * 8) / send_time / 1e9 if send_time > 0 else 0
    log(f"SEND -> Rank {target_rank}: {total_bytes/1e6:.1f} MB, {send_time:.3f}s, {gbps:.2f} Gbps")

elif role == "training":
    torch.cuda.synchronize()
    dist.barrier()

else:
    source_rank = rank - 24
    conn_id = connections[source_rank]
    
    param_dict = {n: p for n, p in model.named_parameters()}
    registered_mrs = {}
    for name, param in param_dict.items():
        tensor = param.data.contiguous()
        ok, mr_id = ep.reg(tensor.data_ptr(), tensor.numel() * tensor.element_size())
        if ok:
            registered_mrs[name] = (mr_id, tensor.data_ptr(), tensor.numel() * tensor.element_size())
    
    torch.cuda.synchronize()
    dist.barrier()
    
    recv_start = time.perf_counter()
    total_bytes = 0
    for name, (mr_id, ptr, size) in registered_mrs.items():
        ok = ep.recv(conn_id, mr_id, ptr, size)
        if ok:
            total_bytes += size
    
    recv_time = time.perf_counter() - recv_start
    gbps = (total_bytes * 8) / recv_time / 1e9 if recv_time > 0 else 0
    log(f"RECV <- Rank {source_rank}: {total_bytes/1e6:.1f} MB, {recv_time:.3f}s, {gbps:.2f} Gbps")

dist.barrier()
transfer_time = time.perf_counter() - transfer_start

if rank == 0:
    log(f"Total Transfer Time: {transfer_time:.3f}s")

# ============ STEP 10: INFERENCE ============
if role == "inference":
    if rank == 24:
        log("")
        log("=" * 70)
        log("PHASE 3: INFERENCE")
        log("=" * 70)
    
    model.eval()
    prompts = ["The meaning of life is", "Artificial intelligence will"]
    
    inference_start = time.perf_counter()
    total_tokens = 0
    
    for prompt in prompts:
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
        
        if rank == 24:
            log(f"Prompt: {prompt}")
            log(f"Output: {text}")
            log(f"Tokens: {tokens}")
            log("-" * 40)
    
    inference_time = time.perf_counter() - inference_start
    
    if rank == 24:
        log(f"Inference Time: {inference_time:.2f}s, Throughput: {total_tokens/inference_time:.1f} tok/s")

dist.barrier()

# ============ FINAL SUMMARY ============
if rank == 0:
    log("")
    log("=" * 70)
    log("FINAL SUMMARY")
    log("=" * 70)
    log(f"Training Time:    {total_training_time:.2f}s")
    log(f"Transfer Time:    {transfer_time:.3f}s")
    log(f"Final Perplexity: {perplexity:.2f}")
    log("=" * 70)
    log("ALL DONE!")

dist.destroy_process_group()