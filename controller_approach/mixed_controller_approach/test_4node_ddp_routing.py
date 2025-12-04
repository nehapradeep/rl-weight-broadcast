# test_4node_ddp_routing.py
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import GPT2LMHeadModel, GPT2Config
from uccl import p2p
import pickle
from dataclasses import dataclass, field
from typing import List, Tuple

rank = int(os.environ.get("RANK", 0))
local_rank = int(os.environ.get("LOCAL_RANK", 0))

print(f"[Rank {rank}] Starting...", flush=True)

# Routing table dataclass
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

# Step 1: GLOO init
dist.init_process_group(backend="gloo")
torch.cuda.set_device(local_rank)
print(f"[Rank {rank}] GLOO initialized", flush=True)

# Step 2: Load model
config = GPT2Config.from_pretrained("gpt2")
model = GPT2LMHeadModel(config).to(f"cuda:{local_rank}")
print(f"[Rank {rank}] Model loaded", flush=True)

# Step 3: DDP for training ranks
TRAIN_RANKS = list(range(0, 24))
TRAIN_GROUP = None
role = "training" if rank < 24 else "inference"

if rank in TRAIN_RANKS:
    TRAIN_GROUP = dist.new_group(ranks=TRAIN_RANKS, backend="nccl")
    model = DDP(model, device_ids=[local_rank], process_group=TRAIN_GROUP)
    print(f"[Rank {rank}] DDP wrapped", flush=True)

dist.barrier()

# Step 4: Compute routing tables (rank 0 only)
print(f"[Rank {rank}] Computing routing tables...", flush=True)

routing_tables = {}
if rank == 0:
    # Get model parameters
    if isinstance(model, DDP):
        params = list(model.module.named_parameters())
    else:
        params = list(model.named_parameters())
    
    print(f"[Rank 0] Model has {len(params)} parameters", flush=True)
    
    # Compute routing for each training rank
    for train_rank in range(24):
        rt = RoutingTable(training_rank=train_rank)
        target_inf = 24 + (train_rank % 8)
        
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
    
    # Print routing summary
    print(f"\n{'='*60}", flush=True)
    print(f"ROUTING TABLE SUMMARY", flush=True)
    print(f"{'='*60}", flush=True)
    
    for inf_rank in range(24, 32):
        sources = [r for r in range(24) if (r % 8) == (inf_rank - 24)]
        print(f"Inference Rank {inf_rank} <- Training Ranks {sources}", flush=True)
    
    print(f"\nEach training rank sends:", flush=True)
    print(f"  - {len(routing_tables[0].transfers)} parameters", flush=True)
    print(f"  - {routing_tables[0].total_bytes / 1e6:.2f} MB total", flush=True)
    
    # Print first 5 params as sample
    print(f"\nSample parameters (first 5):", flush=True)
    for entry in routing_tables[0].transfers[:5]:
        print(f"  - {entry.param_name}: {entry.size_bytes/1e3:.1f} KB", flush=True)
    
    print(f"{'='*60}\n", flush=True)

dist.barrier()

# Step 5: Broadcast routing tables
print(f"[Rank {rank}] Broadcasting routing tables...", flush=True)

routing_table = None
if role == "training":
    if rank == 0:
        for t in range(1, 24):
            data = pickle.dumps(routing_tables[t])
            dist.send(torch.tensor([len(data)], dtype=torch.int64), dst=t)
            dist.send(torch.ByteTensor(list(data)), dst=t)
        routing_table = routing_tables[0]
        print(f"[Rank 0] Broadcast complete", flush=True)
    else:
        size = torch.zeros(1, dtype=torch.int64)
        dist.recv(size, src=0)
        data = torch.zeros(int(size.item()), dtype=torch.uint8)
        dist.recv(data, src=0)
        routing_table = pickle.loads(bytes(data.tolist()))
        print(f"[Rank {rank}] Received: {len(routing_table.transfers)} params, "
              f"{routing_table.total_bytes/1e6:.2f} MB -> Rank {24 + (rank % 8)}", flush=True)
else:
    print(f"[Rank {rank}] Inference - no routing needed", flush=True)

dist.barrier()
print(f"[Rank {rank}] After routing barrier", flush=True)

# Step 6: RDMA setup
print(f"[Rank {rank}] Creating endpoint...", flush=True)
ep = p2p.Endpoint(local_rank, 4)
local_metadata = ep.get_metadata()
ip, port, gpu = p2p.Endpoint.parse_metadata(local_metadata)
print(f"[Rank {rank}] Endpoint: IP={ip}, Port={port}, GPU={gpu}", flush=True)

connections = {}

if rank < 24:
    target_rank = 24 + (rank % 8)
    
    print(f"[Rank {rank}] Sending metadata to {target_rank}...", flush=True)
    dist.send(torch.ByteTensor(list(local_metadata)), dst=target_rank)
    remote_tensor = torch.zeros(len(local_metadata), dtype=torch.uint8)
    dist.recv(remote_tensor, src=target_rank)
    remote_metadata = bytes(remote_tensor.tolist())
    
    r_ip, r_port, r_gpu = p2p.Endpoint.parse_metadata(remote_metadata)
    print(f"[Rank {rank}] Connecting to {target_rank} at {r_ip}:{r_port}...", flush=True)
    ok, conn_id = ep.connect(r_ip, r_gpu, remote_port=r_port)
    print(f"[Rank {rank}] Connect: {ok}", flush=True)
    if ok:
        connections[target_rank] = conn_id

else:
    my_sources = [r for r in range(24) if (r % 8) == (rank - 24)]
    print(f"[Rank {rank}] Accepting from: {my_sources}", flush=True)
    
    for source_rank in my_sources:
        print(f"[Rank {rank}] Recv metadata from {source_rank}...", flush=True)
        remote_tensor = torch.zeros(len(local_metadata), dtype=torch.uint8)
        dist.recv(remote_tensor, src=source_rank)
        dist.send(torch.ByteTensor(list(local_metadata)), dst=source_rank)
        
        print(f"[Rank {rank}] Accepting from {source_rank}...", flush=True)
        ok, c_ip, c_gpu, conn_id = ep.accept()
        print(f"[Rank {rank}] Accept: {ok}", flush=True)
        if ok:
            connections[source_rank] = conn_id

dist.barrier()
print(f"[Rank {rank}] SUCCESS! {len(connections)} connections", flush=True)
dist.destroy_process_group()