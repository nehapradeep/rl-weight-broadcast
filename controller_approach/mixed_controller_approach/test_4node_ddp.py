# test_4node_ddp_routing.py
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import GPT2LMHeadModel, GPT2Config
from uccl import p2p
import pickle

rank = int(os.environ.get("RANK", 0))
local_rank = int(os.environ.get("LOCAL_RANK", 0))

print(f"[Rank {rank}] Starting...", flush=True)

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

# Step 4: Routing table broadcast (like V3)
print(f"[Rank {rank}] Starting routing table broadcast...", flush=True)

routing_table = None
if role == "training":
    if rank == 0:
        # Create dummy routing tables
        routing_tables = {r: f"routing_for_rank_{r}" for r in range(24)}
        for t in range(1, 24):
            data = pickle.dumps(routing_tables[t])
            print(f"[Rank 0] Sending routing to rank {t}...", flush=True)
            dist.send(torch.tensor([len(data)], dtype=torch.int64), dst=t)
            dist.send(torch.ByteTensor(list(data)), dst=t)
        routing_table = routing_tables[0]
        print(f"[Rank 0] Routing broadcast complete", flush=True)
    else:
        print(f"[Rank {rank}] Waiting for routing from rank 0...", flush=True)
        size = torch.zeros(1, dtype=torch.int64)
        dist.recv(size, src=0)
        data = torch.zeros(int(size.item()), dtype=torch.uint8)
        dist.recv(data, src=0)
        routing_table = pickle.loads(bytes(data.tolist()))
        print(f"[Rank {rank}] Received routing table", flush=True)
else:
    print(f"[Rank {rank}] Inference - skipping routing broadcast", flush=True)

dist.barrier()
print(f"[Rank {rank}] After routing barrier", flush=True)

# Step 5: RDMA setup
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
    print(f"[Rank {rank}] Connecting to {target_rank}...", flush=True)
    ok, conn_id = ep.connect(r_ip, r_gpu, remote_port=r_port)
    print(f"[Rank {rank}] Connect: {ok}", flush=True)
    connections[target_rank] = conn_id

else:
    my_sources = [r for r in range(24) if (r % 8) == (rank - 24)]
    print(f"[Rank {rank}] Accepting from: {my_sources}", flush=True)
    
    for source_rank in my_sources:
        print(f"[Rank {rank}] Recv metadata from {source_rank}...", flush=True)
        remote_tensor = torch.zeros(len(local_metadata), dtype=torch.uint8)
        dist.recv(remote_tensor, src=source_rank)
        dist.send(torch.ByteTensor(list(local_metadata)), dst=source_rank)
        
        print(f"[Rank {rank}] Accepting...", flush=True)
        ok, c_ip, c_gpu, conn_id = ep.accept()
        print(f"[Rank {rank}] Accept: {ok}", flush=True)
        connections[source_rank] = conn_id

dist.barrier()
print(f"[Rank {rank}] SUCCESS! {len(connections)} connections", flush=True)
dist.destroy_process_group()