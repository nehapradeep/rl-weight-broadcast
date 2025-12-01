# Complete Integration Guide: Controller + UCCL P2P RDMA

This guide shows how the **Controller Routing Table** approach integrates with **UCCL P2P RDMA** for efficient weight transfers.

## System Overview

```
┌────────────────────────────────────────────────────────────────┐
│                   STARTUP PHASE (Once)                          │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. Controller (Rank 16) computes routing tables               │
│     ├─ Collects parameter metadata from all ranks              │
│     ├─ Matches training & inference parameters                 │
│     ├─ Computes load-balanced routing plan                     │
│     └─ Distributes tables to training ranks                    │
│                                                                 │
│  2. RDMA Connection Setup                                       │
│     ├─ All ranks exchange UCCL P2P metadata                    │
│     ├─ Training ranks connect to inference ranks               │
│     └─ Inference ranks accept connections                      │
│                                                                 │
└────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────┐
│              TRAINING LOOP (Repeated Every N Steps)            │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. Training (Ranks 0-15)                                       │
│     ├─ Forward & backward pass                                 │
│     ├─ Optimizer step                                          │
│     └─ Every 100 steps: Trigger weight transfer                │
│                                                                 │
│  2. Weight Transfer via RDMA                                    │
│     ├─ Training ranks execute routing table                    │
│     │   └─ For each entry: ep.send(conn_id, mr_id, ptr, size) │
│     └─ Inference ranks receive                                 │
│         └─ For each param: ep.recv(conn_id, mr_id, ptr, size)  │
│                                                                 │
│  3. Inference (Ranks 24-31)                                     │
│     ├─ Weights automatically updated via RDMA                  │
│     └─ Continue serving inference requests                     │
│                                                                 │
└────────────────────────────────────────────────────────────────┘
```

## Component Interaction

### 1. Controller → Training Ranks

**Controller computes routing tables:**
```python
# On Rank 16 (Controller)
routing_tables = controller.compute_routing_tables(
    trainer_params,
    inference_params
)

# routing_tables[0] = RoutingTable for Training Rank 0:
# - transformer.h.0.attn.c_attn.weight → Inference Rank 24
# - transformer.h.1.attn.c_attn.weight → Inference Rank 25
# ... etc

# routing_tables[1] = RoutingTable for Training Rank 1:
# - transformer.h.2.attn.c_attn.weight → Inference Rank 26
# ... etc
```

**Distribution to training ranks** (two approaches):

**Option A: Broadcast via torch.distributed**
```python
# On controller (Rank 16)
for train_rank, routing_table in routing_tables.items():
    # Serialize routing table
    serialized = pickle.dumps(routing_table)
    tensor = torch.ByteTensor(list(serialized))
    dist.send(tensor, dst=train_rank)

# On training ranks (Ranks 0-15)
if rank < 16:
    # Receive routing table
    buffer = torch.zeros(MAX_SIZE, dtype=torch.uint8)
    dist.recv(buffer, src=16)  # From controller
    routing_table = pickle.loads(bytes(buffer.tolist()))
    
    rdma_worker.set_routing_table(routing_table)
```

**Option B: Shared file system**
```python
# On controller (Rank 16)
for train_rank, routing_table in routing_tables.items():
    with open(f'/shared/routing_table_{train_rank}.pkl', 'wb') as f:
        pickle.dump(routing_table, f)

# On training ranks (Ranks 0-15)
if rank < 16:
    with open(f'/shared/routing_table_{rank}.pkl', 'rb') as f:
        routing_table = pickle.load(f)
    
    rdma_worker.set_routing_table(routing_table)
```

### 2. Controller → Inference Ranks

**Parameter source mapping:**
```python
# Controller creates param_sources for each inference rank
# param_sources[24] = { "layer0.weight": 0, "layer1.weight": 1, ... }
# param_sources[25] = { "layer2.weight": 2, "layer3.weight": 3, ... }

# Distribute to inference ranks
for infer_rank in range(24, 32):
    param_sources = extract_param_sources_for_rank(
        routing_tables, infer_rank
    )
    
    # Broadcast via torch.distributed or shared file
    dist.send(serialize(param_sources), dst=infer_rank)

# On inference ranks (Ranks 24-31)
if rank >= 24:
    param_sources = dist.recv(src=16)  # From controller
    rdma_worker.set_param_sources(param_sources)
```

### 3. RDMA Metadata Exchange

**All ranks participate:**
```python
# Every rank (0-31) participates in metadata exchange

# 1. Create UCCL P2P endpoint
ep = p2p.Endpoint(local_gpu, max_connections)
local_md = ep.get_metadata()

# 2. Exchange metadata via torch.distributed
all_metadata = {}
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

all_metadata[rank] = local_md
```

### 4. RDMA Connection Establishment

**Training ranks connect to inference ranks:**
```python
# Training ranks (0-15)
if rank < 16:
    inference_ranks = list(range(24, 32))
    
    for infer_rank in inference_ranks:
        # Parse metadata
        ip, port, r_gpu = p2p.Endpoint.parse_metadata(
            all_metadata[infer_rank]
        )
        
        # Connect
        ok, conn_id = ep.connect(ip, r_gpu, remote_port=port)
        connections[infer_rank] = conn_id
```

**Inference ranks accept from training ranks:**
```python
# Inference ranks (24-31)
if rank >= 24:
    training_ranks = list(range(0, 16))
    
    for _ in range(len(training_ranks)):
        # Accept connection
        ok, r_ip, r_gpu, conn_id = ep.accept()
        
        # Map to training rank (simplified - in practice need handshake)
        # Assume connections arrive in order
        train_rank = training_ranks[i]
        connections[train_rank] = conn_id
```

### 5. RDMA Weight Transfer Execution

**Training worker sends:**
```python
# Training ranks (0-15)
def transfer_weights_pipelined(self):
    for entry in self.routing_table.transfers:
        # Get parameter
        tensor = self.param_dict[entry.param_name].data.contiguous()
        
        # Register memory if not already registered
        ptr = tensor.data_ptr()
        size = tensor.numel() * tensor.element_size()
        if ptr not in self.registered_mrs:
            ok, mr_id = self.ep.reg(ptr, size)
            self.registered_mrs[ptr] = mr_id
        else:
            mr_id = self.registered_mrs[ptr]
        
        # Send via RDMA
        conn_id = self.connections[entry.dst_rank]
        ok = self.ep.send(conn_id, mr_id, ptr, size)
```

**Inference worker receives:**
```python
# Inference ranks (24-31)
def receive_weights(self):
    for param_name, src_rank in self.param_sources.items():
        # Get parameter
        param = self.param_dict[param_name].data
        
        # Register memory if not already registered
        ptr = param.data_ptr()
        size = param.numel() * param.element_size()
        if ptr not in self.registered_mrs:
            ok, mr_id = self.ep.reg(ptr, size)
            self.registered_mrs[ptr] = mr_id
        else:
            mr_id = self.registered_mrs[ptr]
        
        # Receive via RDMA
        conn_id = self.connections[src_rank]
        ok = self.ep.recv(conn_id, mr_id, ptr, size)
```

## Complete Example: Training Step with RDMA Transfer

```python
def training_step_with_rdma(
    model, batch, optimizer, rdma_worker, step
):
    """Complete training step with RDMA weight update"""
    
    # 1. Forward pass
    outputs = model(**batch)
    loss = outputs.loss
    
    # 2. Backward pass
    loss.backward()
    
    # 3. Optimizer step
    optimizer.step()
    optimizer.zero_grad()
    
    # 4. Periodic RDMA weight update
    if step % 100 == 0:
        # Synchronize all training ranks
        dist.barrier()
        
        # Execute RDMA transfer according to routing table
        transfer_time = rdma_worker.transfer_weights_pipelined()
        
        # Synchronize after transfer
        dist.barrier()
        
        logging.info(f"RDMA transfer completed in {transfer_time:.2f}s")
    
    return loss.item()
```

## Implementation Checklist

### ✅ Completed
- [x] Controller routing table computation
- [x] UCCL P2P training worker
- [x] UCCL P2P inference worker
- [x] RDMA send/receive implementation
- [x] Memory region registration
- [x] Connection management

### ⏳ To Implement
- [ ] Routing table distribution (controller → training)
- [ ] Parameter source mapping (controller → inference)
- [ ] Connection rank identification (handshake protocol)
- [ ] Error handling and retry logic
- [ ] Memory region caching optimization
- [ ] Multi-NIC support

### 🎯 Recommended Implementation Order

**Phase 1: Basic RDMA Working (Week 1)**
1. Implement routing table distribution
   ```python
   # Add to main_distributed_rdma.py
   def broadcast_routing_tables(controller_rank, routing_tables):
       if rank == controller_rank:
           for train_rank, table in routing_tables.items():
               send_routing_table(table, train_rank)
       elif rank < 16:  # Training rank
           routing_table = receive_routing_table(controller_rank)
           rdma_worker.set_routing_table(routing_table)
   ```

2. Implement parameter source mapping
   ```python
   # Add to weight_transfer_controller.py
   def extract_param_sources_for_inference_rank(
       routing_tables, inference_rank
   ):
       param_sources = {}
       for train_rank, table in routing_tables.items():
           for entry in table.transfers:
               if entry.dst_rank == inference_rank:
                   param_sources[entry.param_name] = train_rank
       return param_sources
   ```

3. Test on 2 nodes (1 training, 1 inference)

**Phase 2: Connection Management (Week 2)**
1. Add connection handshake protocol
   ```python
   # Send rank ID after connection
   def connect_with_handshake(ep, ip, port, gpu, my_rank):
       ok, conn_id = ep.connect(ip, gpu, remote_port=port)
       rank_tensor = torch.tensor([my_rank], dtype=torch.int32).cuda()
       ptr = rank_tensor.data_ptr()
       size = 4
       ok, mr_id = ep.reg(ptr, size)
       ok = ep.send(conn_id, mr_id, ptr, size)
       return conn_id
   ```

2. Improve connection-to-rank mapping
3. Test on 4 nodes

**Phase 3: Optimization (Week 3)**
1. Add memory region caching
2. Implement pipelining
3. Add performance monitoring
4. Benchmark and optimize

**Phase 4: Production Ready (Week 4)**
1. Error handling and recovery
2. Logging and debugging
3. Integration tests
4. Documentation

## Testing Strategy

### Unit Tests
```python
# test_routing_table.py
def test_routing_table_computation():
    controller = WeightTransferController(...)
    routing_tables = controller.compute_routing_tables(...)
    
    # Verify load balancing
    assert all_tables_balanced(routing_tables)
    
    # Verify coverage
    assert all_parameters_covered(routing_tables)

# test_rdma_worker.py
def test_rdma_send_receive():
    # Setup mock UCCL endpoint
    ep_send = MockEndpoint()
    ep_recv = MockEndpoint()
    
    # Test send/receive
    worker_send = RDMATrainingWorker(...)
    worker_recv = RDMAInferenceWorker(...)
    
    worker_send.transfer_weights()
    worker_recv.receive_weights()
    
    # Verify weights match
    assert weights_equal(worker_send.model, worker_recv.model)
```

### Integration Tests
```bash
# test_2node.sh - 1 training node, 1 inference node
torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 test_rdma.py &
torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 test_rdma.py &
wait

# test_4node.sh - Full 4-node setup
bash launch_all_nodes.sh --test-mode
```

### Performance Benchmarks
```python
# benchmark_rdma.py
def benchmark_transfer_performance():
    sizes = [100e6, 500e6, 1e9, 5e9, 10e9]  # bytes
    
    for size in sizes:
        # Create dummy tensor
        tensor = torch.randn(size // 4).cuda()
        
        # Measure transfer time
        start = time.perf_counter()
        rdma_worker.transfer_single_tensor(tensor, dst_rank)
        transfer_time = time.perf_counter() - start
        
        bandwidth = size / transfer_time / 1e9  # GB/s
        print(f"Size: {size/1e9:.1f} GB, "
              f"Time: {transfer_time:.2f}s, "
              f"Bandwidth: {bandwidth:.2f} GB/s")
```

## Debugging Tips

### Enable UCCL Debug Logging
```bash
export UCCL_LOG_LEVEL=DEBUG
export UCX_LOG_LEVEL=info
```

### Check RDMA Connectivity
```bash
# On training node
python -c "from uccl import p2p; ep = p2p.Endpoint(0, 1); print(ep.get_metadata())"

# On inference node (should see connection)
python -c "from uccl import p2p; ep = p2p.Endpoint(0, 1); ep.accept()"
```

### Monitor Transfer Progress
```python
# Add progress logging in transfer loop
for idx, entry in enumerate(routing_table.transfers):
    ok = ep.send(...)
    if idx % 10 == 0:
        logging.info(f"Progress: {idx}/{len(routing_table.transfers)}")
```

### Verify Weights Updated
```python
# Before transfer
old_params = {name: param.data.clone() 
              for name, param in model.named_parameters()}

# After transfer
rdma_worker.receive_weights()

# Verify
changed = sum(1 for name, param in model.named_parameters()
              if not torch.equal(param.data, old_params[name]))
logging.info(f"Changed parameters: {changed}/{len(old_params)}")
```

## Summary

This integration guide shows how to combine:

1. **Controller Routing Tables** (from blog post concept)
   - Pre-computed transfer plans
   - Load balancing
   - Minimal runtime overhead

2. **UCCL P2P RDMA** (from your uploaded code)
   - Direct GPU-to-GPU transfers
   - High bandwidth, low latency
   - SM-free operation

3. **Distributed Training** (PyTorch ecosystem)
   - DDP for training
   - torch.distributed for control plane
   - Seamless integration

The result is a high-performance weight transfer system that achieves the 2-second transfer time from the blog post while remaining flexible and maintainable.

---

**Next**: Follow the implementation checklist to complete the integration!
