# Troubleshooting Guide

Common issues and solutions for the Weight Transfer System.

## Table of Contents

1. [Distributed Setup Issues](#distributed-setup-issues)
2. [NCCL Communication Errors](#nccl-communication-errors)
3. [Memory Issues](#memory-issues)
4. [Performance Problems](#performance-problems)
5. [Weight Transfer Failures](#weight-transfer-failures)
6. [AMD GPU Specific Issues](#amd-gpu-specific-issues)

---

## Distributed Setup Issues

### Issue: "Connection refused" or "Unable to connect to master"

**Symptoms:**
```
RuntimeError: Connection refused
```

**Solutions:**

1. **Check master node IP:**
   ```bash
   # On master node
   hostname -I
   # Use the correct IP, not localhost or 127.0.0.1
   ```

2. **Verify firewall settings:**
   ```bash
   # Allow port 29500
   sudo ufw allow 29500
   # Or disable firewall temporarily for testing
   sudo ufw disable
   ```

3. **Check network connectivity:**
   ```bash
   # From worker node to master
   ping MASTER_IP
   telnet MASTER_IP 29500
   ```

4. **Verify all nodes use same master_addr and master_port:**
   ```bash
   echo $MASTER_ADDR
   echo $MASTER_PORT
   ```

### Issue: "Timeout initializing process group"

**Symptoms:**
```
RuntimeError: Timed out initializing process group
```

**Solutions:**

1. **Increase timeout:**
   ```python
   dist.init_process_group(
       backend="nccl",
       timeout=timedelta(minutes=30)  # Increase from default
   )
   ```

2. **Check that all nodes started:**
   ```bash
   # On each node, verify process is running
   ps aux | grep torchrun
   ```

3. **Verify NCCL_SOCKET_IFNAME:**
   ```bash
   export NCCL_SOCKET_IFNAME=eth0  # or your network interface
   # Check available interfaces:
   ifconfig
   ```

---

## NCCL Communication Errors

### Issue: "NCCL error: unhandled system error"

**Symptoms:**
```
NCCL error: unhandled system error, NCCL version X.Y.Z
```

**Solutions:**

1. **Enable NCCL debugging:**
   ```bash
   export NCCL_DEBUG=INFO
   export NCCL_DEBUG_SUBSYS=ALL
   ```

2. **Check GPU visibility:**
   ```bash
   export ROCR_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
   # Verify GPUs are visible:
   rocm-smi
   ```

3. **Disable InfiniBand if not available:**
   ```bash
   export NCCL_IB_DISABLE=1
   ```

4. **Try different NCCL transport:**
   ```bash
   export NCCL_NET=Socket  # Force socket communication
   ```

### Issue: "NCCL operation timed out"

**Symptoms:**
```
RuntimeError: NCCL operation XXX timed out
```

**Solutions:**

1. **Increase NCCL timeout:**
   ```bash
   export NCCL_TIMEOUT=1800  # 30 minutes in seconds
   export NCCL_BLOCKING_WAIT=1
   ```

2. **Check network bandwidth:**
   ```bash
   # Install iperf3
   sudo apt install iperf3
   
   # On master node:
   iperf3 -s
   
   # On worker node:
   iperf3 -c MASTER_IP -t 30
   ```

3. **Verify no rank is stuck:**
   ```python
   # Add logging before barriers
   print(f"[Rank {rank}] Reaching barrier...")
   dist.barrier()
   print(f"[Rank {rank}] Passed barrier")
   ```

---

## Memory Issues

### Issue: "CUDA out of memory"

**Symptoms:**
```
RuntimeError: CUDA out of memory. Tried to allocate X MiB
```

**Solutions:**

1. **Reduce batch size:**
   ```python
   config.data.batch_size = 2  # Reduce from 4
   ```

2. **Reduce max_tmp_bytes for weight transfer:**
   ```python
   max_tmp_bytes = 1 << 30  # 1 GB instead of 2 GB
   ```

3. **Enable gradient checkpointing:**
   ```python
   model.gradient_checkpointing_enable()
   ```

4. **Clear cache periodically:**
   ```python
   if step % 100 == 0:
       torch.cuda.empty_cache()
   ```

5. **Monitor memory usage:**
   ```python
   import torch
   print(f"Memory allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
   print(f"Memory reserved: {torch.cuda.memory_reserved()/1e9:.2f} GB")
   ```

### Issue: Memory leak during weight transfers

**Symptoms:**
- Memory usage increases over time
- Eventually OOM after many transfers

**Solutions:**

1. **Explicitly delete tensors:**
   ```python
   def transfer_weights(self):
       # ... transfer logic ...
       
       # Clean up
       del tensor
       torch.cuda.empty_cache()
   ```

2. **Use context manager:**
   ```python
   with torch.no_grad():
       # Transfer weights
       pass
   ```

3. **Check for circular references:**
   ```python
   import gc
   gc.collect()
   torch.cuda.empty_cache()
   ```

---

## Performance Problems

### Issue: Very slow weight transfers (>10 seconds)

**Symptoms:**
- Weight transfer takes much longer than expected
- Lower bandwidth than network capacity

**Solutions:**

1. **Profile the transfer:**
   ```python
   import time
   
   start = time.time()
   # GPU ops
   gpu_time = time.time() - start
   
   start = time.time()
   # Network transfer
   net_time = time.time() - start
   
   print(f"GPU: {gpu_time:.2f}s, Network: {net_time:.2f}s")
   ```

2. **Enable pipelining:**
   ```python
   worker = OptimizedTrainingWorker(...)
   worker.transfer_weights_pipelined()  # Instead of transfer_weights()
   ```

3. **Increase pipeline batch size:**
   ```python
   config.transfer.pipeline_batch_size = 8  # From 4
   ```

4. **Check for blocking operations:**
   ```python
   # Make sure using non-blocking transfers
   req = dist.isend(tensor, dst=rank)  # Non-blocking
   # Not: dist.send(tensor, dst=rank)  # Blocking
   ```

5. **Verify network isn't bottleneck:**
   ```bash
   # Check network usage
   nload  # or
   iftop
   ```

### Issue: Training is slower with weight transfers

**Symptoms:**
- Training speed drops when transfers are enabled
- GPUs idle during transfers

**Solutions:**

1. **Increase transfer frequency:**
   ```python
   # Transfer less frequently
   weight_update_frequency = 500  # From 100
   ```

2. **Overlap training with transfer:**
   - Use async transfers
   - Continue training on next batch while transferring

3. **Use separate CUDA streams:**
   ```python
   transfer_stream = torch.cuda.Stream()
   with torch.cuda.stream(transfer_stream):
       # Transfer operations
       pass
   ```

---

## Weight Transfer Failures

### Issue: "Parameter not found" errors

**Symptoms:**
```
Warning: Parameter transformer.h.0.attn.qkv_proj.weight not found
```

**Solutions:**

1. **Check parameter name matching:**
   ```python
   # Print all parameter names
   for name, param in model.named_parameters():
       print(name)
   ```

2. **Update ModelWeightMatcher:**
   ```python
   # Ensure matcher handles all parameter patterns
   def map_trainer_weight_name(self, ...):
       # Add missing mappings
       pass
   ```

3. **Verify model architectures match:**
   ```python
   # Training and inference should use same model
   print("Training params:", len(list(train_model.parameters())))
   print("Inference params:", len(list(infer_model.parameters())))
   ```

### Issue: Weights not actually updating

**Symptoms:**
- Transfer completes but inference still uses old weights
- Verification shows weights unchanged

**Solutions:**

1. **Check receive logic:**
   ```python
   # Ensure inference worker actually receives
   worker.receive_weights()
   
   # Verify
   worker.verify_weights_updated(prev_params)
   ```

2. **Check tag matching:**
   ```python
   # Send and receive must use same tag
   tag = hash(param_name) % 10000
   dist.isend(tensor, dst=rank, tag=tag)  # Sender
   dist.irecv(tensor, src=rank, tag=tag)  # Receiver
   ```

3. **Add explicit barriers:**
   ```python
   # Before transfer
   dist.barrier()
   
   # Send/receive
   
   # After transfer
   dist.barrier()
   ```

---

## AMD GPU Specific Issues

### Issue: ROCm not detected

**Symptoms:**
```
torch.cuda.is_available() returns False
```

**Solutions:**

1. **Verify ROCm installation:**
   ```bash
   rocm-smi
   rocminfo | grep "Name:"
   ```

2. **Install correct PyTorch version:**
   ```bash
   pip install torch torchvision torchaudio \
       --index-url https://download.pytorch.org/whl/rocm5.7
   ```

3. **Set environment variables:**
   ```bash
   export HSA_OVERRIDE_GFX_VERSION=10.3.0  # For your GPU
   export ROCR_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
   ```

### Issue: "RuntimeError: HIP error: invalid device function"

**Symptoms:**
```
RuntimeError: HIP error: invalid device function
```

**Solutions:**

1. **Check GPU architecture compatibility:**
   ```bash
   rocminfo | grep "gfx"
   ```

2. **Set correct GFX version:**
   ```bash
   export HSA_OVERRIDE_GFX_VERSION=10.3.0  # Adjust for your GPU
   ```

3. **Rebuild PyTorch if needed:**
   ```bash
   pip install torch --upgrade --force-reinstall \
       --index-url https://download.pytorch.org/whl/rocm5.7
   ```

### Issue: Poor performance on AMD GPUs

**Solutions:**

1. **Enable TF32:**
   ```python
   torch.backends.cuda.matmul.allow_tf32 = True
   ```

2. **Use optimized kernels:**
   ```bash
   export PYTORCH_ROCM_ARCH="gfx90a"  # Or your arch
   ```

3. **Check GPU clocks:**
   ```bash
   rocm-smi --showclocks
   # Set to high performance mode
   rocm-smi --setperflevel high
   ```

---

## General Debugging Tips

### Enable verbose logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

```bash
export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=DETAIL
```

### Add rank tracking

```python
def log(msg):
    rank = dist.get_rank()
    print(f"[Rank {rank}] {msg}", flush=True)
```

### Use smaller model for debugging

```python
config = get_single_node_config()  # From config.py
```

### Test incrementally

1. First: Single node, single GPU
2. Then: Single node, multiple GPUs
3. Then: Multiple nodes, few GPUs
4. Finally: Full cluster

### Monitor system resources

```bash
# GPU usage
watch -n 1 rocm-smi

# Memory
watch -n 1 free -h

# Network
nload
```

---

## Getting Help

If issues persist:

1. **Check logs carefully:**
   - Look for the first error, not just the last one
   - Check logs on ALL ranks, not just rank 0

2. **Create minimal reproducer:**
   - Simplify to smallest failing case
   - Share code and error message

3. **Gather system info:**
   ```bash
   python -c "import torch; print(torch.__version__)"
   rocm-smi
   ifconfig
   ```

4. **Check GitHub issues:**
   - PyTorch: https://github.com/pytorch/pytorch/issues
   - ROCm: https://github.com/RadeonOpenCompute/ROCm/issues

---

## Quick Diagnostic Script

Save as `diagnose.py`:

```python
import torch
import torch.distributed as dist
import os

print("=== System Diagnostics ===")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"Number of GPUs: {torch.cuda.device_count()}")

if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"    Memory: {torch.cuda.get_device_properties(i).total_memory/1e9:.1f} GB")

print(f"\nEnvironment variables:")
for var in ['MASTER_ADDR', 'MASTER_PORT', 'RANK', 'WORLD_SIZE', 
            'NCCL_SOCKET_IFNAME', 'ROCR_VISIBLE_DEVICES']:
    print(f"  {var}: {os.environ.get(var, 'NOT SET')}")

print(f"\nDistributed backend available: {dist.is_available()}")
print(f"NCCL available: {dist.is_nccl_available()}")
```

Run with: `python diagnose.py`
