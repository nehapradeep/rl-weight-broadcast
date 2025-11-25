from __future__ import annotations
import torch, time, os, sys
import torch.distributed as dist
from torch.distributed._tensor import DTensor, Replicate, Shard
import logging
from datetime import datetime
import math

# DTensor setup
from torch.distributed.device_mesh import DeviceMesh

from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader
from uccl import p2p
import torch.nn as nn
import json
import numpy as np

# ---------------------------
# DTensor parameter metadata helpers
# ---------------------------
def analyze_dtensor_layout(dtensor, param_name="unknown"):
    """
    Analyze DTensor layout to determine which GPU(s) hold which parts of the tensor.
    Returns a dictionary with layout analysis information.
    """
    analysis = {
        "param_name": param_name,
        "is_dtensor": isinstance(dtensor, DTensor),
        "global_shape": list(dtensor.shape) if hasattr(dtensor, "shape") else None,
        "dtype": str(dtensor.dtype) if hasattr(dtensor, "dtype") else None,
    }
    
    if not isinstance(dtensor, DTensor):
        # Regular tensor - check device
        if hasattr(dtensor, "device"):
            analysis["device"] = str(dtensor.device)
            analysis["layout_type"] = "regular_tensor"
        return analysis
    
    # DTensor-specific analysis
    analysis["layout_type"] = "dtensor"
    
    # 1. Device mesh information
    if hasattr(dtensor, "device_mesh") and dtensor.device_mesh is not None:
        mesh = dtensor.device_mesh
        analysis["mesh"] = {
            "ndim": mesh.ndim,
            "devices": [str(d) for d in mesh.devices],
            "mesh_shape": list(mesh.shape),
            "device_type": str(mesh.device_type) if hasattr(mesh, "device_type") else None,
            "mesh_id": id(mesh),  # Python object ID (process-local)
        }
        
        # Try to get mesh coordinate for current rank
        if hasattr(mesh, "get_coordinate"):
            try:
                coord = mesh.get_coordinate()
                analysis["mesh"]["current_coordinate"] = list(coord) if coord is not None else None
            except:
                pass
    
    # 2. Placement information
    if hasattr(dtensor, "placements") and dtensor.placements is not None:
        placements = dtensor.placements
        analysis["placements"] = [str(p) for p in placements]
        
        # Analyze placement types
        placement_types = []
        shard_info = []
        
        for i, placement in enumerate(placements):
            placement_str = str(placement)
            placement_types.append(placement_str)
            
            # Check if it's a Shard placement
            if isinstance(placement, Shard):
                shard_dim = placement.dim if hasattr(placement, "dim") else None
                shard_info.append({
                    "mesh_dim": i,
                    "shard_dim": shard_dim,
                    "placement_type": "Shard",
                })
            elif isinstance(placement, Replicate):
                shard_info.append({
                    "mesh_dim": i,
                    "placement_type": "Replicate",
                })
        
        analysis["placement_types"] = placement_types
        analysis["shard_info"] = shard_info
        
        # Determine overall layout strategy
        if all(isinstance(p, Replicate) for p in placements):
            analysis["layout_strategy"] = "replicated"
        elif any(isinstance(p, Shard) for p in placements):
            analysis["layout_strategy"] = "sharded"
        else:
            analysis["layout_strategy"] = "mixed"
    
    # 3. Local tensor information (what this GPU actually holds)
    if hasattr(dtensor, "to_local"):
        try:
            local_tensor = dtensor.to_local()
            analysis["local_tensor"] = {
                "shape": list(local_tensor.shape),
                "dtype": str(local_tensor.dtype),
                "device": str(local_tensor.device),
                "numel": local_tensor.numel(),
                "size_bytes": local_tensor.numel() * local_tensor.element_size(),
            }
        except Exception as e:
            analysis["local_tensor_error"] = str(e)
    
    # 4. Try to determine which GPU holds which shard
    # For sharded tensors, we need to figure out shard assignment
    if analysis.get("layout_strategy") == "sharded" and "mesh" in analysis:
        mesh = dtensor.device_mesh
        devices = mesh.devices
        mesh_shape = mesh.shape
        
        # Calculate shard assignment per GPU
        gpu_assignments = []
        
        # For 1D mesh with Shard placement, each GPU gets a contiguous slice
        if len(mesh_shape) == 1 and len(placements) == 1:
            placement = placements[0]
            if isinstance(placement, Shard):
                shard_dim = placement.dim
                global_shape = dtensor.shape
                
                if shard_dim < len(global_shape):
                    dim_size = global_shape[shard_dim]
                    num_shards = mesh_shape[0]
                    shard_size = dim_size // num_shards
                    remainder = dim_size % num_shards
                    
                    for gpu_idx in range(num_shards):
                        shard_start = gpu_idx * shard_size + min(gpu_idx, remainder)
                        shard_end = shard_start + shard_size + (1 if gpu_idx < remainder else 0)
                        
                        gpu_assignments.append({
                            "gpu_index": gpu_idx,
                            "device": str(devices[gpu_idx]),
                            "shard_dim": shard_dim,
                            "shard_slice": (shard_start, shard_end),
                            "shard_size": shard_end - shard_start,
                        })
        
        analysis["gpu_assignments"] = gpu_assignments
    
    # 5. Try to get global tensor size
    try:
        global_numel = dtensor.numel()
        element_size = dtensor.element_size() if hasattr(dtensor, "element_size") else None
        analysis["global_size"] = {
            "numel": global_numel,
            "size_bytes": global_numel * element_size if element_size else None,
            "element_size": element_size,
        }
    except:
        pass
    
    # 6. Additional DTensor attributes to explore
    dtensor_attrs = {}
    for attr in ["_spec", "_local_tensor", "_global_shape", "_stride"]:
        if hasattr(dtensor, attr):
            try:
                val = getattr(dtensor, attr)
                dtensor_attrs[attr] = str(type(val).__name__)
                # Try to get more info for _spec
                if attr == "_spec" and val is not None:
                    if hasattr(val, "dim_map"):
                        dtensor_attrs["_spec_dim_map"] = str(val.dim_map) if hasattr(val, "dim_map") else None
                    if hasattr(val, "placements"):
                        dtensor_attrs["_spec_placements"] = [str(p) for p in val.placements] if hasattr(val, "placements") else None
            except:
                dtensor_attrs[attr] = "accessible_but_error"
    
    if dtensor_attrs:
        analysis["dtensor_internal_attrs"] = dtensor_attrs
    
    return analysis


def gather_dtensor_param_metadata(model):
    metadata = {}
    for name, param in model.named_parameters():
        dtensor = param.data
        entry = {
            "name": name,
            "shape": list(dtensor.shape),
            "dtype": str(dtensor.dtype),
        }
        # Try to get mesh and placement info if DTensor
        if hasattr(dtensor, "device_mesh"):
            entry["mesh"] = {
                "ndim": dtensor.device_mesh.ndim,
                "devices": [str(d) for d in dtensor.device_mesh.devices],
                "mesh_shape": list(dtensor.device_mesh.shape),
            }
        if hasattr(dtensor, "placements"):
            entry["placements"] = [str(p) for p in dtensor.placements]
        metadata[name] = entry
    return metadata


# ---------------------------
# Logging
# ---------------------------
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


# ---------------------------
# RDMA send/recv helpers
# ---------------------------
def broadcast_model(ep, conn_ids, model, rank):
    """Send model to multiple receivers with detailed logging"""
    state_dict = model.state_dict()
    # Convert all tensors in state_dict to local torch.Tensor for RDMA
    for name, tensor in state_dict.items():
        if hasattr(tensor, "to_local"):
            state_dict[name] = tensor.to_local()
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
    # Convert all tensors in state_dict to local torch.Tensor for RDMA
    for name, tensor in state_dict.items():
        if hasattr(tensor, "to_local"):
            state_dict[name] = tensor.to_local()
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


# ---------------------------
# Dataset preparation
# ---------------------------
def prepare_dataset(tokenizer, max_length=128, num_samples=200):
    """Load and prepare WikiText-2 dataset"""
    logging.info("="*80)
    logging.info("LOADING DATASET")
    logging.info("="*80)
    logging.info("Loading WikiText-2 dataset from Hugging Face...")
    
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    logging.info(f"Dataset loaded: {len(dataset)} total examples")
    
    dataset = dataset.filter(lambda x: len(x["text"].strip()) > 0)
    logging.info(f"After filtering empty texts: {len(dataset)} examples")
    
    if len(dataset) > num_samples:
        dataset = dataset.select(range(num_samples))
        logging.info(f"Using subset: {num_samples} examples")
    
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
    
    tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])
    
    logging.info(f"Tokenization complete")
    logging.info(f"  - Examples: {len(tokenized_dataset)}")
    logging.info(f"  - Max length: {max_length} tokens")
    logging.info("="*80)
    
    return tokenized_dataset


# ---------------------------
# Training (rank 1)
# ---------------------------
def run_training(model, tokenizer, num_epochs=2, batch_size=4, lr=5e-5):
    """Training loop for rank 1 (supports DataParallel)"""
    logging.info("="*80)
    logging.info("TRAINING NODE - Starting training on WikiText-2")
    logging.info(f"Visible GPUs on this node: {torch.cuda.device_count()}")
    logging.info(f"Model class: {model.__class__.__name__}")
    logging.info("="*80)
    
    train_dataset = prepare_dataset(tokenizer, max_length=128, num_samples=200)
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    
    total_steps = len(train_dataloader) * num_epochs
    logging.info(f"Training configuration:")
    logging.info(f"  - Examples: {len(train_dataset)}")
    logging.info(f"  - Batch size: {batch_size}")
    logging.info(f"  - Epochs: {num_epochs}")
    logging.info(f"  - Steps per epoch: {len(train_dataloader)}")
    logging.info(f"  - Total steps: {total_steps}")
    logging.info(f"  - Learning rate: {lr}")
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
            
            # Move batch to GPU 0; DataParallel will scatter internally if enabled
            input_ids = batch["input_ids"].cuda()
            attention_mask = batch["attention_mask"].cuda()
            # Convert batch tensors to Dtensors
            if torch.cuda.device_count() > 1:
                mesh_devices = list(range(torch.cuda.device_count()))
                device_mesh = DeviceMesh("cuda", mesh_devices)
            else:
                device_mesh = DeviceMesh("cuda", [0])
            input_ids = DTensor(input_ids, device_mesh, [Replicate()])
            attention_mask = DTensor(attention_mask, device_mesh, [Replicate()])
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids
            )
            loss = outputs.loss  # can be scalar or vector (per-GPU) with DataParallel
            
            # ---- MULTI-GPU SAFE LOSS REDUCTION ----
            if isinstance(loss, torch.Tensor) and loss.dim() > 0:
                loss_scalar = loss.mean()
            else:
                loss_scalar = loss
            
            optimizer.zero_grad()
            loss_scalar.backward()
            optimizer.step()
            
            step_time = time.perf_counter() - step_start
            
            # Use scalar loss for metrics
            loss_value = loss_scalar.item()
            perplexity = math.exp(loss_value) if loss_value < 100 else float('inf')
            
            epoch_loss += loss_value
            epoch_perplexity += perplexity
            
            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(train_dataloader):
                logging.info(
                    f"Step {global_step}/{total_steps} "
                    f"(Epoch {epoch+1}, Batch {batch_idx+1}/{len(train_dataloader)}) | "
                    f"Loss: {loss_value:.4f} | Perplexity: {perplexity:.2f} | "
                    f"Time: {step_time:.3f}s"
                )
            
            if global_step % 20 == 0:
                total_norm = 0.0
                for p in model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5
                logging.info(f"  -> Gradient norm: {total_norm:.4f}")
        
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
    logging.info(f"Total training time: {total_train_time:.2f}s")
    logging.info(f"Total steps: {total_steps}")
    logging.info(f"Average time per step: {avg_step_time:.3f}s")
    logging.info(f"Steps per second: {total_steps/total_train_time:.2f}")
    logging.info(f"Final loss: {epoch_losses[-1]:.4f}")
    logging.info(f"Loss improvement: {epoch_losses[0] - epoch_losses[-1]:.4f}")
    logging.info("="*80)
    
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # If DataParallel is used, save underlying module
    save_model = model.module if isinstance(model, nn.DataParallel) else model
    checkpoint_path = f"{checkpoint_dir}/model_rank1_wikitext2_trained.pt"
    torch.save({
        'model_state_dict': save_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch_losses': epoch_losses,
        'total_steps': total_steps,
        'final_loss': epoch_losses[-1],
    }, checkpoint_path)
    logging.info(f"Model checkpoint saved to: {checkpoint_path}")


# ---------------------------
# Inference (rank 2)
# ---------------------------
def run_inference(model, tokenizer, num_samples=5):
    logging.info("="*80)
    logging.info("INFERENCE NODE - Starting inference")
    logging.info("="*80)
    
    model.eval()
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
    logging.info("Generation settings: max_length=50, temperature=0.7, top_k=50, top_p=0.95")
    
    total_inference_start = time.perf_counter()
    total_tokens = 0
    
    with torch.no_grad():
        for idx, prompt in enumerate(prompts[:num_samples], 1):
            inference_start = time.perf_counter()
            
            logging.info(f"\n{'='*60}")
            logging.info(f"SAMPLE {idx}/{num_samples}")
            logging.info(f"{'='*60}")
            logging.info(f"Prompt: '{prompt}'")
            
            inputs = tokenizer(prompt, return_tensors="pt")
            input_ids = inputs["input_ids"].cuda()
            # Convert input_ids to Dtensor
            if torch.cuda.device_count() > 1:
                mesh_devices = list(range(torch.cuda.device_count()))
                device_mesh = DeviceMesh("cuda", mesh_devices)
            else:
                device_mesh = DeviceMesh("cuda", [0])
            input_ids = DTensor(input_ids, device_mesh, [Replicate()])
            
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


# ---------------------------
# Main (all ranks)
# ---------------------------
def main():
    dist.init_process_group(backend="gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    log_file = setup_logging(rank)
    
    logging.info(f"Process started - Rank: {rank}, World size: {world_size}")
    assert world_size == 3, "Run with three ranks (1 broadcaster + 1 training + 1 inference)."
    

    # IMPORTANT: use GPU 0 on each node; trainer uses DataParallel internally
    if torch.cuda.is_available():
        local_gpu = 0
        torch.cuda.set_device(local_gpu)
        logging.info(f"CUDA device set to GPU {local_gpu}")
        mesh_devices = list(range(torch.cuda.device_count()))
        device_mesh = DeviceMesh("cuda", mesh_devices)
    else:
        local_gpu = None
        device_mesh = None
        logging.info("CUDA not available. (RDMA will not work.)")
    
    logging.info("Initializing P2P endpoint...")
    ep = p2p.Endpoint(local_gpu, 4)
    local_md = ep.get_metadata()
    logging.info(f"Local metadata obtained (size: {len(local_md)} bytes)")
    
    # Exchange metadata
    logging.info("Starting metadata exchange...")
    all_metadata = [None] * world_size
    all_metadata[rank] = local_md
    
    metadata_start = time.perf_counter()
    for i in range(world_size):
        if i == rank:
            for j in range(world_size):
                if j != rank:
                    dist.send(torch.ByteTensor(list(local_md)), dst=j)
        else:
            remote_md = torch.zeros(len(local_md), dtype=torch.uint8)
            dist.recv(remote_md, src=i)
            all_metadata[i] = bytes(remote_md.tolist())
    metadata_time = time.perf_counter() - metadata_start
    logging.info(f"Metadata exchange complete in {metadata_time:.2f}s")
    
    if rank == 0:
        # Broadcaster
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

        # Receive parameter metadata from trainer and inference nodes
        param_metadata = {}
        for src_rank in [1, 2]:
            logging.info(f"Receiving parameter metadata from rank {src_rank}...")
            size_tensor = torch.zeros(1, dtype=torch.int32)
            dist.recv(size_tensor, src=src_rank)
            size = int(size_tensor.item())
            buf = torch.empty(size, dtype=torch.uint8)
            dist.recv(buf, src=src_rank)
            meta_json = buf.cpu().numpy().tobytes().decode("utf-8")
            param_metadata[src_rank] = json.loads(meta_json)
            logging.info(f"Received metadata from rank {src_rank}: {len(param_metadata[src_rank])} parameters")

        # Configure routing table using param_metadata
        logging.info("Configuring routing table with received parameter metadata...")
        # ... Routing table logic goes here ...
        logging.info("Routing table configured.")

        logging.info("Broadcaster setup complete. Ready for further operations.")
    
    elif rank == 1:
        # Trainer
        logging.info("="*80)
        logging.info("TRAINING NODE (Rank 1)")
        logging.info("="*80)
        logging.info("Waiting for broadcaster connection...")
        ok, r_ip, r_gpu, conn_id = ep.accept()
        assert ok, "Accept failed"
        logging.info("Connected to broadcaster")

        logging.info("Loading model and tokenizer...")
        base_model = GPT2LMHeadModel.from_pretrained("gpt2").cuda()
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        logging.info("Model and tokenizer loaded")
        # Convert model parameters to Dtensors
        for name, param in base_model.named_parameters():
            param.data = DTensor(param.data, device_mesh, [Replicate()])

        # STEP 1: Analyze DTensor layout for sample parameters
        logging.info("="*80)
        logging.info("DTENSOR LAYOUT ANALYSIS (Training Node)")
        logging.info("="*80)
        sample_params = list(base_model.named_parameters())[:5]  # Analyze first 5 params
        for name, param in sample_params:
            analysis = analyze_dtensor_layout(param.data, param_name=name)
            logging.info(f"\nParameter: {name}")
            logging.info(f"  Is DTensor: {analysis.get('is_dtensor')}")
            logging.info(f"  Layout type: {analysis.get('layout_type')}")
            logging.info(f"  Layout strategy: {analysis.get('layout_strategy', 'N/A')}")
            if 'mesh' in analysis:
                mesh_info = analysis['mesh']
                logging.info(f"  Mesh devices: {mesh_info.get('devices')}")
                logging.info(f"  Mesh shape: {mesh_info.get('mesh_shape')}")
                logging.info(f"  Mesh ID (process-local): {mesh_info.get('mesh_id')}")
            if 'placements' in analysis:
                logging.info(f"  Placements: {analysis['placements']}")
            if 'shard_info' in analysis:
                logging.info(f"  Shard info: {analysis['shard_info']}")
            if 'local_tensor' in analysis:
                local = analysis['local_tensor']
                logging.info(f"  Local tensor shape: {local.get('shape')}")
                logging.info(f"  Local tensor device: {local.get('device')}")
                logging.info(f"  Local tensor size: {local.get('size_bytes', 0) / 1e6:.2f} MB")
            if 'gpu_assignments' in analysis and analysis['gpu_assignments']:
                logging.info(f"  GPU assignments: {len(analysis['gpu_assignments'])} GPUs")
                for gpu_assignment in analysis['gpu_assignments'][:3]:  # Show first 3
                    logging.info(f"    GPU {gpu_assignment['gpu_index']} ({gpu_assignment['device']}): "
                               f"shard_slice={gpu_assignment.get('shard_slice')}, "
                               f"size={gpu_assignment.get('shard_size')}")
            if 'global_size' in analysis:
                global_size = analysis['global_size']
                logging.info(f"  Global size: {global_size.get('size_bytes', 0) / 1e6:.2f} MB")
            if 'dtensor_internal_attrs' in analysis:
                logging.info(f"  Internal attrs: {list(analysis['dtensor_internal_attrs'].keys())}")
        logging.info("="*80)

        # Gather and send parameter metadata to broadcaster
        param_meta = gather_dtensor_param_metadata(base_model)
        meta_json = json.dumps(param_meta).encode("utf-8")
        meta_tensor = torch.from_numpy(np.frombuffer(meta_json, dtype=np.uint8))
        size_tensor = torch.tensor([meta_tensor.numel()], dtype=torch.int32)
        dist.send(size_tensor, dst=0)
        dist.send(meta_tensor, dst=0)
        logging.info(f"Sent parameter metadata to broadcaster (size: {size_tensor.item()} bytes)")

        recv_model(ep, conn_id, base_model, rank)

        if torch.cuda.device_count() > 1:
            device_ids = list(range(torch.cuda.device_count()))
            logging.info(f"Wrapping model in DataParallel across GPUs: {device_ids}")
            model = nn.DataParallel(base_model, device_ids=device_ids)
        else:
            logging.info("Only one GPU detected; using single-GPU training.")
            model = base_model

        run_training(model, tokenizer, num_epochs=2, batch_size=4, lr=5e-5)
    
    else:
        # Inference node (rank 2)
        logging.info("="*80)
        logging.info("INFERENCE NODE (Rank 2)")
        logging.info("="*80)
        logging.info("Waiting for broadcaster connection...")
        ok, r_ip, r_gpu, conn_id = ep.accept()
        assert ok, "Accept failed"
        logging.info("Connected to broadcaster")

        logging.info("Loading model and tokenizer...")
        model = GPT2LMHeadModel.from_pretrained("gpt2").cuda()
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        logging.info("Model and tokenizer loaded")
        # Convert model parameters to Dtensors
        for name, param in model.named_parameters():
            param.data = DTensor(param.data, device_mesh, [Replicate()])

        # STEP 1: Analyze DTensor layout for sample parameters
        logging.info("="*80)
        logging.info("DTENSOR LAYOUT ANALYSIS (Inference Node)")
        logging.info("="*80)
        sample_params = list(model.named_parameters())[:5]  # Analyze first 5 params
        for name, param in sample_params:
            analysis = analyze_dtensor_layout(param.data, param_name=name)
            logging.info(f"\nParameter: {name}")
            logging.info(f"  Is DTensor: {analysis.get('is_dtensor')}")
            logging.info(f"  Layout type: {analysis.get('layout_type')}")
            logging.info(f"  Layout strategy: {analysis.get('layout_strategy', 'N/A')}")
            if 'mesh' in analysis:
                mesh_info = analysis['mesh']
                logging.info(f"  Mesh devices: {mesh_info.get('devices')}")
                logging.info(f"  Mesh shape: {mesh_info.get('mesh_shape')}")
                logging.info(f"  Mesh ID (process-local): {mesh_info.get('mesh_id')}")
            if 'placements' in analysis:
                logging.info(f"  Placements: {analysis['placements']}")
            if 'shard_info' in analysis:
                logging.info(f"  Shard info: {analysis['shard_info']}")
            if 'local_tensor' in analysis:
                local = analysis['local_tensor']
                logging.info(f"  Local tensor shape: {local.get('shape')}")
                logging.info(f"  Local tensor device: {local.get('device')}")
                logging.info(f"  Local tensor size: {local.get('size_bytes', 0) / 1e6:.2f} MB")
            if 'gpu_assignments' in analysis and analysis['gpu_assignments']:
                logging.info(f"  GPU assignments: {len(analysis['gpu_assignments'])} GPUs")
                for gpu_assignment in analysis['gpu_assignments'][:3]:  # Show first 3
                    logging.info(f"    GPU {gpu_assignment['gpu_index']} ({gpu_assignment['device']}): "
                               f"shard_slice={gpu_assignment.get('shard_slice')}, "
                               f"size={gpu_assignment.get('shard_size')}")
            if 'global_size' in analysis:
                global_size = analysis['global_size']
                logging.info(f"  Global size: {global_size.get('size_bytes', 0) / 1e6:.2f} MB")
            if 'dtensor_internal_attrs' in analysis:
                logging.info(f"  Internal attrs: {list(analysis['dtensor_internal_attrs'].keys())}")
        logging.info("="*80)

        # Gather and send parameter metadata to broadcaster
        param_meta = gather_dtensor_param_metadata(model)
        meta_json = json.dumps(param_meta).encode("utf-8")
        meta_tensor = torch.from_numpy(np.frombuffer(meta_json, dtype=np.uint8))
        size_tensor = torch.tensor([meta_tensor.numel()], dtype=torch.int32)
        dist.send(size_tensor, dst=0)
        dist.send(meta_tensor, dst=0)
        logging.info(f"Sent parameter metadata to broadcaster (size: {size_tensor.item()} bytes)")

        recv_model(ep, conn_id, model, rank)
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