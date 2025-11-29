from __future__ import annotations
import torch, time, os, sys
import torch.distributed as dist
from torch.distributed._tensor import DTensor, Replicate, Shard
import logging
from datetime import datetime
import math

# DTensor setup
from torch.distributed.device_mesh import DeviceMesh

# FSDP imports
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    StateDictType,
    FullStateDictConfig,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers.models.gpt2.modeling_gpt2 import GPT2Block

from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from uccl import p2p
import torch.nn as nn
import json
import numpy as np

# DTensor utilities
from utils.dtensor_utils import (
    compute_gpu_shard_assignments,
    get_current_gpu_shard_info,
    mesh_coordinate_to_gpu_index,
    gpu_index_to_mesh_coordinate
)

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
        global_shape = dtensor.shape
        
        try:
            # Use comprehensive shard assignment computation
            gpu_assignments = compute_gpu_shard_assignments(
                dtensor, mesh, placements, global_shape
            )
            
            # Also get current GPU's shard info if available
            current_gpu_info = get_current_gpu_shard_info(dtensor, mesh)
            if current_gpu_info:
                analysis["current_gpu_shard"] = current_gpu_info
            
            analysis["gpu_assignments"] = gpu_assignments
        except Exception as e:
            # Fallback to basic calculation if comprehensive one fails
            logging.warning(f"Comprehensive shard assignment failed for {param_name}: {e}. Using fallback.")
            analysis["gpu_assignments_error"] = str(e)
            
            # Simple fallback for 1D mesh
            mesh_shape = list(mesh.shape)
            if len(mesh_shape) == 1 and len(placements) == 1:
                placement = placements[0]
                if isinstance(placement, Shard):
                    shard_dim = placement.dim
                    if shard_dim < len(global_shape):
                        dim_size = global_shape[shard_dim]
                        num_shards = mesh_shape[0]
                        devices = mesh.devices
                        
                        gpu_assignments = []
                        for gpu_idx in range(num_shards):
                            shard_start = (gpu_idx * dim_size) // num_shards
                            shard_end = ((gpu_idx + 1) * dim_size) // num_shards
                            
                            gpu_assignments.append({
                                "gpu_index": gpu_idx,
                                "device": str(devices[gpu_idx]) if gpu_idx < len(devices) else f"unknown_{gpu_idx}",
                                "shard_slices": {shard_dim: (shard_start, shard_end)},
                                "shard_shape": list(global_shape),
                            })
                            gpu_assignments[-1]["shard_shape"][shard_dim] = shard_end - shard_start
                        
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
# Training (FSDP)
# ---------------------------
def run_training(model, tokenizer, num_epochs=2, batch_size=4, lr=5e-5, train_group=None, rank=None):
    """Training loop with FSDP support"""
    is_fsdp = isinstance(model, FSDP)
    logging.info("="*80)
    logging.info("TRAINING - Starting training on WikiText-2")
    logging.info(f"Model type: {'FSDP' if is_fsdp else 'Standard'}")
    logging.info(f"Visible GPUs on this node: {torch.cuda.device_count()}")
    logging.info(f"Model class: {model.__class__.__name__}")
    
    train_dataset = prepare_dataset(tokenizer, max_length=128, num_samples=200)
    
    # Use DistributedSampler for proper data distribution across training GPUs
    if train_group is not None:
        train_group_size = dist.get_world_size(train_group)
        train_group_rank = dist.get_rank(train_group)
        logging.info(f"Training process group: rank {train_group_rank}/{train_group_size}")
        sampler = DistributedSampler(
            train_dataset,
            num_replicas=train_group_size,
            rank=train_group_rank,
            shuffle=True
        )
        logging.info(f"Using DistributedSampler: rank {train_group_rank}/{train_group_size} in training group")
    else:
        sampler = None
        logging.info("No training group provided, using regular DataLoader (not recommended for FSDP)")
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=(sampler is None),  # Only shuffle if no sampler
        num_workers=0  # Set to 0 to avoid multiprocessing issues with RDMA
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
        
        # Set epoch for DistributedSampler to ensure proper shuffling
        if sampler is not None:
            sampler.set_epoch(epoch)
        
        logging.info(f"\n{'='*80}")
        logging.info(f"EPOCH {epoch + 1}/{num_epochs}")
        logging.info(f"{'='*80}")
        
        for batch_idx, batch in enumerate(train_dataloader):
            step_start = time.perf_counter()
            global_step += 1
            
            # Move batch to GPU 0; DataParallel will scatter internally if enabled
            input_ids = batch["input_ids"].cuda()
            attention_mask = batch["attention_mask"].cuda()
            # Convert batch tensors to Dtensors (using single-GPU mesh since DeviceMesh must match process group)
            # DataParallel handles multi-GPU distribution
            device_mesh = DeviceMesh("cuda", [0])
            input_ids = DTensor(input_ids, device_mesh, [Replicate()])
            attention_mask = DTensor(attention_mask, device_mesh, [Replicate()])
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids
            )
            loss = outputs.loss
            
            # FSDP handles loss reduction automatically, but ensure it's a scalar
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
    
    # Save checkpoint - use FSDP state dict context manager if FSDP is used
    if isinstance(model, FSDP):
        # FSDP requires special handling to gather sharded parameters
        # Get rank in training group for saving on rank 0 of training group
        if train_group is not None:
            train_group_rank = dist.get_rank(train_group)
        else:
            train_group_rank = rank
        
        # Policy: gather full state dict to CPU on rank 0 of training group
        save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        
        # Use the context manager to gather the full state dict
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
            cpu_state = model.state_dict()
        
        # Save on rank 0 of training group
        if train_group_rank == 0:
            checkpoint_path = f"{checkpoint_dir}/model_fsdp_wikitext2_trained.pt"
            state = {
                'model_state_dict': cpu_state,
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch_losses': epoch_losses,
                'total_steps': total_steps,
                'final_loss': epoch_losses[-1],
            }
            torch.save(state, checkpoint_path)
            logging.info(f"Model checkpoint saved to: {checkpoint_path}")
        else:
            logging.info(f"Rank {train_group_rank} in training group - checkpoint saved on rank 0")
    elif isinstance(model, nn.DataParallel):
        # DataParallel: get underlying module
        save_model = model.module
        checkpoint_path = f"{checkpoint_dir}/model_dataparallel_wikitext2_trained.pt"
        torch.save({
            'model_state_dict': save_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch_losses': epoch_losses,
            'total_steps': total_steps,
            'final_loss': epoch_losses[-1],
        }, checkpoint_path)
        logging.info(f"Model checkpoint saved to: {checkpoint_path}")
    else:
        # Regular model
        checkpoint_path = f"{checkpoint_dir}/model_wikitext2_trained.pt"
        torch.save({
            'model_state_dict': model.state_dict(),
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
            # Convert input_ids to Dtensor (using single-GPU mesh since DeviceMesh must match process group)
            # DataParallel handles multi-GPU distribution if needed
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
    # Use NCCL backend for better multi-GPU performance (requires CUDA)
    # Fallback to gloo if NCCL not available (e.g., CPU-only or some edge cases)
    if torch.cuda.is_available():
        try:
            dist.init_process_group(backend="nccl", init_method="env://")
            logging.info("Initialized process group with NCCL backend")
        except Exception as e:
            logging.warning(f"Failed to initialize NCCL backend: {e}. Falling back to gloo.")
            dist.init_process_group(backend="gloo")
    else:
        dist.init_process_group(backend="gloo")
        logging.info("Initialized process group with Gloo backend (CUDA not available)")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    log_file = setup_logging(rank)
    
    logging.info(f"Process started - Rank: {rank}, World size: {world_size}")
    
    # Architecture: Rank 0 = Broadcaster, Ranks 1 to N-2 = Trainers, Rank N-1 = Inference
    # Example: 32 trainer GPUs -> world_size = 34 (rank 0 broadcaster, ranks 1-32 trainers, rank 33 inference)
    assert world_size >= 3, "Need at least 3 ranks: 1 broadcaster + 1+ trainers + 1 inference"
    
    # Determine role based on rank
    broadcaster_rank = 0
    inference_rank = world_size - 1
    trainer_ranks = list(range(1, world_size - 1))
    num_trainers = len(trainer_ranks)
    
    is_broadcaster = (rank == broadcaster_rank)
    is_trainer = (rank in trainer_ranks)
    is_inference = (rank == inference_rank)
    
    logging.info(f"Role: {'Broadcaster' if is_broadcaster else 'Trainer' if is_trainer else 'Inference'}")
    if is_trainer:
        logging.info(f"Trainer ranks: {trainer_ranks} (total: {num_trainers} training GPUs)")

    # Setup CUDA device - use LOCAL_RANK for multi-GPU per node, or rank-based assignment
    if torch.cuda.is_available():
        # Use LOCAL_RANK if set (torchrun sets this), otherwise use rank % GPUs_per_node
        local_rank = int(os.environ.get("LOCAL_RANK", rank % torch.cuda.device_count()))
        torch.cuda.set_device(local_rank)
        local_gpu = torch.cuda.current_device()
        num_gpus = torch.cuda.device_count()
        logging.info(f"CUDA device set to GPU {local_gpu} (local_rank={local_rank})")
        logging.info(f"Total GPUs on this node: {num_gpus}")
        
        # DeviceMesh for DTensor (single GPU per process)
        device_mesh = DeviceMesh("cuda", [0])
        logging.info(f"DeviceMesh created with single GPU")
    else:
        local_gpu = None
        local_rank = None
        device_mesh = None
        logging.info("CUDA not available. (RDMA will not work.)")
    
    # Create process groups for FSDP
    # Training process group: only includes trainer ranks (excludes broadcaster/inference)
    if is_trainer:
        logging.info(f"Creating training process group with ranks: {trainer_ranks}")
        train_group = dist.new_group(ranks=trainer_ranks)
    else:
        train_group = None
    
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
    
    if is_broadcaster:
        # Broadcaster
        logging.info("="*80)
        logging.info("BROADCASTER MODE")
        logging.info("="*80)
        logging.info("Connecting to receivers...")
        conn_ids = []
        # Connect to all trainer ranks and inference rank
        receiver_ranks = trainer_ranks + [inference_rank]
        for receiver_rank in receiver_ranks:
            ip, port, r_gpu = p2p.Endpoint.parse_metadata(all_metadata[receiver_rank])
            logging.info(f"Connecting to rank {receiver_rank}: IP={ip}, Port={port}, GPU={r_gpu}")
            ok, conn_id = ep.connect(ip, r_gpu, remote_port=port)
            assert ok, f"Connect failed to rank {receiver_rank}"
            conn_ids.append(conn_id)
            node_type = "Training GPU" if receiver_rank in trainer_ranks else "Inference Node"
            logging.info(f"Connected to {node_type} (rank {receiver_rank}, conn_id={conn_id})")

        # Receive parameter metadata from all trainer ranks and inference node
        param_metadata = {}
        for src_rank in receiver_ranks:
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
    
    elif is_trainer:
        # Trainer
        logging.info("="*80)
        logging.info(f"TRAINING GPU (Rank {rank})")
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
                    mesh_coord = gpu_assignment.get('mesh_coordinate', 'N/A')
                    shard_slices = gpu_assignment.get('shard_slices', {})
                    shard_shape = gpu_assignment.get('shard_shape', 'N/A')
                    size_bytes = gpu_assignment.get('size_bytes', 0)
                    logging.info(f"    GPU {gpu_assignment['gpu_index']} ({gpu_assignment['device']}):")
                    logging.info(f"      mesh_coord={mesh_coord}, shard_slices={shard_slices}")
                    logging.info(f"      shard_shape={shard_shape}, size={size_bytes/1e6:.2f} MB")
            if 'current_gpu_shard' in analysis:
                current = analysis['current_gpu_shard']
                logging.info(f"  Current GPU shard: GPU {current.get('gpu_index')}, "
                          f"mesh_coord={current.get('mesh_coordinate')}, "
                          f"shard_slices={current.get('shard_slices')}")
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

        # Wrap model with FSDP for distributed training across all trainer GPUs
        logging.info(f"Wrapping model with FSDP across {num_trainers} training GPUs...")
        auto_wrap_policy = transformer_auto_wrap_policy(
            transformer_layer_cls={GPT2Block}
        )
        
        model = FSDP(
            base_model,
            auto_wrap_policy=auto_wrap_policy,
            process_group=train_group,  # CRITICAL: Use training group, not global group
            device_id=local_gpu,
            mixed_precision=None,  # Can enable mixed precision here if needed
        )
        logging.info("Model wrapped with FSDP. Parameters will be sharded across training GPUs.")

        run_training(model, tokenizer, num_epochs=2, batch_size=4, lr=5e-5, train_group=train_group, rank=rank)
    
    elif is_inference:
        # Inference node
        logging.info("="*80)
        logging.info(f"INFERENCE NODE (Rank {rank})")
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
                    mesh_coord = gpu_assignment.get('mesh_coordinate', 'N/A')
                    shard_slices = gpu_assignment.get('shard_slices', {})
                    shard_shape = gpu_assignment.get('shard_shape', 'N/A')
                    size_bytes = gpu_assignment.get('size_bytes', 0)
                    logging.info(f"    GPU {gpu_assignment['gpu_index']} ({gpu_assignment['device']}):")
                    logging.info(f"      mesh_coord={mesh_coord}, shard_slices={shard_slices}")
                    logging.info(f"      shard_shape={shard_shape}, size={size_bytes/1e6:.2f} MB")
            if 'current_gpu_shard' in analysis:
                current = analysis['current_gpu_shard']
                logging.info(f"  Current GPU shard: GPU {current.get('gpu_index')}, "
                          f"mesh_coord={current.get('mesh_coordinate')}, "
                          f"shard_slices={current.get('shard_slices')}")
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