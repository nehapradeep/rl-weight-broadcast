# RL Weight Broadcasting via UCCL RDMA (P2P + Collectives)
**High-throughput, zero-copy GPU-to-GPU weight transfer for distributed reinforcement learning.**  
**Authors:** John Drab, Nathan Kotni, Neha Pradeep, Sakthi Karimanal, Vrushali Harane  

This repository contains our implementation of asynchronous RL weight broadcasting using **UCCL RDMA**, comparing:

- **UCCL Collectives**
- **NCCL Collectives (baseline)**
- **UCCL P2P (one-sided RDMA writes)**


We design a fully distributed reinforcement learning pipeline that integrates **RDMA weight transfer**, **hierarchical collectives**, and a **routing-table–based load balancing algorithm** to achieve **40+ GB/s GPU-to-GPU transfer throughput** across multi-node, multi-GPU clusters.

---

# Motivation

Modern reinforcement learning systems require **frequent, asymmetric, one-to-many weight broadcasts** from a central learner to many rollout workers. These transfers often exceed **hundreds of gigabytes per update**, and typical CPU-mediated or RPC-based methods introduce:

- extra memory copies  
- high latency  
- reduced throughput  
- policy lag → slower convergence  

RDMA enables **direct GPU-to-GPU transfers** with **no CPU involvement**.

**UCCL P2P**, in particular, supports asynchronous *one-sided RDMA writes*, making it a natural fit for RL workloads where worker nodes operate at different rates.

This project evaluates UCCL’s suitability for RL and introduces:

- A multi-node RL architecture integrating NCCL + UCCL  
- A routing algorithm for optimal parameter distribution  
- Zero-copy RDMA weight transfer  
- A scalable RL train → transfer → inference pipeline  

---

# Features

### Zero-copy GPU-to-GPU RDMA weight transfer  
Direct writes into remote GPU memory using UCCL P2P.

### Hierarchical communication design  
- **NCCL** for fast intra-node collectives  
- **UCCL** for inter-node RDMA collectives  
- **Gloo** for rank discovery + metadata exchange

### Routing Table Algorithm  
Load-balanced parameter-to-GPU mapping to maximize parallel transfer bandwidth.

### Complete RL pipeline  
Supports both **DDP** and **FSDP** across multi-node GPU clusters.

### Demonstrated scalability  
Sustains **40+ GB/s** aggregate RDMA bandwidth.

---

# System Architecture

```
Controller Node
    ├── Collects metadata from all workers
    ├── Computes routing table (parameter → sender GPU)
    └── Broadcasts RDMA endpoints + routing table

Trainer Nodes (DDP/FSDP)
    ├── Perform PPO updates
    ├── Intra-node gradient sync via NCCL
    └── RDMA-write updated weights to inference GPUs (UCCL P2P)

Inference Nodes
    ├── Receive weights via one-sided RDMA
    └── Perform rollout generation
```

Control path: **Gloo**  
Data path: **UCCL RDMA**

# Architecture Diagram

```
                        ┌─────────────────────────────┐
                        │        Controller Node      │
                        │  - Gathers metadata         │
                        │  - Computes routing table   │
                        │  - Distributes RDMA info    │
                        └──────────────┬──────────────┘
                                       │
                                       ▼
        ┌──────────────────────────────────────────────────────────────┐
        │                        Training Nodes                        │
        │                          (DDP/FSDP)                          │
        │   ┌────────────────────────────────────────────────────────┐ │
        │   │  - PPO updates                                         │ │
        │   │  - NCCL intra-node gradient AllReduce                  │ │
        │   │  - UCCL P2P → RDMA-write weights to inference GPUs     │ │
        │   └────────────────────────────────────────────────────────┘ │
        └───────────────────────────────┬──────────────────────────────┘
                                        │ RDMA (one-sided)
                                        ▼
                ┌──────────────────────────────────────────────┐
                │             Inference Nodes                  │
                │  - Receive weights via one-sided RDMA        │
                │  - Perform rollouts                          │
                └──────────────────────────────────────────────┘

```

---

# Installation

## Requirements
- RDMA-capable NICs (RoCEv2)  
- GPUs with GPUDirect RDMA  
- PyTorch w/ NCCL  
- UCCL built with RoCm  
- Python ≥ 3.9  

---

# Required Environment Variables

These variables configure both NCCL and UCCL for RDMA-based GPU networking.

```bash
export NCCL_SOCKET_IFNAME=enp49s0f1np1
export NCCL_NSOCKS_PERTHREAD=2
export NCCL_SOCKET_NTHREADS=1
export NCCL_IB_DISABLE=0
export NCCL_IB_HCA='^bnxt_re6'
export NCCL_IB_GID_INDEX=3
export UCCL_IB_GID_INDEX=3
export NCCL_DEBUG=INFO
```

> **Note:** Replace `enp49s0f1np1` with your cluster-specific NIC interface.

---


# Running the System
```bash
export MASTER_ADDR=<IP_OF_NODE_0>
```

## 1. Take git clone of UCCL Repo and build it 
```bash
git clone https://github.com/uccl-project/uccl.git --recursive
cd uccl && bash build_and_install.sh [cuda|rocm] p2p [py_version]
```

## 2. Source the RDMA environment  
```bash
source setup_rdma_env.sh
```

## 3. Enter the directory depending upon p2p/collective
```bash
cd uccl_p2p_/
```
## 4. Execute commands for each node
```bash
torchrun --nproc_per_node=8 --nnodes=4 --node_rank=0 \ --master_addr=$MASTER_ADDR --master_port=29502 \ main_rdma_rl_qwen_fixed.py \ --model_name=Qwen/Qwen2.5-0.5B \ --num_training=24 \ --num_inference=8
```
---

---

# Results Summary

## UCCL P2P Bandwidth (Qwen-2.5 Model)

| Trainers | Transfer Time | Aggregate BW |
|---------|----------------|--------------|
| 8       | 0.055 s        | 39.49 GB/s   |
| 16      | 0.084 s        | 38.27 GB/s   |
| 24      | 0.078 s        | 37.52 GB/s   |
| 32      | 0.064 s        | 27.48 GB/s   |

## Key Findings
- **UCCL Collective ≈ NCCL** in DDP/FSDP training.  
- **UCCL P2P scales linearly** with sender count up to ~24 GPUs.  
- Sustains **40+ GB/s aggregate GPU-to-GPU transfer throughput**.  
- Zero CPU involvement reduces policy lag and improves RL stability.  

---

# Routing Table Algorithm

UCCL P2P uses a **greedy byte-balanced routing strategy**:

1. Controller gathers parameter metadata  
2. For each `(parameter, inference_gpu)` pair:
   - choose the training GPU with **lowest cumulative assigned bytes**
3. Broadcast routing table to all GPUs  
4. Trainers perform RDMA writes with no runtime coordination  

This ensures:
- maximal parallelism  
- balanced NIC usage  
- minimized makespan  

---

# Future Work

- FSDP-aware RDMA transfers (DTensor integration)  
- Multi-NIC, topology-aware routing  
- Scaling to 70B+ parameter models  
- End-to-end RL frameworks integration (e.g., RLlib/VLLM-RL)  
- Inference-stage overlapping of computation + transfers  

---

# References

See our full paper for complete citations.  
Key references include:

- *Journey to 2-Second RL Weight Transfer*  
- UCCL transport layer (Zhou et al., 2025)  


