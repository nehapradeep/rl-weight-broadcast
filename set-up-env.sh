#!/bin/bash

# =================================================================
# AMD GPU + Broadcom RDMA Environment Setup Script
# Usage: source setup_amd_env.sh
# =================================================================

echo "Configuring environment for AMD GPUs with Broadcom RDMA..."

# 0. Activate Conda Environment
if [[ "$CONDA_DEFAULT_ENV" != "rl_rdma_env" ]]; then
    echo "Current environment is '$CONDA_DEFAULT_ENV'. Attempting to activate 'rl_rdma_env'..."
    conda activate rl_rdma_env
    
    if [[ "$CONDA_DEFAULT_ENV" != "rl_rdma_env" ]]; then
        echo "ERROR: Failed to automatically activate 'rl_rdma_env'."
        echo "       Please run 'conda activate rl_rdma_env' manually before sourcing this script."
        return 1 2>/dev/null || exit 1
    fi
    echo "  [OK] Activated 'rl_rdma_env'"
else
    echo "  [OK] Already in 'rl_rdma_env'"
fi

# 1. GPU Visibility
# Unset this to ensure PyTorch finds all GPUs automatically.
# (If set incorrectly, it causes 'no ROCm-capable device detected')
unset HIP_VISIBLE_DEVICES
echo "  [OK] Unset HIP_VISIBLE_DEVICES"

# 2. Network Selection (CRITICAL)
# Exclude slow ethernet (enp/eth), docker bridges, and loopback.
# NCCL will automatically pick up the remaining high-speed Broadcom (bnxt_re) 
# and Mellanox (mlx5) adapters.
export NCCL_IB_HCA="^enp,eth,docker,lo,mlx5_0,mlx5_1"
echo "  [OK] NCCL_IB_HCA set to exclude slow interfaces"

# 3. Control Plane Binding
# Bind the handshake/coordination traffic to the known working interface.
# This prevents hangs during initialization.
export NCCL_SOCKET_IFNAME="enp49s0f1np1"
export GLOO_SOCKET_IFNAME="enp49s0f1np1"
echo "  [OK] Socket interfaces bound to enp49s0f1np1"

# 4. Performance Tuning
# Relaxed Ordering is crucial for high throughput on AMD platforms.
# export NCCL_IB_PCI_RELAXED_ORDERING=1
# Enable GPU Direct RDMA (GDR) for zero-copy transfer between GPU and NIC.
export NCCL_NET_GDR_LEVEL=2
# # Optimize Peer-to-Peer chunk sizes
# export NCCL_P2P_NET_CHUNKSIZE=524288
# export NCCL_BUFFSIZE=8388608
# # Channel and Queue Pair settings standard for scale-out
export NCCL_MIN_NCHANNELS=4
export NCCL_MAX_NCHANNELS=8
# export NCCL_IB_QPS_PER_CONNECTION=4
# export NCCL_IB_SPLIT_DATA_ON_QPS=1
export NCCL_IB_DISABLE=0 
echo "  [OK] Enabled GPU Direct RDMA (GDR)"
# echo "  [OK] Applied performance tuning flags (Relaxed Ordering, GDR)"

# 5. Debugging
# Set to INFO to verify which interfaces are selected at runtime.
# Change to WARN to reduce clutter once verified.
export NCCL_DEBUG=INFO
echo "  [OK] NCCL_DEBUG set to INFO"


export NCCL_IB_GID_INDEX=3
echo "  [OK] NCCL_IB_GID_INDEX set to $NCCL_IB_GID_INDEX"

# 6. Threading
# Prevent CPU contention between PyTorch dataloaders
export OMP_NUM_THREADS=1
echo "  [OK] OMP_NUM_THREADS set to 1"

echo "================================================================="
echo "Environment ready. Run your torchrun command now."
echo "================================================================="