#!/bin/bash
# launch_v2.sh - Launch script for main_distributed_rdma_v2.py

export GLOO_SOCKET_IFNAME=enp49s0f1np1
export NCCL_SOCKET_IFNAME=enp49s0f1np1
export UCCL_NET_DEVICES=mlx5_1
export UCX_NET_DEVICES=mlx5_1:1
export UCX_IB_GID_INDEX=3
export NCCL_IB_HCA=mlx5_1
export NCCL_IB_DISABLE=1

export MASTER_ADDR="${MASTER_ADDR:?Set MASTER_ADDR}"
export NODE_RANK="${NODE_RANK:?Set NODE_RANK (0-3)}"

echo "=========================================="
echo "UCCL P2P RDMA Training V2"
echo "=========================================="
echo "Node Rank: $NODE_RANK"
echo "Master: $MASTER_ADDR"
echo "=========================================="

torchrun --nproc_per_node=8 --nnodes=4 --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --master_port=29501 main_distributed_rdma_v2.py "$@"