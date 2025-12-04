#!/bin/bash
# setup_rdma_env.sh

echo "=== RoCE Environment Setup ==="
echo "Hardware: 8x Broadcom 400Gbps + 2x Mellanox 100Gbps"
echo ""

########################################
# NCCL / RCCL (PyTorch backend='nccl')
########################################

# Control-plane TCP NIC (must reach MASTER_ADDR)
export NCCL_SOCKET_IFNAME=enp49s0f1np1
export NCCL_NSOCKS_PERTHREAD=2      # default is often 4
export NCCL_SOCKET_NTHREADS=1 


# Enable RDMA and use all HCAs except the DOWN one
export NCCL_IB_DISABLE=1
export NCCL_IB_HCA=^bnxt_re6           # exclude bad HCA only
export NCCL_IB_GID_INDEX=3             # RoCEv2 GID index (cluster-specific)
#export NCCL_IB_TIMEOUT=22              # larger timeout for multi-node

# Optional: debug
export NCCL_DEBUG=INFO

########################################
# UCX (for UCCL / custom RDMA engine)
########################################

# All active HCAs (8 Broadcom + 2 Mellanox)
export UCX_NET_DEVICES=bnxt_re0:1,bnxt_re1:1,bnxt_re2:1,bnxt_re3:1,bnxt_re4:1,bnxt_re5:1,bnxt_re7:1,bnxt_re8:1,mlx5_0:1,mlx5_1:1

# Use ROCm-aware or generic transports (adjust to how UCX was built)
# If ROCm-aware:
export UCX_TLS=rc_x,sm,self,rocm_copy,rocm_ipc
# If not ROCm-aware:
# export UCX_TLS=rc_x,ud_x,tcp,sm,self,rocm_copy,rocm_ipc

export UCX_IB_GPU_DIRECT_RDMA=yes
export UCX_IB_TRAFFIC_CLASS=auto
export UCX_MAX_RNDV_RAILS=8

# # Optional: debug
# export UCX_LOG_LEVEL=info

########################################
# Cluster / launcher
########################################

export MASTER_ADDR=10.162.224.129
export MASTER_PORT=29501   # consider randomizing per job to avoid EADDRINUSE

# AMD GPU visibility
export ROCR_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# optionally:
# export HIP_VISIBLE_DEVICES=$ROCR_VISIBLE_DEVICES

echo "✅ Environment configured successfully"
echo ""
echo "Active RDMA interfaces:"
echo "  Broadcom: ens26np0, ens28np0, ens25np0, ens27np0"
echo "            ens22np0, ens24np0, ens21np0, ens23np0"
echo "  Mellanox: enp49s0f0np0, enp49s0f1np1"
echo ""
echo "Total: 3.4 Tbps bandwidth"
