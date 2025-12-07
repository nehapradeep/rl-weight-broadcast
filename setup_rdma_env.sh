#!/bin/bash

# Control-plane TCP NIC (must reach MASTER_ADDR)
export NCCL_SOCKET_IFNAME=enp49s0f1np1
export NCCL_NSOCKS_PERTHREAD=2    
export NCCL_SOCKET_NTHREADS=1 

# Enable RDMA and use all HCAs except the DOWN one
export NCCL_IB_DISABLE=1
export NCCL_IB_HCA=^bnxt_re6           # exclude bad HCA only
export NCCL_IB_GID_INDEX=3             # RoCEv2 GID index (cluster-specific)

# Optional: debug
export NCCL_DEBUG=INFO

export MASTER_ADDR=10.162.224.129
export MASTER_PORT=29501   

# AMD GPU visibility
export ROCR_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
