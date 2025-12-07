#!/bin/bash

# Control-plane TCP NIC (must reach MASTER_ADDR)
export GLOO_SOCKET_IFNAME=enp49s0f1np1
export NCCL_SOCKET_IFNAME=enp49s0f1np1
export NCCL_NSOCKS_PERTHREAD=2    
export NCCL_SOCKET_NTHREADS=1
export NCCL_MIN_NCHANNELS=4
export NCCL_MAX_NCHANNELS=8
export OMP_NUM_THREADS=1

# Enable RDMA and use all HCAs except the DOWN one
export NCCL_IB_DISABLE=0
export NCCL_IB_HCA=^bnxt_re6           # exclude bad HCA only
export NCCL_IB_GID_INDEX=3             # RoCEv2 GID index

# Optional: debug
export NCCL_DEBUG=INFO

#export MASTER_ADDR=10.162.224.129  #set it to your master node address
#export MASTER_PORT=29501    #set it to your master port address

# AMD GPU visibility
export ROCR_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
