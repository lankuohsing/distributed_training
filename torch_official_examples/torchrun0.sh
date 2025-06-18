#!/bin/bash
export GLOO_SOCKET_IFNAME=bond0
export NCCL_SOCKET_IFNAME=bond0

# 强制使用 IPv4
export GLOO_SOCKET_FAMILY=IPv4
export NCCL_P2P_DISABLE=1

# 设置分布式端口范围
export TORCH_DISTRIBUTED_PORT_RANGE="29500-30000"

# 设置日志级别
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_DEBUG=INFO
# 显式设置MASTER地址为IP（重要！）
export MASTER_ADDR="10.205.92.13"
export MASTER_PORT=29603

CUDA_VISIBLE_DEVICES=4,5 torchrun \
--nproc_per_node=2 \
--nnodes=2 \
--node_rank=0 \
--rdzv_id=456 \
--rdzv_backend=c10d \
--rdzv_endpoint=10.205.92.13:29603 \
--rdzv_conf is_host_name=false \
multinode.py 50 10 2>&1 | tee master.log