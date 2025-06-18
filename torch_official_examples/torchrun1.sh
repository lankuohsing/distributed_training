#!/bin/bash
export TORCH_DISTRIBUTED_PORT_RANGE="29500-30000"
export GLOO_SOCKET_IFNAME=bond0
export NCCL_SOCKET_IFNAME=bond0
CUDA_VISIBLE_DEVICES=0,1 torchrun \
--nproc_per_node=2 \
--nnodes=2 \
--node_rank=1 \
--rdzv_id=456 \
--rdzv_backend=c10d \
--rdzv_endpoint=10.205.92.13:29603 \
--rdzv_conf is_host_name=false \
multinode.py 50 10 2>&1 | tee worker.log