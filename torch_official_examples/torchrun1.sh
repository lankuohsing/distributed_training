#!/bin/bash
export GLOO_SOCKET_IFNAME=bond0  # 使用bond0接口
export NCCL_SOCKET_IFNAME=bond0   # 使用bond0接口
CUDA_VISIBLE_DEVICES=0,1 torchrun \
--nproc_per_node=2 \
--nnodes=2 \
--node_rank=1 \
--rdzv_id=456 \
--rdzv_backend=c10d \
--rdzv_endpoint=10.205.92.13:29603 \
--rdzv_conf is_host_name=false \  # 强制使用IP
multinode.py 50 10 2>&1 | tee worker.log