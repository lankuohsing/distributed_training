#!/bin/bash
export MASTER_ADDR="10.205.92.13"
export MASTER_PORT=29500
export WORLD_SIZE=4
export PYTHONPATH=.

# 关键 NCCL 设置
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=bond0  # 替换为你的实际接口
export NCCL_BLOCKING_WAIT=1

# GPU可视设置
export CUDA_VISIBLE_DEVICES=0,1

# 显示网络信息
echo "Worker starting at $(hostname) - $(date)"
# ip addr
# ifconfig

# 启动训练
torchrun \
    --nnodes=2 \
    --nproc_per_node=2 \
    --node_rank=1 \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    scripts/gpu/multi_node_multi_gpu_train_ddp_torchrun.py 2>&1 | tee worker.log