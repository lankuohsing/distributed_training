#!/bin/bash
# 主节点 (10.205.92.13)

# 设置环境变量
export MASTER_ADDR="10.205.92.13"
export MASTER_PORT=29500
export WORLD_SIZE=4
export PYTHONPATH=.

# 启动命令 (主节点使用 node_rank=0)
torchrun \
    --nnodes=2 \
    --nproc_per_node=2 \
    --node_rank=0 \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    scripts/gpu/single_node_multi_gpu_train_ddp_torchrun.py 2>&1 | tee master.log