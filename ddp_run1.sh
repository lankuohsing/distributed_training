#!/bin/bash
# 工作节点 (10.205.58.30)

# 设置环境变量
export MASTER_ADDR="10.205.92.13"  # 指向主节点IP
export MASTER_PORT=29500
export WORLD_SIZE=4
export PYTHONPATH=.

# 启动命令 (工作节点使用 node_rank=1)
torchrun \
    --nnodes=2 \
    --nproc_per_node=2 \
    --node_rank=1 \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    scripts/gpu/single_node_multi_gpu_train_ddp_torchrun.py 2>&1 | tee worker.log