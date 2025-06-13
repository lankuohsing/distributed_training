# distributed_training
distributed_training codes for new learner

## CPU训练

```commandline
PYTHONPATH=. python scripts/cpu/custom_cpu_train.py 2>&1 | tee cpu_train.log
```

## 单GPU训练

```commandline
PYTHONPATH=. CUDA_VISIBLE_DEVICES=4 python scripts/gpu/single_node_single_gpu_train.py  2>&1 | tee  gpu_train.log
```

## 单节点多GPU(dp)
```commandline
PYTHONPATH=. CUDA_VISIBLE_DEVICES=4,5 python scripts/gpu/single_node_multi_gpu_train_dp.py  2>&1 | tee  dp_gpu_train.log
```
## 单节点多GPU(ddp)
```commandline
CUDA_VISIBLE_DEVICES=4,5 PYTHONPATH=. torchrun \
    --nnodes=1 \
    --nproc_per_node=2 \
    --rdzv_id=12345 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=localhost:0 \
    scripts/gpu/single_node_multi_gpu_train_ddp_torchrun.py 2>&1 | tee ddp_torchrun.log
```

## 多节点多GPU(ddp)
master: 
```commandline
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
export CUDA_VISIBLE_DEVICES=4,5

# 显示网络信息
echo "Master starting at $(hostname) - $(date)"
# ip addr
# ifconfig

# 启动训练
torchrun \
    --nnodes=2 \
    --nproc_per_node=2 \
    --node_rank=0 \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    scripts/gpu/multi_node_multi_gpu_train_ddp_torchrun.py 2>&1 | tee master.log
```

worker:
```commandline
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
```