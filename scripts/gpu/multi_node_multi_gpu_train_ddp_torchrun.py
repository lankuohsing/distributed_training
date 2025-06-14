import torch
from scripts.utils.custom_train_base import train_multi_node
from scripts.utils.config import input_dim, hidden_dim, output_dim, batch_size_per_device
from torch.utils.data import DataLoader
from scripts.utils.data_utils import get_dataset
from scripts.utils.model import CustomModel
import os
import time
import torch.distributed as dist

'''
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

# 显示网络信息
echo "Master starting at $(hostname) - $(date)"
ip addr
ifconfig

# 启动训练
torchrun \
    --nnodes=2 \
    --nproc_per_node=2 \
    --node_rank=0 \# master为0，其他则为正整数
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    scripts/gpu/multi_node_multi_gpu_train_ddp_torchrun.py 2>&1 | tee master.log
'''
def ddp_train(train_dataset, batch_size_per_device=32, output_dir="outputs/ddp/"):
    # 从环境变量获取分布式信息
    print(f"Initializing process group: rank={os.environ['RANK']}, "
          f"local_rank={os.environ['LOCAL_RANK']}, "
          f"world_size={os.environ['WORLD_SIZE']}")
    '''
    master上打印：
    Initializing process group: rank=1, local_rank=1, world_size=4
    Initializing process group: rank=0, local_rank=0, world_size=4
    worker上打印：
    Initializing process group: rank=2, local_rank=0, world_size=4
    Initializing process group: rank=3, local_rank=1, world_size=4
    '''
    '''
    WORLD_SIZE = nnodes * nproc_per_node
    RANK = node_rank * nproc_per_node + LOCAL_RANK
    LOCAL_RANK: 当前节点内的进程编号（范围：0 到 nproc_per_node - 1）。
    CUDA_VISIBLE_DEVICES=4,5 → LOCAL_RANK=0 使用 GPU4，LOCAL_RANK=1 使用 GPU5。
    '''
    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])

    # 初始化进程组 - torchrun 已设置环境变量
    dist.init_process_group(
        backend='nccl',
        init_method='env://'
    )
    dist.barrier()# 进程同步屏障：所有进程执行到此处时会互相等待，直到全部进程都到达该屏障后才会继续执行后续代码。
    if int(os.environ.get('RANK', 0)) == 0:
        print("All processes joined the process group")
    torch.cuda.set_device(local_rank)

    # 创建模型并移动至当前GPU
    model = CustomModel(input_dim, hidden_dim, output_dim)
    model = model.to(local_rank)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

    # 创建分布式数据加载器
    sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size_per_device,
        sampler=sampler,
        shuffle=False,# ddp模式（多设备）下，这里必须为False，
        num_workers=4,
        pin_memory=True
    )

    # 执行训练
    train_multi_node(
        model,
        train_loader,
        device=local_rank,
        global_rank=rank,
        output_dir=output_dir
    )

    # 清理进程组
    dist.destroy_process_group()


if __name__ == '__main__':
    start = time.perf_counter()
    train_dataset = get_dataset()
    output_dir = "outputs/ddp/"
    os.makedirs(output_dir, exist_ok=True)

    # 直接调用训练函数 - torchrun 会处理进程创建
    ddp_train(train_dataset, batch_size_per_device, output_dir)

    end = time.perf_counter()
    # 只在 rank 0 打印时间
    if int(os.environ.get('RANK', 0)) == 0:
        print(f'time_cost: {end - start}')
    # 8.757532767951488