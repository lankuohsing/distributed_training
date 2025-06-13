import torch
from scripts.utils.custom_train_base import train_single_node
from scripts.utils.config import input_dim, hidden_dim, output_dim, batch_size_per_device
from torch.utils.data import DataLoader
from scripts.utils.data_utils import get_dataset
from scripts.utils.model import CustomModel
import os
import time
import torch.distributed as dist
import torch.multiprocessing as mp

'''
PYTHONPATH=. CUDA_VISIBLE_DEVICES=4,5 python scripts/gpu/single_node_multi_gpu_train_ddp_spawn.py  2>&1 | tee  ddp_gpu_train.log

'''
def ddp_train(rank, world_size, train_dataset, batch_size_per_device=32, output_dir="outputs/gpu/"):
    # 初始化进程组设置
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank
    )

    torch.cuda.set_device(rank)

    # 创建模型并移动至当前GPU
    model = CustomModel(input_dim, hidden_dim, output_dim)
    model = model.to(rank)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])

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
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # 执行训练
    train_single_node(
        model,
        train_loader,
        device=rank,
        local_rank=rank,
        output_dir=output_dir
    )

    # 清理进程组
    dist.destroy_process_group()


if __name__ == '__main__':
    start = time.perf_counter()
    train_dataset = get_dataset()
    world_size = 2  # 使用两个GPU
    output_dir="outputs/ddp/"
    os.makedirs(output_dir,exist_ok=True)
    # 启动多进程训练
    mp.spawn(
        ddp_train,
        args=(world_size, train_dataset, batch_size_per_device, output_dir),
        nprocs=world_size,
        join=True
    )

    end = time.perf_counter()
    print(f'time_cost: {end - start}')
    # 47.47289187926799
