import os
import time
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from scripts.utils.custom_train_base import train
from scripts.utils.config import input_dim, hidden_dim, output_dim
from scripts.utils.data_utils import get_dataset
from scripts.utils.model import CustomModel


def setup_ddp(rank, world_size, master_addr='10.205.58.30', master_port='12355'):
    # 初始化进程组
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup_ddp():
    dist.destroy_process_group()


def ddp_train(rank, world_size, output_dir="outputs/ddp/"):
    os.makedirs(output_dir, exist_ok=True)

    # DDP初始化
    setup_ddp(rank, world_size)
    torch.cuda.set_device(rank)  # 每个进程绑定到自己的GPU

    # 模型和数据
    model = CustomModel(input_dim, hidden_dim, output_dim).to(rank)
    model = DDP(model, device_ids=[rank])

    # 分布式Sampler
    train_dataset = get_dataset()
    sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    batch_size = 32  # 每个GPU的batch size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)

    # 训练（只在rank 0记录输出）
    if rank == 0:
        print(f"Training on {world_size} nodes with {torch.cuda.device_count()} GPUs per node")

    train(model, train_loader, device=rank, output_dir=output_dir)

    cleanup_ddp()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--rank', type=int, required=True, help='Global rank of the process')
    parser.add_argument('--world_size', type=int, required=True, help='Total number of processes')
    args = parser.parse_args()

    start_time = time.perf_counter()
    ddp_train(args.rank, args.world_size)

    if args.rank == 0:  # 只在主节点输出时间
        end_time = time.perf_counter()
        print(f'\nTotal time: {end_time - start_time:.2f} seconds\n')
