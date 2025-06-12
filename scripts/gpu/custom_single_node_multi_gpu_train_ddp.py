from scripts.utils.custom_train_base import train
from scripts.utils.config import input_dim, hidden_dim, output_dim, batch_size_per_device
from torch.utils.data import DataLoader
from scripts.utils.data_utils import get_dataset
from scripts.utils.model import CustomModel
import os
import time
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp


def setup(rank, world_size):
    # 设置分布式环境
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size
    )


def cleanup():
    dist.destroy_process_group()


def ddp_train(rank, world_size, model_params, batch_size, output_dir="outputs/ddp/"):
    setup(rank, world_size)
    torch.cuda.set_device(rank)

    # 创建模型和数据加载器
    model = CustomModel(*model_params).to(rank)
    model = DDP(model, device_ids=[rank])

    dataset = get_dataset()
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )

    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # 修改保存路径为rank专属路径（可选）
    rank_output_dir = os.path.join(output_dir, f"rank_{rank}")
    os.makedirs(rank_output_dir, exist_ok=True)

    # 调用训练函数
    train(
        model,
        train_loader,
        device=rank,
        local_rank=rank,
        output_dir=output_dir,  # 或使用rank_output_dir分开保存
        num_epochs=20
    )

    cleanup()


if __name__ == '__main__':
    start = time.perf_counter()

    # 注意：这里不要提前加载数据集或模型到主进程
    world_size = 2
    model_params = (input_dim, hidden_dim, output_dim)

    mp.spawn(
        ddp_train,
        args=(world_size, model_params, batch_size_per_device),
        nprocs=world_size,
        join=True
    )

    end = time.perf_counter()
    if dist.get_rank() == 0:  # 确保只在主进程打印
        print(f'Total time cost: {end - start}')
