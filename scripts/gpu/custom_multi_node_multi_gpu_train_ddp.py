import os
import time
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from scripts.utils.custom_train_base import train
from scripts.utils.config import input_dim, hidden_dim, output_dim
from scripts.utils.data_utils import get_dataset
from scripts.utils.model import CustomModel


def setup_ddp(rank, world_size, master_addr='10.205.58.30', master_port='12355'):
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup_ddp():
    dist.destroy_process_group()


def ddp_train(rank, world_size, output_dir="outputs/ddp/"):
    os.makedirs(output_dir, exist_ok=True)

    local_rank = int(os.environ["LOCAL_RANK"])  # 关键修改点
    setup_ddp(rank, world_size)
    torch.cuda.set_device(local_rank)  # 使用本地GPU索引

    model = CustomModel(input_dim, hidden_dim, output_dim).to(local_rank)
    model = DDP(model, device_ids=[local_rank])

    train_dataset = get_dataset()
    sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)

    if rank == 0:
        print(f"Training on {world_size} nodes with {torch.cuda.device_count()} GPUs per node")

    train(model, train_loader, device=local_rank, output_dir=output_dir)
    cleanup_ddp()


if __name__ == '__main__':
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    start_time = time.perf_counter()
    ddp_train(rank, world_size)

    if rank == 0:
        end_time = time.perf_counter()
        print(f'\nTotal time: {end_time - start_time:.2f} seconds\n')
