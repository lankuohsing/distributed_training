import os
import time
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from scripts.utils.config import input_dim, hidden_dim, output_dim, batch_size_per_device
from scripts.utils.data_utils import get_dataset
from scripts.utils.model import CustomModel
from scripts.utils.custom_train_base import train
'''
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=12345 scripts/gpu/custom_single_node_multi_gpu_train_dp.py  2>&1 | tee  ddp_gpu_train.log

'''
def setup_for_ddp(local_rank, world_size):
    # os.environ['MASTER_ADDR'] = 'localhost'
    # os.environ['MASTER_PORT'] = '12345'  # 确保端口没被占用

    dist.init_process_group("nccl", rank=local_rank, world_size=world_size)
    torch.cuda.set_device(local_rank)

def cleanup_ddp():
    dist.destroy_process_group()

def custom_ddp_train(local_rank, world_size, output_dir="outputs/ddp_gpu/"):
    setup_for_ddp(local_rank, world_size)

    train_dataset = get_dataset()
    sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=local_rank, shuffle=True)

    model = CustomModel(input_dim, hidden_dim, output_dim).to(local_rank)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size_per_device,
        shuffle=False,
        sampler=sampler,
        num_workers=2,
        pin_memory=True
    )

    train(
        model,
        train_loader,
        device=local_rank,
        local_rank=local_rank,
        output_dir=output_dir
    )

    cleanup_ddp()

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    # launch工具会传递--local_rank参数
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--world_size", type=int, default=2)
    args = parser.parse_args()
    start = time.perf_counter()
    custom_ddp_train(args.local_rank, args.world_size)
    end = time.perf_counter()

    if args.local_rank == 0:
        print(f'''
        time_cost: {end - start}
        ''')
