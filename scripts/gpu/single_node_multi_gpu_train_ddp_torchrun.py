import torch
from scripts.utils.custom_train_base import train_single_node
from scripts.utils.config import input_dim, hidden_dim, output_dim, batch_size_per_device
from torch.utils.data import DataLoader
from scripts.utils.data_utils import get_dataset
from scripts.utils.model import CustomModel
import os
import time
import torch.distributed as dist

'''
CUDA_VISIBLE_DEVICES=4,5 PYTHONPATH=. torchrun \
    --nnodes=1 \
    --nproc_per_node=2 \
    --rdzv_id=12345 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=localhost:0 \
    scripts/gpu/single_node_multi_gpu_train_ddp_torchrun.py 2>&1 | tee ddp_torchrun.log
'''
def ddp_train(train_dataset, batch_size_per_device=32, output_dir="outputs/ddp/"):
    # 从环境变量获取分布式信息
    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])

    # 初始化进程组 - torchrun 已设置环境变量
    dist.init_process_group(
        backend='nccl',
        init_method='env://'
    )

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
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # 执行训练
    train_single_node(
        model,
        train_loader,
        device=local_rank,
        local_rank=local_rank,
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
    # 14.761768782045692