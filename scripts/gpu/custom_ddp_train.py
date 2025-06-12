import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler,DataLoader
import torch.distributed as dist
import torch.multiprocessing as mp
from scripts.utils.data_utils import get_dataset
from scripts.utils.model import CustomModel
from scripts.utils.custom_train_base import train
from scripts.utils.config import input_dim, hidden_dim, output_dim, batch_size_per_device
import os
def _ddp_train(local_rank, world_size, model, train_dataset, num_gpus_per_node=None, batch_size_per_device=32,output_dir="outputs/ddp_gpu/"):
    node_rank = int(os.getenv('NODE_RANK'))
    num_gpus_per_node = num_gpus_per_node if num_gpus_per_node is not None else torch.cuda.device_count()
    global_rank = node_rank * num_gpus_per_node + local_rank
    #初始化分布式环境，init_method='env://'表示使用环境变量来初始化进程组
    dist.init_process_group(backend='nccl',
                            init_method='env://',  # 显式指定主节点地址
                            rank=global_rank,
                            world_size=world_size)

    device = f'cuda:{local_rank}'
    model = model.to(device) #如果模型不在GPU卡上，无法对其进行封装
    model = DDP(model, device_ids=[local_rank])

    #每个进程都会单独加载全量的train_dataset
    #然后利用全局的num_replicas和当前进程的全局rank来获取自己的那一份数据
    sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=global_rank)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size_per_device, sampler=sampler)

    train(model, train_loader, device=device, local_rank=local_rank,output_dir=output_dir)

    #销毁分布式环境
    dist.destroy_process_group()

def custom_ddp_train(model, train_dataset, batch_size_per_device=32,output_dir="outputs/ddp_gpu/"):
    world_size = int(os.getenv('WORLD_SIZE'))
    num_gpus_per_node = torch.cuda.device_count()
    #local_rank是自动传递的，范围是：range(nprocs)
    #如果想自定义devices：
        #方法1：使用环境变量CUDA_VISIBLE_DEVICES将自定义devices映射到range(nprocs)
        #方法2：在给_train函数传递device_ids，rank只是用来作为index
    #mp.spawn(_ddp_train, args=(world_size, model, train_dataset, num_gpus_per_node,output_dir), nprocs=num_gpus_per_node, join=True)
    mp.spawn(_ddp_train,
             args=(world_size, model, train_dataset, num_gpus_per_node, batch_size_per_device, output_dir),
             nprocs=num_gpus_per_node,
             join=True)

if __name__ == '__main__':
    train_dataset = get_dataset()
    model = CustomModel(input_dim, hidden_dim, output_dim)

    # 单机多GPU简单训练
    custom_ddp_train(model, train_dataset)
