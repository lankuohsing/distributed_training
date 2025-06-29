from scripts.utils.custom_train_base import train_single_node
from scripts.utils.config import input_dim, hidden_dim, output_dim, batch_size_per_device
from torch.utils.data import DataLoader
from scripts.utils.data_utils import get_dataset
from scripts.utils.model import CustomModel
import os
import time
'''
PYTHONPATH=. CUDA_VISIBLE_DEVICES=4,5 python scripts/gpu/single_node_multi_gpu_train_dp.py  2>&1 | tee  dp_gpu_train.log
多GPU训练
采用DataParallel进行分布式计算。
原理：当给定model时，主要实现功能是将input数据依据batch的这个维度，将数据平均划分到指定的设备（GPU）上。
其他的对象(objects，例如模型)复制到每个设备上。
在每次前向传播的过程中，module被复制到每个设备上，每个复制的副本处理这个设备上的输入数据。
在每次反向传播过程中，每个副本module的梯度被汇聚到原始的module上进行反向传播计算(一般为第0块GPU)。
'''


def custom_dp_train(model, train_dataset, batch_size_per_device=32,output_dir="outputs/gpu/"):
    os.makedirs(output_dir, exist_ok=True)
    from torch.nn import DataParallel as DP
    device_ids = [0, 1]
    model = DP(model, device_ids=device_ids)
    batch_size = batch_size_per_device * len(device_ids)#全局batch_size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    train_single_node(model, train_loader, device=device_ids[0], output_dir=output_dir)


if __name__ == '__main__':
    start=time.perf_counter()
    train_dataset = get_dataset()
    model = CustomModel(input_dim, hidden_dim, output_dim)

    # 单机多GPU简单训练
    custom_dp_train(model, train_dataset)
    end = time.perf_counter()
    print(f'''
time_cost: {end - start}
''')
# 23.619675908237696
