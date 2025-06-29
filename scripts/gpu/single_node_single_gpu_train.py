from scripts.utils.custom_train_base import train_single_device
from scripts.utils.config import input_dim, hidden_dim, output_dim, batch_size_per_device
from torch.utils.data import DataLoader
from scripts.utils.data_utils import get_dataset
from scripts.utils.model import CustomModel
import time
import os
'''
单GPU训练
只需要管理device即可
PYTHONPATH=. CUDA_VISIBLE_DEVICES=4 python scripts/gpu/single_node_single_gpu_train.py  2>&1 | tee  gpu_train.log
'''
def custom_gpu_train(model, train_dataset, batch_size_per_device=32,output_dir="outputs/gpu/"):
    os.makedirs(output_dir,exist_ok=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size_per_device, shuffle=True)
    train_single_device(model, train_loader, device='cuda:0', output_dir=output_dir)


if __name__ == '__main__':
    start=time.perf_counter()
    train_dataset = get_dataset()
    model = CustomModel(input_dim, hidden_dim, output_dim)
    # CPU简单训练模型
    custom_gpu_train(model, train_dataset, batch_size_per_device)
    end = time.perf_counter()
    print(f'''
time_cost: {end - start}
''')
    # 13.19159437622875
