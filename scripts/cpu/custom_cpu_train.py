import os
import time
from scripts.utils.custom_train_base import train0
from scripts.utils.config import input_dim, hidden_dim, output_dim, batch_size_per_device
from torch.utils.data import DataLoader
from scripts.utils.data_utils import get_dataset
from scripts.utils.model import CustomModel
'''
CPU训练
只需要管理device即可
PYTHONPATH=. python scripts/cpu/custom_cpu_train.py 2>&1 | tee cpu_train.log
'''
def custom_cpu_train(model, train_dataset, batch_size_per_device=32,output_dir="outputs/cpu/"):
    os.makedirs(output_dir,exist_ok=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size_per_device, shuffle=True)
    train0(model, train_loader, device='cpu',output_dir=output_dir)


if __name__ == '__main__':
    start=time.perf_counter()
    train_dataset = get_dataset()
    model = CustomModel(input_dim, hidden_dim, output_dim)
    # CPU简单训练模型
    custom_cpu_train(model, train_dataset, batch_size_per_device)
    end=time.perf_counter()
    print(f'''
time_cost: {end-start}
''')
    # time_cost: 5.773849678225815
