from scripts.utils.custom_train_base import train
from scripts.utils.config import input_dim, hidden_dim, output_dim, batch_size_per_device
from torch.utils.data import DataLoader
from scripts.utils.data_utils import get_dataset
from scripts.utils.model import CustomModel
import os
'''
单GPU训练
只需要管理device即可
'''
def custom_gpu_train(model, train_dataset, batch_size_per_device=32,output_dir="outputs/cpu/"):
    os.makedirs(output_dir,exist_ok=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size_per_device, shuffle=True)
    train(model, train_loader, device='cuda:0',output_dir=output_dir)


if __name__ == '__main__':
    train_dataset = get_dataset()
    model = CustomModel(input_dim, hidden_dim, output_dim)
    # CPU简单训练模型
    custom_gpu_train(model, train_dataset, batch_size_per_device)
