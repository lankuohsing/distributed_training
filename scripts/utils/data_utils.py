import os
import torch
from torch.utils.data import Dataset
from scripts.utils.config import input_dim, output_dim, data_num

class CustomDataset(Dataset):
    '''
    Map-style datasets
    重写基类torch.utils.data import Dataset
    是个iterable object
    '''
    def __init__(self, x, y):
        '''
        :param x: Tensor, [num_samples, input_dim]
        :param y: Tensor, [num_samples, output_dim]
        '''
        self.x = x
        self.y = y
    def __len__(self):
        return len(self.x)
    def __getitem__(self, idx):
        return {'inputs': self.x[idx], 'labels': self.y[idx]}

def get_dataset(data_num=10000):
    data_path = '/opt/data2/languoxing/datasets/train_data.pt'
    if os.path.exists(data_path):
        data_dict = torch.load(data_path)
        x_train = data_dict['x_train']
        y_train = data_dict['y_train']
    else:
        x_train = torch.randn(data_num, input_dim)
        gold_w = torch.randn(input_dim, output_dim)
        gold_b = torch.randn(output_dim)
        y_train = torch.matmul(x_train, gold_w) + gold_b

    # 创建数据集和数据加载器
    train_dataset = CustomDataset(x_train, y_train)
    return train_dataset

if __name__ == '__main__':
    def setup_seed(seed):
        import numpy as np
        import random
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
    # 设置随机数种子
    setup_seed(20)
    '''
    随机初始化X，W，b
    构造输出Y=WX+b
    '''
    x_train = torch.randn(data_num, input_dim)
    gold_w = torch.randn(input_dim, output_dim)
    gold_b = torch.randn(output_dim)
    y_train = torch.matmul(x_train, gold_w) + gold_b
    torch.save({'x_train': x_train, 'y_train': y_train}, 'datasets/train_data.pt')