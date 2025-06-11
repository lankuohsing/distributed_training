import torch
import torch.nn as nn
# 自定义模型
class CustomModel(nn.Module):
    def __init__(self, input_dim=10, hidden_dim=50, output_dim=1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.criterion = nn.MSELoss()#当前的模型是用来解决回归问题的

    def forward(self, inputs, labels=None):
        y = self.fc1(inputs)
        y = self.relu(y)
        y = self.fc2(y)
        if labels is None:
            loss = None
        else:
            loss = self.criterion(y, labels)
        return {
            'loss': loss,
            'outputs': y
        }