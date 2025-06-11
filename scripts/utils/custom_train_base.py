import os
import torch
import torch.nn as nn
from torch import optim


def save_model(model, optimizer, epoch, loss, only_save_model,output_dir="outputs"):
    '''
    不推荐使用torch.save(model, path)
    :param model:
    :param optimizer:
    :param epoch:
    :param loss:
    :param only_save_model:
    :return:
    '''
    os.makedirs("outputs/", exist_ok=True)
    if only_save_model:
        torch.save(model.state_dict(), os.path.join(output_dir,f'only-model_epoch-{epoch}.pth'))
    else:

        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss.item()
        }
        torch.save(checkpoint, os.path.join(output_dir,f'outputs/checkpoint_epoch-{epoch}.pth'))


def train(model, train_loader, lr=1e-3, num_epochs=20, device='cpu', local_rank=0,
          only_save_model=True,output_dir="outputs"):
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()  # 设置模型为训练模式
    for epoch in range(num_epochs):
        for i, batch in enumerate(train_loader):
            inputs = batch['inputs'].to(device)
            labels = batch['labels'].to(device)
            # 前向传播
            model_result = model(inputs, labels)
            outputs = model_result['outputs']
            loss = criterion(outputs, labels)
            '''
            用下面的loss_0来反向传播是等价的
            '''
            loss_0 = model_result['loss']
            # 反向传播和优化
            optimizer.zero_grad()  # 清空梯度
            loss.backward()  # 计算梯度
            optimizer.step()  # 更新参数

            # 打印训练信息
            if (i + 1) % 10 == 0 and local_rank == 0:# 在多进程训练中，只打印一个进程的log，避免日志重复混杂影响debug
                print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}')

    save_model(model, optimizer, epoch, loss, only_save_model,output_dir=output_dir)