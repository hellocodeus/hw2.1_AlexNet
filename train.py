import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import torch
from torch.utils.tensorboard import SummaryWriter
import time
import copy

# 训练函数
def train_model(model, dataloaders, dataset_sizes, criterion, optimizer, scheduler, num_epochs=50, device='cuda:0', log_dir='./logs'):
    since = time.time()
    
    # 创建TensorBoard写入器
    writer = SummaryWriter(log_dir)
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)
        
        # 每个epoch都有一个训练和验证阶段
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # 训练模式
            else:
                model.eval()   # 评估模式
            
            running_loss = 0.0
            running_corrects = 0
            
            # 迭代数据
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # 零梯度
                optimizer.zero_grad()
                
                # 前向传播
                # 只有在训练时才跟踪历史
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    # 只有在训练阶段才进行反向传播和优化
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                # 统计
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            if phase == 'train' and scheduler:
                scheduler.step()
            
            # 修正：使用dataset_sizes字典获取正确的数据集大小
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            # 记录到TensorBoard
            writer.add_scalar(f'Loss/{phase}', epoch_loss, epoch)
            writer.add_scalar(f'Accuracy/{phase}', epoch_acc, epoch)
            
            # 深拷贝模型
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
        
        print()
    
    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')
    
    # 加载最佳模型权重
    model.load_state_dict(best_model_wts)
    
    # 关闭TensorBoard写入器
    writer.close()
    
    return model
