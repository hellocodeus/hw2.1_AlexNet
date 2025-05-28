import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import torch.nn as nn
from torchvision import models

# 模型定义
def create_model(num_classes, pretrained=True):
    # 加载预训练的ResNet-18模型
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    
    # 修改最后的全连接层
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    return model