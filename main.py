import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

from data import Caltech101
from model import create_model  
from train import train_model
from evaluate import evaluate_model


# 设置随机种子，保证结果可复现
torch.manual_seed(42)
np.random.seed(42)

# 主函数
def main():
    # 设置设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 加载数据
    data = Caltech101(batch_size=32,data_dir='C:\\Users\\hello\\Documents\\AlexNet_GPU')
    print(f"Number of classes: {data.num_classes}")
    
    # 实验配置
    experiments = {
        'pretrained_finetune': {
            'pretrained': True,
            'freeze_layers': False,  # 不冻结参数，微调所有层
            'log_dir': './logs/pretrained_finetune'
        },
        'pretrained_freeze': {
            'pretrained': True,
            'freeze_layers': True,   # 冻结除最后一层外的参数
            'log_dir': './logs/pretrained_freeze'
        },
        'random_init': {
            'pretrained': False,
            'freeze_layers': False,  # 随机初始化，不冻结参数
            'log_dir': './logs/random_init'
        }
    }
    
    results = {}
    
    # 运行实验
    for exp_name, exp_config in experiments.items():
        print(f"\n=== Running experiment: {exp_name} ===")
        
        # 创建模型，正确传递freeze_layers参数
        model = create_model(
            num_classes=data.num_classes,
            pretrained=exp_config['pretrained'],
            freeze_layers=exp_config['freeze_layers']
        )
        model = model.to(device)
        
        # 定义损失函数
        criterion = nn.CrossEntropyLoss()
        
        # 根据是否冻结层来设置优化器
        if exp_config['freeze_layers']:
            # 只训练最后一层
            optimizer = optim.Adam(model.fc.parameters(), lr=1e-3)
        else:
            # 微调所有层或训练随机初始化的模型
            if exp_config['pretrained']:
                # 预训练模型微调，为不同层设置不同学习率
                params_to_update = []
                for name, param in model.named_parameters():
                    if name.startswith('fc'):
                        params_to_update.append({'params': param, 'lr': 1e-3})
                    else:
                        params_to_update.append({'params': param, 'lr': 1e-5})
                optimizer = optim.Adam(params_to_update)
            else:
                # 随机初始化模型，所有层使用相同学习率
                optimizer = optim.Adam(model.parameters(), lr=1e-3)
        
        # 学习率调度器
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
        
        # 训练模型
        model = train_model(
            model=model,
            dataloaders=data.dataloaders,
            dataset_sizes=data.dataset_sizes,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            num_epochs=50,
            device=device,
            log_dir=exp_config['log_dir']
        )
        
        # 评估模型
        test_accuracy = evaluate_model(model, data.test_loader, device)
        results[exp_name] = test_accuracy
    
    # 打印最终结果
    print("\n=== Summary ===")
    for exp_name, accuracy in results.items():
        print(f"{exp_name}: Test accuracy = {accuracy:.4f}")
    
    # 可视化结果
    plt.figure(figsize=(10, 6))
    plt.bar(results.keys(), results.values())
    plt.ylim(0, 1)
    plt.title('Test Accuracy Comparison')
    plt.xlabel('Experiment')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('accuracy_comparison.png')
    plt.show()

if __name__ == "__main__":
    main()    
