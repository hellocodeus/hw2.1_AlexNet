import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import torch

# 评估模型
def evaluate_model(model, dataloader, device='cuda:0'):
    model.eval()
    running_corrects = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            total += labels.size(0)
            running_corrects += (preds == labels).sum().item()
    
    accuracy = running_corrects / total
    print(f'Accuracy: {accuracy:.4f}')
    
    return accuracy