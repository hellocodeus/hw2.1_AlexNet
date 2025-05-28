import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.datasets.utils import download_and_extract_archive
import tarfile


# 数据下载和预处理
class Caltech101:
    def __init__(self, data_dir='C:\\Users\\hello\\Documents\\AlexNet_GPU', batch_size=32, num_workers=4):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        
        self.data_folder= os.path.join(self.data_dir, "data")
        os.makedirs(self.data_folder, exist_ok=True)
        
        # 定义数据转换
        self.data_transforms = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std)
            ]),
            'val': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std)
            ]),
            'test': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std)
            ]),
        }
        
        # 下载和准备数据
        self._download_data()
        tar_file_path = os.path.join(self.data_folder, "caltech-101\\101_ObjectCategories.tar.gz")
        extract_folder = os.path.join(self.data_folder, "101_ObjectCategories")
        os.makedirs(extract_folder, exist_ok=True)
        # 解压文件
        with tarfile.open(tar_file_path, 'r:gz') as tar:
            tar.extractall(path=extract_folder)
            print(f"成功解压文件")
        self._prepare_datasets()
        
    def _download_data(self):
        url = 'https://data.caltech.edu/records/mzrjq-6wc02/files/caltech-101.zip?download=1'
        filename = 'caltech-101.zip'
        download_and_extract_archive(url, self.data_folder, filename=filename)

    def _prepare_datasets(self):
        # 加载完整数据集
        full_dataset = ImageFolder(os.path.join(self.data_folder, '101_ObjectCategories'), 
                                   transform=self.data_transforms['train'])
        
        # 移除背景类
        non_background_indices = [i for i, (path, label) in enumerate(full_dataset.samples) 
                                  if full_dataset.classes[label] != 'BACKGROUND_Google']
        non_background_dataset = torch.utils.data.Subset(full_dataset, non_background_indices)
        
        # 重新映射类别标签
        self.classes = [c for c in full_dataset.classes if c != 'BACKGROUND_Google']
        self.num_classes = len(self.classes)
        
        # 划分训练集、验证集和测试集
        train_size = int(0.7 * len(non_background_dataset))
        val_size = int(0.15 * len(non_background_dataset))
        test_size = len(non_background_dataset) - train_size - val_size
        
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            non_background_dataset, [train_size, val_size, test_size]
        )
        
        # 为验证集和测试集设置正确的transform
        self.val_dataset.dataset.transform = self.data_transforms['val']
        self.test_dataset.dataset.transform = self.data_transforms['test']
        
        # 创建数据加载器
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size,
                                      shuffle=True, num_workers=self.num_workers)
        self.val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size,
                                    shuffle=False, num_workers=self.num_workers)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size,
                                     shuffle=False, num_workers=self.num_workers)
        
        # 数据加载器字典
        self.dataloaders = {
            'train': self.train_loader,
            'val': self.val_loader,
            'test': self.test_loader
        }
        
        # 数据集大小字典
        self.dataset_sizes = {
            'train': len(self.train_dataset),
            'val': len(self.val_dataset),
            'test': len(self.test_dataset)
        }
