# 微调在ImageNet上预训练的卷积神经网络实现Caltech-101分类
修改 ResNet-18 模型输出层以适配 Caltech-101 数据集的 101 个类别，利用 ImageNet 预训练参数初始化模型。
对比从零开始随机初始化训练和预训练后微调两种策略的模型性能。
探究不同超参数（训练步数、学习率等）及其组合对模型性能的影响，寻找最优超参数配置。

## 训练方法
1. 修改main.py中data_dir变量，使之指向Caltech-101训练数据所存放的文件路径
2. 运行main.py即可进行模型训练和测试，并且训练结束后可以输出Loss和Accuracy的变化曲线并支持Tensorboard可视化训练结果。

项目地址：https://github.com/hellocodeus/hw2.1_AlexNet.git
模型权重下载地址：https://drive.google.com/file/d/1dmky7jHl0n8WRKrM5uM_T_GtHLQPmenS/view?usp=sharing
