# import tensorflow as tf

# if tf.test.is_gpu_available():
#     print("GPU is available.")
# else:
#     print("GPU is not available.")

# print(tf.__version__)  # 打印TensorFlow版本号
# print(tf.__file__)  # 打印TensorFlow安装路径



# import torch
 
# cuda_version = torch.version.cuda
# print("CUDA版本:", cuda_version)

# import torch

# print(f"CUDA是否可用: {torch.cuda.is_available()}")
# print(f"可用GPU数量: {torch.cuda.device_count()}")
# print(f"当前GPU名称: {torch.cuda.get_device_name(0)}")

import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))