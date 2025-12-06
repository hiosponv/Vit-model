import torch

IN_CHANNELS = 3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PATCH_SIZE = 16
EMB_SIZE = 256
NUM_HEADS = 4
NUM_CLASSES = 10
NUM_LAYERS = 6
EPOCHES = 50
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
NUM_WORKERS = 10
Droupout_RATE = 0.1
PERSISTENT_WORKERS = True

# ViT 模型通常使用 ImageNet 的均值和标准差进行归一化
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMAGE_SIZE = 224 # ViT 的标准输入尺寸