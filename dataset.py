import torchvision
from torchvision.transforms import ToTensor, Resize, Normalize, Compose
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from config import *
import torch.distributed as dist
from torch.utils.data import Dataset


class CIFAR10Dataset(Dataset):
    """
    一个针对 CIFAR-10 的 PyTorch Dataset 类，集成了分布式下载同步和 ViT 标准预处理。
    """
    def __init__(self):
        super().__init__()
        # Compose 序列，用于 ViT 的微调预处理
        self.img_transform = Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
    def prepare_data(self, rank, is_train):
        if rank == 0:
            print("Rank 0 is downloading CIFAR-10 data...")
            torchvision.datasets.CIFAR10(
                root='./cifar10/',
                train=True,
                download=True,
                transform=self.img_transform
            )
            # 下载测试集
            torchvision.datasets.CIFAR10(
                root='./cifar10/',
                train=False,
                download=True,
                transform=self.img_transform
            )
        # 2. 分布式同步：确保所有进程等待下载完成
        if dist.is_initialized():
            print(f"Rank {rank} waiting for download sync...")
            dist.barrier()
        # 3. 所有进程加载数据
        if is_train:
            self.dataset = torchvision.datasets.CIFAR10(
                root='./cifar10/',
                train=True,
                download=False,
                transform=self.img_transform
            )
        else:
            self.dataset = torchvision.datasets.CIFAR10(
                root='./cifar10/',
                train=False,
                download=False,
                transform=self.img_transform
            )
        print(f"Rank {rank} loaded {len(self.dataset)} samples.")
        return self.dataset
        
    def __len__(self):
        if not hasattr(self, 'dataset'):
            raise RuntimeError("Dataset not prepared. Call prepare_data first.")
        return len(self.dataset)
    
    def __getitem__(self,idx):
        img,label=self.dataset[idx]
        # 图像经过 transform 已经完成了 Tensor 转换、缩放和归一化
        return img, label
