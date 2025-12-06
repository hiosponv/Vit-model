import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast_mode, grad_scaler
from torch.utils.data import DataLoader, DistributedSampler

# 初始化分布式环境
def setup_ddp():
    """"初始化分布式环境"""
    dist.init_process_group(backend='nccl') # 建立通信组（多卡多机核心）
    # torchrun 自动注入 LOCAL_RANK，指示当前进程绑定的 GPU ID
    local_rank = int(os.environ["LOCAL_RANK"])
    # 设置当前进程使用的 GPU（非常重要）
    torch.cuda.set_device(local_rank)
    # RANK：全局唯一进程 ID（0 ～ world_size-1）
    # WORLD_SIZE：全局进程数（GPU数量）
    return local_rank, int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])

def cleanup_ddp():
    """销毁分布式进程组（训练结束时必须调用）"""
    dist.destroy_process_group()

# DDP 专用 DataLoader（要带 DistributedSampler）
def create_dataloader(dataset, batch_size, num_workers, world_size, rank, shuffle=True):
    """
    创建带 DistributedSampler 的 DataLoader。

    dataset: 原始数据集
    world_size: GPU 数量（即并行进程数）
    rank: 当前进程 ID
    DistributedSampler 会自动让不同 GPU 读取不同数据，避免重复。
    """

    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,   # 参与训练的 GPU 数
        rank=rank,                 # 本 GPU 对应的进程 ID
        shuffle=shuffle            # 每 epoch 打乱
    )

    # 关键：不使用 shuffle=True，而由 sampler 控制随机性
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,           # 必须加 sampler，否则每卡读取同样的数据
        num_workers=num_workers,
        persistent_workers=True
    )

    return loader, sampler

# 将模型包装成分布式 DDP
def  wrap_ddp_model(model, local_rank):
    """
    将模型放到对应 GPU,并用 DistributedDataParallel 包装。
    local_rank: 当前进程对应的 GPU ID
    """
    # 先把模型放到当前 GPU
    model = model.to(local_rank)
    # 再包成 DDP，device_ids 必须是列表
    return DDP(module=model, device_ids=[local_rank], output_device=local_rank)

# 保存模型（仅 rank 0 进程保存一次）
def model_save(model, path, rank):
    """
    DDP 模型需要保存 model.module
    只有 rank 0(主进程)才能保存文件。
    """
    if rank == 0:
        torch.save(model.state_dict(), path)