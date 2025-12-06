import torch
import torch.distributed as dist
import os
import random
import numpy as np

# 从训练代码中导入所需的配置和工具
from config import * 
from dataset import CIFAR10Dataset
from distribute import setup_ddp, cleanup_ddp, create_dataloader, wrap_ddp_model
from vit import ViT 
from torch.nn.utils.clip_grad import clip_grad_norm_ # 即使推理不用，也导入以保持环境一致

# ---------------------------- DDP 初始化 ----------------------------
locals_rank, rank, world_size = setup_ddp()
device = torch.device(f'cuda:{locals_rank}')

if rank == 0:
    print("--------------------------------------------------")
    print(f"开始推理阶段,DDP World Size: {world_size}")
    print("--------------------------------------------------")


# ------------------------------ 数据加载 ------------------------------
# 使用验证集或测试集进行推理
dataset = CIFAR10Dataset()
val_dataset = dataset.prepare_data(rank, is_train=False)

# 创建 DataLoader (推理时 shuffle=False)
inference_dataloader, _ = create_dataloader(
    val_dataset,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    world_size=world_size,
    rank=rank,
    shuffle=False
)

# ----------------------------- 模型初始化与加载 -----------------------------
model = ViT(emb_size=EMB_SIZE).to(device) # 实例化模型

# 仅 rank 0 负责加载权重并广播给其他进程
checkpoint_path = 'model.pth'
if not os.path.exists(checkpoint_path) and rank == 0:
    print(f"错误：未找到模型权重文件 '{checkpoint_path}'。请确保训练后已保存。")

if os.path.exists(checkpoint_path):
    if rank == 0:
        print(f"Rank {rank}: 正在加载模型权重...")
        # 加载时使用 map_location='cpu' 可以节省 GPU 内存
        state_dict = torch.load(checkpoint_path, map_location='cpu') 
        model.load_state_dict(state_dict)
    
    # 使用 barrier 确保所有进程等待 rank 0 完成加载
    if dist.is_initialized():
        dist.barrier()
        
# 包装 DDP 模型 (即使是推理，也需要 DDP 包装来确保一致性)
model = wrap_ddp_model(model, device)

criterion = torch.nn.CrossEntropyLoss()

# ----------------------------- 推理/评估函数 (DDP 同步) -----------------------------
@torch.no_grad()
def infer_and_evaluate(model, dataloader, device):
    """
    在分布式环境下执行推理并聚合评估指标。
    """
    model.eval()
    
    # 使用 float64 以避免在累加大量数据时出现精度问题
    total_correct = 0.0
    total_samples = 0.0
    total_loss_sum = 0.0 
    
    # 用于收集所有预测结果和真实标签 (可选)
    all_preds = []
    all_labels = []

    for imgs, labels in dataloader:
        imgs, labels = imgs.to(device), labels.to(device)
        
        # 前向传播
        logits = model(imgs)
        preds = logits.argmax(dim=1)
        
        # 计算损失
        loss = criterion(logits, labels)
        
        # 累计当前进程的指标
        total_loss_sum += loss.item() * labels.size(0)
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)

        # 收集预测和标签 (在 CPU 上收集以避免 GPU 内存爆炸)
        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())

    # ============== DDP 同步区域 (聚合全局指标) ==============
    if dist.is_initialized():
        # 将局部统计量转换为张量
        correct_tensor = torch.tensor(total_correct, dtype=torch.float64).to(device)
        samples_tensor = torch.tensor(total_samples, dtype=torch.float64).to(device)
        loss_tensor = torch.tensor(total_loss_sum, dtype=torch.float64).to(device)

        # 聚合（求和）
        dist.all_reduce(correct_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(samples_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        
        # 提取全局聚合值
        global_correct = correct_tensor.item()
        global_samples = samples_tensor.item()
        global_loss_sum = loss_tensor.item()

        # 收集完整的预测和标签需要使用 gather (这里只是示例，实际收集逻辑更复杂，通常仅在 rank 0 上进行)
        # 如果需要完整的预测结果，需要使用 dist.gather 或 dist.all_gather
        
    else:
        # 非 DDP 环境下直接使用局部值
        global_correct = total_correct
        global_samples = total_samples
        global_loss_sum = total_loss_sum
    # ==========================================

    # 避免除以零
    if global_samples == 0:
         return 0.0, 0.0, None, None

    avg_loss = global_loss_sum / global_samples
    accuracy = global_correct / global_samples
    
    # 将所有进程的预测和标签结果合并成一个张量 (仅在 rank 0 上有意义)
    final_preds = torch.cat(all_preds, dim=0)
    final_labels = torch.cat(all_labels, dim=0)

    return accuracy, avg_loss, final_preds, final_labels

# ----------------------------- 执行推理 -----------------------------

# 确保在开始评估前，所有进程都已完成模型加载
if dist.is_initialized():
    dist.barrier()

# 执行推理
global_accuracy, global_avg_loss, all_preds, all_labels = infer_and_evaluate(model, inference_dataloader, device)

# ----------------------------- 结果报告 -----------------------------
if rank == 0:
    print("\n================ 最终推理结果 ================")
    print(f"总样本数 (Total Samples): {len(val_dataset)}")
    print(f"全局平均损失 (Global Average Loss): {global_avg_loss:.4f}")
    print(f"全局准确率 (Global Accuracy): {global_accuracy:.4f}")
    print("================================================")
    
    # 额外：如果您需要将预测结果保存到文件，可以在这里进行
    # 例如：np.save('predictions.npy', all_preds.numpy())
    
# ----------------------------- DDP 清理 -----------------------------
cleanup_ddp()