import torch
from config import *
import torch.distributed as dist
from dataset import CIFAR10Dataset
from distribute import setup_ddp, cleanup_ddp, create_dataloader, wrap_ddp_model, model_save
from vit import ViT
import torch.optim as optim 
import os
import swanlab
import random
import sys
from torch.nn.utils.clip_grad import clip_grad_norm_

locals_rank, rank, world_size = setup_ddp()
device = torch.device(f'cuda:{locals_rank}')

if rank == 0:
    swanlab.init(project_name='CIFAR-10-Vit-DDP', config={
        'EPOCHS': EPOCHES,
        'BATCH_SIZE': BATCH_SIZE,
        'LR': LEARNING_RATE
    })

# 创建分布式 DataLoader
dataset = CIFAR10Dataset()
train_dataset = dataset.prepare_data(rank, is_train=True)
val_dataset = dataset.prepare_data(rank, is_train=False)

dataloader, sampler = create_dataloader(
    train_dataset,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    world_size=world_size,
    rank=rank,
)

val_dataloader, _ = create_dataloader(
    val_dataset,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    world_size=world_size,
    rank=rank,
    shuffle=False
)

model = ViT(emb_size=EMB_SIZE).to(device)

# 仅在 rank 0 上加载权重，并使用 barrier 同步
checkpoint_path = 'model.pth'
if rank == 0 and os.path.exists(checkpoint_path):
    print(f"Loading model from {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    print(f"Model loaded from {checkpoint_path}")

# 使用 barrier 确保所有进程等待 rank 0 完成加载
if dist.is_initialized():
    dist.barrier()

if dist.is_initialized() and world_size > 1:
    # 仅在多 GPU 场景下转换
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

model = wrap_ddp_model(model, device)

optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHES)
criterion = torch.nn.CrossEntropyLoss()

@torch.no_grad()
def validate(model, dataloader, device):
    model.eval()
    
    # 使用 float64 以避免在累加大量数据时出现精度问题
    total_correct = 0.0
    total_samples = 0.0
    total_loss_sum = 0.0 
    
    for imgs, labels in dataloader:
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        preds = logits.argmax(dim=1)
        loss = criterion(logits, labels)
        
        # 累计当前进程的损失和样本数
        total_loss_sum += loss.item() * labels.size(0)
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)

    # ============== DDP 同步区域 ==============
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
    else:
        # 非 DDP 环境下直接使用局部值
        global_correct = total_correct
        global_samples = total_samples
        global_loss_sum = total_loss_sum

    avg_loss = global_loss_sum / global_samples
    accuracy = global_correct / global_samples
    return accuracy, avg_loss

def main():
    iter_count = 0
    for epoch in range(EPOCHES):
        model.train()
        sampler.set_epoch(epoch)
        
        for imgs, labels in dataloader:
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            # clip_grad_norm_(model.parameters(), max_norm=0.8)
            optimizer.step()
            
            scheduler.step()
            
            # 仅在第一次迭代时打印调试信息    
            if iter_count == 0 and rank == 0:
                with torch.no_grad():
                    print("=== Early debug info ===")
                    print("logits mean/std/min/max:", logits.mean().item(), logits.std().item(),
                          logits.min().item(), logits.max().item())
                    preds = logits.argmax(dim=1)
                    try:
                        print("preds bincount:", torch.bincount(preds.cpu()))
                        print("labels bincount:", torch.bincount(labels.cpu()))
                    except Exception as e:
                        print("bincount error:", e)
                    print("labels dtype/min/max:", labels.dtype, labels.min().item(), labels.max().item())
                    # 检查模型是否有梯度
                    grad_norm = 0.0
                    for p in model.parameters():
                        if p.grad is not None:
                            grad_norm += p.grad.detach().float().norm().item()
                    print("sum grad norms (some nonzero?)", grad_norm)
                    print("=========================")

            if iter_count % 100 == 0 and rank == 0:
                print(f'Epoch: {epoch}, Iteration: {iter_count}, Loss: {loss.item():.4f}')
                swanlab.log({'epoch': epoch, 'iteration': iter_count, 'loss': loss.item()})

            iter_count += 1

        # 验证
        val_accuracy, val_avg_loss = validate(model, val_dataloader, device)
        if rank == 0:
            print(f'Epoch: {epoch}, Average Loss: {val_avg_loss:.4f} Validation Accuracy: {val_accuracy:.4f}')
            swanlab.log({'epoch': epoch, 'val_accuracy': val_accuracy, 'val_avg_loss':val_avg_loss})
            model_save(model.module if hasattr(model, "module") else model, '.model.pth', rank)
            os.replace('.model.pth', 'model.pth')

    if rank == 0:
        swanlab.finish()  
    cleanup_ddp()

if __name__ == '__main__':
    # 确保只在 torchrun 启动的进程中运行 main
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        main()
    else:
        print("Please run this script using 'torchrun' or set the DDP environment variables.")
        sys.exit(1)
