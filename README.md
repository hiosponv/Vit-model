
# CIFAR-10 ViT 分布式训练

本项目使用 **PyTorch** 实现 **Vision Transformer (ViT)** 在 **CIFAR-10** 数据集上的训练，支持 **多 GPU 分布式训练 (DDP)**，并集成数据加载、日志记录和模型保存功能。

---

## 📦 功能
- 基于 CIFAR-10 的 ViT 模型。
- PyTorch **DDP** 分布式训练，支持多 GPU 加速。
- **CosineAnnealingLR** 学习率调度。
- 梯度裁剪（Gradient Clipping）保证训练稳定。
- 使用 **SwanLab** 进行实验日志记录（可选）。
- 模型检查点保存与恢复。

---

## 🖼 数据集
使用 **CIFAR-10**：
- 训练集：50,000 张图片
- 测试集：10,000 张图片
- 将图片统一缩放到 **224×224**，适配 ViT 输入。
- 图像归一化参数：
```python
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
```

---

## ⚙️ 环境
- Python >= 3.9
- PyTorch >= 2.0
- torchvision
- CUDA GPU（支持 DDP）
- SwanLab（可选，用于日志记录）

---

## 🚀 训练
<img width="2085" height="767" alt="image" src="https://github.com/user-attachments/assets/e3a7a55c-41b0-479f-bf37-1f5b6c1761d5" />

<img width="2084" height="850" alt="image" src="https://github.com/user-attachments/assets/fbc6cb49-104f-4e55-aa99-ec4623886bba" />

<img width="2081" height="844" alt="image" src="https://github.com/user-attachments/assets/6bf8f737-6b3a-4ce1-be2a-6425ee36edc6" />

## 推理
```
Rank 0 loaded 10000 samples.
Rank 0: 正在加载模型权重...
Rank 2 loaded 10000 samples.
Rank 3 loaded 10000 samples.
Rank 1 loaded 10000 samples.
```

```
================ 最终推理结果 ================
总样本数 (Total Samples): 10000
全局平均损失 (Global Average Loss): 0.7909
全局准确率 (Global Accuracy): 0.7300
================================================
```

### 1. 多 GPU 启动
使用 `torchrun`：
```bash
torchrun --nproc_per_node=NUM_GPUS train.py
```
将 `NUM_GPUS` 替换为实际使用的 GPU 数量。

### 2. 超参数配置
在 `config.py` 中：
- `EPOCHES = 50`
- `BATCH_SIZE = 256`
- `LEARNING_RATE = 1e-3`
- `EMB_SIZE = 768`
- `NUM_HEADS = 12`
- `NUM_LAYERS = 12`
- Patch 大小：16

### 3. 学习率调度器
使用 **CosineAnnealingLR**：
```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHES)
```
在每个 epoch 结束后调用：
```python
scheduler.step()
```

---

## 🧠 模型结构（提供官方和自己实现的）
- **Patch Embedding**：通过 `Conv2d` 将图片划分 patch 并映射到 embedding。
- **CLS Token** + 位置编码。
- **Transformer 编码器** (`nn.TransformerEncoder`)。
- **分类头**：仅使用 [CLS] token 输出。

---

## 💾 模型保存
- 每个 epoch 保存一次（仅 rank 0）：
```python
model_save(model.module if hasattr(model, "module") else model, '.model.pth', rank)
```
- 最新模型重命名为 `model.pth`。

---

## 📊 验证
- `validate()` 在 **无梯度**下运行。
- 支持 DDP 全局聚合。
- 输出 **平均损失** 和 **准确率**。

---

## ⚡ 使用提示
- CIFAR-10 数据量较小，ViT-Base 容易过拟合，可调整：
  - 减小 embedding 尺寸 (`emb_size=256`)
  - 减少层数 (`num_layers=6`)
  - 减少注意力头 (`num_heads=8`)
- 梯度裁剪保证训练稳定：

```python
clip_grad_norm_(model.parameters(), max_norm=1.0)
```

---

## 📜 参考资料
- [Vision Transformer (ViT)](https://arxiv.org/abs/2010.11929)
- [PyTorch Distributed Data Parallel](https://pytorch.org/docs/stable/notes/ddp.html)
- [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)
- https://github.com/owenliang/mnist-vit
- https://www.bilibili.com/video/BV1fH4y1H7mV/?spm_id_from=333.1391.0.0




本项目仅用于科研与学习。


