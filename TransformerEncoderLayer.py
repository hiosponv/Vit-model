import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    """
    多头自注意力机制 (Multi-Head Self-Attention)
    负责计算序列中每个Token之间的关系。
    """
    def __init__(self, emb_size: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        # 确保嵌入维度能被头数整除
        self.head_dim = emb_size // num_heads
        assert emb_size % num_heads == 0, "嵌入维度必须能被头数整除"

        # QKV 投影层
        # 使用一个大层，然后拆分为 Q, K, V，比分别创建三个层更高效
        self.qkv_proj = nn.Linear(emb_size, emb_size * 3, bias=True)
        # 最终的输出投影层
        self.out_proj = nn.Linear(emb_size, emb_size)

        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5 # 缩放因子

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x 形状: (Batch_size, Seq_len, Emb_size) -> (B, L, D)

        # 1. QKV 投影和分割
        # qkv_proj(x) 形状: (B, L, D*3)
        # view 操作将其重塑为 (B, L, 3, N_heads, Head_dim)
        # permute 操作将其转置为 (3, B, N_heads, L, Head_dim)
        qkv = self.qkv_proj(x).reshape(
            x.shape[0], x.shape[1], 3, self.num_heads, self.head_dim
        ).permute(2, 0, 3, 1, 4)
        
        # 分离 Q, K, V。每个形状为 (B, N_heads, L, Head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # 2. 计算注意力权重
        # q @ k.transpose(-2, -1) 形状: (B, N_heads, L, L)
        attn = (q @ k.transpose(-2, -1)) * self.scale

        # 3. Softmax 归一化和 Dropout
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        # 4. 加权求和
        # attn @ v 形状: (B, N_heads, L, Head_dim)
        weighted_output = (attn @ v)

        # 5. 拼接多头并进行最终投影
        # 将多头结果重新拼接 (B, L, D)
        weighted_output = weighted_output.transpose(1, 2).reshape(x.shape[0], x.shape[1], self.emb_size)
        
        # 最终线性投影
        x = self.out_proj(weighted_output)
        x = self.proj_drop(x)
        return x


class FeedForwardBlock(nn.Module):
    """
    前馈网络模块 (MLP Head)
    在 MHA 之后对每个Token独立进行处理。
    """
    def __init__(self, emb_size: int, mlp_ratio: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        # 内部隐藏层维度，通常是嵌入维度的 4 倍
        self.hidden_dim = int(emb_size * mlp_ratio) 
        
        self.net = nn.Sequential(
            # 第一层：放大维度
            nn.Linear(emb_size, self.hidden_dim),
            nn.GELU(), # ViT 中通常使用 GELU 激活函数
            nn.Dropout(dropout),
            # 第二层：还原维度
            nn.Linear(self.hidden_dim, emb_size),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EncoderBlock(nn.Module):
    """
    单个 Transformer 编码器层 (替换 nn.TransformerEncoderLayer)
    遵循 LayerNorm -> MHA -> Add&Norm -> MLP -> Add&Norm 的结构。
    """
    def __init__(self, emb_size: int, num_heads: int, mlp_ratio: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        
        # 第一部分：MHA 和残差连接
        self.norm1 = nn.LayerNorm(emb_size)
        self.attn = MultiHeadAttention(emb_size, num_heads=num_heads, dropout=dropout)
        
        # 第二部分：MLP 和残差连接
        self.norm2 = nn.LayerNorm(emb_size)
        self.mlp = FeedForwardBlock(emb_size, mlp_ratio=mlp_ratio, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. MHA 子层 (LayerNorm -> MHA -> Residual)
        # 注意：这里采用了 Pre-Norm 结构，即先进行 LayerNorm
        x = x + self.attn(self.norm1(x)) 
        
        # 2. MLP 子层 (LayerNorm -> MLP -> Residual)
        x = x + self.mlp(self.norm2(x))
        return x