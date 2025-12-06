import torch
import torch.nn as nn
from config import *
from TransformerEncoderLayer import MultiHeadAttention, FeedForwardBlock, EncoderBlock


class ViTTransformerEncoder(nn.Module):
    """
    完整的 Transformer 编码器 (替换 nn.TransformerEncoder)
    堆叠 NUM_LAYERS 个 EncoderBlock。
    """
    def __init__(self, emb_size: int, num_layers: int, num_heads: int, mlp_ratio: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        # 堆叠 NUM_LAYERS 个编码器层
        self.blocks = nn.ModuleList([
            EncoderBlock(
                emb_size=emb_size, 
                num_heads=num_heads, 
                mlp_ratio=mlp_ratio, 
                dropout=dropout
            )
            for _ in range(num_layers)
        ])
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x