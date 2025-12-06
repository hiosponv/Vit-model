import torch 
import torch.nn as nn
import torch.nn.functional as F
from config import *
from ViTTransformerEncoder import ViTTransformerEncoder


class ViT(nn.Module):
    def __init__(self, emb_size) -> None:
        super().__init__()
        self.emb_size = emb_size # (768)
        self.patch_size = PATCH_SIZE
        self.patch_count = IMAGE_SIZE // self.patch_size  # 14

        # 图片转patch(其实和线性层效果一样，使用卷积可能更能抽取特征)
        self.conv = nn.Conv2d(in_channels=IN_CHANNELS, out_channels=self.emb_size,
                              kernel_size=self.patch_size, padding=0, stride=self.patch_size) # (batch_size,channel=256,width=14,height=14)
        # 单个patch做emb
        # self.patch_emb = nn.Linear(in_features=self.patch_size**2, out_features=self.emb_size) # 转成768维, ( batch_size,seq_len=196,emb_size=768)
        # 分类头输入
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.emb_size))  # (1,1,emb_size)
        # position位置向量 (1,seq_len,emb_size)
        self.pos_emb = nn.Parameter(torch.zeros(1, self.patch_count**2+1, self.emb_size)) # (1,197,emb_size)
        # transformer编码器
        # self.transformer_enc = ViTTransformerEncoder(
        #     emb_size=emb_size,
        #     num_layers=NUM_LAYERS,
        #     num_heads=NUM_HEADS,
        #     mlp_ratio=4,
        #     dropout=0.1
        # )
        self.transformer_enc=nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=emb_size, nhead=NUM_HEADS, dim_feedforward=emb_size*4, 
                                                                              dropout=Droupout_RATE, batch_first=True),num_layers=NUM_LAYERS)   # transformer编码器
        # 分类线性层
        self.cls_linear = nn.Linear(in_features=self.emb_size, out_features=NUM_CLASSES)

    def forward(self, x):  # (batch_size, channel=3, width=224, height=224)
        # 1. Patch Projection: (B, C, H, W) -> (B, E, P_H, P_W)
        x = self.conv(x) # (batch_size, emb_size=768, width=14, height=14)
        
        # 2. Flatten: (B, E, P_H, P_W) -> (B, E, Seq_len)
        x = x.flatten(2) # (batch_size, emb_size=768, seq_len=196)
        
        # 3. Transpose: (B, E, Seq_len) -> (B, Seq_len, E)
        x = x.transpose(1, 2) # (batch_size, seq_len=196, emb_size=768)

        # 4. Add CLS Token
        cls_token = self.cls_token.expand(x.size(0), -1, -1) # (batch_size, 1, emb_size)
        x = torch.cat((cls_token, x), dim=1) # (batch_size, seq_len=197, emb_size)
        
        # 5. Add Position Embedding
        self.pos_emb = self.pos_emb.to(x.device)  # 确保位置向量在同一设备上
        x = self.pos_emb + x  # (batch_size, seq_len=197, emb_size=768)
        
        # 6. Transformer Encoder
        y = self.transformer_enc(x) 
        
        # 7. Classification Head: 只使用 [CLS] token 的输出
        # 输出 Logits (未经 Softmax)
        return self.cls_linear(y[:, 0, :])  # (batch_size, NUM_CLASSES)
