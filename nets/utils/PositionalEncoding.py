import torch
import torch.nn as nn

import math

class PositionalEncoding(nn.Module):
    """
    注入绝对位置信息的正弦波编码器。
    """
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 预先在内存中分配好全零矩阵
        pe = torch.zeros(max_len, d_model)
        
        # position 维度: (max_len, 1)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        
        # div_term 维度: (d_model/2,)
        # 利用对数恒等式保证 fp32/fp16 精度下的数值稳定性
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        )
        
        # 偶数维度填充正弦，奇数维度填充余弦
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # 扩充 batch 维度并注册为不可训练的 buffer
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # 运行时直接截取序列长度并相加，无需操作 autograd 图
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)