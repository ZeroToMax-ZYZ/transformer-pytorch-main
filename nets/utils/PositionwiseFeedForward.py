import torch
import torch.nn as nn
from torch.nn import functional as F

import math
import copy


class PositionwiseFeedForward(nn.Module):
    """
    逐位置前馈网络。

    结构与论文一致：
        Linear(d_model -> d_ff) -> ReLU -> Dropout -> Linear(d_ff -> d_model)
    """

    def __init__(self, d_model, d_ff, dropout=0.1):
        """
        初始化 FFN 模块。
        
        参数:
        d_model (int): 模型的输入/输出基准维度 (例如 512)。
        d_ff (int): 隐藏层膨胀维度 (通常设为 d_model 的 4 倍，例如 2048)。
        dropout (float): 正则化丢弃率。
        """
        super(PositionwiseFeedForward, self).__init__()
        
        # 步骤 1：升维映射 (Dimensionality Expansion)
        # 物理意义：将特征从 512 维的低维流形，投影到 2048 维的高维空间，
        # 从而暴露出更多可能线性可分的特征组合。
        self.w_1 = nn.Linear(d_model, d_ff)
        
        # 步骤 2：降维映射 (Dimensionality Reduction)
        # 物理意义：将经过非线性激活后的高维特征，重新压缩回 512 维，
        # 保证输出维度与输入严格一致，以满足后续残差连接 (x + FFN(x)) 的严格要求。
        self.w_2 = nn.Linear(d_ff, d_model)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        输入 `x` 形状为 `(B, T, d_model)`，输出形状不变。
        """
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


def test_feed_forward():
    # 1. 设定超参数
    batch_size = 2
    seq_len = 5          # 句子长度（Token 数量）
    d_model = 512        # 模型的输入/输出基准维度
    d_ff = 2048          # 隐藏层膨胀维度 (通常是 d_model 的 4 倍)
    dropout = 0.1

    # 2. 实例化 FFN 模块
    print("正在初始化 PositionwiseFeedForward 模块...")
    ffn = PositionwiseFeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout)
    
    # 3. 构造随机输入的 Tensor (模拟经过 Attention 和 LayerNorm 后的特征)
    # 维度: (batch_size, seq_len, d_model)
    x = torch.randn(batch_size, seq_len, d_model)
    
    print(f"输入张量维度: {x.shape}")
    print("-" * 40)

    # 4. 执行前向传播
    output = ffn(x)

    # 5. 验证输出与中间维度
    print(f"输出张量维度: {output.shape}")
    
    # 检查维度是否符合预期：输出维度必须与输入维度完全一致
    assert output.shape == (batch_size, seq_len, d_model), "输出维度错误！必须与输入一致才能进行残差相加。"
    
    # 我们可以通过打印内部权重的 shape 来验证升降维逻辑
    print(f"w_1 (升维) 权重矩阵维度: {ffn.w_1.weight.shape}  -> 期望为 (2048, 512)")
    print(f"w_2 (降维) 权重矩阵维度: {ffn.w_2.weight.shape}  -> 期望为 (512, 2048)")
    
    print("\n测试通过！FFN 模块成功维持了张量的输入输出维度，可以完美接入残差连接。")

# 运行测试
if __name__ == "__main__":
    test_feed_forward()
