import torch
import torch.nn as nn

import math
import copy


def attention(query, key, value, mask=None, dropout=None):
    """
    计算缩放点积注意力。

    约定输入已经完成多头切分：
    - query: (B, H, T_q, d_k)
    - key:   (B, H, T_k, d_k)
    - value: (B, H, T_k, d_v)
    - mask:  可广播到 (B, H, T_q, T_k)
    """

    # 提取特征维度 d_k
    # size(-1) 取最后一个维度的长度，这是进行缩放的数学基准。
    d_k = query.size(-1) 
    
    # 步骤 1 & 2：计算点积并立刻缩放 (MatMul & Scale)
    # key.transpose(-2, -1) 将 key 的最后两个维度互换，变为 (..., d_k, seq_len_k)
    # torch.matmul 执行批量矩阵乘法，(..., seq_len_q, d_k) @ (..., d_k, seq_len_k)
    # 输出 scores 维度: (batch_size, num_heads, seq_len_q, seq_len_k)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    
    # 步骤 3：掩码注入 (Mask)
    if mask is not None:
        # AMP 下直接写死 -inf 容易引入数值问题，因此这里使用当前 dtype 的最小值。
        min_value = torch.finfo(scores.dtype).min
        scores = scores.masked_fill(mask == 0, min_value)
        
    # 步骤 4：概率归一化 (Softmax)
    # 沿着最后一个维度 (seq_len_k) 做 softmax。
    # 物理意义：对于 Query 中的每一个词，它分配给 Key 中所有词的注意力权重之和必须等于 1。
    # -1e9 经过 softmax 后 e^{-1e9} 严格等于 0。
    p_attn = scores.softmax(dim=-1)
    
    # 步骤 5：正则化 (Dropout)
    # 随机丢弃一部分注意力连接，防止模型过度依赖某几个特定的词频共现对，增强泛化性。
    if dropout is not None:
        p_attn = dropout(p_attn)
        
    # 步骤 6：特征加权融合 (MatMul with Value)
    # (..., seq_len_q, seq_len_k) @ (..., seq_len_k, d_v)
    # 输出维度: (batch_size, num_heads, seq_len_q, d_v)
    return torch.matmul(p_attn, value), p_attn

class MultiHeadedAttention(nn.Module):
    """
    多头注意力。

    负责：
    1. 线性投影得到 Q / K / V。
    2. 将 `d_model` 切成 `h` 个头。
    3. 调用底层 `attention` 完成注意力计算。
    4. 重新拼接各头并映射回 `d_model`。
    """

    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        # 边界条件检查：特征维度必须能被头数整除，否则无法均匀切分张量
        assert d_model % h == 0
        
        self.d_k = d_model // h 
        self.h = h 
        
        # 工程设计决断：为什么是 4 个大小为 (d_model, d_model) 的 Linear？
        # 前 3 个分别对应 W_q, W_k, W_v，第 4 个对应最终的输出融合矩阵 W_o。
        # 理论上，多头注意力要求有 h 个独立的 (d_model, d_k) 小矩阵。
        # 但在工程落地时，为了极致的 GPU 并行效率，我们将 h 个小矩阵合并成了一个大矩阵 (d_model, d_model)。
        self.linears = nn.ModuleList([copy.deepcopy(nn.Linear(d_model, d_model)) for _ in range(4)])
        self.attn = None 
        self.dropout = nn.Dropout(p=dropout) 

    def forward(self, query, key, value, mask=None):
        """
        前向传播。

        常见 mask 形状：
        - src self-attention / cross-attention: (B, 1, S)
        - tgt self-attention:                  (B, T, T)
        """
        if mask is not None:
            # 为 head 维补一个维度，让同一个 mask 广播到所有注意力头。
            mask = mask.unsqueeze(1)
            
        nbatches = query.size(0)

        # 1) 线性映射与多头切分 (Linear Projection & Head Splitting)
        query, key, value = [
            # 核心张量体操 (Tensor Gymnastics)：
            # x 经过 lin(x) 后维度为 (batch, seq_len, 512)
            # .view(...) 将其在物理内存视角上切分为 (batch, seq_len, 8, 64)
            # .transpose(1, 2) 将 seq_len 和 head 维度互换，变为 (batch, 8, seq_len, 64)
            lin(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for lin, x in zip(self.linears[:3], (query, key, value)) # 注意：这里切片了前3个linear
        ]
        
        # 2) 核心注意力计算 (Scaled Dot-Product Attention)
        # 此时送入 attention 的张量维度为 (batch, 8, seq_len, 64)
        # attention 算子内部的 matmul 会将前两维 (batch, 8) 视为独立的 Batch 空间，
        # 在底层 CUDA 算子中并行发起 batch * 8 个矩阵乘法。
        x, self.attn = attention(
            query, key, value, mask=mask, dropout=self.dropout
        ) 
        
        # 3) 多头重组与连续化内存 (Recombination & Memory Contiguity)
        x = (
            x.transpose(1, 2)
            .contiguous() # 【极其关键的内存操作，详见下文拆解】
            .view(nbatches, -1, self.h * self.d_k)
        )
        
        # 释放局部变量引用，提示 Python 垃圾回收机制 (GC) 尽早释放显存
        del query
        del key
        del value
        
        # 4) 最终线性投影 (Final Linear Projection)
        # 使用第 4 个 Linear 层 (W_o) 将拼接后的多头信息进行特征混合
        return self.linears[-1](x)
    

def test_multi_head_attention():
    # 1. 设定超参数
    batch_size = 2
    seq_len = 5          # 句子长度（Token 数量）
    d_model = 512        # 模型的隐藏层维度
    num_heads = 8        # 多头注意力的头数
    dropout = 0.1

    # 2. 实例化多头注意力模块
    print("正在初始化 MultiHeadedAttention 模块...")
    mha = MultiHeadedAttention(h=num_heads, d_model=d_model, dropout=dropout)
    
    # 3. 构造随机输入的 Tensor (模拟 Self-Attention，Q=K=V=X)
    # 维度: (batch_size, seq_len, d_model)
    x = torch.randn(batch_size, seq_len, d_model)
    
    # 4. 构造掩码 (Mask)
    # 假设我们想屏蔽掉 batch 0 中最后一个 token，以及 batch 1 中最后两个 token
    # 维度: (batch_size, 1, seq_len) 
    # 注意：这里的 1 是为了留出 head 的维度位置，配合代码里的 unsqueeze(1)
    mask = torch.tensor([
        [[1, 1, 1, 1, 0]],  # 0 表示被 mask 掉
        [[1, 1, 1, 0, 0]]
    ], dtype=torch.bool)    # 使用 bool 类型更符合现代 PyTorch 习惯

    print(f"输入张量维度: {x.shape}")
    print(f"掩码张量维度: {mask.shape}")
    print("-" * 40)

    # 5. 执行前向传播
    # 在自注意力机制中，Query, Key, Value 通常来自同一个输入 X
    output = mha(query=x, key=x, value=x, mask=mask)

    # 6. 验证输出
    print(f"输出张量维度: {output.shape}")
    print(f"内部注意力权重矩阵 (p_attn) 维度: {mha.attn.shape}")
    
    # 检查维度是否符合预期：输出维度应该与输入维度完全一致
    assert output.shape == (batch_size, seq_len, d_model), "输出维度错误！"
    assert mha.attn.shape == (batch_size, num_heads, seq_len, seq_len), "注意力权重矩阵维度错误！"
    
    print("\n测试通过！维度校验完全正确。")
    
    # 7. 打印部分注意力权重，验证 Mask 是否生效
    print("\n观察 Batch 1, Head 0 的注意力权重 (最后两列应该全为 0):")
    # 取消科学计数法打印，方便观察
    torch.set_printoptions(sci_mode=False, precision=4)
    print(mha.attn[1, 0, :, :])

# 运行测试
if __name__ == "__main__":
    test_multi_head_attention()
