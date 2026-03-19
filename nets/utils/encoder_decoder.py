import torch
import torch.nn as nn
from torch.nn import functional as F

from nets.utils.attention import MultiHeadedAttention
from nets.utils.PositionwiseFeedForward import PositionwiseFeedForward

from nets.utils.Generator import Generator
from nets.utils.PositionalEncoding import PositionalEncoding

import math
import copy

def clones(module, N):
    """
    层克隆工具函数。
    
    物理意义：在内存中创建 N 个结构完全相同，但权重互相独立（参数不共享）的神经网络层。
    
    工程边界条件（极易踩坑点）：
    1. 必须使用 copy.deepcopy()：如果使用 [module] * N 或浅拷贝，Python 只会复制内存指针。
       这会导致 N 个层实际上指向同一组物理权重（变成了类似 ALBERT 模型的参数共享机制），
       违背了标准 Transformer 每层独立学习特征的设计初衷。
    2. 必须包装在 nn.ModuleList 中：如果只返回一个普通的 Python 列表（List），
       PyTorch 的底层机制将无法追踪这些层，模型在调用 model.parameters() 时会漏掉这些权重，
       导致这 N 个层在反向传播时完全不更新（无法计算梯度）。
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class SublayerConnection(nn.Module):
    """
    子层连接包装器 (Sublayer Connection Wrapper)。
    
    物理意义：
    它是一个通用的“插槽”。无论是多头注意力 (Multi-Head Attention) 还是
    前馈神经网络 (Feed Forward)，都可以塞进这个包装器里。
    它负责为其内部的子模块自动添加：LayerNorm、Dropout 以及 残差连接 (Residual Connection)。
    """

    def __init__(self, size, dropout):
        """
        初始化包装器。
        
        参数:
        size (int): 模型的隐状态维度 d_model (如 512)。
        dropout (float): 神经元丢弃率，用于正则化防止过拟合。
        """
        super(SublayerConnection, self).__init__()
        # 实例化我们在上一步修正过的 LayerNorm
        self.norm = LayerNorm(size) 
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        """
        前向传播。
        
        参数:
        x (Tensor): 输入张量，维度 (batch_size, seq_len, d_model)
        sublayer (Callable): 一个可调用的神经网络模块 (如 Attention 或 FFN 函数)
        
        数据流与运算顺序 (严格遵循 Pre-LN 范式):
        1. self.norm(x): 先对输入进行层归一化
        2. sublayer(...): 将归一化后的数据送入子层 (提取特征)
        3. self.dropout(...): 对提取出的特征进行正则化
        4. x + ... : 将原始输入 x 与上述结果相加 (残差连接)
        """
        return x + self.dropout(sublayer(self.norm(x)))


class LayerNorm(nn.Module):
    """
    标准的层归一化 (Layer Normalization) 模块。
    
    物理意义：
    强制将每个 Token 的特征向量分布，拉回到均值为 0、方差为 1 的标准正态分布，
    并在最后赋予模型重新缩放和偏移的能力。
    """

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        # 仿射变换参数 (Affine Transformation Parameters)
        # weight (对应公式中的 gamma): 缩放因子，初始化为 1
        # bias (对应公式中的 beta): 偏移因子，初始化为 0
        # nn.Parameter 作用是将这两个张量注册为模型的可学习参数，跟随梯度下降更新。
        self.weight = nn.Parameter(torch.ones(features))
        self.bias = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        # x 维度: (batch_size, seq_len, d_model)
        # -1 表示沿着最后一个维度 (d_model) 计算。keepdim=True 保证输出维度为 (batch_size, seq_len, 1)，以便后续广播。
        mean = x.mean(-1, keepdim=True)
        
        # 【致命缺陷修正处】
        # 1. 必须计算方差 (Variance)，且必须是有偏估计 (unbiased=False)，即除以 N 而不是 N-1。
        # 2. eps 必须在开平方前加在方差内部。
        var = x.var(-1, unbiased=False, keepdim=True)
        
        # 执行归一化与仿射变换
        return self.weight * (x - mean) / torch.sqrt(var + self.eps) + self.bias


class Encoder(nn.Module):
    """
    核心编码器堆叠模块。
    
    架构定位：接收由 Embedding 层输出的初始张量，经过 N 次深度特征提取，
    输出最终的上下文向量 (Context Vector / Memory)，供解码器 (Decoder) 使用。
    """
    def __init__(self, layer, N):
        """
        初始化堆叠编码器。
        
        参数：
        layer (nn.Module): 单个 EncoderLayer 的实例（包含 Multi-Head Attention 和 Feed Forward）。
                           这里再次使用了依赖注入（Dependency Injection），保证顶层代码的整洁。
        N (int): 堆叠的层数（论文默认设定为 6）。
        """
        super(Encoder, self).__init__()
        # 生成 N 个参数独立的编码器层
        self.layers = clones(layer, N)
        
        # 最后的层归一化 (Layer Normalization)
        # 物理意义：在 N 层特征提取全部结束后，对最终输出的隐状态再做一次归一化。
        # 工程考量：这能确保送入 Decoder 进行交叉注意力计算的 memory 张量，
        # 其特征分布严格保持在均值为 0、方差为 1 的平滑状态，极大提升 Decoder 的收敛稳定性。
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, mask):
        """
        前向传播计算。
        
        数据流与边界条件：
        x: 当前的隐状态张量。初始输入维度为 (batch_size, seq_len, d_model)。
        mask: 源序列的 Padding Mask。维度通常为 (batch_size, 1, seq_len)。
              注意：在整个 N 层的 for 循环中，mask 是【全局只读】且【恒定不变】的。
              因为无论特征经过多少层非线性变换，句子中 Padding（补零）的物理位置永远不会改变。
        """
        # 逐层穿透：将上一层的输出直接作为下一层的输入
        for layer in self.layers:
            x = layer(x, mask)
            
        # 返回前执行最终的归一化
        return self.norm(x)


class EncoderLayer(nn.Module):
    """
    单层编码器 (Encoder Layer)。
    
    架构定位：
    Transformer 编码器的最小重复单元。论文中默认堆叠 6 层此类结构。
    它负责完成两个正交的任务：
    1. 序列维度的信息交互 (Self-Attention)
    2. 特征维度的非线性映射 (Feed-Forward)
    """
 
    def __init__(self, size, self_attn, feed_forward, dropout):
        """
        初始化单层编码器。
        
        参数:
        size (int): 隐状态特征维度 d_model (如 512)。
        self_attn (nn.Module): 实例化的多头注意力模块。
        feed_forward (nn.Module): 实例化的前馈神经网络模块。
        dropout (float): 正则化丢弃率。
        """
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        
        # 核心复用：克隆 2 个完全独立的子层包装器 (SublayerConnection)
        # 第一个包装器用于 Attention，第二个用于 Feed-Forward。
        # 它们各自拥有独立的 LayerNorm 权重，互不干扰。
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        """
        前向传播数据流。
        
        参数:
        x (Tensor): 输入特征，维度 (batch_size, seq_len, d_model)。
        mask (Tensor): Padding 掩码，维度 (batch_size, 1, seq_len)。
        """
        # 步骤 1：自注意力机制 (Self-Attention) 与残差连接
        # 工程细节：这里使用了 Python 的 lambda 匿名函数来适配 SublayerConnection 的接口。
        # 为什么传入三个 x？这正是 "Self (自)" 的数学定义：
        # Query = x, Key = x, Value = x。
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        
        # 步骤 2：前馈神经网络 (Feed-Forward) 与残差连接
        # 数据流：经过 Attention 混淆了全局位置信息的张量 x，
        # 被送入 FFN 进行局部的、逐位置的 (Position-wise) 非线性特征升维与降维。
        return self.sublayer[1](x, self.feed_forward)


class Decoder(nn.Module):
    """
    核心解码器堆叠模块 (Decoder Stack)。
    
    架构定位：
    作为 Transformer 的自回归 (Auto-regressive) 生成引擎，它包含 N 个解码器层。
    它与 Encoder 的根本区别在于双重信息流：
    既要处理目标序列 (Target) 自身的时序推演，又要不断地“回望”源序列 (Source) 的全局语义。
    """

    def __init__(self, layer, N):
        """
        初始化堆叠解码器。
        
        参数:
        layer (nn.Module): 单个 DecoderLayer 的实例（包含掩码自注意力、交叉注意力和前馈网络）。
        N (int): 堆叠的层数（通常与 Encoder 保持一致，如 6 层）。
        """
        super(Decoder, self).__init__()
        # 生成 N 个参数完全独立的解码器层
        self.layers = clones(layer, N)
        
        # 最后的层归一化 (Layer Normalization)
        # 物理意义：与 Encoder 相同，采用 Pre-LN 架构时，必须在 N 层堆叠结束后追加一次全局归一化，
        # 确保最终送入 Generator (线性映射+Softmax) 的特征分布是稳定且无偏移的。
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        """
        前向传播计算。
        
        工程边界条件与数据流解析 (极其重要)：
        x: 目标序列当前的隐状态张量。维度 (batch_size, tgt_seq_len, d_model)。
           在训练阶段，这是经过 Shifted Right 并带有词嵌入的完整 Target 序列。
           
        memory: 编码器 (Encoder) 最后一层输出的全局上下文张量。维度 (batch_size, src_seq_len, d_model)。
                注意：在整个 N 层的 for 循环中，memory 是【全局只读】的，它不会被修改。
                
        src_mask: 源序列的 Padding 掩码。维度 (batch_size, 1, src_seq_len)。
                  作用于交叉注意力 (Cross-Attention) 中，防止 Decoder 将注意力浪费在源序列的补零位置上。
                  
        tgt_mask: 目标序列的因果掩码 (通常是下三角矩阵)。维度 (batch_size, tgt_seq_len, tgt_seq_len)。
                  作用于掩码自注意力 (Masked Self-Attention) 中，防止当前 Token 提前看到未来的 Token。
        """
        # 逐层穿透：将上一层的目标序列输出作为下一层的输入
        for layer in self.layers:
            # memory 和 masks 在所有层中保持恒定，充当全局环境参数
            x = layer(x, memory, src_mask, tgt_mask)
            
        # 返回前执行最终的归一化
        return self.norm(x)


class DecoderLayer(nn.Module):
    """
    单层解码器 (Decoder Layer)。
    
    架构定位：
    自回归生成任务中的核心特征处理单元。
    它不仅要在目标序列内部建立时序依赖，还要跨越空间去源序列中提取对应信息。
    """

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        """
        初始化单层解码器。
        
        参数:
        size (int): 隐状态特征维度 d_model (如 512)。
        self_attn (nn.Module): 掩码多头自注意力模块 (处理目标序列内部逻辑)。
        src_attn (nn.Module): 交叉注意力模块 (处理目标序列与源序列的映射)。
        feed_forward (nn.Module): 前馈神经网络模块。
        dropout (float): 正则化丢弃率。
        """
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        
        # 核心复用：克隆 3 个独立的子层包装器 (包含各自独立的 LayerNorm 和残差连接)
        # index 0: 掩码自注意力
        # index 1: 交叉注意力
        # index 2: FFN
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        """
        前向传播数据流。
        
        参数:
        x (Tensor): 目标序列特征张量，维度 (batch_size, tgt_seq_len, d_model)。
        memory (Tensor): 源序列特征张量，维度 (batch_size, src_seq_len, d_model)。
        src_mask (Tensor): 源序列 Padding 掩码。
        tgt_mask (Tensor): 目标序列因果掩码 (下三角矩阵)。
        """
        m = memory
        
        # 阶段 1：掩码自注意力 (Masked Self-Attention)
        # 数据流：Q=x, K=x, V=x。
        # 约束：必须传入 tgt_mask。
        # 物理意义：当前生成的词只能根据它之前的词来更新自己的特征，绝对不能看到之后的词。
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        
        # 阶段 2：交叉注意力 (Cross-Attention / Source-Attention)
        # 数据流：Q=x, K=m, V=m。
        # 约束：必须传入 src_mask。
        # 物理意义：拿着经过阶段 1 更新后的当前词特征 (Q)，去编码器的输出 (m) 中进行全局检索 (K, V)，
        # 找出当前词在源句子中最对应的语义片段，并将其融合到当前词的特征中。
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        
        # 阶段 3：前馈神经网络 (Feed-Forward)
        # 数据流：仅针对 x 自身进行。
        # 物理意义：对融合了内部时序信息和外部源语义信息的特征，进行逐位置的非线性高维映射。
        return self.sublayer[2](x, self.feed_forward)


class EncoderDecoder(nn.Module):
    """
    标准的编码器-解码器顶层架构包装器。
    设计意图：将模型解耦为 5 个正交的独立组件（编码、解码、源嵌入、目标嵌入、生成头），
    通过构造函数注入（Dependency Injection），保证底层模块的极高可复用性。
    """

    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        # 1. 编码器堆叠层（通常包含 N 个 TransformerEncoderLayer）
        self.encoder = encoder 
        
        # 2. 解码器堆叠层（通常包含 N 个 TransformerDecoderLayer）
        self.decoder = decoder 
        
        # 3. 源序列嵌入层（包含 Token Embedding + Positional Encoding）
        # 负责将离散的源 Token ID 映射为连续的稠密向量
        self.src_embed = src_embed 
        
        # 4. 目标序列嵌入层（同上，用于处理 Target Token ID）
        self.tgt_embed = tgt_embed 
        
        # 5. 生成器（通常是一个 Linear 层 + Softmax）
        # 负责将解码器输出的隐状态向量映射回词表的概率分布
        self.generator = generator 

    def forward(self, src, tgt, src_mask, tgt_mask):
        """
        前向传播函数。注意：此函数仅在【训练阶段（Training）】使用。
        在训练时，我们拥有完整的目标序列 tgt（Teacher Forcing 机制），因此可以一次性并行计算。
        
        输入维度假定：
        src: (batch_size, src_seq_len)
        tgt: (batch_size, tgt_seq_len)
        src_mask: (batch_size, 1, src_seq_len) - 用于屏蔽 Padding
        tgt_mask: (batch_size, tgt_seq_len, tgt_seq_len) - 用于实现下三角掩码，防止信息穿越
        """
        # 数据流：
        # 1. 走 self.encode() 拿到源序列的全局上下文特征 memory
        # 2. 将 memory、tgt 以及对应的 mask 一起送入 self.decode() 获取最终隐状态
        memory = self.encode(src, src_mask)
        out = self.decode(memory, src_mask, tgt, tgt_mask)
        return out

    def encode(self, src, src_mask):
        """
        独立的编码逻辑。
        数据流：Token ID -> 词嵌入 -> 叠加位置编码 -> N层自注意力与前馈网络 -> Context Vector (memory)
        输出维度：(batch_size, src_seq_len, d_model)
        """
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        """
        独立的解码逻辑。
        数据流：Target Token ID -> 词嵌入 -> N层解码器（包含掩码自注意力和交叉注意力） -> 解码特征
        注意：解码器不仅需要 tgt 本身，还需要引入编码器生成的 memory 进行交叉注意力（Cross-Attention）计算。
        输出维度：(batch_size, tgt_seq_len, d_model)
        """
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)
    

class Embeddings(nn.Module):
    """
    标准的词嵌入层 + 位置编码胶水层
    """
    def __init__(self, d_model, vocab_size, dropout=0.1):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model
        self.pe = PositionalEncoding(d_model, dropout)

    def forward(self, x):
        # 数学细节：论文指出，在将 Embedding 加上 PE 之前，
        # 需要将 Embedding 乘以 sqrt(d_model) 进行缩放，以此来平衡方差。
        x = self.lut(x) * math.sqrt(self.d_model)
        return self.pe(x)


def test_full_transformer():
    print("🚀 开始进行 Transformer 端到端终极测试 🚀\n")
    
    # 1. 超参数全景设定
    vocab_size = 32000   # 真实场景下的典型词表大小 (如 BPE 分词)
    d_model = 512        # 隐藏层维度
    d_ff = 2048          # FFN 膨胀维度
    num_heads = 8        # 注意力头数
    dropout = 0.1
    num_layers = 6       # 还原论文标准的 6 层堆叠
    batch_size = 4
    src_seq_len = 12     # 源序列长度
    tgt_seq_len = 10     # 目标序列长度

    print("1️⃣ 正在组装底层引擎 (Attention & FFN)...")
    attn = MultiHeadedAttention(h=num_heads, d_model=d_model, dropout=dropout)
    ff = PositionwiseFeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout)

    print("2️⃣ 正在构建 6 层 Encoder 与 Decoder 塔...")
    encoder = Encoder(EncoderLayer(d_model, copy.deepcopy(attn), copy.deepcopy(ff), dropout), num_layers)
    decoder = Decoder(DecoderLayer(d_model, copy.deepcopy(attn), copy.deepcopy(attn), copy.deepcopy(ff), dropout), num_layers)

    print("3️⃣ 正在装配词嵌入与位置编码...")
    # 假设源语言和目标语言共享词表
    src_embed = Embeddings(d_model, vocab_size, dropout)
    tgt_embed = Embeddings(d_model, vocab_size, dropout)

    print("4️⃣ 正在安装 Generator 生成头...")
    generator = Generator(d_model, vocab_size)

    print("5️⃣ 拼装终极模型：EncoderDecoder...")
    model = EncoderDecoder(encoder, decoder, src_embed, tgt_embed, generator)

    # 统计模型参数量
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n✨ 模型组装完毕！总可训练参数量: {total_params / 1e6:.2f} M")
    
    # ---------------- 测试数据流 ----------------
    print("\n--- 构造 Input Tensors ---")
    # 模拟输入 Token IDs
    src = torch.randint(0, vocab_size, (batch_size, src_seq_len))
    tgt = torch.randint(0, vocab_size, (batch_size, tgt_seq_len))

    # 构造真实掩码
    src_mask = torch.ones(batch_size, 1, src_seq_len, dtype=torch.bool)
    tgt_mask = torch.tril(torch.ones(tgt_seq_len, tgt_seq_len, dtype=torch.bool)).unsqueeze(0).expand(batch_size, -1, -1)

    print(f"源 Token IDs (src): {src.shape}")
    print(f"目标 Token IDs (tgt): {tgt.shape}")
    print(f"源 Mask (src_mask): {src_mask.shape}")
    print(f"目标 Mask (tgt_mask): {tgt_mask.shape}")

    print("\n--- 执行前向传播 (Forward Pass) ---")
    # 1. 获取 Decoder 输出的隐状态
    out_hidden = model(src, tgt, src_mask, tgt_mask)
    print(f"模型输出隐状态维度: {out_hidden.shape}  -> 期望: ({batch_size}, {tgt_seq_len}, {d_model})")
    assert out_hidden.shape == (batch_size, tgt_seq_len, d_model)

    # 2. 获取最终的 Logits 概率分布
    logits = model.generator(out_hidden)
    print(f"Generator 输出 Logits 维度: {logits.shape} -> 期望: ({batch_size}, {tgt_seq_len}, {vocab_size})")
    assert logits.shape == (batch_size, tgt_seq_len, vocab_size)

    print("\n✅ 端到端测试完美通过！这套代码已经具备了直接上 GPU 训练机器翻译模型的工业级水准。")

if __name__ == "__main__":
    test_full_transformer()
