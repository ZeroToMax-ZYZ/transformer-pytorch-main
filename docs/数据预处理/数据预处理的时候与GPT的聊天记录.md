2017-Transformer.pdf

PDF

transformer\_project\_code.txt

文档

这是我当前的transformer代码，我期望使用论文的英德数据集，复现。 ================================================== File Path: .\\packed.py ================================================== import os def merge\_code\_for\_gemini(repo\_path, output\_file): # 想包含的文件后缀 allowed\_extensions = ('.py', '.yaml', '.md', '.txt', '.json') with open(output\_file, 'w', encoding='utf-8') as outfile: for root, dirs, files in os.walk(repo\_path): # 排除隐藏文件夹（比如 .git）和缓存文件夹 if '.git' in root or '\_\_pycache\_\_' in root or 'runs' in root: continue for file in files: if file.endswith(allowed\_extensions): filepath = os.path.join(root, file) outfile.write(f"\\n\\n{'='\*50}\\n") outfile.write(f"File Path: {filepath}\\n") outfile.write(f"{'='\*50}\\n") try: with open(filepath, 'r', encoding='utf-8') as f: outfile.write(f.read()) except Exception as e: outfile.write(f"Error reading file: {e}\\n") print(f"打包完成！文件已保存为: {output\_file}") merge\_code\_for\_gemini('.', 'transformer\_project\_code.txt') ================================================== File Path: .\\requirements.txt ================================================== ================================================== File Path: .\\transformer\_project\_code.txt ================================================== ================================================== File Path: .\\.vscode\\settings.json ================================================== { "python-envs.defaultEnvManager": "ms-python.python:system" } ================================================== File Path: .\\nets\\build\_transformer.py ================================================== import torch import torch.nn as nn from nets.utils.attention import MultiHeadedAttention from nets.utils.PositionwiseFeedForward import PositionwiseFeedForward from nets.utils.Generator import Generator from nets.utils.PositionalEncoding import PositionalEncoding from nets.utils.encoder\_decoder import Encoder, Decoder, EncoderLayer, DecoderLayer, EncoderDecoder, Embeddings import copy import math def make\_model(src\_vocab, tgt\_vocab, N=6, d\_model=512, d\_ff=2048, h=8, dropout=0.1): """ 构建完整 Transformer 模型的工厂函数。 参数: src\_vocab (int): 源语言词表大小。 tgt\_vocab (int): 目标语言词表大小。 N (int): Encoder 和 Decoder 的堆叠层数 (默认 6)。 d\_model (int): 模型的隐藏层特征维度 (默认 512)。 d\_ff (int): FFN 的中间膨胀维度 (默认 2048)。 h (int): 多头注意力的头数 (默认 8)。 dropout (float): 正则化丢弃率 (默认 0.1)。 """ # 核心工具：使用深拷贝生成结构相同但内存物理隔离的模块 c = copy.deepcopy # 1. 实例化基础算子 (注意：这里不需要再实例化 PositionalEncoding 了) attn = MultiHeadedAttention(h, d\_model, dropout) ff = PositionwiseFeedForward(d\_model, d\_ff, dropout) # 2. 组装端到端模型 (依赖注入) # 此时 src\_embed 和 tgt\_embed 直接使用你封装好的高级 Embeddings 类即可 model = EncoderDecoder( encoder=Encoder(EncoderLayer(d\_model, c(attn), c(ff), dropout), N), decoder=Decoder(DecoderLayer(d\_model, c(attn), c(attn), c(ff), dropout), N), src\_embed=Embeddings(d\_model, src\_vocab, dropout), tgt\_embed=Embeddings(d\_model, tgt\_vocab, dropout), generator=Generator(d\_model, tgt\_vocab) ) # 3. 极其关键的工程细节：Xavier / Glorot 参数初始化 # 物理意义：打破网络权重的对称性，保证前向传播时激活值的方差不至于过大或过小， # 防止深层 Transformer 出现梯度消失或梯度爆炸。 for p in model.parameters(): if p.dim() > 1: # 只对二维以上的权重矩阵（如 Linear 的 weight，Embedding 的 weight）做均匀分布初始化 nn.init.xavier\_uniform\_(p) return model def test\_factory\_method(): print("=== 开始测试工厂函数 make\_model ===") # 1. 设定测试环境超参数 src\_vocab\_size = 5000 tgt\_vocab\_size = 5000 batch\_size = 3 src\_len = 7 tgt\_len = 5 print("正在构建微型 Transformer 模型 (2层)...") # 为了测试速度和显存，我们生成一个 2 层的微型版本 model = make\_model( src\_vocab=src\_vocab\_size, tgt\_vocab=tgt\_vocab\_size, N=2, d\_model=256, d\_ff=1024, h=4, dropout=0.1 ) # 打印参数量，验证模型是否成功拼装 total\_params = sum(p.numel() for p in model.parameters() if p.requires\_grad) print(f"模型构建完成！总可训练参数量: {total\_params / 1e6:.2f} M") print("-" \* 40) # 2. 构造假数据 (模拟真实训练中的 Batch) src\_tokens = torch.randint(0, src\_vocab\_size, (batch\_size, src\_len)) tgt\_tokens = torch.randint(0, tgt\_vocab\_size, (batch\_size, tgt\_len)) # 构造简单的全局可见 Mask (源语言) 和 因果 Mask (目标语言) src\_mask = torch.ones(batch\_size, 1, src\_len, dtype=torch.bool) tgt\_mask = torch.tril(torch.ones(tgt\_len, tgt\_len, dtype=torch.bool)).unsqueeze(0).expand(batch\_size, -1, -1) # 3. 执行一次完整的前向传播 print("执行前向传播获取隐状态...") hidden\_states = model(src\_tokens, tgt\_tokens, src\_mask, tgt\_mask) print(f"Decoder 输出隐状态维度: {hidden\_states.shape} -> 期望: ({batch\_size}, {tgt\_len}, 256)") assert hidden\_states.shape == (batch\_size, tgt\_len, 256), "隐状态维度错误！" # 4. 执行 Generator 映射到词表 print("执行 Generator 投影...") logits = model.generator(hidden\_states) print(f"最终预测概率分布维度: {logits.shape} -> 期望: ({batch\_size}, {tgt\_len}, {tgt\_vocab\_size})") assert logits.shape == (batch\_size, tgt\_len, tgt\_vocab\_size), "预测概率维度错误！" print("\\n✅ 工厂函数完美打通！输入 -> 编码 -> 解码 -> 生成，全链路畅通无阻。") if \_\_name\_\_ == "\_\_main\_\_": test\_factory\_method() ================================================== File Path: .\\nets\\utils\\attention.py ================================================== import torch import torch.nn as nn import math import copy def attention(query, key, value, mask=None, dropout=None): """ 计算缩放点积注意力 (Scaled Dot-Product Attention)。 物理意义： 计算 Query 序列中每个 Token 对 Key 序列中所有 Token 的匹配度， 并以此权重对 Value 序列进行加权求和。 张量维度预设 (假设已完成多头切分)： query: (batch\_size, num\_heads, seq\_len\_q, d\_k) key: (batch\_size, num\_heads, seq\_len\_k, d\_k) value: (batch\_size, num\_heads, seq\_len\_k, d\_v) 注意：通常 d\_k == d\_v，且 seq\_len\_k == seq\_len\_v。 """ # 提取特征维度 d\_k # size(-1) 取最后一个维度的长度，这是进行缩放的数学基准。 d\_k = query.size(-1) # 步骤 1 & 2：计算点积并立刻缩放 (MatMul & Scale) # key.transpose(-2, -1) 将 key 的最后两个维度互换，变为 (..., d\_k, seq\_len\_k) # torch.matmul 执行批量矩阵乘法，(..., seq\_len\_q, d\_k) @ (..., d\_k, seq\_len\_k) # 输出 scores 维度: (batch\_size, num\_heads, seq\_len\_q, seq\_len\_k) scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d\_k) # 步骤 3：掩码注入 (Mask) if mask is not None: # masked\_fill 是一个 out-of-place 极速算子。 # 当 mask 为 0 (或 False) 时，将 scores 中的对应位置替换为 -1e9。 # 为什么是 -1e9 而不是 -inf？ # 在 FP16 混合精度训练中，-inf 经过某些算子可能会溢出为 NaN，-1e9 是一个工程上绝对安全的极小值。 scores = scores.masked\_fill(mask == 0, -1e9) # 步骤 4：概率归一化 (Softmax) # 沿着最后一个维度 (seq\_len\_k) 做 softmax。 # 物理意义：对于 Query 中的每一个词，它分配给 Key 中所有词的注意力权重之和必须等于 1。 # -1e9 经过 softmax 后 e^{-1e9} 严格等于 0。 p\_attn = scores.softmax(dim=-1) # 步骤 5：正则化 (Dropout) # 随机丢弃一部分注意力连接，防止模型过度依赖某几个特定的词频共现对，增强泛化性。 if dropout is not None: p\_attn = dropout(p\_attn) # 步骤 6：特征加权融合 (MatMul with Value) # (..., seq\_len\_q, seq\_len\_k) @ (..., seq\_len\_k, d\_v) # 输出维度: (batch\_size, num\_heads, seq\_len\_q, d\_v) return torch.matmul(p\_attn, value), p\_attn class MultiHeadedAttention(nn.Module): """ 多头注意力机制 (Multi-Head Attention)。 物理意义： 底层的 `attention` 算子本身是无状态的（没有可学习参数）。 这个类是它的参数化容器，负责维护 Q、K、V 的特征映射矩阵， 并将高维特征切分为多个低维子空间（Heads）以捕捉不同角度的语义，最后再融合。 """ def \_\_init\_\_(self, h, d\_model, dropout=0.1): super(MultiHeadedAttention, self).\_\_init\_\_() # 边界条件检查：特征维度必须能被头数整除，否则无法均匀切分张量 assert d\_model % h == 0 self.d\_k = d\_model // h self.h = h # 工程设计决断：为什么是 4 个大小为 (d\_model, d\_model) 的 Linear？ # 前 3 个分别对应 W\_q, W\_k, W\_v，第 4 个对应最终的输出融合矩阵 W\_o。 # 理论上，多头注意力要求有 h 个独立的 (d\_model, d\_k) 小矩阵。 # 但在工程落地时，为了极致的 GPU 并行效率，我们将 h 个小矩阵合并成了一个大矩阵 (d\_model, d\_model)。 self.linears = nn.ModuleList(\[copy.deepcopy(nn.Linear(d\_model, d\_model)) for \_ in range(4)\]) self.attn = None self.dropout = nn.Dropout(p=dropout) def forward(self, query, key, value, mask=None): """ 前向传播。 """ if mask is not None: # 维度扩充 (Broadcasting 准备) # 原始 mask 维度: (batch\_size, 1, seq\_len) # 扩充后维度: (batch\_size, 1, 1, seq\_len) # 这里的第二个 1 是为了对齐新切分出来的 num\_heads 维度， # 确保在 attention 内部执行 scores.masked\_fill 时，同一个掩码能广播到所有头。 mask = mask.unsqueeze(1) nbatches = query.size(0) # 1) 线性映射与多头切分 (Linear Projection & Head Splitting) query, key, value = \[ # 核心张量体操 (Tensor Gymnastics)： # x 经过 lin(x) 后维度为 (batch, seq\_len, 512) # .view(...) 将其在物理内存视角上切分为 (batch, seq\_len, 8, 64) # .transpose(1, 2) 将 seq\_len 和 head 维度互换，变为 (batch, 8, seq\_len, 64) lin(x).view(nbatches, -1, self.h, self.d\_k).transpose(1, 2) for lin, x in zip(self.linears\[:3\], (query, key, value)) # 注意：这里切片了前3个linear \] # 2) 核心注意力计算 (Scaled Dot-Product Attention) # 此时送入 attention 的张量维度为 (batch, 8, seq\_len, 64) # attention 算子内部的 matmul 会将前两维 (batch, 8) 视为独立的 Batch 空间， # 在底层 CUDA 算子中并行发起 batch \* 8 个矩阵乘法。 x, self.attn = attention( query, key, value, mask=mask, dropout=self.dropout ) # 3) 多头重组与连续化内存 (Recombination & Memory Contiguity) x = ( x.transpose(1, 2) .contiguous() # 【极其关键的内存操作，详见下文拆解】 .view(nbatches, -1, self.h \* self.d\_k) ) # 释放局部变量引用，提示 Python 垃圾回收机制 (GC) 尽早释放显存 del query del key del value # 4) 最终线性投影 (Final Linear Projection) # 使用第 4 个 Linear 层 (W\_o) 将拼接后的多头信息进行特征混合 return self.linears\[-1\](x) def test\_multi\_head\_attention(): # 1. 设定超参数 batch\_size = 2 seq\_len = 5 # 句子长度（Token 数量） d\_model = 512 # 模型的隐藏层维度 num\_heads = 8 # 多头注意力的头数 dropout = 0.1 # 2. 实例化多头注意力模块 print("正在初始化 MultiHeadedAttention 模块...") mha = MultiHeadedAttention(h=num\_heads, d\_model=d\_model, dropout=dropout) # 3. 构造随机输入的 Tensor (模拟 Self-Attention，Q=K=V=X) # 维度: (batch\_size, seq\_len, d\_model) x = torch.randn(batch\_size, seq\_len, d\_model) # 4. 构造掩码 (Mask) # 假设我们想屏蔽掉 batch 0 中最后一个 token，以及 batch 1 中最后两个 token # 维度: (batch\_size, 1, seq\_len) # 注意：这里的 1 是为了留出 head 的维度位置，配合代码里的 unsqueeze(1) mask = torch.tensor(\[ \[\[1, 1, 1, 1, 0\]\], # 0 表示被 mask 掉 \[\[1, 1, 1, 0, 0\]\] \], dtype=torch.bool) # 使用 bool 类型更符合现代 PyTorch 习惯 print(f"输入张量维度: {x.shape}") print(f"掩码张量维度: {mask.shape}") print("-" \* 40) # 5. 执行前向传播 # 在自注意力机制中，Query, Key, Value 通常来自同一个输入 X output = mha(query=x, key=x, value=x, mask=mask) # 6. 验证输出 print(f"输出张量维度: {output.shape}") print(f"内部注意力权重矩阵 (p\_attn) 维度: {mha.attn.shape}") # 检查维度是否符合预期：输出维度应该与输入维度完全一致 assert output.shape == (batch\_size, seq\_len, d\_model), "输出维度错误！" assert mha.attn.shape == (batch\_size, num\_heads, seq\_len, seq\_len), "注意力权重矩阵维度错误！" print("\\n测试通过！维度校验完全正确。") # 7. 打印部分注意力权重，验证 Mask 是否生效 print("\\n观察 Batch 1, Head 0 的注意力权重 (最后两列应该全为 0):") # 取消科学计数法打印，方便观察 torch.set\_printoptions(sci\_mode=False, precision=4) print(mha.attn\[1, 0, :, :\]) # 运行测试 if \_\_name\_\_ == "\_\_main\_\_": test\_multi\_head\_attention() ================================================== File Path: .\\nets\\utils\\encoder\_decoder.py ================================================== import torch import torch.nn as nn from torch.nn import functional as F from nets.utils.attention import MultiHeadedAttention from nets.utils.PositionwiseFeedForward import PositionwiseFeedForward from nets.utils.Generator import Generator from nets.utils.PositionalEncoding import PositionalEncoding import math import copy def clones(module, N): """ 层克隆工具函数。 物理意义：在内存中创建 N 个结构完全相同，但权重互相独立（参数不共享）的神经网络层。 工程边界条件（极易踩坑点）： 1. 必须使用 copy.deepcopy()：如果使用 \[module\] \* N 或浅拷贝，Python 只会复制内存指针。 这会导致 N 个层实际上指向同一组物理权重（变成了类似 ALBERT 模型的参数共享机制）， 违背了标准 Transformer 每层独立学习特征的设计初衷。 2. 必须包装在 nn.ModuleList 中：如果只返回一个普通的 Python 列表（List）， PyTorch 的底层机制将无法追踪这些层，模型在调用 model.parameters() 时会漏掉这些权重， 导致这 N 个层在反向传播时完全不更新（无法计算梯度）。 """ return nn.ModuleList(\[copy.deepcopy(module) for \_ in range(N)\]) class SublayerConnection(nn.Module): """ 子层连接包装器 (Sublayer Connection Wrapper)。 物理意义： 它是一个通用的“插槽”。无论是多头注意力 (Multi-Head Attention) 还是 前馈神经网络 (Feed Forward)，都可以塞进这个包装器里。 它负责为其内部的子模块自动添加：LayerNorm、Dropout 以及 残差连接 (Residual Connection)。 """ def \_\_init\_\_(self, size, dropout): """ 初始化包装器。 参数: size (int): 模型的隐状态维度 d\_model (如 512)。 dropout (float): 神经元丢弃率，用于正则化防止过拟合。 """ super(SublayerConnection, self).\_\_init\_\_() # 实例化我们在上一步修正过的 LayerNorm self.norm = LayerNorm(size) self.dropout = nn.Dropout(dropout) def forward(self, x, sublayer): """ 前向传播。 参数: x (Tensor): 输入张量，维度 (batch\_size, seq\_len, d\_model) sublayer (Callable): 一个可调用的神经网络模块 (如 Attention 或 FFN 函数) 数据流与运算顺序 (严格遵循 Pre-LN 范式): 1. self.norm(x): 先对输入进行层归一化 2. sublayer(...): 将归一化后的数据送入子层 (提取特征) 3. self.dropout(...): 对提取出的特征进行正则化 4. x + ... : 将原始输入 x 与上述结果相加 (残差连接) """ return x + self.dropout(sublayer(self.norm(x))) class LayerNorm(nn.Module): """ 标准的层归一化 (Layer Normalization) 模块。 物理意义： 强制将每个 Token 的特征向量分布，拉回到均值为 0、方差为 1 的标准正态分布， 并在最后赋予模型重新缩放和偏移的能力。 """ def \_\_init\_\_(self, features, eps=1e-6): super(LayerNorm, self).\_\_init\_\_() # 仿射变换参数 (Affine Transformation Parameters) # weight (对应公式中的 gamma): 缩放因子，初始化为 1 # bias (对应公式中的 beta): 偏移因子，初始化为 0 # nn.Parameter 作用是将这两个张量注册为模型的可学习参数，跟随梯度下降更新。 self.weight = nn.Parameter(torch.ones(features)) self.bias = nn.Parameter(torch.zeros(features)) self.eps = eps def forward(self, x): # x 维度: (batch\_size, seq\_len, d\_model) # -1 表示沿着最后一个维度 (d\_model) 计算。keepdim=True 保证输出维度为 (batch\_size, seq\_len, 1)，以便后续广播。 mean = x.mean(-1, keepdim=True) # 【致命缺陷修正处】 # 1. 必须计算方差 (Variance)，且必须是有偏估计 (unbiased=False)，即除以 N 而不是 N-1。 # 2. eps 必须在开平方前加在方差内部。 var = x.var(-1, unbiased=False, keepdim=True) # 执行归一化与仿射变换 return self.weight \* (x - mean) / torch.sqrt(var + self.eps) + self.bias class Encoder(nn.Module): """ 核心编码器堆叠模块。 架构定位：接收由 Embedding 层输出的初始张量，经过 N 次深度特征提取， 输出最终的上下文向量 (Context Vector / Memory)，供解码器 (Decoder) 使用。 """ def \_\_init\_\_(self, layer, N): """ 初始化堆叠编码器。 参数： layer (nn.Module): 单个 EncoderLayer 的实例（包含 Multi-Head Attention 和 Feed Forward）。 这里再次使用了依赖注入（Dependency Injection），保证顶层代码的整洁。 N (int): 堆叠的层数（论文默认设定为 6）。 """ super(Encoder, self).\_\_init\_\_() # 生成 N 个参数独立的编码器层 self.layers = clones(layer, N) # 最后的层归一化 (Layer Normalization) # 物理意义：在 N 层特征提取全部结束后，对最终输出的隐状态再做一次归一化。 # 工程考量：这能确保送入 Decoder 进行交叉注意力计算的 memory 张量， # 其特征分布严格保持在均值为 0、方差为 1 的平滑状态，极大提升 Decoder 的收敛稳定性。 self.norm = LayerNorm(layer.size) def forward(self, x, mask): """ 前向传播计算。 数据流与边界条件： x: 当前的隐状态张量。初始输入维度为 (batch\_size, seq\_len, d\_model)。 mask: 源序列的 Padding Mask。维度通常为 (batch\_size, 1, seq\_len)。 注意：在整个 N 层的 for 循环中，mask 是【全局只读】且【恒定不变】的。 因为无论特征经过多少层非线性变换，句子中 Padding（补零）的物理位置永远不会改变。 """ # 逐层穿透：将上一层的输出直接作为下一层的输入 for layer in self.layers: x = layer(x, mask) # 返回前执行最终的归一化 return self.norm(x) class EncoderLayer(nn.Module): """ 单层编码器 (Encoder Layer)。 架构定位： Transformer 编码器的最小重复单元。论文中默认堆叠 6 层此类结构。 它负责完成两个正交的任务： 1. 序列维度的信息交互 (Self-Attention) 2. 特征维度的非线性映射 (Feed-Forward) """ def \_\_init\_\_(self, size, self\_attn, feed\_forward, dropout): """ 初始化单层编码器。 参数: size (int): 隐状态特征维度 d\_model (如 512)。 self\_attn (nn.Module): 实例化的多头注意力模块。 feed\_forward (nn.Module): 实例化的前馈神经网络模块。 dropout (float): 正则化丢弃率。 """ super(EncoderLayer, self).\_\_init\_\_() self.self\_attn = self\_attn self.feed\_forward = feed\_forward # 核心复用：克隆 2 个完全独立的子层包装器 (SublayerConnection) # 第一个包装器用于 Attention，第二个用于 Feed-Forward。 # 它们各自拥有独立的 LayerNorm 权重，互不干扰。 self.sublayer = clones(SublayerConnection(size, dropout), 2) self.size = size def forward(self, x, mask): """ 前向传播数据流。 参数: x (Tensor): 输入特征，维度 (batch\_size, seq\_len, d\_model)。 mask (Tensor): Padding 掩码，维度 (batch\_size, 1, seq\_len)。 """ # 步骤 1：自注意力机制 (Self-Attention) 与残差连接 # 工程细节：这里使用了 Python 的 lambda 匿名函数来适配 SublayerConnection 的接口。 # 为什么传入三个 x？这正是 "Self (自)" 的数学定义： # Query = x, Key = x, Value = x。 x = self.sublayer\[0\](x, lambda x: self.self\_attn(x, x, x, mask)) # 步骤 2：前馈神经网络 (Feed-Forward) 与残差连接 # 数据流：经过 Attention 混淆了全局位置信息的张量 x， # 被送入 FFN 进行局部的、逐位置的 (Position-wise) 非线性特征升维与降维。 return self.sublayer\[1\](x, self.feed\_forward) class Decoder(nn.Module): """ 核心解码器堆叠模块 (Decoder Stack)。 架构定位： 作为 Transformer 的自回归 (Auto-regressive) 生成引擎，它包含 N 个解码器层。 它与 Encoder 的根本区别在于双重信息流： 既要处理目标序列 (Target) 自身的时序推演，又要不断地“回望”源序列 (Source) 的全局语义。 """ def \_\_init\_\_(self, layer, N): """ 初始化堆叠解码器。 参数: layer (nn.Module): 单个 DecoderLayer 的实例（包含掩码自注意力、交叉注意力和前馈网络）。 N (int): 堆叠的层数（通常与 Encoder 保持一致，如 6 层）。 """ super(Decoder, self).\_\_init\_\_() # 生成 N 个参数完全独立的解码器层 self.layers = clones(layer, N) # 最后的层归一化 (Layer Normalization) # 物理意义：与 Encoder 相同，采用 Pre-LN 架构时，必须在 N 层堆叠结束后追加一次全局归一化， # 确保最终送入 Generator (线性映射+Softmax) 的特征分布是稳定且无偏移的。 self.norm = LayerNorm(layer.size) def forward(self, x, memory, src\_mask, tgt\_mask): """ 前向传播计算。 工程边界条件与数据流解析 (极其重要)： x: 目标序列当前的隐状态张量。维度 (batch\_size, tgt\_seq\_len, d\_model)。 在训练阶段，这是经过 Shifted Right 并带有词嵌入的完整 Target 序列。 memory: 编码器 (Encoder) 最后一层输出的全局上下文张量。维度 (batch\_size, src\_seq\_len, d\_model)。 注意：在整个 N 层的 for 循环中，memory 是【全局只读】的，它不会被修改。 src\_mask: 源序列的 Padding 掩码。维度 (batch\_size, 1, src\_seq\_len)。 作用于交叉注意力 (Cross-Attention) 中，防止 Decoder 将注意力浪费在源序列的补零位置上。 tgt\_mask: 目标序列的因果掩码 (通常是下三角矩阵)。维度 (batch\_size, tgt\_seq\_len, tgt\_seq\_len)。 作用于掩码自注意力 (Masked Self-Attention) 中，防止当前 Token 提前看到未来的 Token。 """ # 逐层穿透：将上一层的目标序列输出作为下一层的输入 for layer in self.layers: # memory 和 masks 在所有层中保持恒定，充当全局环境参数 x = layer(x, memory, src\_mask, tgt\_mask) # 返回前执行最终的归一化 return self.norm(x) class DecoderLayer(nn.Module): """ 单层解码器 (Decoder Layer)。 架构定位： 自回归生成任务中的核心特征处理单元。 它不仅要在目标序列内部建立时序依赖，还要跨越空间去源序列中提取对应信息。 """ def \_\_init\_\_(self, size, self\_attn, src\_attn, feed\_forward, dropout): """ 初始化单层解码器。 参数: size (int): 隐状态特征维度 d\_model (如 512)。 self\_attn (nn.Module): 掩码多头自注意力模块 (处理目标序列内部逻辑)。 src\_attn (nn.Module): 交叉注意力模块 (处理目标序列与源序列的映射)。 feed\_forward (nn.Module): 前馈神经网络模块。 dropout (float): 正则化丢弃率。 """ super(DecoderLayer, self).\_\_init\_\_() self.size = size self.self\_attn = self\_attn self.src\_attn = src\_attn self.feed\_forward = feed\_forward # 核心复用：克隆 3 个独立的子层包装器 (包含各自独立的 LayerNorm 和残差连接) # index 0: 掩码自注意力 # index 1: 交叉注意力 # index 2: FFN self.sublayer = clones(SublayerConnection(size, dropout), 3) def forward(self, x, memory, src\_mask, tgt\_mask): """ 前向传播数据流。 参数: x (Tensor): 目标序列特征张量，维度 (batch\_size, tgt\_seq\_len, d\_model)。 memory (Tensor): 源序列特征张量，维度 (batch\_size, src\_seq\_len, d\_model)。 src\_mask (Tensor): 源序列 Padding 掩码。 tgt\_mask (Tensor): 目标序列因果掩码 (下三角矩阵)。 """ m = memory # 阶段 1：掩码自注意力 (Masked Self-Attention) # 数据流：Q=x, K=x, V=x。 # 约束：必须传入 tgt\_mask。 # 物理意义：当前生成的词只能根据它之前的词来更新自己的特征，绝对不能看到之后的词。 x = self.sublayer\[0\](x, lambda x: self.self\_attn(x, x, x, tgt\_mask)) # 阶段 2：交叉注意力 (Cross-Attention / Source-Attention) # 数据流：Q=x, K=m, V=m。 # 约束：必须传入 src\_mask。 # 物理意义：拿着经过阶段 1 更新后的当前词特征 (Q)，去编码器的输出 (m) 中进行全局检索 (K, V)， # 找出当前词在源句子中最对应的语义片段，并将其融合到当前词的特征中。 x = self.sublayer\[1\](x, lambda x: self.src\_attn(x, m, m, src\_mask)) # 阶段 3：前馈神经网络 (Feed-Forward) # 数据流：仅针对 x 自身进行。 # 物理意义：对融合了内部时序信息和外部源语义信息的特征，进行逐位置的非线性高维映射。 return self.sublayer\[2\](x, self.feed\_forward) class EncoderDecoder(nn.Module): """ 标准的编码器-解码器顶层架构包装器。 设计意图：将模型解耦为 5 个正交的独立组件（编码、解码、源嵌入、目标嵌入、生成头）， 通过构造函数注入（Dependency Injection），保证底层模块的极高可复用性。 """ def \_\_init\_\_(self, encoder, decoder, src\_embed, tgt\_embed, generator): super(EncoderDecoder, self).\_\_init\_\_() # 1. 编码器堆叠层（通常包含 N 个 TransformerEncoderLayer） self.encoder = encoder # 2. 解码器堆叠层（通常包含 N 个 TransformerDecoderLayer） self.decoder = decoder # 3. 源序列嵌入层（包含 Token Embedding + Positional Encoding） # 负责将离散的源 Token ID 映射为连续的稠密向量 self.src\_embed = src\_embed # 4. 目标序列嵌入层（同上，用于处理 Target Token ID） self.tgt\_embed = tgt\_embed # 5. 生成器（通常是一个 Linear 层 + Softmax） # 负责将解码器输出的隐状态向量映射回词表的概率分布 self.generator = generator def forward(self, src, tgt, src\_mask, tgt\_mask): """ 前向传播函数。注意：此函数仅在【训练阶段（Training）】使用。 在训练时，我们拥有完整的目标序列 tgt（Teacher Forcing 机制），因此可以一次性并行计算。 输入维度假定： src: (batch\_size, src\_seq\_len) tgt: (batch\_size, tgt\_seq\_len) src\_mask: (batch\_size, 1, src\_seq\_len) - 用于屏蔽 Padding tgt\_mask: (batch\_size, tgt\_seq\_len, tgt\_seq\_len) - 用于实现下三角掩码，防止信息穿越 """ # 数据流： # 1. 走 self.encode() 拿到源序列的全局上下文特征 memory # 2. 将 memory、tgt 以及对应的 mask 一起送入 self.decode() 获取最终隐状态 memory = self.encode(src, src\_mask) out = self.decode(memory, src\_mask, tgt, tgt\_mask) return out def encode(self, src, src\_mask): """ 独立的编码逻辑。 数据流：Token ID -> 词嵌入 -> 叠加位置编码 -> N层自注意力与前馈网络 -> Context Vector (memory) 输出维度：(batch\_size, src\_seq\_len, d\_model) """ return self.encoder(self.src\_embed(src), src\_mask) def decode(self, memory, src\_mask, tgt, tgt\_mask): """ 独立的解码逻辑。 数据流：Target Token ID -> 词嵌入 -> N层解码器（包含掩码自注意力和交叉注意力） -> 解码特征 注意：解码器不仅需要 tgt 本身，还需要引入编码器生成的 memory 进行交叉注意力（Cross-Attention）计算。 输出维度：(batch\_size, tgt\_seq\_len, d\_model) """ return self.decoder(self.tgt\_embed(tgt), memory, src\_mask, tgt\_mask) class Embeddings(nn.Module): """ 标准的词嵌入层 + 位置编码胶水层 """ def \_\_init\_\_(self, d\_model, vocab\_size, dropout=0.1): super(Embeddings, self).\_\_init\_\_() self.lut = nn.Embedding(vocab\_size, d\_model) self.d\_model = d\_model self.pe = PositionalEncoding(d\_model, dropout) def forward(self, x): # 数学细节：论文指出，在将 Embedding 加上 PE 之前， # 需要将 Embedding 乘以 sqrt(d\_model) 进行缩放，以此来平衡方差。 x = self.lut(x) \* math.sqrt(self.d\_model) return self.pe(x) def test\_full\_transformer(): print("🚀 开始进行 Transformer 端到端终极测试 🚀\\n") # 1. 超参数全景设定 vocab\_size = 32000 # 真实场景下的典型词表大小 (如 BPE 分词) d\_model = 512 # 隐藏层维度 d\_ff = 2048 # FFN 膨胀维度 num\_heads = 8 # 注意力头数 dropout = 0.1 num\_layers = 6 # 还原论文标准的 6 层堆叠 batch\_size = 4 src\_seq\_len = 12 # 源序列长度 tgt\_seq\_len = 10 # 目标序列长度 print("1️⃣ 正在组装底层引擎 (Attention & FFN)...") attn = MultiHeadedAttention(h=num\_heads, d\_model=d\_model, dropout=dropout) ff = PositionwiseFeedForward(d\_model=d\_model, d\_ff=d\_ff, dropout=dropout) print("2️⃣ 正在构建 6 层 Encoder 与 Decoder 塔...") encoder = Encoder(EncoderLayer(d\_model, copy.deepcopy(attn), copy.deepcopy(ff), dropout), num\_layers) decoder = Decoder(DecoderLayer(d\_model, copy.deepcopy(attn), copy.deepcopy(attn), copy.deepcopy(ff), dropout), num\_layers) print("3️⃣ 正在装配词嵌入与位置编码...") # 假设源语言和目标语言共享词表 src\_embed = Embeddings(d\_model, vocab\_size, dropout) tgt\_embed = Embeddings(d\_model, vocab\_size, dropout) print("4️⃣ 正在安装 Generator 生成头...") generator = Generator(d\_model, vocab\_size) print("5️⃣ 拼装终极模型：EncoderDecoder...") model = EncoderDecoder(encoder, decoder, src\_embed, tgt\_embed, generator) # 统计模型参数量 total\_params = sum(p.numel() for p in model.parameters() if p.requires\_grad) print(f"\\n✨ 模型组装完毕！总可训练参数量: {total\_params / 1e6:.2f} M") # ---------------- 测试数据流 ---------------- print("\\n--- 构造 Input Tensors ---") # 模拟输入 Token IDs src = torch.randint(0, vocab\_size, (batch\_size, src\_seq\_len)) tgt = torch.randint(0, vocab\_size, (batch\_size, tgt\_seq\_len)) # 构造真实掩码 src\_mask = torch.ones(batch\_size, 1, src\_seq\_len, dtype=torch.bool) tgt\_mask = torch.tril(torch.ones(tgt\_seq\_len, tgt\_seq\_len, dtype=torch.bool)).unsqueeze(0).expand(batch\_size, -1, -1) print(f"源 Token IDs (src): {src.shape}") print(f"目标 Token IDs (tgt): {tgt.shape}") print(f"源 Mask (src\_mask): {src\_mask.shape}") print(f"目标 Mask (tgt\_mask): {tgt\_mask.shape}") print("\\n--- 执行前向传播 (Forward Pass) ---") # 1. 获取 Decoder 输出的隐状态 out\_hidden = model(src, tgt, src\_mask, tgt\_mask) print(f"模型输出隐状态维度: {out\_hidden.shape} -> 期望: ({batch\_size}, {tgt\_seq\_len}, {d\_model})") assert out\_hidden.shape == (batch\_size, tgt\_seq\_len, d\_model) # 2. 获取最终的 Logits 概率分布 logits = model.generator(out\_hidden) print(f"Generator 输出 Logits 维度: {logits.shape} -> 期望: ({batch\_size}, {tgt\_seq\_len}, {vocab\_size})") assert logits.shape == (batch\_size, tgt\_seq\_len, vocab\_size) print("\\n✅ 端到端测试完美通过！这套代码已经具备了直接上 GPU 训练机器翻译模型的工业级水准。") if \_\_name\_\_ == "\_\_main\_\_": test\_full\_transformer() ================================================== File Path: .\\nets\\utils\\Generator.py ================================================== import torch import torch.nn as nn class Generator(nn.Module): """ 生成器 (Generator) / 语言模型头 (LM Head) 物理意义： 将解码器输出的连续、高维隐状态空间 (Latent Space) 映射回离散的词汇表概率空间。 """ def \_\_init\_\_(self, d\_model, vocab\_size): """ 初始化映射层。 参数: d\_model (int): Transformer 内部的隐状态特征维度 (例如 512)。 vocab\_size (int): 目标语言的词表总大小 (例如 30000)。 """ super(Generator, self).\_\_init\_\_() # 线性投影层 (Linear Projection Layer) # 底层包含一个权重矩阵 W，其形状为 (vocab\_size, d\_model)，以及一个偏置项 b。 # 物理意义：这个矩阵 W 实际上存储了词汇表中所有词的“输出词向量”。 # 当执行 x @ W^T 时，本质上是在计算当前位置的隐状态 x，与词表中每一个词向量的内积。 # 内积越大，说明当前特征与该词的语义越匹配。 self.proj = nn.Linear(d\_model, vocab\_size) def forward(self, x): """ 前向传播计算。 参数: x (Tensor): 解码器 (Decoder) 最后一层的输出张量。 期望维度: (batch\_size, seq\_len, d\_model) 返回: Tensor: 词表上的对数概率分布 (Log Probabilities)。 输出维度: (batch\_size, seq\_len, vocab\_size) """ # 步骤 1: 线性特征映射 # 执行 self.proj(x)，张量维度从 (batch\_size, seq\_len, d\_model) # 变换为 (batch\_size, seq\_len, vocab\_size)。 # 此时的输出被称为 Logits（未归一化的原始得分），值域为 (-inf, +inf)。 logits = self.proj(x) return logits ================================================== File Path: .\\nets\\utils\\PositionalEncoding.py ================================================== import torch import torch.nn as nn import math class PositionalEncoding(nn.Module): """ 注入绝对位置信息的正弦波编码器。 """ def \_\_init\_\_(self, d\_model, dropout, max\_len=5000): super(PositionalEncoding, self).\_\_init\_\_() self.dropout = nn.Dropout(p=dropout) # 预先在内存中分配好全零矩阵 pe = torch.zeros(max\_len, d\_model) # position 维度: (max\_len, 1) position = torch.arange(0, max\_len).unsqueeze(1).float() # div\_term 维度: (d\_model/2,) # 利用对数恒等式保证 fp32/fp16 精度下的数值稳定性 div\_term = torch.exp( torch.arange(0, d\_model, 2).float() \* -(math.log(10000.0) / d\_model) ) # 偶数维度填充正弦，奇数维度填充余弦 pe\[:, 0::2\] = torch.sin(position \* div\_term) pe\[:, 1::2\] = torch.cos(position \* div\_term) # 扩充 batch 维度并注册为不可训练的 buffer pe = pe.unsqueeze(0) self.register\_buffer("pe", pe) def forward(self, x): # 运行时直接截取序列长度并相加，无需操作 autograd 图 x = x + self.pe\[:, :x.size(1)\] return self.dropout(x) ================================================== File Path: .\\nets\\utils\\PositionwiseFeedForward.py ================================================== import torch import torch.nn as nn from torch.nn import functional as F import math import copy class PositionwiseFeedForward(nn.Module): """ 逐位置前馈神经网络 (Position-wise FFN)。 架构定位： 如果说 Attention 负责在全局视角下“寻找关联” (跨 Token 的信息混合)， 那么 FFN 就负责在局部视角下“升华特征” (单个 Token 内部的非线性映射)。 两者是完全正交的设计。 """ def \_\_init\_\_(self, d\_model, d\_ff, dropout=0.1): """ 初始化 FFN 模块。 参数: d\_model (int): 模型的输入/输出基准维度 (例如 512)。 d\_ff (int): 隐藏层膨胀维度 (通常设为 d\_model 的 4 倍，例如 2048)。 dropout (float): 正则化丢弃率。 """ super(PositionwiseFeedForward, self).\_\_init\_\_() # 步骤 1：升维映射 (Dimensionality Expansion) # 物理意义：将特征从 512 维的低维流形，投影到 2048 维的高维空间， # 从而暴露出更多可能线性可分的特征组合。 self.w\_1 = nn.Linear(d\_model, d\_ff) # 步骤 2：降维映射 (Dimensionality Reduction) # 物理意义：将经过非线性激活后的高维特征，重新压缩回 512 维， # 保证输出维度与输入严格一致，以满足后续残差连接 (x + FFN(x)) 的严格要求。 self.w\_2 = nn.Linear(d\_ff, d\_model) self.dropout = nn.Dropout(dropout) def forward(self, x): """ 前向传播计算。 张量流转： 输入 x: (batch\_size, seq\_len, d\_model) 执行逻辑：x -> Linear(d\_model->d\_ff) -> ReLU -> Dropout -> Linear(d\_ff->d\_model) 工程细节 (关于 Position-wise 的底层实现)： PyTorch 的 nn.Linear 默认只作用于张量的最后一个维度。 因此，当传入 3D 张量 (batch\_size, seq\_len, d\_model) 时， 矩阵乘法是独立且等价地作用于 seq\_len 中的每一个 Token 上的。 相当于一个 kernel\_size=1 的 1D 卷积。 """ # 注意：在较新的 PyTorch 规范中，推荐直接使用 F.relu 而不是对张量调用 .relu() return self.w\_2(self.dropout(F.relu(self.w\_1(x)))) def test\_feed\_forward(): # 1. 设定超参数 batch\_size = 2 seq\_len = 5 # 句子长度（Token 数量） d\_model = 512 # 模型的输入/输出基准维度 d\_ff = 2048 # 隐藏层膨胀维度 (通常是 d\_model 的 4 倍) dropout = 0.1 # 2. 实例化 FFN 模块 print("正在初始化 PositionwiseFeedForward 模块...") ffn = PositionwiseFeedForward(d\_model=d\_model, d\_ff=d\_ff, dropout=dropout) # 3. 构造随机输入的 Tensor (模拟经过 Attention 和 LayerNorm 后的特征) # 维度: (batch\_size, seq\_len, d\_model) x = torch.randn(batch\_size, seq\_len, d\_model) print(f"输入张量维度: {x.shape}") print("-" \* 40) # 4. 执行前向传播 output = ffn(x) # 5. 验证输出与中间维度 print(f"输出张量维度: {output.shape}") # 检查维度是否符合预期：输出维度必须与输入维度完全一致 assert output.shape == (batch\_size, seq\_len, d\_model), "输出维度错误！必须与输入一致才能进行残差相加。" # 我们可以通过打印内部权重的 shape 来验证升降维逻辑 print(f"w\_1 (升维) 权重矩阵维度: {ffn.w\_1.weight.shape} -> 期望为 (2048, 512)") print(f"w\_2 (降维) 权重矩阵维度: {ffn.w\_2.weight.shape} -> 期望为 (512, 2048)") print("\\n测试通过！FFN 模块成功维持了张量的输入输出维度，可以完美接入残差连接。") # 运行测试 if \_\_name\_\_ == "\_\_main\_\_": test\_feed\_forward()

  

## 结论

你的这份代码，现在更准确的定位是：

`Transformer 核心网络骨架已经基本搭起来了，但还不是“按论文复现 WMT14 英德翻译实验”的完整工程。`

原因很直接：

-   从**网络主干**看，你已经实现了多头注意力、FFN、正弦位置编码、Encoder/Decoder 堆叠，这些核心部件是对的。
    
-   但从**论文复现**看，真正决定能不能接近论文结果的，往往不是“网络能不能前向跑通”，而是**训练范式、mask 细节、词表构建、权重共享、优化器与学习率调度、label smoothing、beam search、checkpoint averaging、BLEU 评估口径**这些外围机制。你当前上传的内容里，主要还是模型定义与测试，没有形成完整复现闭环。论文的英德设置明确写了：WMT14 English-German，约 4.5M 句对，共享源/目标 BPE 词表约 37k，按近似长度分桶，每个 batch 大约 25k source tokens + 25k target tokens，Adam 的参数为 $ \\beta\_1=0.9,\\beta\_2=0.98,\\epsilon=10^{-9} $，学习率采用 warmup=4000 的 Noam schedule，训练时使用 $ \\epsilon\_{ls}=0.1 $ 的 label smoothing，推理时用 beam size=4、length penalty $ \\alpha=0.6 $，并对最后若干 checkpoint 做平均。
    

* * *

## 一、你当前代码里，已经对齐论文主干的部分

### 1\. Base 模型超参数骨架基本是对的

论文 base 配置的关键参数是：

-   $N=6$
    
-   $d\_{model}=512$
    
-   $d\_{ff}=2048$
    
-   $h=8$
    
-   dropout $=0.1$
    

你的 `make_model()` 默认值正好就是这一组，这说明你在**模型尺度**上是朝论文 base 版对齐的。

### 2\. Scaled Dot-Product Attention 的数学实现基本正确

论文给出的核心公式是：

$Attention(Q,K,V)=softmax(QKTdk)V\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)VAttention(Q,K,V)=softmax(dk​​QKT​)V$

你在 `attention()` 中实现了：

-   `query @ key.transpose(-2, -1)`
    
-   再除以 `sqrt(d_k)`
    
-   再 `softmax`
    
-   再乘 `value`
    

这一点是对齐论文的。

### 3\. 多头注意力拆头逻辑是对的

论文中多头注意力的核心思想是：先把 $Q,K,V$ 线性映射到多个子空间，再并行做 attention，最后 concat 回来再过一个输出投影。你现在的 `MultiHeadedAttention` 也是这个思路：先过线性层，再 `view` 成多头，再 `transpose`，最后拼回去。这个结构没有本质性问题。

### 4\. FFN 和正弦位置编码也基本正确

论文的 FFN 是：

$FFN(x)=max⁡(0,xW1+b1)W2+b2\mathrm{FFN}(x)=\max(0, xW_1+b_1)W_2+b_2FFN(x)=max(0,xW1​+b1​)W2​+b2​$

你现在是两层 `Linear` 中间接 `ReLU`，这一点是对的。位置编码你也用了正弦/余弦编码，并且 embedding 之后乘了 $ \\sqrt{d\_{model}} $，这也与论文描述一致。

### 5\. Decoder 的三段结构也对

论文 decoder 每层是：

1.  masked self-attention
    
2.  encoder-decoder attention
    
3.  FFN
    

你当前 `DecoderLayer` 的顺序也是这样，主干没跑偏。

* * *

## 二、你现在距离“论文复现”还差的关键点

下面这些不是“锦上添花”，而是**决定你能不能复现论文结果的关键项**。

### 1\. 你现在实现的是 **Pre-LN**，论文原文是 **Post-LN**

论文在第 3.1 节写得非常明确：

$LayerNorm(x+Sublayer(x))\mathrm{LayerNorm}(x + \mathrm{Sublayer}(x))LayerNorm(x+Sublayer(x))$

这表示它的规范是：

-   先子层
    
-   再残差相加
    
-   最后 LayerNorm
    

这就是 **Post-LN**。

而你现在的 `SublayerConnection.forward()` 是：

```
return x + self.dropout(sublayer(self.norm(x)))
```

这表示：

-   先 LayerNorm
    
-   再子层
    
-   再残差
    

这是 **Pre-LN**。

### 这个差异为什么重要？

因为如果你的目标是：

-   **工程上更稳定地训起来**：Pre-LN 通常更稳
    
-   **尽量按原论文复现**：你应该改成 Post-LN
    

所以这里要非常明确：

> 你当前代码不是“论文严格同构”，而是“现代更常见的稳定训练变体”。

这不是小差异，是**架构定义层面的差异**。

* * *

### 2\. 你没有实现论文里的 **权重共享（weight tying）**

论文第 3.4 节明确写了：

-   源 embedding
    
-   目标 embedding
    
-   pre-softmax linear transformation
    

三者共享同一组权重矩阵（前提是共享词表）。同时 embedding 处乘以 $ \\sqrt{d\_{model}} $。

而你现在的实现是：

-   `src_embed = Embeddings(...)`
    
-   `tgt_embed = Embeddings(...)`
    
-   `generator = Generator(...)`
    

三者是独立参数，没有 tying。

### 这个差异的后果

这会带来两个问题：

1.  参数量和论文不完全一致
    
2.  语义空间对齐方式不同，可能影响最终翻译效果
    

如果你要走“论文尽量一致”的路线，这个一定要补。

* * *

### 3\. 你还没有体现 **target shifted right**

论文 decoder 训练时有一个关键条件：

> 输出 embedding 要右移一位，确保位置 $i$ 的预测只能依赖于小于 $i$ 的已知输出。

也就是训练时必须区分：

-   `tgt_input = [BOS, y1, y2, ..., y_{n-1}]`
    
-   `tgt_label = [y1, y2, ..., yn, EOS]`
    

而你当前代码里只是把一个 `tgt` 直接喂给模型测试前向，没有展示任何 `shift_right` 的训练逻辑。

### 这是致命问题

如果训练代码里没有严格做右移，那么 decoder 会“偷看答案”，训练目标就错了，最后 BLEU 没法看。

* * *

### 4\. 你的 `tgt_mask` 现在只有“因果 mask”，没有和 “padding mask” 正确组合

你测试代码中的目标 mask 是：

```
tgt_mask = torch.tril(torch.ones(...))
```

这只是**下三角因果 mask**。

但真实训练里，目标 mask 应该是：

$tgt_mask=pad_mask∧subsequent_mask\text{tgt\_mask} = \text{pad\_mask} \land \text{subsequent\_mask}tgt_mask=pad_mask∧subsequent_mask$

也就是同时屏蔽：

1.  未来位置
    
2.  PAD 位置
    

否则 decoder 在 self-attention 中仍然可能看到 target 里的 padding token，这会污染训练。论文明确要求 decoder self-attention 不能看到后面的位置，同时真实 batch 中又一定存在 padding，所以这两个 mask 必须联合处理。

* * *

### 5\. 你的 `src_mask` 也还只是测试用的全 1 mask，不是基于 PAD 的真实 mask

你现在测试里：

```
src_mask = torch.ones(batch_size, 1, src_seq_len, dtype=torch.bool)
```

这只是为了验证前向维度。

真实翻译任务里，源序列长度不同，一定要 padding，因此 `src_mask` 必须来自：

$src_mask=(src≠PAD)\text{src\_mask} = (src \neq PAD)src_mask=(src=PAD)$

否则 encoder self-attention 和 decoder cross-attention 都会去关注 padding 位置。

* * *

### 6\. 你还没有实现论文的数据管线：WMT14 英德 + 共享 BPE 约 37k 词表

论文在训练数据部分写得很清楚：

-   数据集：WMT 2014 English-German
    
-   规模：约 4.5M sentence pairs
    
-   编码：byte-pair encoding
    
-   词表：shared source-target vocabulary，约 37000 tokens
    

你当前上传的代码里，还看不到：

-   WMT14 数据读取
    
-   BPE/SentencePiece 训练或加载
    
-   共享词表构建
    
-   BOS/EOS/PAD/UNK 特殊符号定义
    
-   长度分桶与 batch sampler
    

也就是说，**你现在只有网络，没有数据系统**。

* * *

### 7\. 你还没有实现论文最关键的训练策略

论文第 5.3、5.4 节规定了训练细节：

-   Adam
    
-   $ \\beta\_1=0.9,\\beta\_2=0.98,\\epsilon=10^{-9} $
    
-   Noam 学习率：
    

$lrate=dmodel−0.5⋅min⁡(step−0.5, step⋅warmup−1.5)\mathrm{lrate}=d_{model}^{-0.5}\cdot \min(step^{-0.5},\ step\cdot warmup^{-1.5})lrate=dmodel−0.5​⋅min(step−0.5, step⋅warmup−1.5)$

-   warmup\_steps = 4000
    
-   label smoothing = 0.1
    
-   dropout = 0.1（base）
    

而你当前代码中还没有看到：

-   优化器封装
    
-   step 级 lr scheduler
    
-   label smoothing loss
    
-   train / valid loop
    
-   gradient accumulation 或 token-based batching
    

这意味着，即使模型定义完全正确，你也还没有进入“可复现实验”的阶段。

* * *

### 8\. 你还没有实现论文里的推理与评估闭环

论文翻译实验结果还依赖这些推理细节：

-   checkpoint averaging：base 模型平均最后 5 个 checkpoint
    
-   beam search：beam size = 4
    
-   length penalty：$ \\alpha = 0.6 $
    
-   最大输出长度：input length + 50，且可提前结束
    

你当前代码还没有看到：

-   greedy decode
    
-   beam search
    
-   length penalty
    
-   checkpoint averaging
    
-   BLEU 评估脚本
    

所以你现在还无法判断模型是否真的“复现成功”。

* * *

## 三、我对你当前代码的严格评价

### 可以肯定的部分

你的代码不是胡写，主干理解是对的，特别是：

-   attention 主公式没问题
    
-   encoder / decoder 拆分没问题
    
-   embedding + sinusoidal PE 没问题
    
-   base 超参数方向对
    

这说明你已经跨过了“看懂论文结构图”的阶段，进入了“能自己手写 Transformer 核心模块”的阶段。

### 但必须直说的部分

你现在的 `test_full_transformer()` 只能证明一件事：

> 你的张量维度流是通的。

它**不能证明**：

-   训练范式正确
    
-   mask 逻辑完整
    
-   权重共享正确
    
-   loss 定义对齐
    
-   学习率调度对齐
    
-   解码策略对齐
    
-   BLEU 可复现
    

所以代码里“已经具备工业级水准”“可以直接上 GPU 训练机器翻译模型”这种表述，当前阶段是过度结论。

* * *

## 四、如果你的目标是“尽量按论文复现”，我建议你按这个顺序改

### 第一阶段：先把“模型定义”修到论文一致

-    把 `SublayerConnection` 从 Pre-LN 改成 Post-LN
    
-    实现 shared embedding + generator weight tying
    
-    明确 `BOS / EOS / PAD` 机制
    
-    实现 `shift_right(tgt)`
    
-    实现 `src_mask` 与 `tgt_mask` 的真实构造
    
-    写一个标准 `Batch` 类，把 `src`、`tgt_input`、`tgt_y`、`ntokens`、mask 全部封装起来
    

这一步做完，你的网络才叫“论文结构基本对齐”。

* * *

### 第二阶段：把“训练机制”补齐

-    实现 label smoothing，$ \\epsilon\_{ls}=0.1 $
    
-    实现 Adam，参数固定为 $ (0.9, 0.98, 10^{-9}) $
    
-    实现 Noam scheduler，warmup=4000
    
-    训练日志里记录 `step / lr / tokens/sec / loss / ppl`
    
-    验证集定期评估
    

这一步做完，你才进入“论文训练协议对齐”的阶段。

* * *

### 第三阶段：把“数据系统”补齐

-    下载并整理 WMT14 En-De
    
-    构建共享 BPE 词表，目标规模约 37k
    
-    句对按近似长度分桶
    
-    batch 按 token 数控制，而不是按固定句子数控制
    
-    尽量接近论文的 25k src tokens + 25k tgt tokens / batch
    

这一步做完，你的数据输入分布才接近论文。

* * *

### 第四阶段：把“推理评估”补齐

-    实现 greedy decode
    
-    实现 beam search
    
-    加入 length penalty，$ \\alpha=0.6 $
    
-    限制最大输出长度为 `src_len + 50`
    
-    实现 checkpoint averaging
    
-    跑 dev/test BLEU
    

这一步做完，你才真正拥有“复现结果”的闭环。

* * *

## 五、最关键的一个判断

如果你现在直接在这份代码上开训，我的判断是：

> 能跑，但不能称为论文复现；即使 loss 下降，也很难说明你复现了论文结果。

核心原因不是网络层数不对，而是下面这几个“复现杀手”还没解决：

1.  Post-LN / Pre-LN 不一致
    
2.  没做权重共享
    
3.  没有 shift-right 训练输入
    
4.  mask 还不是完整训练 mask
    
5.  没有 Noam lr + label smoothing
    
6.  没有 WMT14 shared BPE 数据管线
    
7.  没有 beam search + checkpoint averaging + BLEU 评估
    

这些问题里，前四个属于“模型定义 correctness”，后面三个属于“实验 protocol correctness”。两边缺一都不叫真正复现。

* * *

## 六、最终结论

我给你的最终判断是：

**你的代码适合作为 Transformer 论文复现的起点，但绝对还不是终点。**

更准确地说：

-   **主干模块：70 分**
    
-   **论文一致性：40 分**
    
-   **可训练复现闭环：20 分**
    

如果你的目标是“严格按论文复现 WMT14 英德 base 版”，那你下一步最该做的不是继续解释 attention，而是先把下面四个文件补出来：

-   `dataset.py`
    
-   `batch.py`
    
-   `loss.py`
    
-   `train.py`
    

然后再补：

-   `scheduler.py`
    
-   `decode.py`
    
-   `bleu_eval.py`
    

下一条消息里，我可以直接按你当前项目风格，给你生成一套**严格面向论文复现**的最小可用版本框架，先从 `Batch + Mask + LabelSmoothing + NoamOpt` 四个核心模块开始。

  

1.关于Pre-LN和Post-LN我先不修了，因为我参考的是斯坦福的那个解读transformer的ipynb。 2.帮我修一下权重共享 3.帮我加上target shifted right 4.tgt\_mask帮我也加上padding mask。 5.src\_mask帮我也加上正确的mask。 6.修复好上面的内容之后，我们再进一步的帮我写如何加载数据，我应该去哪里下载数据，以及数据加载的代码部分。 7.剩余的训练细节，推理，损失等等我们稍后讨论。

  

## 好，先按你现在的路线来

这次我**不动 Pre-LN / Post-LN**，保持你参考的斯坦福 `Annotated Transformer` 风格不变。

下面我把你要的 5 个点一次性收拢成一个统一约定：

1.  `tgt` 在进入 batch 构造之前，**已经包含** `<bos>` 和 `<eos>`
    
2.  训练时通过
    
    -   `tgt_input = tgt[:, :-1]`
        
    -   `tgt_y = tgt[:, 1:]`  
        来实现 **target shifted right**
        
3.  `src_mask` 使用真实 padding mask：
    
    $src_mask=(src≠pad_idx)\text{src\_mask} = (src \neq pad\_idx)src_mask=(src=pad_idx)$
4.  `tgt_mask` 同时包含：
    
    -   padding mask
        
    -   causal mask（下三角 mask）
        
5.  当你后面用**joint source-target vocabulary**时，把：
    
    -   `src_embed.lut.weight`
        
    -   `tgt_embed.lut.weight`
        
    -   `generator.proj.weight`  
        绑成同一个权重矩阵
        

这套做法和论文里“decoder 不能看未来位置、target 输出右移一位、源/目标 embedding 与 pre-softmax 共享权重矩阵”的描述是一致的；而你当前代码里，`src_embed`、`tgt_embed`、`Generator` 还是独立参数，测试中的 `src_mask` 也是全 1，`tgt_mask` 只有因果下三角，还没有真正落地到训练态。

* * *

## 一、先说清楚这次改动的核心思想

### 1\. `target shifted right` 到底怎么落地

最稳的工程写法不是手工挪 tensor，而是：

假设 target 原始序列已经是：

$[BOS,y1,y2,…,yn,EOS,PAD,… ][\text{BOS}, y_1, y_2, \dots, y_n, \text{EOS}, \text{PAD}, \dots][BOS,y1​,y2​,…,yn​,EOS,PAD,…]$

那么训练时：

-   `tgt_input = tgt[:, :-1]`
    
-   `tgt_y = tgt[:, 1:]`
    

于是：

$tgt_input=[BOS,y1,y2,…,yn,EOS]tgt\_input = [\text{BOS}, y_1, y_2, \dots, y_n, \text{EOS}]tgt_input=[BOS,y1​,y2​,…,yn​,EOS]$ $tgt_y=[y1,y2,…,yn,EOS,PAD]tgt\_y = [y_1, y_2, \dots, y_n, \text{EOS}, \text{PAD}]tgt_y=[y1​,y2​,…,yn​,EOS,PAD]$

这就是“右移一位”的实际工程化写法。  
论文里说的 _“the output embeddings are offset by one position”_，本质上就是这个意思。

* * *

### 2\. `tgt_mask` 为什么必须把 padding mask 也并进去

你当前只有下三角 mask，这只解决了：

-   不能看未来 token
    

但没解决：

-   不能看 target 里的 PAD
    

所以正确形式是：

$tgt_mask=pad_mask∧causal_mask\text{tgt\_mask} = \text{pad\_mask} \land \text{causal\_mask}tgt_mask=pad_mask∧causal_mask$

其中：

$pad_mask=(tgt_input≠pad_idx)\text{pad\_mask} = (tgt\_input \neq pad\_idx)pad_mask=(tgt_input=pad_idx)$ $causal_mask=LowerTriangular(T,T)\text{causal\_mask} = \text{LowerTriangular}(T, T)causal_mask=LowerTriangular(T,T)$

最终两者按位与，得到 decoder 自注意力真正该用的 mask。

* * *

### 3\. `src_mask` 应该是什么形状

在你当前 `MultiHeadedAttention.forward()` 里，传进来的 `mask` 会先做一次：

```
mask = mask.unsqueeze(1)
```

所以为了和你现有实现兼容，最合适的输入形状是：

-   `src_mask`: `(B, 1, S)`
    
-   `tgt_mask`: `(B, T, T)`
    

这样进入 `attention()` 时广播正好成立。

这个点很关键。  
因为如果你把 `tgt_mask` 先做成 `(B, 1, T, T)`，再进 `MultiHeadedAttention`，它又会 `unsqueeze(1)` 一次，维度就错了。

* * *

### 4\. 权重共享为什么要现在修

论文明确说了：

-   两个 embedding 层共享同一个权重矩阵
    
-   pre-softmax linear transformation 也共享这组权重
    

同时论文的英德任务还用了**共享的 source-target BPE 词表**，大约 37k token。

所以你现在如果后面走 joint BPE / shared vocab 路线，那么现在就应该把：

```
src_embed.lut.weighttgt_embed.lut.weightgenerator.proj.weight
```

绑在一起。

* * *

## 二、你应该新增的文件：`data/batch.py`

这个文件专门负责：

-   target shifted right
    
-   src\_mask
    
-   tgt\_mask
    
-   统一 batch 封装
    

### `data/batch.py`

```
from __future__ import annotationsfrom dataclasses import dataclassfrom typing import Tupleimport torch"""功能概述：1. 构造 Transformer 训练阶段所需的 src_mask。2. 构造同时包含 padding mask + causal mask 的 tgt_mask。3. 将完整 target 序列切分为：   - tgt_input: 送入 Decoder 的输入（右移后的序列）   - tgt_y:     监督标签4. 用 Seq2SeqBatch 统一封装 batch，方便后续训练代码直接调用。重要约定：1. 传入的 tgt 必须已经包含 <bos> 和 <eos>。2. 这里的 shifted right 采用最稳妥的切片写法：   tgt_input = tgt[:, :-1]   tgt_y     = tgt[:, 1:]3. 当前实现与用户现有的 MultiHeadedAttention 兼容：   - src_mask 形状: (B, 1, S)   - tgt_mask 形状: (B, T, T)"""def subsequent_mask(size: int, device: torch.device) -> torch.Tensor:    """    生成下三角因果 Mask。    输入：        size: 目标序列长度 T        device: 张量所在设备    输出：        mask: 形状为 (1, T, T) 的 bool Tensor              True  表示允许关注              False 表示禁止关注    """    return torch.tril(        torch.ones((1, size, size), dtype=torch.bool, device=device)    )def make_src_mask(src: torch.Tensor, pad_idx: int) -> torch.Tensor:    """    根据源序列构造 src_mask。    输入：        src: 形状 (B, S) 的源序列 token id        pad_idx: PAD 的词表 id    输出：        src_mask: 形状 (B, 1, S) 的 bool Tensor    """    return (src != pad_idx).unsqueeze(1)def shift_right(tgt: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:    """    将完整 target 序列切分为 Decoder 输入和监督标签。    输入：        tgt: 形状 (B, T_full) 的完整目标序列             要求已经包含 <bos> 和 <eos>    输出：        tgt_input: 形状 (B, T_full - 1)                   送入 Decoder 的输入        tgt_y:     形状 (B, T_full - 1)                   训练监督标签    示例：        tgt = [BOS, w1, w2, EOS, PAD]        tgt_input = [BOS, w1, w2, EOS]        tgt_y     = [w1,  w2, EOS, PAD]    """    if tgt.dim() != 2:        raise ValueError(f"tgt 必须是 2 维张量，当前维度为: {tgt.dim()}")    if tgt.size(1) < 2:        raise ValueError(            "tgt 的序列长度至少要 >= 2，"            "因为必须能够切分出 tgt_input 和 tgt_y。"        )    tgt_input = tgt[:, :-1].contiguous()    tgt_y = tgt[:, 1:].contiguous()    return tgt_input, tgt_ydef make_tgt_mask(tgt_input: torch.Tensor, pad_idx: int) -> torch.Tensor:    """    构造 Decoder 自注意力所需的 tgt_mask。    输入：        tgt_input: 形状 (B, T) 的 Decoder 输入序列        pad_idx: PAD 的词表 id    输出：        tgt_mask: 形状 (B, T, T) 的 bool Tensor    构造逻辑：        1. padding mask: 屏蔽掉 PAD 位置           形状先为 (B, 1, T)        2. causal mask: 只允许看当前位置及之前位置           形状为 (1, T, T)        3. 两者按位与，广播后得到 (B, T, T)    """    if tgt_input.dim() != 2:        raise ValueError(            f"tgt_input 必须是 2 维张量，当前维度为: {tgt_input.dim()}"        )    pad_mask = (tgt_input != pad_idx).unsqueeze(1)          # (B, 1, T)    causal_mask = subsequent_mask(tgt_input.size(1), tgt_input.device)  # (1, T, T)    tgt_mask = pad_mask & causal_mask                       # -> (B, T, T)    return tgt_mask@dataclassclass Seq2SeqBatch:    """    统一封装 Transformer 训练所需的一个 batch。    字段说明：        src:       源序列，形状 (B, S)        tgt_input: Decoder 输入，形状 (B, T)        tgt_y:     监督标签，形状 (B, T)        src_mask:  源序列 mask，形状 (B, 1, S)        tgt_mask:  目标序列 mask，形状 (B, T, T)        ntokens:   tgt_y 中非 PAD token 的数量    """    src: torch.Tensor    tgt_input: torch.Tensor    tgt_y: torch.Tensor    src_mask: torch.Tensor    tgt_mask: torch.Tensor    ntokens: int    @classmethod    def from_tensors(        cls,        src: torch.Tensor,        tgt: torch.Tensor,        pad_idx: int,    ) -> "Seq2SeqBatch":        """        从 src / tgt 原始张量构造训练 batch。        输入：            src: 形状 (B, S)            tgt: 形状 (B, T_full)，要求已包含 <bos> 和 <eos>            pad_idx: PAD 的词表 id        """        src_mask = make_src_mask(src, pad_idx)        tgt_input, tgt_y = shift_right(tgt)        tgt_mask = make_tgt_mask(tgt_input, pad_idx)        ntokens = int((tgt_y != pad_idx).sum().item())        return cls(            src=src,            tgt_input=tgt_input,            tgt_y=tgt_y,            src_mask=src_mask,            tgt_mask=tgt_mask,            ntokens=ntokens,        )    def to(self, device: torch.device) -> "Seq2SeqBatch":        """        将 batch 移动到指定设备。        """        return Seq2SeqBatch(            src=self.src.to(device, non_blocking=True),            tgt_input=self.tgt_input.to(device, non_blocking=True),            tgt_y=self.tgt_y.to(device, non_blocking=True),            src_mask=self.src_mask.to(device, non_blocking=True),            tgt_mask=self.tgt_mask.to(device, non_blocking=True),            ntokens=self.ntokens,        )
```

* * *

## 三、修改 `nets/utils/Generator.py`

这里建议你把输出投影层写得更干净一点，方便做 weight tying。

### `nets/utils/Generator.py`

```
import torchimport torch.nn as nnclass Generator(nn.Module):    """    Generator / 词表投影头    功能：        将 Decoder 最后一层输出的隐状态映射到词表维度。    说明：        这里默认使用 bias=False，原因是后续要做 weight tying，        即让 generator.proj.weight 与 embedding.weight 共享同一组参数。    """    def __init__(self, d_model: int, vocab_size: int, bias: bool = False):        super().__init__()        self.proj = nn.Linear(d_model, vocab_size, bias=bias)    def forward(self, x: torch.Tensor) -> torch.Tensor:        """        输入：            x: (B, T, d_model)        输出：            logits: (B, T, vocab_size)        """        return self.proj(x)
```

* * *

## 四、修改 `nets/build_transformer.py`

这里做三件事：

1.  增加 `share_embeddings=True`
    
2.  在构图后做权重绑定
    
3.  测试代码改成真正的训练态输入方式
    

### `nets/build_transformer.py`

```
import copyimport mathimport torchimport torch.nn as nnfrom nets.utils.attention import MultiHeadedAttentionfrom nets.utils.PositionwiseFeedForward import PositionwiseFeedForwardfrom nets.utils.Generator import Generatorfrom nets.utils.encoder_decoder import (    Encoder,    Decoder,    EncoderLayer,    DecoderLayer,    EncoderDecoder,    Embeddings,)def tie_transformer_weights(model: EncoderDecoder) -> None:    """    将 Transformer 的三处权重绑定为同一组参数。    绑定对象：        1. src_embed.lut.weight        2. tgt_embed.lut.weight        3. generator.proj.weight    说明：        这要求 src/tgt 使用同一个 joint vocabulary，        也就是同一套 token id 体系。    """    shared_weight = model.tgt_embed.lut.weight    # 源 embedding 与目标 embedding 共享    model.src_embed.lut.weight = shared_weight    # Generator 输出投影与 embedding 共享    model.generator.proj.weight = shared_weightdef make_model(    src_vocab: int,    tgt_vocab: int,    N: int = 6,    d_model: int = 512,    d_ff: int = 2048,    h: int = 8,    dropout: float = 0.1,    share_embeddings: bool = True,) -> EncoderDecoder:    """    构建完整 Transformer 模型。    参数：        src_vocab: 源语言词表大小        tgt_vocab: 目标语言词表大小        N: Encoder / Decoder 层数        d_model: 隐状态维度        d_ff: FFN 中间维度        h: 多头注意力头数        dropout: dropout 概率        share_embeddings: 是否共享 src/tgt embedding 以及 generator 权重    返回：        完整的 EncoderDecoder 模型    """    c = copy.deepcopy    if share_embeddings and src_vocab != tgt_vocab:        raise ValueError(            "当 share_embeddings=True 时，src_vocab 和 tgt_vocab 必须一致。"            "这通常意味着你使用的是 joint source-target vocabulary。"        )    attn = MultiHeadedAttention(h, d_model, dropout)    ff = PositionwiseFeedForward(d_model, d_ff, dropout)    model = EncoderDecoder(        encoder=Encoder(            EncoderLayer(d_model, c(attn), c(ff), dropout),            N,        ),        decoder=Decoder(            DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout),            N,        ),        src_embed=Embeddings(d_model, src_vocab, dropout),        tgt_embed=Embeddings(d_model, tgt_vocab, dropout),        generator=Generator(d_model, tgt_vocab, bias=False),    )    # 先做权重绑定，再做初始化。    # 这样共享参数只会保留一份物理权重。    if share_embeddings:        tie_transformer_weights(model)    # Xavier / Glorot 初始化    for p in model.parameters():        if p.dim() > 1:            nn.init.xavier_uniform_(p)    return modeldef test_factory_method() -> None:    """    测试：    1. 权重共享是否成功    2. src_mask / tgt_mask 是否按训练态正确构造    3. 前向维度是否正确    """    from data.batch import Seq2SeqBatch    print("=== 开始测试工厂函数 make_model ===")    src_vocab_size = 5000    tgt_vocab_size = 5000    batch_size = 3    pad_idx = 0    bos_idx = 1    eos_idx = 2    print("正在构建微型 Transformer 模型 (2层)...")    model = make_model(        src_vocab=src_vocab_size,        tgt_vocab=tgt_vocab_size,        N=2,        d_model=256,        d_ff=1024,        h=4,        dropout=0.1,        share_embeddings=True,    )    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)    print(f"模型构建完成！总可训练参数量: {total_params / 1e6:.2f} M")    print("-" * 60)    # 构造带 PAD 的源序列    src_tokens = torch.tensor([        [11, 12, 13, 14, 15, pad_idx, pad_idx],        [21, 22, 23, 24, pad_idx, pad_idx, pad_idx],        [31, 32, 33, 34, 35, 36, 37],    ], dtype=torch.long)    # 构造完整目标序列：必须已经包含 BOS / EOS / PAD    tgt_tokens = torch.tensor([        [bos_idx, 101, 102, 103, eos_idx, pad_idx, pad_idx],        [bos_idx, 201, 202, eos_idx, pad_idx, pad_idx, pad_idx],        [bos_idx, 301, 302, 303, 304, eos_idx, pad_idx],    ], dtype=torch.long)    batch = Seq2SeqBatch.from_tensors(        src=src_tokens,        tgt=tgt_tokens,        pad_idx=pad_idx,    )    print(f"src.shape      = {batch.src.shape}")    print(f"tgt_input.shape= {batch.tgt_input.shape}")    print(f"tgt_y.shape    = {batch.tgt_y.shape}")    print(f"src_mask.shape = {batch.src_mask.shape}")    print(f"tgt_mask.shape = {batch.tgt_mask.shape}")    print(f"ntokens        = {batch.ntokens}")    # 检查共享权重是否真的指向同一块内存    src_ptr = model.src_embed.lut.weight.data_ptr()    tgt_ptr = model.tgt_embed.lut.weight.data_ptr()    gen_ptr = model.generator.proj.weight.data_ptr()    assert src_ptr == tgt_ptr == gen_ptr, "权重共享失败：三者没有绑定到同一块内存。"    # 前向传播：注意这里必须传入 tgt_input，而不是完整 tgt    hidden_states = model(        batch.src,        batch.tgt_input,        batch.src_mask,        batch.tgt_mask,    )    print(f"Decoder 输出隐状态维度: {hidden_states.shape}")    assert hidden_states.shape == (batch_size, batch.tgt_input.size(1), 256)    logits = model.generator(hidden_states)    print(f"最终 logits 维度: {logits.shape}")    assert logits.shape == (batch_size, batch.tgt_input.size(1), tgt_vocab_size)    print("\n✅ 测试通过：")    print("1. 权重共享正常")    print("2. target shifted right 正常")    print("3. src_mask / tgt_mask 正常")    print("4. 前向传播正常")if __name__ == "__main__":    test_factory_method()
```

* * *

## 五、`encoder_decoder.py` 你暂时不用大改

这个文件目前**逻辑上不需要改结构**。  
你只要在调用时，把原来传进去的 `tgt` 改成 `tgt_input` 即可。

也就是说训练阶段应该从：

```
hidden = model(src, tgt, src_mask, tgt_mask)
```

变成：

```
hidden = model(batch.src, batch.tgt_input, batch.src_mask, batch.tgt_mask)
```

也就是：

-   model 输入的是 `tgt_input`
    
-   loss 对齐的是 `tgt_y`
    

这是 seq2seq teacher forcing 的标准训练接口。

* * *

## 六、训练侧现在应该怎么用

先给你一个最小使用例子。

```
from data.batch import Seq2SeqBatchpad_idx = 0# src: (B, S)# tgt: (B, T_full)，要求已经包含 BOS / EOS / PADbatch = Seq2SeqBatch.from_tensors(src, tgt, pad_idx=pad_idx)hidden = model(    batch.src,    batch.tgt_input,    batch.src_mask,    batch.tgt_mask,)logits = model.generator(hidden)# 后面你的损失函数要和 batch.tgt_y 对齐# logits: (B, T, V)# tgt_y : (B, T)
```

* * *

## 七、关于数据：你应该去哪里下载

这里我直接给你结论。

### 路线 A：严格朝论文靠拢

优先用 **WMT14 官方原始数据**。  
WMT14 官方页面给出了训练数据来源，包括：

-   Europarl
    
-   News Commentary
    
-   Common Crawl
    

而且官方页面还明确说了：

-   数据是 UTF-8
    
-   **未 tokenized**
    
-   Moses 的 `tokenizer.perl`、`lowercase.perl`、`detokenizer.perl` 等脚本可用于预处理。Statistical Machine Translation
    

论文里对应的英德设置是：

-   WMT 2014 English-German
    
-   大约 4.5M sentence pairs
    
-   使用 **shared source-target BPE vocabulary**
    
-   词表规模约 **37000 tokens**。
    

所以如果你的目标是“尽量复现论文”，最稳的路线就是：

1.  下官方 WMT14 原始平行语料
    
2.  自己做 tokenizer / clean / joint BPE
    
3.  自己产出 `train.en / train.de / valid.en / valid.de / test.en / test.de`
    

* * *

### 路线 B：先把工程打通

也可以用 Hugging Face `datasets` 体系先快速打通。  
Hugging Face 的 `datasets` 文档里明确列出了 `wmt14` 这个数据集名，并说明了 `load_dataset()` 的标准加载机制。Hugging Face

但是你这次要的是“论文复现”，所以我更建议：

-   **调试阶段**可以用 `datasets`
    
-   **正式复现阶段**还是自己走官方 WMT14 原始数据 + 自己的预处理链
    

* * *

### 一个必须提醒你的坑

现在 fairseq README 里那个 `prepare-wmt14en2de.sh`，当前默认行为并不是“纯 WMT14 原样复现”。README 明确写了：默认会产出一个“参考 Attention Is All You Need”的数据版本，但**额外带上了 WMT'17 的 news-commentary-v12**。GitHub

所以如果你追求“尽量按论文原始数据”，**不要无脑把 fairseq 默认脚本输出当成论文原版数据**。

* * *

## 八、先给你一套本地文本文件版的数据加载代码

我这里先给你一个**不和训练细节绑定**、可复用性很高的版本。

这个版本假设你后续已经准备好了下面这样的目录：

```
data/wmt14_ende/├── train.en├── train.de├── valid.en├── valid.de├── test.en└── test.de
```

或者你后面做完 joint BPE，也可以换成：

```
data/wmt14_ende_bpe/├── train.en├── train.de├── valid.en├── valid.de├── test.en└── test.de
```

也就是：  
**文件内容是逐行对齐的平行语料**，至于是不是 BPE 文本，这个 loader 不关心。

* * *

## 九、数据集代码：`data/parallel_text_dataset.py`

```
from __future__ import annotationsfrom typing import Callable, List, Optional, Sequence, Tupleimport torchfrom torch.utils.data import Dataset, DataLoaderfrom data.batch import Seq2SeqBatch"""功能概述：1. 读取本地逐行对齐的平行语料文件。2. 在 collate_fn 中完成：   - 文本 -> token id   - 添加特殊符号   - padding   - 构造 shifted-right 的 tgt_input / tgt_y   - 构造 src_mask / tgt_mask3. 返回统一封装后的 Seq2SeqBatch。当前版本定位：- 这是“先把数据接口打通”的版本。- 目前仍然是按 batch_size（句子数）组 batch。- 还没有做到论文里那种“按 token 数近似分桶”的 batching。- 这个部分你后面做训练细节时再继续升级。"""TokenizeFn = Callable[[str], List[int]]class ParallelTextDataset(Dataset):    """    逐行对齐的平行语料数据集。    每个样本返回：        (src_text, tgt_text)    """    def __init__(        self,        src_path: str,        tgt_path: str,        skip_empty: bool = True,        encoding: str = "utf-8",    ) -> None:        super().__init__()        with open(src_path, "r", encoding=encoding) as f:            src_lines = f.readlines()        with open(tgt_path, "r", encoding=encoding) as f:            tgt_lines = f.readlines()        if len(src_lines) != len(tgt_lines):            raise ValueError(                f"源文件和目标文件行数不一致："                f"{src_path} -> {len(src_lines)} 行, "                f"{tgt_path} -> {len(tgt_lines)} 行"            )        pairs: List[Tuple[str, str]] = []        for src_text, tgt_text in zip(src_lines, tgt_lines):            src_text = src_text.strip()            tgt_text = tgt_text.strip()            if skip_empty and (not src_text or not tgt_text):                continue            pairs.append((src_text, tgt_text))        self.pairs = pairs    def __len__(self) -> int:        return len(self.pairs)    def __getitem__(self, index: int) -> Tuple[str, str]:        return self.pairs[index]def pad_sequences(sequences: Sequence[List[int]], pad_idx: int) -> torch.Tensor:    """    将变长 id 序列 padding 成统一长度的 LongTensor。    输入：        sequences: List[List[int]]        pad_idx: PAD 的词表 id    输出：        tensor: (B, L_max)    """    if len(sequences) == 0:        raise ValueError("sequences 不能为空。")    max_len = max(len(seq) for seq in sequences)    batch_size = len(sequences)    out = torch.full((batch_size, max_len), pad_idx, dtype=torch.long)    for i, seq in enumerate(sequences):        out[i, :len(seq)] = torch.tensor(seq, dtype=torch.long)    return outdef build_translation_collate_fn(    src_encode: TokenizeFn,    tgt_encode: TokenizeFn,    pad_idx: int,    bos_idx: int,    eos_idx: int,    max_src_len: Optional[int] = None,    max_tgt_len: Optional[int] = None,    add_src_eos: bool = True,) -> Callable[[Sequence[Tuple[str, str]]], Seq2SeqBatch]:    """    构造 DataLoader 所需的 collate_fn。    说明：        1. 源序列默认可选地追加 EOS。        2. 目标序列一定追加 BOS / EOS。        3. 最终返回 Seq2SeqBatch，内部已经包含：           - tgt_input           - tgt_y           - src_mask           - tgt_mask    参数：        src_encode: 源语言文本编码函数，输入 str，输出 List[int]        tgt_encode: 目标语言文本编码函数，输入 str，输出 List[int]        pad_idx: PAD id        bos_idx: BOS id        eos_idx: EOS id        max_src_len: 源序列最大长度（不含额外加的 EOS 时的截断长度）        max_tgt_len: 目标序列最大长度（不含 BOS/EOS 时的截断长度）        add_src_eos: 是否给源序列末尾加 EOS    """    def collate_fn(batch: Sequence[Tuple[str, str]]) -> Seq2SeqBatch:        src_id_list: List[List[int]] = []        tgt_id_list: List[List[int]] = []        for src_text, tgt_text in batch:            src_ids = src_encode(src_text)            tgt_ids = tgt_encode(tgt_text)            if max_src_len is not None:                src_ids = src_ids[:max_src_len]            if max_tgt_len is not None:                tgt_ids = tgt_ids[:max_tgt_len]            if add_src_eos:                src_ids = src_ids + [eos_idx]            # 目标序列必须显式带 BOS / EOS，后续才能做 shifted right            tgt_ids = [bos_idx] + tgt_ids + [eos_idx]            src_id_list.append(src_ids)            tgt_id_list.append(tgt_ids)        src_tensor = pad_sequences(src_id_list, pad_idx=pad_idx)        tgt_tensor = pad_sequences(tgt_id_list, pad_idx=pad_idx)        batch_obj = Seq2SeqBatch.from_tensors(            src=src_tensor,            tgt=tgt_tensor,            pad_idx=pad_idx,        )        return batch_obj    return collate_fndef build_parallel_dataloader(    src_path: str,    tgt_path: str,    src_encode: TokenizeFn,    tgt_encode: TokenizeFn,    pad_idx: int,    bos_idx: int,    eos_idx: int,    batch_size: int,    shuffle: bool,    num_workers: int = 0,    pin_memory: bool = True,    max_src_len: Optional[int] = None,    max_tgt_len: Optional[int] = None,    add_src_eos: bool = True,) -> DataLoader:    """    构建平行语料 DataLoader。    返回：        DataLoader，每次迭代返回一个 Seq2SeqBatch    """    dataset = ParallelTextDataset(        src_path=src_path,        tgt_path=tgt_path,        skip_empty=True,    )    collate_fn = build_translation_collate_fn(        src_encode=src_encode,        tgt_encode=tgt_encode,        pad_idx=pad_idx,        bos_idx=bos_idx,        eos_idx=eos_idx,        max_src_len=max_src_len,        max_tgt_len=max_tgt_len,        add_src_eos=add_src_eos,    )    loader = DataLoader(        dataset=dataset,        batch_size=batch_size,        shuffle=shuffle,        num_workers=num_workers,        pin_memory=pin_memory,        collate_fn=collate_fn,    )    return loader
```

* * *

## 十、这个 DataLoader 现在怎么接

### 如果你后面用 joint tokenizer

最常见的情况是：

-   源和目标共用一个 BPE / SentencePiece 模型
    
-   所以 `src_encode` 和 `tgt_encode` 可以是同一个函数
    

例如：

```
import sentencepiece as spmfrom data.parallel_text_dataset import build_parallel_dataloadersp = spm.SentencePieceProcessor(model_file="data/joint_bpe.model")def encode_fn(text: str):    return sp.encode(text, out_type=int)pad_idx = sp.pad_id()bos_idx = sp.bos_id()eos_idx = sp.eos_id()train_loader = build_parallel_dataloader(    src_path="data/wmt14_ende_bpe/train.en",    tgt_path="data/wmt14_ende_bpe/train.de",    src_encode=encode_fn,    tgt_encode=encode_fn,    pad_idx=pad_idx,    bos_idx=bos_idx,    eos_idx=eos_idx,    batch_size=64,    shuffle=True,    num_workers=4,    pin_memory=True,    max_src_len=128,    max_tgt_len=128,)
```

* * *

## 十一、这个版本和论文还有什么差距

这里我要明确一下，避免你误以为“数据部分已经完全复现”。

### 现在已经解决的

-    shared embedding / generator weight tying
    
-    target shifted right
    
-    `src_mask`
    
-    `tgt_mask = padding + causal`
    
-    本地平行语料 DataLoader 框架
    

### 现在还没解决的

-    官方 WMT14 原始数据的下载脚本
    
-    tokenizer / joint BPE 训练脚本
    
-    Moses 清洗与标准预处理
    
-    论文风格的按长度近似分桶 / 按 token 数 batching
    
-    label smoothing
    
-    Noam lr
    
-    beam search
    
-    BLEU 评估闭环
    

其中，论文明确提到英德训练集大约 4.5M 句对、shared BPE 约 37k、batch 是按近似长度聚合并控制 source / target token 数量，而不是简单固定句子数。

所以我上面给你的 `build_parallel_dataloader()`，定位是：

> 先把**正确的数据接口和 mask 逻辑**打通，后面再升级到论文风格 batching。

* * *

## 十二、你现在建议的落地顺序

我建议你下一步就按这个顺序做，不要乱跳：

1.  把我上面这三个代码块先落到项目里
    
2.  先跑通 `test_factory_method()`
    
3.  再用一个很小的人造文本数据集，跑通 `DataLoader -> Seq2SeqBatch -> model forward`
    
4.  确认 shape 全部正确以后，再开始做：
    
    -   官方 WMT14 下载
        
    -   joint BPE / tokenizer
        
    -   真实 train/valid/test 文件生成
        

* * *

## 十三、最后给你一个非常直接的判断

你这一步修完之后，你的项目会从：

> “模型结构能前向跑通”

提升到：

> “模型结构 + seq2seq 训练接口语义是对的”

这两者差别很大。

真正的 Transformer 复现里，**target shifted right + 正确的 padding/causal mask + 正确的 weight tying**，都属于不能含糊的基础设施。论文本身也明确强调了 decoder 的未来位置屏蔽、输出序列右移，以及共享 embedding / pre-softmax 权重矩阵；英德实验则明确基于 WMT14 约 4.5M 句对和共享 BPE 约 37k。

下一条我建议直接进入：

## `WMT14 官方数据下载 + 预处理 + joint BPE 生成 train/valid/test 文本文件`

这一步接上去最顺。

  

wget : Forbidden You don't have permission to access /projects/nmt/data/wmt14.en-de/train.en on this server. Apache/2.2.15 (CentOS) Server at nlp.stanford.edu Port 443 所在位置 行:1 字符: 1 + wget https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/train.en + ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ + CategoryInfo : InvalidOperation: (System.Net.HttpWebRequest:HttpWebRequest) \[Invoke-WebRequest\]，We bException + FullyQualifiedErrorId : WebCmdletWebResponseException,Microsoft.PowerShell.Commands.InvokeWebRequestCommand

  

  

我尝试通过hugging face下载wmt英德数据集： 这是代码： import os from datasets import load\_dataset from tqdm import tqdm def export\_wmt14\_to\_raw\_text(): print("🚀 正在从 Hugging Face 下载 WMT14 英德数据集 (这可能需要一会儿)...") # 使用你找到的那个数据集标识 dataset = load\_dataset("wmt14", "de-en") # 我们主要导出训练集 (train) train\_data = dataset\["train"\] total\_lines = len(train\_data) print(f"✅ 下载成功！训练集共包含 {total\_lines} 对句子。") print("\\n📝 正在将数据导出为原始纯文本格式 (raw.en 和 raw.de)...") # 打开两个文本文件，准备写入 with open("raw.en", "w", encoding="utf-8") as f\_en, \\ open("raw.de", "w", encoding="utf-8") as f\_de: # 遍历数据集，提取英文和德文 # 使用 tqdm 加上进度条，因为 450 万行需要跑一小会儿 for item in tqdm(train\_data, desc="导出进度"): en\_text = item\["translation"\]\["en"\] de\_text = item\["translation"\]\["de"\] # 清理掉句子内部可能带有的换行符，保证"一行一句"的严格对齐 en\_text = en\_text.replace('\\n', ' ').strip() de\_text = de\_text.replace('\\n', ' ').strip() # 写入文件 f\_en.write(en\_text + "\\n") f\_de.write(de\_text + "\\n") print("\\n🎉 导出完成！") print("生成了 raw.en 和 raw.de") print(" 后续可以Tokenizer -> Clean -> Joint BPE ") if \_\_name\_\_ == "\_\_main\_\_": export\_wmt14\_to\_raw\_text() 这是终端结果： PS D:\\1AAAAAstudy\\python\_base\\pytorch\\my\_github\_workspace\\transformer-pytorch-main> python script/download\_dataset.py 🚀 正在从 Hugging Face 下载 WMT14 英德数据集 (这可能需要一会儿)... README.md: 10.5kB \[00:00, 10.5MB/s\] C:\\Users\\zyzking\\AppData\\Local\\Programs\\Python\\Python38\\lib\\site-packages\\huggingface\_hub\\file\_download.py:143: UserWarning: `huggingface_hub` cache-system uses symlinks by default to efficiently store duplicated files but your machine does not support them in C:\\Users\\zyzking\\.cache\\huggingface\\hub\\datasets--wmt14. Caching files will still work but in a degraded version that might require more space on your disk. This warning can be disabled by setting the `HF_HUB_DISABLE_SYMLINKS_WARNING` environment variable. For more details, see https://huggingface.co/docs/huggingface\_hub/how-to-cache#limitations. To support symlinks on Windows, you either need to activate Developer Mode or to run Python as an administrator. In order to activate developer mode, see this article: https://docs.microsoft.com/en-us/windows/apps/get-started/enable-your-device-for-development warnings.warn(message) Xet Storage is enabled for this repo, but the 'hf\_xet' package is not installed. Falling back to regular HTTP download. For better performance, install the package with: `pip install huggingface_hub[hf_xet]` or `pip install hf_xet` train-00000-of-00003.parquet: 100%|█████████████████████████████████████████████████| 280M/280M \[00:11<00:00, 25.3MB/s\] Xet Storage is enabled for this repo, but the 'hf\_xet' package is not installed. Falling back to regular HTTP download. For better performance, install the package with: `pip install huggingface_hub[hf_xet]` or `pip install hf_xet` train-00001-of-00003.parquet: 100%|█████████████████████████████████████████████████| 265M/265M \[00:09<00:00, 27.5MB/s\] Xet Storage is enabled for this repo, but the 'hf\_xet' package is not installed. Falling back to regular HTTP download. For better performance, install the package with: `pip install huggingface_hub[hf_xet]` or `pip install hf_xet` train-00002-of-00003.parquet: 100%|█████████████████████████████████████████████████| 273M/273M \[00:09<00:00, 28.4MB/s\] Xet Storage is enabled for this repo, but the 'hf\_xet' package is not installed. Falling back to regular HTTP download. Xet Storage is enabled for this repo, but the 'hf\_xet' package is not installed. Falling back to regular HTTP download. For better performance, install the package with: `pip install huggingface_hub[hf_xet]` or `pip install hf_xet` validation-00000-of-00001.parquet: 100%|████████████████████████████████████████████| 474k/474k \[00:00<00:00, 8.63MB/s\] Xet Storage is enabled for this repo, but the 'hf\_xet' package is not installed. Falling back to regular HTTP download. For better performance, install the package with: `pip install huggingface_hub[hf_xet]` or `pip install hf_xet` test-00000-of-00001.parquet: 100%|██████████████████████████████████████████████████| 509k/509k \[00:00<00:00, 29.8MB/s\] Generating train split: 100%|█████████████████████████████████████| 4508785/4508785 \[00:05<00:00, 867708.83 examples/s\] Generating validation split: 100%|██████████████████████████████████████| 3000/3000 \[00:00<00:00, 588674.25 examples/s\] Generating test split: 100%|████████████████████████████████████████████| 3003/3003 \[00:00<00:00, 594127.12 examples/s\] ✅ 下载成功！训练集共包含 4508785 对句子。 📝 正在将数据导出为原始纯文本格式 (raw.en 和 raw.de)... 导出进度: 100%|███████████████████████████████████████████████████████████| 4508785/4508785 \[02:02<00:00, 36671.82it/s\] 🎉 导出完成！ 生成了 raw.en 和 raw.de 后续可以Tokenizer -> Clean -> Joint BPE 我这一步正确吗？后续干什么？

  

## 结论

**这一步基本是正确的，而且是一个可用的起点。**

你现在做成了这几件事：

-   成功用 `load_dataset("wmt14", "de-en")` 拉下了 Hugging Face 上的 WMT14 英德数据；
    
-   训练集样本数是 `4,508,785`，验证集 `3,000`，测试集 `3,003`，这和 Hugging Face 这个 `de-en` 配置的公开数据统计是一致的；论文里对 WMT14 English-German 的描述也是“about 4.5 million sentence pairs”。Hugging Face
    
-   你把训练集导出成了逐行对齐的 `raw.en` / `raw.de`，并且做了句内换行替换，这个做法本身没有问题，适合作为后续预处理的输入。Hugging Face+1
    

但是要非常明确地说：

> **“下载成功”不等于“已经严格对齐论文复现数据”。**

原因有两个：

1.  你现在只导出了 `train`，还没有把 `validation` 和 `test` 同样落成文本文件。
    
2.  Hugging Face 的 `wmt14` 数据卡明确提示：**Common Crawl 语料存在问题**，包括非英语文件中混入大量英语句子，以及所谓的“平行句”存在未对齐的问题。这个警告是公开写在数据卡里的。Hugging Face
    

所以，**你的当前结果可以继续往下做工程**，但如果你的目标是“尽量贴近论文实验口径”，后面必须再加一层数据检查和预处理。

* * *

## 一、你这一步为什么说“基本正确”

### 1\. 你拿到的数据规模是对的

Hugging Face 的 `wmt14 / de-en` 数据卡写的是：

-   train: `4,508,785`
    
-   validation: `3,000`
    
-   test: `3,003` Hugging Face
    

你终端里打印出来的训练集数量也是 `4508785`，所以这一步没有跑偏。Hugging Face

### 2\. 你的导出方式是对的

你做的事情本质上是：

-   从 `dataset["train"]`
    
-   取 `item["translation"]["en"]`
    
-   取 `item["translation"]["de"]`
    
-   清理内部换行
    
-   分别写进两个文本文件
    

这就是标准的“平行语料一行一句”导出方式。  
后面无论你用：

-   `subword-nmt`
    
-   `SentencePiece`
    
-   自己写 tokenizer / loader
    

这种文本格式都最方便接。

### 3\. 从工程上看，这比你之前直接 `wget Stanford` 更稳

你前面那个 Stanford 链接 403，本质上说明：

-   那个地址现在不能直接这样抓
    
-   即使历史上很多教程提过，也不该把它当成当前稳定来源
    

Hugging Face 至少现在是可用的、可编程访问的，而且 `load_dataset()` 也是官方推荐入口之一。Hugging Face+1

* * *

## 二、但这里有一个重要风险：HF 数据卡的 Common Crawl 警告

这点你一定要记住，不然后面你会误以为“数据已经完全干净”。

Hugging Face `wmt14` 数据卡写得很直接：

-   Common Crawl 语料有问题；
    
-   非英语文件里混入很多英语句子；
    
-   英文那一侧与对应句子**不对齐**。Hugging Face
    

这意味着什么？

### 你可以这样理解

你现在下载到的是一个“**可用版本**的 WMT14 英德数据接口”，但它并不自动保证：

-   每一对句子都高质量对齐；
    
-   每一条都适合直接拿去训练；
    
-   与论文当年的预处理链完全一致。
    

所以后面你至少要做：

-   空行过滤
    
-   极端长度过滤
    
-   长度比过滤
    
-   简单语言噪声检查
    

否则训练时会吃进去不少脏样本。

* * *

## 三、你现在最该做的，不是立刻训练 tokenizer，而是先补齐数据落盘

你现在只导出了 `train`。  
下一步应该先把三个 split 都导出来，至少形成下面这个目录：

```
data/wmt14_raw/├── train.en├── train.de├── valid.en├── valid.de├── test.en└── test.de
```

因为论文训练/验证/测试是一个完整闭环。Transformer 论文提到 WMT14 English-German 约 4.5M 句对，并在英德翻译任务上报告的是 `newstest2014` 的结果；训练部分也明确提到 shared BPE vocab 约 37k。

你如果现在只保存 `train`，后面很多环节都会卡住：

-   BPE 是否只用 train 训练
    
-   valid/test 如何编码
    
-   后续 BLEU 怎么对齐
    
-   推理时如何还原目标句
    

* * *

## 四、我建议你下一步按这个顺序走

## 1\. 先把导出脚本改成“三个 split 全导出”

你当前脚本逻辑对，但应该升级成同时导出：

-   `train`
    
-   `validation`
    
-   `test`
    

并且建议按数据集原生 split 分文件，不要自己再手切。

### 推荐目录命名

```
data/wmt14_raw/    train.en    train.de    valid.en    valid.de    test.en    test.de
```

### 这里为什么很重要

因为 Hugging Face 这个 `de-en` 配置已经给了：

-   train 4,508,785
    
-   validation 3,000
    
-   test 3,003 Hugging Face
    

你直接沿用这三个 split，后续最省事。

* * *

## 2\. 导出之后做一次“原始语料体检”

这一步很有必要。你至少检查下面几项：

-    `train.en` 和 `train.de` 行数是否完全一致
    
-    是否存在空行
    
-    是否存在超长句
    
-    是否存在明显长度比异常的句对
    
-    是否存在德语侧明显是英语的噪声样本
    

### 为什么要做长度比过滤

因为平行语料里常见脏样本是：

-   一边很短，一边特别长
    
-   一边是标题，一边是整段正文
    
-   一边不是同语种
    

对 seq2seq 来说，这种噪声很伤。

### 一个常见经验阈值

你后面可以考虑先用这样的规则做清洗：

-   源句和目标句长度都要 `>= 1`
    
-   token 数都要 `<= 250` 或 `<= 200`
    
-   长度比不超过 `1:3` 或 `1:4`
    

这不是论文原文硬性写出来的规则，而是机器翻译里常见、实用的清洗策略。HF 数据卡对 Common Crawl 的问题警告，本身就说明你做这一步是合理的。Hugging Face

* * *

## 3\. 然后再做 shared BPE / tokenizer

这里你要先定路线：

### 路线 A：更贴近论文

使用 **shared BPE vocabulary，大小约 37000**。论文明确写了 WMT14 English-German 用的是 byte-pair encoding，并且是 shared source-target vocabulary，约 37000 token。

这条路线最适合你的“复现论文”目标。

你可以选：

-   `subword-nmt`：更接近当年的经典 BPE 工作流
    
-   `SentencePiece BPE`：工程更现代，接口更舒服，但不是最历史同构
    

如果你现在目标是“论文复现优先”，我建议：

> **优先用 BPE，不要先上 unigram。**

### 路线 B：工程优先

直接用 `SentencePiece` 训练一个 joint tokenizer。  
这样后面：

-   `pad_id`
    
-   `bos_id`
    
-   `eos_id`
    
-   `unk_id`
    

都会更统一，接你前面做的 `shifted right / mask / shared embedding` 也更顺。

* * *

## 4\. tokenizer 训练时，应该只用 train，不要把 valid/test 混进去

这是一个很容易犯的错误。

你后面训练 BPE / tokenizer 时：

-   **只用 `train.en + train.de`**
    
-   不要把 `valid` / `test` 混进去
    

原因很简单：

-   valid/test 是评估集，不应该参与训练任何数据驱动的文本建模步骤
    
-   tokenizer / BPE 虽然不是模型参数，但它本质上也是“从文本统计中学习规则”
    

如果你把 test 也喂进去，会造成评估泄露。

* * *

## 5\. tokenizer 训练完以后，编码全部 split

你最终应该得到类似这样的一组文件：

```
data/wmt14_bpe/├── train.en├── train.de├── valid.en├── valid.de├── test.en└── test.de
```

注意这里的 `.en/.de` 文件内容已经不再是原始自然语言，而是 BPE 后的 token 文本，或者你也可以直接在 DataLoader 中动态编码。

* * *

## 6\. 然后再接你前面刚修好的 batch/mask 逻辑

你前面已经在模型侧准备做：

-   `target shifted right`
    
-   `src_mask`
    
-   `tgt_mask = padding + causal`
    
-   shared embedding / generator weight tying
    

而你的当前代码里，`make_model()` 还没有做 embedding / generator 的共享，测试里的 `src_mask` 还是全 1，`tgt_mask` 也只是简单下三角，这些都还是“测试态写法”。

所以你后面把真实文本编码成 id 之后，流程应该变成：

1.  原始文本
    
2.  tokenizer/BPE 编码成 id
    
3.  target 加 `BOS/EOS`
    
4.  `tgt_input = tgt[:, :-1]`
    
5.  `tgt_y = tgt[:, 1:]`
    
6.  构造 `src_mask` 和 `tgt_mask`
    
7.  喂给模型
    

这时你整个“数据 → 模型接口”的语义才真正闭环。

* * *

## 五、我建议你立刻把当前脚本升级成下面这个思路

你现在的脚本不要只导 `train`，改成导三份。逻辑上大概这样：

```
def export_split(split_name, dataset, out_dir):    ...
```

然后分别导：

-   `dataset["train"] -> train.en / train.de`
    
-   `dataset["validation"] -> valid.en / valid.de`
    
-   `dataset["test"] -> test.en / test.de`
    

并且顺手统计：

-   行数
    
-   空句数量
    
-   平均长度
    
-   最长句长度
    

这样你马上就能知道这批数据干不干净。

* * *

## 六、你现在这一步之后，“正确的下一步”是什么

不是训练模型。  
也不是先写 loss。  
而是下面这三步：

### 第一步：补齐三个 split 的导出

把 `train/valid/test` 都导出来。

### 第二步：做数据清洗与体检

最少做：

-   空行过滤
    
-   长度截断
    
-   长度比过滤
    
-   噪声检查
    

### 第三步：训练 joint BPE / tokenizer

目标对齐论文时，用 **shared BPE 37000**。

* * *

## 七、非常直接的判断

### 你的当前步骤是否正确？

**对，基本正确。**

但更精确地说：

> 你现在完成的是“拿到一个可用的 WMT14 de-en 原始平行语料版本”，还没有完成“论文复现级数据准备”。

### 你后面干什么？

按优先级排序：

1.  导出 `valid/test`
    
2.  做数据清洗
    
3.  训练 joint BPE/tokenizer
    
4.  编码全部 split
    
5.  再把真实 token id 接入你前面修过的 `batch + mask + shifted-right + shared-weight` 流程
    

* * *

## 八、我给你的建议

下一条你直接让我来做这件事最合适：

**我帮你生成一套完整的数据预处理代码**，包括：

-   从 Hugging Face 导出 `train/valid/test`
    
-   统计数据质量
    
-   清洗平行语料
    
-   用 `SentencePiece BPE` 训练 **joint 37000 vocab**
    
-   编码全部 split
    
-   最终产出可直接给 Transformer DataLoader 用的文件结构
    

这样你下一步就能直接落地。

  

有人推荐我说先用Moses给这两个txt做一下Tokenize和清理，合理吗？ 那BPE我也用Moses还是说自己写py呢？ 先用Moses处理还是先划分呢？

  

  

补充一下地址是这个：https://github.com/moses-smt/mosesdecoder.git

  

我先核对一下 Moses 和常见 BPE 工具的当前状态，然后直接给你一个可执行的处理顺序。

## 结论先说

**合理。**  
如果你现在的目标是：

-   尽量贴近经典 NMT / Transformer 早期复现流程；
    
-   数据是英德这种以空格分词为基础的语言；
    
-   后面还想和很多公开脚本的处理口径尽量一致；
    

那么：

1.  **先用 Moses 做基础文本预处理**，是合理的。
    
2.  **BPE 不用 Moses 做**，而是用 **`subword-nmt`** 或 **SentencePiece**。
    
3.  **先保留官方 split，再做预处理**。也就是说：
    
    -   不要把 `train/valid/test` 混在一起后再自己乱切；
        
    -   你已经从 Hugging Face 拿到了官方的 `train / validation / test`，那就直接沿用这个划分。
        
4.  如果走“经典复现路线”，推荐顺序是：
    

$官方 split 固定→Moses normalize/tokenize→清洗 train（和可选 valid）→只在 train 上学习 joint BPE→把同一套 BPE 应用到 train/valid/test\text{官方 split 固定} \rightarrow \text{Moses normalize/tokenize} \rightarrow \text{清洗 train（和可选 valid）} \rightarrow \text{只在 train 上学习 joint BPE} \rightarrow \text{把同一套 BPE 应用到 train/valid/test}官方 split 固定→Moses normalize/tokenize→清洗 train（和可选 valid）→只在 train 上学习 joint BPE→把同一套 BPE 应用到 train/valid/test$

Transformer 论文明确写了英德任务使用 **shared source-target BPE vocabulary**，规模大约 **37000**。  
而 fairseq 的 `prepare-wmt14en2de.sh` 也明确是：

-   用 **Moses** 的 `tokenizer.perl`、`clean-corpus-n.perl`、`normalize-punctuation.perl`、`remove-non-printing-char.perl`
    
-   再用 **subword-nmt** 做 BPE 预处理。GitHub+2GitHub+2
    

* * *

## 一、先回答你最核心的三个问题

## 1\. 先用 Moses 给 txt 做 tokenize 和清理，合理吗？

**合理，而且很常见。**

特别是你现在做的是 **WMT14 英德翻译复现**，Moses 这套脚本在传统 MT/NMT 里几乎就是标准基础设施。fairseq 官方的 WMT14 En-De 准备脚本就直接克隆了 `mosesdecoder`，并把下面这些脚本拿来用：

-   `tokenizer.perl`
    
-   `clean-corpus-n.perl`
    
-   `normalize-punctuation.perl`
    
-   `remove-non-printing-char.perl` GitHub+1
    

所以如果你的目标是“尽量经典、尽量和老论文/老仓库对齐”，  
**Moses 预处理是对路的**。

* * *

## 2\. BPE 用 Moses 还是自己写 Python？

### 直接结论

**BPE 不用 Moses。**

更准确地说：

-   **Moses**：负责
    
    -   标点规范化
        
    -   去掉不可打印字符
        
    -   tokenization
        
    -   平行语料清洗
        
-   **BPE**：负责子词切分  
    通常用：
    
    -   **`subword-nmt`**：更经典，更贴近早期 NMT / Transformer 复现流程
        
    -   **SentencePiece**：更现代，更省事
        

fairseq 的 WMT14 En-De 脚本就是这样分工的：  
先 clone `mosesdecoder`，再 clone `subword-nmt`，然后用 `learn_bpe.py` 和 `apply_bpe.py` 做 BPE。GitHub+2GitHub+2

`subword-nmt` 官方 README 也明确写了标准用法：

-   `learn-bpe`
    
-   `apply-bpe` GitHub
    

* * *

## 3\. 先用 Moses 处理还是先划分？

### 对你现在这个场景，正确答案是：

**先固定官方划分，再分别处理。**

也就是：

-   先把 Hugging Face 的：
    
    -   `train`
        
    -   `validation`
        
    -   `test`
        
    
    分别导出成：
    
    -   `train.en / train.de`
        
    -   `valid.en / valid.de`
        
    -   `test.en / test.de`
        

然后再对这三组分别做 tokenize / BPE apply。

### 为什么不是“先合并再划分”？

因为你已经有官方 split 了。  
这时候再把它们混起来重新划分，等于破坏评估口径。

fairseq 脚本里之所以会出现“先 tokenization，再自己从训练语料里切 valid”，是因为它下载的是原始训练语料和测试集，然后**自己构造 valid**。你这里不是那个场景。你已经从 Hugging Face 直接拿到了 `validation` 和 `test`。所以你没必要再自己重切。GitHub+1

* * *

## 二、我建议你的路线：分成两种

* * *

## 路线 A：尽量贴近经典 Transformer / fairseq 复现

这是我更推荐你的路线。

### 处理顺序

#### 第 1 步：先保留官方 split

先把文件整理成：

```
data/wmt14_raw/├── train.en├── train.de├── valid.en├── valid.de├── test.en└── test.de
```

不要把这三份混起来。

* * *

#### 第 2 步：Moses 基础预处理

对 `train/valid/test` 的每个 split 分别做：

1.  `normalize-punctuation.perl`
    
2.  `remove-non-printing-char.perl`
    
3.  `tokenizer.perl`
    

这是非常合理的经典流程，fairseq 脚本也是这么干的。GitHub+1

* * *

#### 第 3 步：清洗

这里要分开说：

### 对 `train`

**建议清洗。**

比如：

-   去空行
    
-   最大长度过滤
    
-   长度比例过滤
    

Moses 的 `clean-corpus-n.perl` 就是干这个的。fairseq 脚本在 BPE 之后对 `train` 和 `valid` 都调用了 `clean-corpus-n.perl`，并用了 `1 250` 的长度范围、`ratio 1.5`。GitHub+1

### 对 `valid`

**可以轻度清洗，但要谨慎。**

如果你追求工程稳定，可以像 fairseq 那样也清一下。GitHub+1  
但如果你更看重“官方评估口径尽量不动”，那 valid 最多只去掉明显空行/坏行，不要大规模删。

### 对 `test`

**一般不要用 `clean-corpus-n.perl` 大清洗。**

fairseq 脚本就是直接复制 test 的 BPE 结果，没有对 test 再做 `clean-corpus-n.perl`。GitHub+1

这个逻辑很合理：

-   test 是评估集；
    
-   不要因为清洗规则把测试集改得面目全非。
    

* * *

#### 第 4 步：只在 train 上学习 joint BPE

Transformer 论文英德任务写的是 **shared source-target vocabulary**，大约 **37000** token。

所以你应该：

-   把 **tokenized + cleaned 的 `train.en` 和 `train.de` 拼起来**
    
-   只用这份 train 文本学习一套 joint BPE codes
    
-   再把**同一套 codes** 应用到：
    
    -   `train.en / train.de`
        
    -   `valid.en / valid.de`
        
    -   `test.en / test.de`
        

### 这里不要犯两个错误

#### 错误 1：拿 valid/test 一起学 BPE

不要。  
BPE 规则必须只从 training split 学。

#### 错误 2：英德各学一套 BPE

如果你想贴近论文，就不要。  
论文强调的是 **shared source-target vocabulary**。

* * *

## 路线 B：更现代、更省事

如果你不执着于 Moses 传统流程，可以直接用 **SentencePiece BPE**。

SentencePiece README 明确说了：

-   它支持 **BPE** 和 unigram；
    
-   可以**直接从 raw sentences 训练**；
    
-   **不一定需要** Moses 这类预分词。GitHub
    

所以现代简化流程可以是：

$官方 split 固定→轻度清洗→SentencePiece BPE 直接在 raw train.en+train.de 上训练→应用到 train/valid/test\text{官方 split 固定} \rightarrow \text{轻度清洗} \rightarrow \text{SentencePiece BPE 直接在 raw train.en+train.de 上训练} \rightarrow \text{应用到 train/valid/test}官方 split 固定→轻度清洗→SentencePiece BPE 直接在 raw train.en+train.de 上训练→应用到 train/valid/test$

### 这种方案的优点

-   代码更干净
    
-   不依赖 Perl 脚本
    
-   Python 接口更舒服
    
-   后面直接拿 token id 更方便
    

### 这种方案的缺点

-   和经典 Moses + subword-nmt 的老论文/老仓库流程不完全同构
    
-   你如果特别追求“老派复现味道”，它就不如路线 A
    

* * *

## 三、结合你的目标，我给你的明确建议

你现在是：

-   想复现 Transformer 论文的英德任务
    
-   已经拿到了 Hugging Face 的 WMT14 de-en
    
-   现在还在搭数据与模型基础设施
    

### 我的建议是：

> **先走路线 A：Moses + subword-nmt。**

原因很简单：

1.  你的目标不是“随便做个翻译模型”，而是“尽量复现论文”
    
2.  fairseq 的公开准备脚本也是这个口径：Moses 负责 tokenization/clean，subword-nmt 负责 BPE。GitHub+2GitHub+2
    
3.  你后面排查问题时，参考资料会更多，和历史代码库更容易对齐
    

* * *

## 四、你现在最合适的具体顺序

我给你一个你现在就能照着做的顺序。

## 第一步：先导出三个 split

不要只留 `raw.en/raw.de` 两个大文件。  
先整理成：

```
data/wmt14_raw/├── train.en├── train.de├── valid.en├── valid.de├── test.en└── test.de
```

* * *

## 第二步：对三组都做 Moses tokenize 前处理

对每个 split、每个语言做：

-   `normalize-punctuation.perl`
    
-   `remove-non-printing-char.perl`
    
-   `tokenizer.perl`
    

得到：

```
data/wmt14_tok/├── train.en├── train.de├── valid.en├── valid.de├── test.en└── test.de
```

* * *

## 第三步：清洗

### 推荐策略

-   `train`：清洗
    
-   `valid`：轻清洗或按同规则清洗
    
-   `test`：不要做 `clean-corpus-n.perl` 这种强过滤，只做最基本的一致性检查
    

这个做法和 fairseq 脚本精神是一致的：它清 train 和 valid，不清 test。GitHub+1

* * *

## 第四步：训练 joint BPE

用 **train.en + train.de** 拼接出来的训练文本学习 joint BPE。

如果你更想贴论文，vocab / merge 规模就朝 **约 37k** 靠。论文写的是 shared source-target vocabulary about 37000 tokens。

注意：

-   `subword-nmt` 的经典接口是按 merge operations 学习；
    
-   论文写的是最终 shared vocab 约 37k，不是直接告诉你 merge 次数必须是多少；
    
-   所以工程上你可以先设一个接近这个规模的配置，然后统计最终词表。
    

* * *

## 第五步：把同一套 BPE 应用到全部 split

应用到：

-   train.en / train.de
    
-   valid.en / valid.de
    
-   test.en / test.de
    

最终得到：

```
data/wmt14_bpe/├── train.en├── train.de├── valid.en├── valid.de├── test.en└── test.de
```

* * *

## 五、我替你把几个容易混淆的点直接说死

## 1\. “先用 Moses 还是先划分？”

**对你来说：先保留官方划分，再用 Moses。**

不是重新划分。  
你已经有 HF 的 `train/validation/test` 了，别再打散重切。

* * *

## 2\. “BPE 也用 Moses 吗？”

**不。**  
BPE 用：

-   `subword-nmt`：更经典
    
-   或 SentencePiece：更现代
    

Moses 不负责 BPE。fairseq 也是 Moses + subword-nmt 分工。GitHub+2GitHub+2

* * *

## 3\. “先 clean 再 tokenize，还是先 tokenize 再 clean？”

如果你走经典 Moses/fairseq 路线，我建议：

$normalize→remove-non-printing→tokenize→clean\text{normalize} \rightarrow \text{remove-non-printing} \rightarrow \text{tokenize} \rightarrow \text{clean}normalize→remove-non-printing→tokenize→clean$

这和 fairseq 脚本的公开流程一致。它先做 normalize/remove-non-printing/tokenize，然后学 BPE，再 clean train/valid。GitHub+1

* * *

## 4\. “valid/test 也参与 BPE 学习吗？”

**绝对不要。**

BPE codes 只从 train 学。  
然后 apply 到 valid/test。

* * *

## 5\. “我需要 truecase / lowercase 吗？”

**当前阶段先不要急着加。**

因为你现在最关键的是先把：

-   split
    
-   tokenize
    
-   clean
    
-   joint BPE
    
-   DataLoader 接口
    

这一整条链打通。

而且你现在参考的经典 pipeline 里，fairseq 这个 WMT14 En-De 脚本并没有在你看到的主流程里加 lowercase/truecase 这一步。GitHub

* * *

## 六、最终建议

如果你问我“现在最合适怎么做”，我的答案是：

### 你现在就按这个方案走

-    先把 HF 的 `train / validation / test` 全部导成独立文件
    
-    用 Moses 做 normalize + remove-non-printing + tokenize
    
-    对 `train` 做清洗；`valid` 轻清洗可选；`test` 不做强过滤
    
-    用 `subword-nmt` 在 **train.en + train.de** 上学 **joint BPE**
    
-    把同一套 BPE 应用到 `train/valid/test`
    
-    然后再接你前面修好的 `shifted right + src_mask + tgt_mask + weight tying`
    

这条路线最稳，也最符合你当前“尽量贴 Transformer 论文复现”的目标。论文明确写了英德任务使用 shared BPE 约 37k；fairseq 公开脚本则清楚展示了 Moses 做 tokenization/clean、subword-nmt 做 BPE 的经典工程分工。 GitHub+2GitHub+2

下一条我建议直接进入这一项：  
**我给你生成一套完整可运行的数据预处理脚本**，包括：

-   导出 HF 三个 split
    
-   调用 Moses
    
-   训练 joint BPE
    
-   应用到 train/valid/test
    
-   生成最终给你 Transformer DataLoader 用的文件结构
    

  

也就是我之前的脚本只下载了训练集的raw.en和raw.de，不够用，我得先下载好全部的数据，然后再用Moses进行基础预处理，然后再对train清洗、val轻度清洗、test一致性检查。第四步再训练bpe，训练好bpe之后，应用到原始数据上，得到最终的可以用于transformer的训练数据。 我期望按照原始论文的做法来。 请你逐步指导我。 这是我新的下载数据的脚本，你先看下可以用不： import os from datasets import load\_dataset from tqdm import tqdm def export\_split\_to\_text(dataset\_split, out\_en\_path, out\_de\_path, desc\_name): """ 通用导出函数：将指定的 Hugging Face 数据集 split 导出为两个对齐的 txt 文件 """ print(f"\\n📝 正在将 {desc\_name} 导出为 {out\_en\_path} 和 {out\_de\_path}...") with open(out\_en\_path, "w", encoding="utf-8") as f\_en, \\ open(out\_de\_path, "w", encoding="utf-8") as f\_de: for item in tqdm(dataset\_split, desc=f"导出 {desc\_name}"): # 提取并清理换行符，确保严格的单行对齐 en\_text = item\["translation"\]\["en"\].replace('\\n', ' ').strip() de\_text = item\["translation"\]\["de"\].replace('\\n', ' ').strip() f\_en.write(en\_text + "\\n") f\_de.write(de\_text + "\\n") print(f"✅ {desc\_name} 导出完成！共 {len(dataset\_split)} 对句子。") def export\_all\_wmt14(): print("🚀 正在从 Hugging Face 加载完整的 WMT14 英德数据集 (如果已缓存会直接读取)...") # 加载包含 train, validation, test 的完整字典 dataset = load\_dataset("wmt14", "de-en") # 1. 导出训练集 (约 450万条) -> 对应原始训练数据 export\_split\_to\_text(dataset\["train"\], "raw.en", "raw.de", "训练集 (Train)") # 2. 导出验证集 (3000条) -> 对应论文中的 newstest2013 export\_split\_to\_text(dataset\["validation"\], "val.en", "val.de", "验证集 (Val / newstest2013)") # 3. 导出测试集 (3003条) -> 对应论文中的 newstest2014 export\_split\_to\_text(dataset\["test"\], "test.en", "test.de", "测试集 (Test / newstest2014)") print("\\n🎉 所有数据集切分导出大功告成！") print("--------------------------------------------------") print("⚠️ 极其重要的数据预处理铁律 (Data Engineering Rules)：") print("1. 【训练 BPE 词表】：只允许使用 raw.en 和 raw.de (训练集) 来 learn-bpe。绝对不能让模型提前看到验证集和测试集的词汇分布！") print("2. 【Clean 清洗】：只允许对训练集进行长度过滤 (clean-corpus-n.perl)。验证集和测试集绝对不能做 Clean 操作，因为评估 BLEU 分数时必须逐句对应，不能偷偷删掉长句子！") print("3. 【应用 BPE】：用训练集生成的 BPE 规则 (bpe\_rules.codes) 和词汇表 (vocab.txt)，分别 apply-bpe 到验证集和测试集上。") if \_\_name\_\_ == "\_\_main\_\_": export\_all\_wmt14()

  

我先核对一下论文里对英德数据和 shared BPE 的表述，以及公开脚本里对 Moses / BPE / valid/test 的处理方式，然后我会直接按“你当前脚本能不能用 → 哪里要改 → 下一步做什么”的顺序给你。

我已经核对完了。你这个脚本能用，但有两个点需要马上纠正：一是文件命名与目录结构，二是不要把 `validation` 直接硬编码成 `newstest2013`。另外，如果你要走“更贴近原论文/经典 pipeline”的路线，BPE 应该应用在 **Moses 预处理后的文本** 上，不是直接对 raw 文本做 `subword-nmt`。

## 可以用，但要改

你的新脚本**总体可用**，能正确把 Hugging Face 的 `wmt14/de-en` 三个 split 导出成对齐文本。  
你当前这一步的方向是对的：先固定官方 split，再做后续预处理。Transformer 论文对英德任务写的是标准 WMT14 English-German，约 4.5M 句对，使用 shared source-target BPE，词表约 37k；论文结果是在 `newstest2014` 上报告，开发集实验在 `newstest2013` 上做。

但你这个脚本里有 **3 个需要立刻修正** 的点。

* * *

## 一、你这个脚本里需要修的 3 个点

### 1\. 不要再用 `raw.en / raw.de` 这种名字保存训练集

因为你现在已经不只是导出训练集了，而是导出：

-   train
    
-   validation
    
-   test
    

这时最清晰、最不容易出错的命名应该是：

```
data/wmt14_raw/├── train.en├── train.de├── valid.en├── valid.de├── test.en└── test.de
```

否则后面你会很容易混淆：

-   `raw.en` 到底是 train 还是全量拼接？
    
-   后面 Moses 输出是不是 `tok.en`？
    
-   BPE 输出是不是 `bpe.en`？
    

这个命名问题看起来小，后面会非常伤工程可维护性。

* * *

### 2\. 不要把 Hugging Face 的 `validation` 在代码注释里硬写成 `newstest2013`

这点你要谨慎。

论文里英德开发实验确实是看 `newstest2013`，最终测试是 `newstest2014`。  
但 Hugging Face 的 `wmt14.py` 构建脚本里，`VALIDATION` 这个 split 的来源集合写的是：

-   `newsdev2014`
    
-   `newstest2013`
    

而 `TEST` 写的是 `newstest2014`。Hugging Face

所以你现在这行注释：

```
# 2. 导出验证集 (3000条) -> 对应论文中的 newstest2013
```

**不够严谨。**

更稳的写法应该是：

```
# 2. 导出验证集 (Validation split from HF wmt14/de-en)
```

也就是说：

-   `test -> newstest2014`，这个基本可以确认；
    
-   `validation -> 直接说 HF validation split`，不要在脚本注释里擅自等同成单一来源。
    

* * *

### 3\. 你最后一条“应用 BPE 到原始数据上”的表达要改

如果你走的是 **Moses + subword-nmt** 这种经典路线，那么 BPE 的输入不应该是最原始的 raw 文本，而应该是：

$raw→normalize/remove-non-printing/tokenize→BPE\text{raw} \rightarrow \text{normalize/remove-non-printing/tokenize} \rightarrow \text{BPE}raw→normalize/remove-non-printing/tokenize→BPE$

fairseq 的 `prepare-wmt14en2de.sh` 就是这样分工的：

-   先 clone `mosesdecoder`
    
-   再 clone `subword-nmt`
    
-   Moses 负责 `normalize-punctuation.perl`、`remove-non-printing-char.perl`、`tokenizer.perl`、`clean-corpus-n.perl`
    
-   `subword-nmt` 负责 BPE。GitHub+1
    

所以你后面应该说：

> 用训练集生成的 BPE 规则，应用到 **Moses 预处理后的 train/valid/test 文本** 上。

而不是“应用到原始数据上”。

* * *

## 二、如果你要尽量贴近原论文/经典做法，正确顺序是什么

你刚才那段总结，方向对了，但我帮你再收紧一下。

## 推荐顺序

-    第 1 步：先固定官方 split
    
-    第 2 步：对每个 split 做 Moses 基础预处理
    
-    第 3 步：只对 `train` 做正式清洗；`valid/test` 只做一致性检查，不删行
    
-    第 4 步：只在 `train` 上学习 **joint BPE**
    
-    第 5 步：把同一套 BPE 应用到 `train/valid/test`
    
-    第 6 步：再进入你自己的 DataLoader / shifted-right / mask / weight tying 管线
    

* * *

## 三、这里有一个关键修正：你前面说“val 轻度清洗”，如果按论文味道走，我建议改成“只做一致性检查，不删行”

这是因为：

-   论文本身没有给出“对 valid 做 clean-corpus-n 删除句子”的硬性规范；它只说用了标准 WMT14 En-De、BPE、按近似长度 batching。
    
-   fairseq 的公开脚本确实会对 `train` 和 `valid` 调 `clean-corpus-n.perl`，但那个脚本 README 也明确说了：默认产出的数据是“modeled after Attention Is All You Need”，而且默认还额外用了 WMT'17 的 `news-commentary-v12`；若只想用 WMT'14，可用 `--icml17`。这说明它是一个“工程复现脚本”，不是论文原始数据协议本体。GitHub+1
    

所以如果你现在的目标是：

> **尽量按原论文口径，而不是尽量按 fairseq 脚本复刻**

那我建议你这样处理：

### `train`

可以清洗。  
因为训练集允许你去掉明显脏样本、超长样本、极端长度比样本。

### `valid`

只做一致性检查，不删行。  
例如：

-   是否空行
    
-   是否编码异常
    
-   是否源目标行数一致
    

但不要用 `clean-corpus-n.perl` 按长度直接删句对。

### `test`

同样只做一致性检查，不删行。  
因为 test 对应最终评估集，最不该擅自删样本。

* * *

## 四、所以你现在这份脚本，我给你的结论是

## 能用的部分

-   `load_dataset("wmt14", "de-en")` 是对的。
    
-   用统一函数导出 split 是对的。
    
-   句内换行替换成空格是对的。
    
-   保持一行一句、英德严格对齐，这一步是对的。
    

* * *

## 需要改的部分

### 改 1：加目录

不要把文件直接输出到项目根目录。

### 改 2：改文件名

不要用 `raw.en/raw.de`，改成 `train.en/train.de`。

### 改 3：改注释

不要把 HF validation 硬写成 `newstest2013`。

### 改 4：改最后的“铁律”文案

把 “BPE 应用到原始数据” 改成 “BPE 应用到 Moses 预处理后的文本”。

* * *

## 五、我建议你把当前脚本改成这个版本

```
import osfrom datasets import load_datasetfrom tqdm import tqdmdef export_split_to_text(dataset_split, out_en_path, out_de_path, desc_name):    """    将指定的 Hugging Face 数据集 split 导出为两个严格对齐的 txt 文件。    """    os.makedirs(os.path.dirname(out_en_path), exist_ok=True)    print(f"\n正在将 {desc_name} 导出为:")    print(f"  {out_en_path}")    print(f"  {out_de_path}")    with open(out_en_path, "w", encoding="utf-8") as f_en, \         open(out_de_path, "w", encoding="utf-8") as f_de:        for item in tqdm(dataset_split, desc=f"导出 {desc_name}"):            en_text = item["translation"]["en"].replace("\n", " ").strip()            de_text = item["translation"]["de"].replace("\n", " ").strip()            f_en.write(en_text + "\n")            f_de.write(de_text + "\n")    print(f"完成：{desc_name}，共 {len(dataset_split)} 对句子。")def export_all_wmt14(out_dir: str = "data/wmt14_raw"):    """    导出 Hugging Face wmt14/de-en 的 train / validation / test 三个 split。    """    print("正在从 Hugging Face 加载完整的 WMT14 de-en 数据集...")    dataset = load_dataset("wmt14", "de-en")    export_split_to_text(        dataset["train"],        os.path.join(out_dir, "train.en"),        os.path.join(out_dir, "train.de"),        "训练集 train",    )    export_split_to_text(        dataset["validation"],        os.path.join(out_dir, "valid.en"),        os.path.join(out_dir, "valid.de"),        "验证集 validation",    )    export_split_to_text(        dataset["test"],        os.path.join(out_dir, "test.en"),        os.path.join(out_dir, "test.de"),        "测试集 test",    )    print("\n全部导出完成。")    print("后续推荐流程：")    print("1. 对 train/valid/test 分别做 Moses normalize + remove-non-printing + tokenize")    print("2. 仅对 train 做正式清洗；valid/test 只做一致性检查")    print("3. 只在 train 上学习 joint BPE")    print("4. 将同一套 BPE 应用到 train/valid/test")if __name__ == "__main__":    export_all_wmt14()
```

* * *

## 六、你接下来应该怎么做：我按步骤带你走

下面开始进入“逐步指导”。

* * *

## 第一步：先把全部 raw 数据导出来

### 你的目标结果应该是

```
data/wmt14_raw/├── train.en├── train.de├── valid.en├── valid.de├── test.en└── test.de
```

### 你现在要做的检查

-    `train.en` 和 `train.de` 行数相同
    
-    `valid.en` 和 `valid.de` 行数相同
    
-    `test.en` 和 `test.de` 行数相同
    
-    文件都不是空的
    

### Windows 下最简单的 Python 检查脚本

```
def count_lines(path: str) -> int:    with open(path, "r", encoding="utf-8") as f:        return sum(1 for _ in f)pairs = [    ("data/wmt14_raw/train.en", "data/wmt14_raw/train.de"),    ("data/wmt14_raw/valid.en", "data/wmt14_raw/valid.de"),    ("data/wmt14_raw/test.en", "data/wmt14_raw/test.de"),]for en_path, de_path in pairs:    en_n = count_lines(en_path)    de_n = count_lines(de_path)    print(en_path, en_n)    print(de_path, de_n)    assert en_n == de_n, f"行数不一致: {en_path} vs {de_path}"
```

### 这一阶段的验收标准

只要三个 split 都能稳定导出，并且双语行数一一对应，这一步就算通过。

* * *

## 第二步：安装并准备 Moses

你给的地址是对的：

`https://github.com/moses-smt/mosesdecoder.git`

fairseq 的预处理脚本也是直接 clone 这个仓库来拿 tokenization / cleaning 脚本。GitHub

### 你需要用到的几个脚本

主要是这几个：

-   `scripts/tokenizer/normalize-punctuation.perl`
    
-   `scripts/tokenizer/remove-non-printing-char.perl`
    
-   `scripts/tokenizer/tokenizer.perl`
    
-   `scripts/training/clean-corpus-n.perl` GitHub
    

### 你下一步的目标目录

```
data/wmt14_tok/├── train.en├── train.de├── valid.en├── valid.de├── test.en└── test.de
```

也就是：  
**raw → Moses 基础预处理 → tok**

* * *

## 第三步：对所有 split 做 Moses 基础预处理

### 顺序建议

对每个文件都做：

$normalize-punctuation→remove-non-printing-char→tokenizer\text{normalize-punctuation} \rightarrow \text{remove-non-printing-char} \rightarrow \text{tokenizer}normalize-punctuation→remove-non-printing-char→tokenizer$

这是经典 Moses/fairseq 风格流程。GitHub

### 注意

这里的目标还不是“删句子”，只是做：

-   标点规范化
    
-   去掉不可打印字符
    
-   分词
    

### 这一阶段的输出

例如：

-   `data/wmt14_tok/train.en`
    
-   `data/wmt14_tok/train.de`
    
-   `data/wmt14_tok/valid.en`
    
-   `data/wmt14_tok/valid.de`
    
-   `data/wmt14_tok/test.en`
    
-   `data/wmt14_tok/test.de`
    

* * *

## 第四步：只对 train 做正式清洗

### 这一点请你先固定下来

如果我们按你现在“尽量贴近原论文评估口径”的目标走，那么建议是：

-   `train`：可以清洗
    
-   `valid`：只做一致性检查
    
-   `test`：只做一致性检查
    

### train 清洗要做什么

可以考虑：

-   去空行
    
-   长度上限
    
-   源/目标长度比例限制
    

Moses 的 `clean-corpus-n.perl` 就是典型工具。fairseq 脚本里也用了它。GitHub

### 这一阶段的输出

```
data/wmt14_clean/├── train.en└── train.de
```

valid/test 这时通常不重新产出“删行版”，而是保留 `tok` 版本用于后续 BPE apply。

* * *

## 第五步：只在 train 上学习 joint BPE

这里是最关键的一步之一。

Transformer 论文明确写的是：

-   byte-pair encoding
    
-   shared source-target vocabulary
    
-   about 37000 tokens
    

所以你这里要做的是：

### 输入

`train.en` 和 `train.de` 的 **Moses 预处理后、清洗后的训练文本**

### 学习方式

把 `train.en` 和 `train.de` 拼起来，学 **一套 joint BPE rules**

### 不是这样做

-   不是分别学两套 en/de BPE
    
-   不是把 valid/test 也混进去学
    

* * *

## 第六步：把同一套 BPE 应用到 train/valid/test

这里要强调一句：

> 如果你走 Moses + subword-nmt 路线，BPE 是应用在 **tokenized 文本** 上，不是应用在最原始 raw 文本上。

### 这一步的最终输出目录建议

```
data/wmt14_bpe/├── train.en├── train.de├── valid.en├── valid.de├── test.en└── test.de
```

到这里，你才真正得到了**可以给 Transformer DataLoader 用的文本级训练数据**。

* * *

## 七、你当前项目和这一步如何接上

你现在的 Transformer 代码里：

-   `make_model()` 还没做 shared embedding / generator weight tying；
    
-   `src_mask` 测试时还是全 1；
    
-   `tgt_mask` 只是简单下三角；
    
-   `decode()` 还是直接用传入的 `tgt`，还没和 `tgt_input / tgt_y` 训练接口正式绑上。
    

所以当前这条数据预处理链的目标，不是立刻训练，而是先把：

$WMT14 raw→Moses→clean train→joint BPE→DataLoader\text{WMT14 raw} \rightarrow \text{Moses} \rightarrow \text{clean train} \rightarrow \text{joint BPE} \rightarrow \text{DataLoader}WMT14 raw→Moses→clean train→joint BPE→DataLoader$

这条链打通。

等这一步通了，你后面再接：

-   `target shifted right`
    
-   `src_mask`
    
-   `tgt_mask = padding + causal`
    
-   weight tying
    

才会非常顺。

* * *

## 八、你现在这一步的最终判断

### 你的理解里，正确的部分

你现在的理解中，下面这些是对的：

-   之前只导出训练集，不够用
    
-   应该先把全部 split 准备好
    
-   然后做 Moses 基础预处理
    
-   然后只在训练集上学 BPE
    
-   再把 BPE 应用到全部 split
    

* * *

### 需要你修正的部分

你刚才这句话里有一处要改：

> “第四步再训练 bpe，训练好 bpe 之后，应用到原始数据上”

这里应改成：

> **训练好 BPE 之后，应用到 Moses 预处理后的文本上。**

如果你要尽量贴近经典 Transformer / fairseq 复现，这是关键差别。GitHub

* * *

## 九、你现在的下一步

你先做这两件事：

-    把导出脚本改成我上面那个版本，真正得到 `data/wmt14_raw/train|valid|test.{en,de}`
    
-    跑一个行数一致性检查，确认 6 个文件都没问题
    

等你做完这一步，下一条我直接带你做：

## `Moses 基础预处理脚本`

我会按你的 Windows 环境来写，尽量避免你在 PowerShell / Perl 路径上踩坑。

  

我下载完成了，然后我运行了检查脚本，发现： data/wmt14\_raw\_en\_de/train.en 4509489 data/wmt14\_raw\_en\_de/train.de 4509342 Traceback (most recent call last): File "d:\\1AAAAAstudy\\python\_base\\pytorch\\my\_github\_workspace\\transformer-pytorch-main\\script\\dataset\_part\\02\_check\_download.py", line 16, in <module> assert en\_n == de\_n, f"行数不一致: {en\_path} vs {de\_path}" AssertionError: 行数不一致: data/wmt14\_raw\_en\_de/train.en vs data/wmt14\_raw\_en\_de/train.de 是不是应该先Moses处理好，在检查是否一一对应呢

  

我先不往 Moses 上跳。你这个问题更像是**导出阶段就把“句内换行/回车”带进文件里了**，导致“按行计数”失真。先把原始导出修正到严格一行一句，再谈 Moses。

## 直接回答

**不是。不要先上 Moses。**

你现在这个现象说明：**导出阶段就已经没有做到“严格一行一句”**。  
一旦原始平行文件的第 $n$ 行和第 $n$ 行不再对应，后面的 Moses 不会帮你“自动恢复对齐”，反而会在错误对齐的基础上继续处理，问题会被放大。

所以正确顺序是：

-    先修复导出脚本，重新导出严格对齐的 `train/valid/test`
    
-    再检查三组文件的英德行数是否完全一致
    
-    确认一致后，再进入 Moses 基础预处理
    
-    然后再做 train 清洗、joint BPE、apply BPE
    

* * *

## 一、为什么我判断是“导出阶段的问题”，而不是“Moses 之前不用检查”

你给出的检查结果是：

```
train.en 4509489train.de 4509342
```

而你之前下载日志里，Hugging Face 的训练集样本数是：

```
4508785
```

所以可以直接算出：

-   英文文件比样本数**多了 704 行**
    
-   德文文件比样本数**多了 557 行**
    
-   英德之间又**相差 147 行**
    

这说明什么？

### 这说明不是“少写了一些样本”

因为你的导出代码是在**同一个循环**里同时写：

```
f_en.write(en_text + "\n")f_de.write(de_text + "\n")
```

如果循环次数是一样的，那么正常情况下：

-   英文写多少条
    
-   德文就写多少条
    

两边行数理论上必须一致。

### 所以更可能的真实原因是

你虽然做了：

```
replace('\n', ' ')
```

但是**还残留了其他行分隔符**，最常见的是：

-   `\r`
    
-   或其他特殊换行符
    

这样写入文件后，后续按“行”去读取时，就会把本来的一条样本拆成多行。

也就是说：

> 你的数据集记录数是对的，但“落盘成 txt 后的一行一句格式”被破坏了。

* * *

## 二、为什么 `replace('\n', ' ')` 不够

因为文本中的“换行类字符”不一定只有 `\n`。

最常见的坑是：

-   Windows 风格回车：`\r`
    
-   某些文本里的特殊行分隔符
    
-   混合换行格式
    

你现在只替换了 `\n`，但如果某条句子里有 `\r`，那么写入 txt 后，后续计数时仍然可能被当成新的一行。

### 更稳的做法

不要手工只替换 `\n`，而应该用：

```
" ".join(text.splitlines()).strip()
```

这个写法的意义是：

-   不管原文本里混着什么“按行切分”的符号
    
-   先全部按“行片段”拆开
    
-   再用一个空格重新拼回单行
    

这比只 `replace('\n', ' ')` 稳很多。

* * *

## 三、结论：当前这批 `train.en/train.de` 不建议继续用

因为你现在已经发现：

-   英文行数和德文行数不一致
    
-   说明平行对齐已经坏了
    

### 这意味着什么

你**不能再拿当前这两个文件继续做 Moses**，因为：

-   Moses 默认认为第 $i$ 行英语和第 $i$ 行德语是一对
    
-   但你现在第 $i$ 行已经未必对应了
    

### 是否可以“补救现有文件”？

**不建议。**

因为一旦句内换行把文件切坏，你无法可靠地知道：

-   哪几行原本应该合并回同一条样本
    
-   英德两侧应如何重新对齐
    

最稳妥的方法就是：

> **直接从 Hugging Face 数据集重新导出。**

* * *

## 四、你现在应该怎么做

## 第一步：删掉当前导出的坏文件

比如你现在目录下这几个：

```
data/wmt14_raw_en_de/train.endata/wmt14_raw_en_de/train.dedata/wmt14_raw_en_de/valid.endata/wmt14_raw_en_de/valid.dedata/wmt14_raw_en_de/test.endata/wmt14_raw_en_de/test.de
```

建议全部删掉，重新导。

* * *

## 第二步：把导出脚本改成“严格单行化”

下面我给你一个更稳的版本。

* * *

## 五、推荐你直接替换成这个导出脚本

```
from __future__ import annotationsimport osfrom typing import Dictfrom datasets import load_datasetfrom tqdm import tqdm"""功能：1. 从 Hugging Face 加载 wmt14 / de-en 数据集。2. 将 train / validation / test 三个 split 导出为严格对齐的 txt 文件。3. 通过“强制单行化”保证每条样本最终只占一行，避免 \r / \n / 其他行分隔符破坏对齐。输出目录结构：data/wmt14_raw_en_de/├── train.en├── train.de├── valid.en├── valid.de├── test.en└── test.de"""def sanitize_to_single_line(text: str) -> str:    """    将任意文本强制压成单行文本。    输入：        text: 原始字符串    输出：        单行字符串    设计说明：        1. splitlines() 会按各种常见“换行类字符”切分，而不仅仅是 '\n'        2. 再用一个空格拼接，保证最终严格一行一句        3. 最后 strip() 去掉首尾空白    """    single_line = " ".join(text.splitlines()).strip()    # 额外保险：如果仍然残留最常见的回车/换行字符，继续替换    single_line = single_line.replace("\r", " ").replace("\n", " ").strip()    return single_linedef export_split_to_text(    dataset_split,    out_en_path: str,    out_de_path: str,    desc_name: str,) -> None:    """    将指定 split 导出为严格对齐的英德文本文件。    """    os.makedirs(os.path.dirname(out_en_path), exist_ok=True)    print(f"\n正在导出 {desc_name}")    print(f"  EN -> {out_en_path}")    print(f"  DE -> {out_de_path}")    num_pairs = 0    # newline="\n" 用来统一输出换行风格，避免平台差异带来的额外混乱    with open(out_en_path, "w", encoding="utf-8", newline="\n") as f_en, \         open(out_de_path, "w", encoding="utf-8", newline="\n") as f_de:        for item in tqdm(dataset_split, desc=f"导出 {desc_name}"):            en_text = sanitize_to_single_line(item["translation"]["en"])            de_text = sanitize_to_single_line(item["translation"]["de"])            # 不在这里跳过空样本，先保持原始对齐完整            # 后续 train 清洗阶段再统一处理            f_en.write(en_text + "\n")            f_de.write(de_text + "\n")            num_pairs += 1    print(f"完成：{desc_name}，共导出 {num_pairs} 对句子。")def count_lines(path: str) -> int:    """    统计文本文件的行数。    """    with open(path, "r", encoding="utf-8") as f:        return sum(1 for _ in f)def check_parallel_files(en_path: str, de_path: str) -> None:    """    检查双语文件行数是否一致。    """    en_n = count_lines(en_path)    de_n = count_lines(de_path)    print(f"{en_path}: {en_n}")    print(f"{de_path}: {de_n}")    if en_n != de_n:        raise AssertionError(f"行数不一致: {en_path} vs {de_path}")    print("行数一致，检查通过。")def export_all_wmt14(out_dir: str = "data/wmt14_raw_en_de") -> None:    """    导出 Hugging Face 的 WMT14 de-en 全部 split。    """    print("正在加载 Hugging Face 数据集: wmt14 / de-en")    dataset = load_dataset("wmt14", "de-en")    export_split_to_text(        dataset["train"],        os.path.join(out_dir, "train.en"),        os.path.join(out_dir, "train.de"),        "训练集 train",    )    export_split_to_text(        dataset["validation"],        os.path.join(out_dir, "valid.en"),        os.path.join(out_dir, "valid.de"),        "验证集 validation",    )    export_split_to_text(        dataset["test"],        os.path.join(out_dir, "test.en"),        os.path.join(out_dir, "test.de"),        "测试集 test",    )    print("\n开始做导出后的一致性检查...\n")    check_parallel_files(        os.path.join(out_dir, "train.en"),        os.path.join(out_dir, "train.de"),    )    check_parallel_files(        os.path.join(out_dir, "valid.en"),        os.path.join(out_dir, "valid.de"),    )    check_parallel_files(        os.path.join(out_dir, "test.en"),        os.path.join(out_dir, "test.de"),    )    print("\n全部导出并检查完成。")if __name__ == "__main__":    export_all_wmt14()
```

* * *

## 六、为什么这个版本更稳

### 1\. `splitlines()` 比 `replace('\n', ' ')` 更合适

因为它不是只盯着 `\n`，而是从“按行切分”的角度处理字符串。  
你这个场景里，本质需求就是：

> 不管原句里混了什么换行类字符，最后都必须压成单行。

这正是 `splitlines()` 更适合的原因。

* * *

### 2\. `newline="\n"` 可以统一输出风格

这样你最终落盘的 txt 文件换行更一致，后面在不同系统下检查也更稳。

* * *

### 3\. 不在导出阶段跳过空样本

这点很重要。

如果你现在在导出阶段擅自：

-   跳过某些空英文
    
-   或跳过某些空德文
    

就有可能破坏原始对齐。

更稳的策略是：

-   先完整导出
    
-   后面在 `train` 清洗阶段统一过滤
    

* * *

## 七、你下一步的正确操作顺序

你现在就按这个顺序来：

-    先删除当前坏掉的 `train/valid/test` 文本文件
    
-    用上面这个新脚本重新导出
    
-    让脚本自己做导出后行数检查
    
-    如果 `train/valid/test` 三组都通过一致性检查，再进入 Moses
    

* * *

## 八、关于你问的那个核心问题，我再明确一句

### 不是“先 Moses 再检查”

而是：

$导出严格对齐 raw→检查一一对应→再 Moses\text{导出严格对齐 raw} \rightarrow \text{检查一一对应} \rightarrow \text{再 Moses}导出严格对齐 raw→检查一一对应→再 Moses$

原因很简单：

-   Moses 处理的是**已经按行对齐的平行语料**
    
-   不是“帮你修复错位文件”的工具
    

所以对齐检查必须发生在 Moses 之前。

* * *

## 九、你可以把当前问题理解成一个工程类比

就像你做目标检测时，标注文件和图片顺序已经错位了。  
这时候你不会说：

> “我先做 Mosaic / Normalize / Resize，再看看标注是不是对应吧。”

因为增强不会把错位标注修回来。

这里也是同理：

-   `train.en`
    
-   `train.de`
    

就是你的“平行标签对”

如果导出时已经错位，后续 Moses/BPE 都只是对错误数据做加工。

* * *

## 十、你现在的阶段性目标

你先不要急着进 Moses。  
你当前这一阶段唯一目标就是：

> **得到 6 个严格对齐、行数完全一致的 raw 文件。**

只有这一步过了，后面才值得继续。

你先按我给的脚本重新导出并检查。  
等你确认 `train/valid/test` 三组全部行数一致之后，我下一步直接带你做：

## `Windows 环境下的 Moses 基础预处理脚本（normalize / remove-non-printing / tokenize）`

  

我现在检查脚本通过了： data/wmt14\_raw\_en\_de/train.en 4508785 data/wmt14\_raw\_en\_de/train.de 4508785 data/wmt14\_raw\_en\_de/valid.en 3000 data/wmt14\_raw\_en\_de/valid.de 3000 data/wmt14\_raw\_en\_de/test.en 3003 data/wmt14\_raw\_en\_de/test.de 3003 请你继续指导我进行第二步和第三步。每一步详细说明如何做，以及简要说明这一步是干什么的

  

## 很好，第一步已经通过

你现在拿到了严格对齐的：

-   `train.en / train.de`
    
-   `valid.en / valid.de`
    
-   `test.en / test.de`
    

这说明“原始导出”这一层已经没问题了。接下来就进入你要的**第二步**和**第三步**。先说总目标：

-   **第二步：Moses 基础预处理**  
    作用是把原始文本规范化成更稳定的 token 序列，后面做长度统计、清洗、BPE 都会更可靠。
    
-   **第三步：训练集清洗**  
    作用是把明显异常、极端过长、长度比例失衡的句对从训练集里去掉，避免污染后续 BPE 学习和模型训练。
    

经典公开脚本里，fairseq 的 `prepare-wmt14en2de.sh` 确实就是围绕 Moses 的 `normalize-punctuation.perl`、`remove-non-printing-char.perl`、`tokenizer.perl`、`clean-corpus-n.perl` 以及 `subword-nmt` 来做这条链的。GitHub+2GitHub+2

* * *

## 先给你一个总判断：这两步怎么安排

你当前最合适的顺序是：

-    第二步：对 `train/valid/test` 全部做 Moses 基础预处理
    
-    第三步：**只对 `train`** 做正式清洗
    
-    `valid/test` 暂时**不删句子**，只做一致性检查
    

这样做的理由是：

1.  论文最终报告的是 WMT14 英德翻译结果，测试集口径要尽量稳定，论文里明确提到英德任务使用 WMT14、共享源目标 BPE 词表约 37k，并在 `newstest2014` 上报告结果。
    
2.  fairseq 的公开脚本虽然会自己从训练语料中切出 valid，并对 `train/valid` 都执行 `clean-corpus-n.perl`，但那是它自己的工程脚本设计；同时它默认还会用到额外的 `news-commentary-v12`，并默认 `BPE_TOKENS=40000`，这和你当前“尽量贴近原论文、且已经有官方 split”的场景并不完全一致。GitHub+1
    

* * *

# 第二步：Moses 基础预处理

## 这一步是干什么的

这一步的核心作用有三个：

1.  **规范标点**：把不同来源语料中的标点写法拉到更统一的分布。
    
2.  **移除不可打印字符**：减少后续分词和统计时的脏字符干扰。
    
3.  **分词**：把句子转成“空格分开的 token 序列”，为后面的长度过滤和 BPE 学习做准备。
    

Moses 仓库里就有这些脚本，fairseq 的 WMT14 En-De 准备脚本也明确使用了：

-   `normalize-punctuation.perl`
    
-   `remove-non-printing-char.perl`
    
-   `tokenizer.perl` GitHub+1
    

* * *

## 这一步做完后，你应该得到什么

你应该新建一个目录，例如：

```
data/wmt14_tok_en_de/├── train.en├── train.de├── valid.en├── valid.de├── test.en└── test.de
```

也就是说：

-   输入：`data/wmt14_raw_en_de/*.en|de`
    
-   输出：`data/wmt14_tok_en_de/*.en|de`
    

* * *

## 第二步具体怎么做

## 2.1 准备 Moses

你已经给了仓库地址 `moses-smt/mosesdecoder`，这个地址是对的。fairseq 的脚本也是直接 `git clone https://github.com/moses-smt/mosesdecoder.git`。 GitHub

### 你现在需要确认两件事

### 第一件：本地有 Perl 运行环境

因为这些脚本本质上都是 `.perl` / Perl 脚本。

你可以在终端里先测一下：

```
perl -v
```

如果能正常打印版本信息，说明 Perl 运行环境没问题。

### 第二件：把 Moses clone 到项目里

例如放到：

```
tools/mosesdecoder/
```

命令示例：

```
git clone https://github.com/moses-smt/mosesdecoder.git tools/mosesdecoder
```

* * *

## 2.2 你要用到的脚本路径

假设你 clone 到了 `tools/mosesdecoder`，那么后面要用到的是：

```
tools/mosesdecoder/scripts/tokenizer/normalize-punctuation.perltools/mosesdecoder/scripts/tokenizer/remove-non-printing-char.perltools/mosesdecoder/scripts/tokenizer/tokenizer.perltools/mosesdecoder/scripts/training/clean-corpus-n.perl
```

这些脚本路径本身也和 fairseq 脚本里定义的变量一致。GitHub

* * *

## 2.3 单文件处理命令先跑通

先不要一次处理 6 个文件。  
先拿 `train.en` 跑通，确认环境和路径都没问题。

### 处理英文训练集

```
perl tools/mosesdecoder/scripts/tokenizer/normalize-punctuation.perl -l en < data/wmt14_raw_en_de/train.en ^| perl tools/mosesdecoder/scripts/tokenizer/remove-non-printing-char.perl ^| perl tools/mosesdecoder/scripts/tokenizer/tokenizer.perl -threads 8 -a -l en ^> data/wmt14_tok_en_de/train.en
```

### 处理德文训练集

```
perl tools/mosesdecoder/scripts/tokenizer/normalize-punctuation.perl -l de < data/wmt14_raw_en_de/train.de ^| perl tools/mosesdecoder/scripts/tokenizer/remove-non-printing-char.perl ^| perl tools/mosesdecoder/scripts/tokenizer/tokenizer.perl -threads 8 -a -l de ^> data/wmt14_tok_en_de/train.de
```

这里的处理顺序和 fairseq 的公开脚本是一致的：先 `normalize-punctuation`，再 `remove-non-printing-char`，最后 `tokenizer.perl -threads 8 -a -l $l`。GitHub

### 关于 `^`

如果你在 **PowerShell** 里跑，续行符和 `cmd` 不完全一样。最稳妥的方式其实是：

-   要么写成一整行；
    
-   要么放到 `.bat` / Git Bash / WSL 里执行。
    

在 PowerShell 里，最省事是写成一整行：

```
perl tools/mosesdecoder/scripts/tokenizer/normalize-punctuation.perl -l en < data/wmt14_raw_en_de/train.en | perl tools/mosesdecoder/scripts/tokenizer/remove-non-printing-char.perl | perl tools/mosesdecoder/scripts/tokenizer/tokenizer.perl -threads 8 -a -l en > data/wmt14_tok_en_de/train.en
```

* * *

## 2.4 确认单文件处理成功

这一步不要跳。

### 你要检查什么

-    `data/wmt14_tok_en_de/train.en` 已生成
    
-    `data/wmt14_tok_en_de/train.de` 已生成
    
-    两个文件都不是空的
    
-    行数仍然一致
    
-    行数通常应该与 raw 保持一致
    

因为第二步只是逐行变换，不应该主动删句子。fairseq 脚本里这一步也只是规范化和分词，真正删句对是在后面的 `clean-corpus-n.perl`。GitHub

### 检查脚本

你可以直接复用之前的行数检查脚本，把目录改成 `wmt14_tok_en_de`：

```
def count_lines(path: str) -> int:    with open(path, "r", encoding="utf-8") as f:        return sum(1 for _ in f)pairs = [    ("data/wmt14_tok_en_de/train.en", "data/wmt14_tok_en_de/train.de"),]for en_path, de_path in pairs:    en_n = count_lines(en_path)    de_n = count_lines(de_path)    print(en_path, en_n)    print(de_path, de_n)    assert en_n == de_n, f"行数不一致: {en_path} vs {de_path}"
```

* * *

## 2.5 再处理 valid / test

确认 `train` 没问题后，再把同样流程应用到：

-   `valid.en / valid.de`
    
-   `test.en / test.de`
    

最终得到：

```
data/wmt14_tok_en_de/├── train.en├── train.de├── valid.en├── valid.de├── test.en└── test.de
```

### 第二步完成后的验收标准

-    `tok` 目录下 6 个文件都存在
    
-    `train.en == train.de` 行数一致
    
-    `valid.en == valid.de` 行数一致
    
-    `test.en == test.de` 行数一致
    
-    与 `raw` 相比，行数不应减少
    

* * *

# 第三步：训练集清洗

## 这一步是干什么的

第二步解决的是“格式规范化”。  
第三步解决的是“训练样本质量”。

这一步的主要目标是：

-   去掉空句对
    
-   去掉极端过长句对
    
-   去掉源目标长度比例严重失衡的句对
    

这些异常样本会污染：

-   后面的 joint BPE 学习
    
-   后面的 batch 长度统计
    
-   后面的 Transformer 训练稳定性
    

Moses 的 `clean-corpus-n.perl` 就是专门干这个的，fairseq 的 WMT14 En-De 脚本也明确用了它，并设置了 `-ratio 1.5` 和长度范围 `1 250`。GitHub

* * *

## 先说你这里的策略

### 你的场景下，我建议：

-   **只清洗 `train`**
    
-   `valid/test` 暂时不删句子，只做一致性检查
    

理由前面已经说过：  
你现在已经有了 Hugging Face 给出的 `validation` 和 `test`，而且你想尽量贴近原论文评估口径，所以不建议在开发集和测试集上擅自删样本。论文最终报告的是固定评估集结果；fairseq 那种“clean valid”的做法，更像工程脚本选择，而不是论文必须如此。 GitHub+1

* * *

## 第三步具体怎么做

## 3.1 新建清洗输出目录

建议建立：

```
data/wmt14_clean_en_de/└── train.en└── train.de
```

注意这里只有 `train`，因为我们这一步只清洗训练集。

* * *

## 3.2 理解 `clean-corpus-n.perl` 的输入方式

这个脚本吃的不是“两个完整文件名”，而是**文件前缀**。

例如你现在有：

```
data/wmt14_tok_en_de/train.endata/wmt14_tok_en_de/train.de
```

那么它的输入前缀是：

```
data/wmt14_tok_en_de/train
```

语言对参数是：

-   `en`
    
-   `de`
    

输出前缀例如：

```
data/wmt14_clean_en_de/train
```

* * *

## 3.3 推荐先用经典阈值跑第一版

你可以先用下面这个命令：

```
perl tools/mosesdecoder/scripts/training/clean-corpus-n.perl -ratio 1.5 data/wmt14_tok_en_de/train en de data/wmt14_clean_en_de/train 1 250
```

它的含义是：

-   `-ratio 1.5`：源/目标长度比例不能太离谱
    
-   `1`：最短长度阈值
    
-   `250`：最长长度阈值
    

这组参数就是 fairseq WMT14 En-De 脚本公开使用的参数。GitHub

* * *

## 3.4 这一步输出什么

运行后，通常会得到：

```
data/wmt14_clean_en_de/train.endata/wmt14_clean_en_de/train.de
```

这两个文件就是**清洗后的训练集**。

* * *

## 3.5 清洗后要做什么检查

### 检查 1：行数仍然一致

这个是必须的。

```
pairs = [    ("data/wmt14_clean_en_de/train.en", "data/wmt14_clean_en_de/train.de"),]
```

用你前面的计数脚本检查即可。

### 检查 2：行数减少是正常的

清洗后行数通常会少于 `4,508,785`。  
这是正常的，因为你删掉了：

-   空样本
    
-   极端长样本
    
-   比例异常样本
    

### 检查 3：不要减少得离谱

如果你清洗后一下子少掉几十万甚至更多，就要回头怀疑：

-   输入文件是不是路径写错了
    
-   是否语言顺序写反了
    
-   是否长度阈值太严
    

* * *

## 3.6 `valid/test` 在第三步怎么处理

### `valid`

不做 `clean-corpus-n.perl`。  
只做：

-   行数一致性检查
    
-   随机抽样看几行是否正常
    

### `test`

同样不做 `clean-corpus-n.perl`。  
只做：

-   行数一致性检查
    
-   文件非空检查
    
-   随机抽样检查
    

这和 fairseq 脚本的一个关键精神是一致的：它对 test 并没有做 `clean-corpus-n.perl`，而是直接复制 test 的 BPE 结果到最终目录。GitHub

* * *

# 第二步和第三步之间的逻辑关系

你可以这样理解：

## 第二步

把“原始句子”变成“规范 token 序列”。

## 第三步

在“规范 token 序列”上删掉不适合训练的异常句对。

这两步合起来的结果是：

-   `train`：质量更高、格式更稳定
    
-   `valid/test`：评估口径尽量不被人为改动
    

* * *

# 一个你现在必须知道的细节

fairseq 的公开脚本里，其实是：

1.  tokenization
    
2.  自己从 train 里切 valid
    
3.  学 BPE
    
4.  apply BPE
    
5.  再 clean train/valid
    

也就是说，**它的 clean 是放在 BPE 后面**。GitHub

但你当前这个项目，我仍然建议你按我们这里的路线：

1.  先把 raw → tok 做对
    
2.  只清 train
    
3.  再学 joint BPE
    

原因是：

-   论文并没有强制规定 clean 必须在 BPE 后；
    
-   你当前最重要的是保证“官方 split 不乱、评估集不被改写、BPE 只从高质量 train 学到规则”；
    
-   这条路线对你后面接自己写的 DataLoader 更干净。
    

* * *

# 你现在就该做什么

## 先完成第二步

按我上面的命令，先把 `train.en/train.de` 跑成 `tok` 版，再检查行数。

## 第二步确认无误后，再做第三步

执行：

```
perl tools/mosesdecoder/scripts/training/clean-corpus-n.perl -ratio 1.5 data/wmt14_tok_en_de/train en de data/wmt14_clean_en_de/train 1 250
```

然后检查：

-   `data/wmt14_clean_en_de/train.en`
    
-   `data/wmt14_clean_en_de/train.de`
    

行数是否一致。

* * *

# 这两步做完后，你会得到什么

到那时你的数据结构应该大致变成：

```
data/├── wmt14_raw_en_de/│   ├── train.en│   ├── train.de│   ├── valid.en│   ├── valid.de│   ├── test.en│   └── test.de├── wmt14_tok_en_de/│   ├── train.en│   ├── train.de│   ├── valid.en│   ├── valid.de│   ├── test.en│   └── test.de└── wmt14_clean_en_de/    ├── train.en    └── train.de
```

接下来第四步才是：

-   用 `wmt14_clean_en_de/train.en + train.de`
    
-   学一套 **joint BPE**
    
-   再把 BPE 应用到：
    
    -   clean train
        
    -   tok valid
        
    -   tok test
        

subword-nmt 官方也明确给出了 `learn-bpe` 和 `apply-bpe` 的标准用法。GitHub+1

* * *

你现在先做第二步。跑完以后，把下面这些结果发给我就够了：

-   `wmt14_tok_en_de/train.en` 和 `train.de` 的行数
    
-   `wmt14_tok_en_de/valid.en` 和 `valid.de` 的行数
    
-   `wmt14_tok_en_de/test.en` 和 `test.de` 的行数
    
-   如果第三步也做了，再发 `wmt14_clean_en_de/train.en` 和 `train.de` 的行数
    

  

我在win安装好了perl，也git好了moses的项目，但是我运行2.3的脚本报错了： PS D:\\1AAAAAstudy\\python\_base\\pytorch\\my\_github\_workspace\\transformer-pytorch-main> perl tools/mosesdecoder/scripts/tokenizer/normalize-punctuation.perl -l en < data/wmt14\_raw\_en\_de/train.en ^ >> | perl tools/mosesdecoder/scripts/tokenizer/remove-non-printing-char.perl ^ >> | perl tools/mosesdecoder/scripts/tokenizer/tokenizer.perl -threads 8 -a -l en ^ >> > data/wmt14\_tok\_en\_de/train.en 所在位置 行:1 字符: 76 + ... sdecoder/scripts/tokenizer/normalize-punctuation.perl -l en < data/wm ... + ~ “<”运算符是为将来使用而保留的。 所在位置 行:2 字符: 1 + | perl tools/mosesdecoder/scripts/tokenizer/remove-non-printing-char. ... + ~ 不允许使用空管道元素。 所在位置 行:3 字符: 1 + | perl tools/mosesdecoder/scripts/tokenizer/tokenizer.perl -threads 8 ... + ~ 不允许使用空管道元素。 + CategoryInfo : ParserError: (:) \[\], ParentContainsErrorRecordException + FullyQualifiedErrorId : RedirectionNotSupported

  

## 原因已经很明确了

你现在的问题**不是 Moses 脚本本身有问题**，而是你把 **CMD 的命令写法** 拿到 **PowerShell** 里执行了。

你这条命令里有两个关键点：

1.  `<`  
    这是 **CMD/bash 风格的标准输入重定向**
    
2.  `^`  
    这是 **CMD 的续行符**
    

但你当前终端提示符是：

```
PS D:\...
```

这说明你在 **PowerShell** 里。

而在 PowerShell 里：

-   `<` 不是这样用的
    
-   `^` 也不是续行符
    
-   所以第一行一解析就炸了
    
-   后面 `|` 变成“空管道元素”，于是继续报错
    

* * *

# 一、先解释报错是什么意思

你看到的核心报错是：

```
“<”运算符是为将来使用而保留的。
```

这句话本质上就是：

> PowerShell 不支持你这种 `< file` 的输入重定向写法。

然后后面的：

```
不允许使用空管道元素。
```

是因为第一段已经解析失败了，后面又以 `|` 开头，PowerShell 认为你前面根本没有合法的管道输入，所以继续报错。

* * *

# 二、你现在最稳的做法

## 推荐做法：不要强行用 PowerShell 原生命令链

因为你现在做的是：

-   大文本文件
    
-   Perl 脚本串联
    
-   需要稳定的 stdin/stdout 重定向
    
-   后面还要做 clean-corpus、BPE
    

这种场景下，在 Windows 上最稳的是：

-   **方案 A：在 PowerShell 里调用 `cmd /c` 执行 CMD 命令**
    
-   **方案 B：写成 `.bat` 批处理文件再运行**
    
-   **方案 C：用 Git Bash / WSL 执行**
    

* * *

## 我最推荐你现在用的方案

### 直接在 PowerShell 里包一层 `cmd /c`

这样你不用切终端，也不用重学 PowerShell 的对象管道语义，而且和后面大量 Moses 教程最接近。

* * *

# 三、你现在立刻可用的正确命令

## 0\. 先创建输出目录

先确保目录存在：

```
mkdir data\wmt14_tok_en_de
```

如果目录已经存在，也没关系。

* * *

## 1\. 处理英文训练集

在 **PowerShell** 里执行下面这一条：

```
cmd /c "perl tools/mosesdecoder/scripts/tokenizer/normalize-punctuation.perl -l en < data/wmt14_raw_en_de/train.en | perl tools/mosesdecoder/scripts/tokenizer/remove-non-printing-char.perl | perl tools/mosesdecoder/scripts/tokenizer/tokenizer.perl -threads 8 -a -l en > data/wmt14_tok_en_de/train.en"
```

* * *

## 2\. 处理德文训练集

```
cmd /c "perl tools/mosesdecoder/scripts/tokenizer/normalize-punctuation.perl -l de < data/wmt14_raw_en_de/train.de | perl tools/mosesdecoder/scripts/tokenizer/remove-non-printing-char.perl | perl tools/mosesdecoder/scripts/tokenizer/tokenizer.perl -threads 8 -a -l de > data/wmt14_tok_en_de/train.de"
```

* * *

## 3\. 为什么这样能行

因为：

-   外层是 PowerShell
    
-   但真正执行管道、重定向的是 `cmd.exe`
    
-   `<`、`>`、`|`、`^` 这些都按 CMD 规则解释
    
-   所以 Moses 这类 Perl 脚本链就能正常工作
    

* * *

# 四、如果你想分多行写，也可以这样写

如果一行太长，你可以在 PowerShell 里先拼成一个字符串，再交给 `cmd /c`。

例如英文：

```
$cmd = 'perl tools/mosesdecoder/scripts/tokenizer/normalize-punctuation.perl -l en < data/wmt14_raw_en_de/train.en | perl tools/mosesdecoder/scripts/tokenizer/remove-non-printing-char.perl | perl tools/mosesdecoder/scripts/tokenizer/tokenizer.perl -threads 8 -a -l en > data/wmt14_tok_en_de/train.en'cmd /c $cmd
```

德文同理：

```
$cmd = 'perl tools/mosesdecoder/scripts/tokenizer/normalize-punctuation.perl -l de < data/wmt14_raw_en_de/train.de | perl tools/mosesdecoder/scripts/tokenizer/remove-non-printing-char.perl | perl tools/mosesdecoder/scripts/tokenizer/tokenizer.perl -threads 8 -a -l de > data/wmt14_tok_en_de/train.de'cmd /c $cmd
```

这个写法在 PowerShell 里更稳，更不容易因为换行或引号出问题。

* * *

# 五、为什么我不推荐你现在直接用 PowerShell 原生管道来替代

你可能会想：

```
Get-Content train.en | perl xxx | perl yyy | perl zzz > out.txt
```

理论上能拼，但我**不建议你在这个任务里这么做**，原因有三个：

## 1\. PowerShell 管道是“对象管道”，不是传统 shell 的字节流思维

虽然文本也能流过去，但中间会多很多编码和换行层面的不确定性。

## 2\. 你现在处理的是几百万行的大语料

`train` 是 450 多万句。  
这种场景下你应该优先选：

-   最接近传统 shell stdin/stdout 的方式
    
-   最少“魔法语义”的方式
    

## 3\. 后面你还要继续跑：

-   `clean-corpus-n.perl`
    
-   `subword-nmt`
    
-   可能还有其他 shell 风格脚本
    

所以现在统一用 `cmd /c`，后面会更省事。

* * *

# 六、你现在第二步应该怎么继续

## 第二步的目标回顾

第二步是 **Moses 基础预处理**，包括：

1.  `normalize-punctuation`
    
2.  `remove-non-printing-char`
    
3.  `tokenizer`
    

作用是：

-   统一文本格式
    
-   去掉脏字符
    
-   让句子变成空格分词后的 token 序列
    

* * *

## 你现在正确的执行顺序

### 先只跑 `train`

先执行：

-    `train.en`
    
-    `train.de`
    

也就是我上面给你的两条 `cmd /c` 命令。

* * *

## 跑完后立刻检查

你要检查：

-    `data/wmt14_tok_en_de/train.en` 是否生成
    
-    `data/wmt14_tok_en_de/train.de` 是否生成
    
-    两个文件都不是空的
    
-    两个文件行数是否一致
    
-    行数应该仍然是 `4508785`
    

因为第二步只是逐行变换，不应该删句子。

* * *

## 检查脚本

你可以继续用之前那个：

```
def count_lines(path: str) -> int:    with open(path, "r", encoding="utf-8") as f:        return sum(1 for _ in f)pairs = [    ("data/wmt14_tok_en_de/train.en", "data/wmt14_tok_en_de/train.de"),]for en_path, de_path in pairs:    en_n = count_lines(en_path)    de_n = count_lines(de_path)    print(en_path, en_n)    print(de_path, de_n)    assert en_n == de_n, f"行数不一致: {en_path} vs {de_path}"
```

* * *

## 如果 `train` 没问题，再跑 `valid` 和 `test`

### 英文 valid

```
cmd /c "perl tools/mosesdecoder/scripts/tokenizer/normalize-punctuation.perl -l en < data/wmt14_raw_en_de/valid.en | perl tools/mosesdecoder/scripts/tokenizer/remove-non-printing-char.perl | perl tools/mosesdecoder/scripts/tokenizer/tokenizer.perl -threads 8 -a -l en > data/wmt14_tok_en_de/valid.en"
```

### 德文 valid

```
cmd /c "perl tools/mosesdecoder/scripts/tokenizer/normalize-punctuation.perl -l de < data/wmt14_raw_en_de/valid.de | perl tools/mosesdecoder/scripts/tokenizer/remove-non-printing-char.perl | perl tools/mosesdecoder/scripts/tokenizer/tokenizer.perl -threads 8 -a -l de > data/wmt14_tok_en_de/valid.de"
```

### 英文 test

```
cmd /c "perl tools/mosesdecoder/scripts/tokenizer/normalize-punctuation.perl -l en < data/wmt14_raw_en_de/test.en | perl tools/mosesdecoder/scripts/tokenizer/remove-non-printing-char.perl | perl tools/mosesdecoder/scripts/tokenizer/tokenizer.perl -threads 8 -a -l en > data/wmt14_tok_en_de/test.en"
```

### 德文 test

```
cmd /c "perl tools/mosesdecoder/scripts/tokenizer/normalize-punctuation.perl -l de < data/wmt14_raw_en_de/test.de | perl tools/mosesdecoder/scripts/tokenizer/remove-non-printing-char.perl | perl tools/mosesdecoder/scripts/tokenizer/tokenizer.perl -threads 8 -a -l de > data/wmt14_tok_en_de/test.de"
```

* * *

# 七、第三步你暂时先别做，等第二步确认完全正确

第三步是：

-   对 `train` 做清洗
    
-   用 `clean-corpus-n.perl`
    

但现在你还没把第二步在 Windows 环境里跑顺，所以**先不要跳**。

因为第三步依赖第二步的输出：

```
data/wmt14_tok_en_de/train.endata/wmt14_tok_en_de/train.de
```

如果第二步还没稳定产出，第三步没有意义。

* * *

# 八、我帮你把这两步的逻辑再串一下

## 第二步：Moses 基础预处理

输入：

```
data/wmt14_raw_en_de/train.endata/wmt14_raw_en_de/train.de
```

输出：

```
data/wmt14_tok_en_de/train.endata/wmt14_tok_en_de/train.de
```

特点：

-   行数不该变
    
-   对齐不该变
    
-   只是文本形式更规范
    

* * *

## 第三步：训练集清洗

输入：

```
data/wmt14_tok_en_de/train.endata/wmt14_tok_en_de/train.de
```

输出：

```
data/wmt14_clean_en_de/train.endata/wmt14_clean_en_de/train.de
```

特点：

-   行数会减少
    
-   但英德两边仍必须严格一致
    
-   这是在删异常训练样本
    

* * *

# 九、你当前要做的唯一事情

先执行我给你的这两条命令：

-   `train.en`
    
-   `train.de`
    

然后把下面这 4 个结果发给我：

1.  `data/wmt14_tok_en_de/train.en` 是否成功生成
    
2.  `data/wmt14_tok_en_de/train.de` 是否成功生成
    
3.  两个文件的行数
    
4.  如果命令还有报错，把完整报错贴出来
    

等这一步确认通过，我下一条就直接带你做：

## 第三步：`clean-corpus-n.perl` 清洗训练集

并且我会继续按 **Windows + PowerShell + cmd /c** 的方式给你命令，保证你可以直接接着跑。

  

我都运行了，然后也用你给的脚本检查了行数： data/wmt14\_tok\_en\_de/train.en 4508785 data/wmt14\_tok\_en\_de/train.de 4508785 data/wmt14\_tok\_en\_de/valid.en 3000 data/wmt14\_tok\_en\_de/valid.de 3000 data/wmt14\_tok\_en\_de/test.en 3003 data/wmt14\_tok\_en\_de/test.de 3003 如果没问题的话，就进行下一步吧

  

## 很好，第二步通过了

这说明你现在已经拿到了 **Moses 基础预处理后的、严格对齐的 tokenized 平行语料**：

-   `wmt14_tok_en_de/train.en`
    
-   `wmt14_tok_en_de/train.de`
    
-   `wmt14_tok_en_de/valid.en`
    
-   `wmt14_tok_en_de/valid.de`
    
-   `wmt14_tok_en_de/test.en`
    
-   `wmt14_tok_en_de/test.de`
    

而且行数完全一致。  
这意味着你现在可以安全进入 **第三步：训练集清洗**。

* * *

# 第三步：只清洗训练集

## 这一步是干什么的

这一步的目标是：

-   去掉空句对
    
-   去掉极端长句对
    
-   去掉源/目标长度比例明显失衡的句对
    

你可以把这一步理解成：

> 在正式训练前，把明显不适合作为监督信号的训练样本剔除掉。

这些异常样本如果不处理，后面会影响：

-   joint BPE 学习
    
-   batch 长度分布
    
-   Transformer 训练稳定性
    

* * *

## 为什么只清洗训练集

因为：

-   **训练集**允许做样本过滤，它本来就是用来学习参数的
    
-   **验证集 / 测试集**更偏向评估口径，不能随便删句子
    

所以我们现在的策略是：

-    `train`：正式清洗
    
-    `valid`：不删句子，只保留 tokenized 版本
    
-    `test`：不删句子，只保留 tokenized 版本
    

* * *

# 你现在会得到什么

这一步做完后，你应该得到一个新目录：

```
data/wmt14_clean_en_de/├── train.en└── train.de
```

注意这里只有训练集，因为我们这一步只处理 `train`。

* * *

# 具体怎么做

## 3.1 先创建输出目录

在 PowerShell 里执行：

```
mkdir data\wmt14_clean_en_de
```

如果目录已经存在，也没关系。

* * *

## 3.2 运行训练集清洗命令

你现在在 PowerShell 里，仍然建议用 `cmd /c` 包一层。

直接执行：

```
cmd /c "perl tools/mosesdecoder/scripts/training/clean-corpus-n.perl -ratio 1.5 data/wmt14_tok_en_de/train en de data/wmt14_clean_en_de/train 1 250"
```

* * *

# 这条命令是什么意思

我们逐项拆开。

## 命令主体

```
perl tools/mosesdecoder/scripts/training/clean-corpus-n.perl
```

表示调用 Moses 的训练语料清洗脚本。

* * *

## `-ratio 1.5`

表示限制源语言和目标语言的长度比例不能太离谱。

直观理解：

-   如果一句英文很短，但对应德文特别特别长
    
-   或反过来
    

那么这类句对很可能是：

-   对齐有问题
    
-   内容不匹配
    
-   噪声样本
    

所以要剔除。

你可以把它理解成一个“长度平衡约束”。

* * *

## `data/wmt14_tok_en_de/train`

这不是具体文件名，而是**前缀**。

它对应的是：

-   `data/wmt14_tok_en_de/train.en`
    
-   `data/wmt14_tok_en_de/train.de`
    

Moses 的这个脚本就是这么设计的：  
它吃的是“文件前缀 + 语言后缀”。

* * *

## `en de`

表示语言对顺序是：

-   源语言：`en`
    
-   目标语言：`de`
    

所以脚本会去找：

-   `train.en`
    
-   `train.de`
    

并按一一对应的方式清洗。

* * *

## `data/wmt14_clean_en_de/train`

这是**输出前缀**。

运行完成后，最终会生成：

-   `data/wmt14_clean_en_de/train.en`
    
-   `data/wmt14_clean_en_de/train.de`
    

* * *

## `1 250`

这两个数字表示长度范围：

-   最短长度：`1`
    
-   最长长度：`250`
    

也就是说：

-   长度小于 1 的句对会被丢掉
    
-   长度大于 250 的句对会被丢掉
    

这里的“长度”是在 **tokenized 之后的 token 数量** 上衡量的，不是原始字符数。

* * *

# 这一步背后的直觉

你可以这样理解：

## 第二步做了什么

第二步只是把文本“整理干净并切词”。

相当于把原始句子整理成：

```
This is a sentence .Das ist ein Satz .
```

这种适合统计和建模的格式。

* * *

## 第三步做了什么

第三步是在这些 token 序列上做筛选。

比如下面这些样本就可能被删掉：

-   空句子
    
-   只有一个符号的句子
    
-   一侧 5 个 token，另一侧 200 个 token
    
-   一侧 300 多个 token 的超长句子
    

* * *

# 运行完成后，你要做哪些检查

这一步非常重要，不要跳。

* * *

## 3.3 检查输出文件是否生成

你要确认这两个文件存在：

-   `data/wmt14_clean_en_de/train.en`
    
-   `data/wmt14_clean_en_de/train.de`
    

* * *

## 3.4 检查英德行数是否一致

继续用你之前的检查脚本即可：

```
def count_lines(path: str) -> int:    with open(path, "r", encoding="utf-8") as f:        return sum(1 for _ in f)pairs = [    ("data/wmt14_clean_en_de/train.en", "data/wmt14_clean_en_de/train.de"),]for en_path, de_path in pairs:    en_n = count_lines(en_path)    de_n = count_lines(de_path)    print(en_path, en_n)    print(de_path, de_n)    assert en_n == de_n, f"行数不一致: {en_path} vs {de_path}"
```

* * *

## 3.5 行数减少是正常的

和第二步不同，第三步**就是会删句子**。

所以你看到：

-   `train.en` 行数少于 `4508785`
    
-   `train.de` 行数少于 `4508785`
    

这是正常现象。

只要：

-   英文和德文行数一致
    
-   不是少得特别离谱
    

那就说明基本正常。

* * *

## 3.6 最好再做一个“清洗比例”检查

我建议你顺手算一下：

-   原始 tokenized 训练集有多少行
    
-   清洗后剩多少行
    
-   删除了多少比例
    

你可以用这个脚本：

```
raw_n = 4508785  # 你前面已经确认过clean_n = 0with open("data/wmt14_clean_en_de/train.en", "r", encoding="utf-8") as f:    clean_n = sum(1 for _ in f)removed = raw_n - clean_nremoved_ratio = removed / raw_nprint(f"原始训练集行数: {raw_n}")print(f"清洗后训练集行数: {clean_n}")print(f"删除句对数量: {removed}")print(f"删除比例: {removed_ratio:.4%}")
```

* * *

# 这一步完成后的目录结构

如果一切正常，你的数据目录现在应该变成：

```
data/├── wmt14_raw_en_de/│   ├── train.en│   ├── train.de│   ├── valid.en│   ├── valid.de│   ├── test.en│   └── test.de├── wmt14_tok_en_de/│   ├── train.en│   ├── train.de│   ├── valid.en│   ├── valid.de│   ├── test.en│   └── test.de└── wmt14_clean_en_de/    ├── train.en    └── train.de
```

* * *

# 这一步完成后，下一步是什么

第三步完成之后，下一步就是：

# 第四步：训练 joint BPE，并应用到 train / valid / test

这里的输入会是：

-   `wmt14_clean_en_de/train.en`
    
-   `wmt14_clean_en_de/train.de`
    

也就是说：

-   **BPE 学习只用清洗后的训练集**
    
-   然后把学到的同一套 BPE 规则应用到：
    
    -   clean train
        
    -   tokenized valid
        
    -   tokenized test
        

这一步才会真正产出你后面 Transformer DataLoader 要吃的 BPE 文本。

* * *

# 你现在要做的事

直接执行这条命令：

```
cmd /c "perl tools/mosesdecoder/scripts/training/clean-corpus-n.perl -ratio 1.5 data/wmt14_tok_en_de/train en de data/wmt14_clean_en_de/train 1 250"
```

然后把下面这些结果发给我：

-   `data/wmt14_clean_en_de/train.en` 的行数
    
-   `data/wmt14_clean_en_de/train.de` 的行数
    
-   删除比例大概是多少
    

等你把这个结果发给我，我就继续带你做 **第四步：joint BPE 训练与应用**。

  

这是检查的结果： data/wmt14\_clean\_en\_de/train.en 3927488 data/wmt14\_clean\_en\_de/train.de 3927488 比例合理吗？ 合理的话我们继续第四步

  

我先核对一下 `subword-nmt` 官方仓库当前的安装与命令形式，避免我给你过时命令。然后我直接把第四步按 Windows 可执行的方式展开。

## 结论

你的清洗结果是：

-   原始 `train`：`4,508,785`
    
-   清洗后 `train`：`3,927,488`
    

也就是：

-   删掉了 `581,297` 对句子
    
-   保留比例约 `87.11%`
    
-   删除比例约 `12.89%`
    

这个比例**偏严格，但仍然是合理区间**。你这一步用的是 `-ratio 1.5` 和 `1 250`，这和 fairseq 的公开 `prepare-wmt14en2de.sh` 脚本参数是一致的。公平地说，fairseq 的这套脚本本身就是一个比较“工程化”的清洗口径；你现在删掉大约 13% 训练样本，不算离谱，可以继续。GitHub

* * *

# 第四步：训练 joint BPE，并应用到 train / valid / test

## 这一步是干什么的

这一步的作用是：

1.  在**训练集上**学习一套共享的 BPE 规则。
    
2.  用这套**同一个** BPE 规则去切分：
    
    -   清洗后的训练集
        
    -   tokenized 的验证集
        
    -   tokenized 的测试集
        
3.  最终得到可以直接进入 Transformer DataLoader 的子词级文本数据。
    

Transformer 原论文对 WMT14 英德任务写的是：句子使用 byte-pair encoding，并且使用**共享的 source-target vocabulary**，规模约 **37000 tokens**。

* * *

## 先说明一个容易混淆的点

### 论文说的是“最终共享词表大约 37000”

但 `subword-nmt` 里的 `-s` 参数表示的是：

-   **BPE merge operations 数量**
    

不是“最终词表大小”。

所以：

> **不能简单地说 `-s 37000` 就等于“完全复现论文的 37k 词表”。**

不过如果你当前目标是先把整条链打通，同时尽量朝论文靠拢，那么第一版可以先用：

```
-s 37000
```

然后在 BPE 应用完之后，**再统计最终 train 的共享词表大小**，看是不是接近论文里说的 “about 37000”。这样做更严谨。论文给的是最终 shared vocabulary 的量级，而不是 `subword-nmt` 的 merge 次数定义。

* * *

## 为什么我这里不直接照 fairseq 的 40000

因为 fairseq 的公开 `prepare-wmt14en2de.sh` 里写的是：

```
BPE_TOKENS=40000
```

而且它的数据准备流程也不是你现在这条线的 1:1 镜像。它默认脚本还会下载并使用 `news-commentary-v12`，并且是**先 apply BPE，再 clean train/valid**。这和你现在“已有固定 split、先 clean train、再学 BPE、尽量贴近原论文”的路线不同。GitHub+1

所以这里我们采取的策略是：

-   **目标向论文靠拢**
    
-   **工具使用 `subword-nmt` 的官方 joint workflow**
    
-   **先做出一版可复现实验链**
    
-   之后如果你想更抠细节，再去微调 `-s`
    

* * *

# 这一步的最终输出应该是什么

你现在应该准备两个目录：

```
data/├── wmt14_bpe_model/│   ├── codes.bpe│   ├── vocab.en│   └── vocab.de└── wmt14_bpe_en_de/    ├── train.en    ├── train.de    ├── valid.en    ├── valid.de    ├── test.en    └── test.de
```

含义是：

-   `wmt14_bpe_model/`：保存 BPE 规则和词表过滤文件
    
-   `wmt14_bpe_en_de/`：保存真正给模型使用的 BPE 后文本
    

* * *

# 第四步具体怎么做

## 4.1 安装 `subword-nmt`

`subword-nmt` 官方 README 目前仍然给出的是 `pip install subword-nmt`，并且官方推荐的基本命令包括：

-   `subword-nmt learn-bpe`
    
-   `subword-nmt apply-bpe`
    
-   `subword-nmt learn-joint-bpe-and-vocab` GitHub+1
    

你先在 PowerShell 里执行：

```
pip install subword-nmt
```

安装完之后，先测试一下：

```
subword-nmt -h
```

如果能打印帮助信息，说明安装成功。

* * *

## 4.2 创建输出目录

在 PowerShell 里执行：

```
mkdir data\wmt14_bpe_modelmkdir data\wmt14_bpe_en_de
```

* * *

## 4.3 学习 joint BPE 规则

### 这一步做什么

这一步只使用：

-   `data/wmt14_clean_en_de/train.en`
    
-   `data/wmt14_clean_en_de/train.de`
    

来学习**一套共享的 BPE rules**。

`subword-nmt` 官方 README 明确给出了 `learn-joint-bpe-and-vocab` 这个推荐工作流：它会同时学习 joint BPE，并且写出两个语言各自的 vocabulary 文件，后续可以配合 `--vocabulary` 和 `--vocabulary-threshold` 进行更稳的 apply。GitHub

### 直接执行这条命令

```
subword-nmt learn-joint-bpe-and-vocab --input data/wmt14_clean_en_de/train.en data/wmt14_clean_en_de/train.de -s 37000 -o data/wmt14_bpe_model/codes.bpe --write-vocabulary data/wmt14_bpe_model/vocab.en data/wmt14_bpe_model/vocab.de
```

* * *

## 4.4 这一步执行完后，你应该看到什么

应该出现这 3 个文件：

```
data/wmt14_bpe_model/codes.bpedata/wmt14_bpe_model/vocab.endata/wmt14_bpe_model/vocab.de
```

### 它们分别是什么

#### `codes.bpe`

这是 joint BPE 的合并规则文件。

#### `vocab.en`

这是英语侧词表统计。

#### `vocab.de`

这是德语侧词表统计。

这正是 `learn-joint-bpe-and-vocab` 官方推荐流程的输出。GitHub

* * *

## 4.5 将 BPE 应用到训练集

### 为什么训练集用 `clean train`

因为你前面第三步已经把训练集清洗过了。  
所以训练集的 BPE 输入应该是：

-   `data/wmt14_clean_en_de/train.en`
    
-   `data/wmt14_clean_en_de/train.de`
    

而不是 tokenized 但未清洗的版本。

### 英文训练集

在 PowerShell 里执行：

```
cmd /c "subword-nmt apply-bpe -c data/wmt14_bpe_model/codes.bpe --vocabulary data/wmt14_bpe_model/vocab.en --vocabulary-threshold 50 < data/wmt14_clean_en_de/train.en > data/wmt14_bpe_en_de/train.en"
```

### 德文训练集

```
cmd /c "subword-nmt apply-bpe -c data/wmt14_bpe_model/codes.bpe --vocabulary data/wmt14_bpe_model/vocab.de --vocabulary-threshold 50 < data/wmt14_clean_en_de/train.de > data/wmt14_bpe_en_de/train.de"
```

这里我保留了 `--vocabulary` 和 `--vocabulary-threshold 50`，因为这是 `subword-nmt` README 在 joint BPE 推荐流程里明确给出的再应用方式。GitHub

* * *

## 4.6 将 BPE 应用到验证集

### 为什么验证集用 `tok valid`

因为我们前面没有对 `valid` 做删句清洗，只做了：

-   Moses normalize
    
-   remove-non-printing
    
-   tokenize
    

所以验证集的 BPE 输入应该是：

-   `data/wmt14_tok_en_de/valid.en`
    
-   `data/wmt14_tok_en_de/valid.de`
    

### 英文验证集

```
cmd /c "subword-nmt apply-bpe -c data/wmt14_bpe_model/codes.bpe --vocabulary data/wmt14_bpe_model/vocab.en --vocabulary-threshold 50 < data/wmt14_tok_en_de/valid.en > data/wmt14_bpe_en_de/valid.en"
```

### 德文验证集

```
cmd /c "subword-nmt apply-bpe -c data/wmt14_bpe_model/codes.bpe --vocabulary data/wmt14_bpe_model/vocab.de --vocabulary-threshold 50 < data/wmt14_tok_en_de/valid.de > data/wmt14_bpe_en_de/valid.de"
```

* * *

## 4.7 将 BPE 应用到测试集

测试集和验证集同理，输入使用 tokenized 版本：

-   `data/wmt14_tok_en_de/test.en`
    
-   `data/wmt14_tok_en_de/test.de`
    

### 英文测试集

```
cmd /c "subword-nmt apply-bpe -c data/wmt14_bpe_model/codes.bpe --vocabulary data/wmt14_bpe_model/vocab.en --vocabulary-threshold 50 < data/wmt14_tok_en_de/test.en > data/wmt14_bpe_en_de/test.en"
```

### 德文测试集

```
cmd /c "subword-nmt apply-bpe -c data/wmt14_bpe_model/codes.bpe --vocabulary data/wmt14_bpe_model/vocab.de --vocabulary-threshold 50 < data/wmt14_tok_en_de/test.de > data/wmt14_bpe_en_de/test.de"
```

* * *

# 第四步做完后要检查什么

## 4.8 先检查 6 个输出文件是否都生成了

你应该有：

-   `train.en / train.de`
    
-   `valid.en / valid.de`
    
-   `test.en / test.de`
    

都在 `data/wmt14_bpe_en_de/` 目录下。

* * *

## 4.9 检查行数是否一致

这一步非常重要。

### 预期应该是

-   `train.en == train.de == 3927488`
    
-   `valid.en == valid.de == 3000`
    
-   `test.en == test.de == 3003`
    

因为 BPE 只是逐行切分，不应该改变文件行数。  
你可以继续用之前的检查脚本，把目录改成 `wmt14_bpe_en_de` 即可。

* * *

## 4.10 再做一个“最终共享词表大小”的统计

因为你希望尽量贴近原论文，而论文写的是：

-   shared source-target vocabulary
    
-   about 37000 tokens
    

所以 BPE 应用完之后，建议你统计一下 **训练集 BPE 文本的共享词表大小**。

### 统计脚本

```
def collect_vocab(path: str) -> set:    vocab = set()    with open(path, "r", encoding="utf-8") as f:        for line in f:            vocab.update(line.strip().split())    return vocabv_en = collect_vocab("data/wmt14_bpe_en_de/train.en")v_de = collect_vocab("data/wmt14_bpe_en_de/train.de")v_union = v_en | v_deprint(f"train.en vocab size: {len(v_en)}")print(f"train.de vocab size: {len(v_de)}")print(f"shared union vocab size: {len(v_union)}")
```

### 这一步的意义

如果最终 `union vocab size` 明显偏离 37000 很多，例如：

-   2 万出头
    
-   5 万多
    

那你后面就可以回头微调 `-s`。

但如果它已经在 **3 万多、接近 37000** 的量级，那第一版就足够继续往下走。

* * *

# 这一步和原论文、fairseq 的关系

为了让你脑子里不乱，我给你一句非常明确的定位：

## 你现在的流程是：

-   **工具**上：沿用经典 `Moses + subword-nmt`
    
-   **目标**上：尽量向原论文的 “shared source-target vocab about 37k” 靠拢
    
-   **工程路径**上：不是 1:1 复刻 fairseq 的 `prepare-wmt14en2de.sh`
    

这是有意为之。

原因是：

-   原论文强调的是 **shared vocabulary about 37000**。
    
-   `subword-nmt` 官方支持 `learn-joint-bpe-and-vocab` 这套 joint workflow，并推荐用 `--vocabulary` 与 `--vocabulary-threshold` 进行一致的再应用。GitHub
    
-   fairseq 的公开脚本则用了 `BPE_TOKENS=40000`，并且它整个数据准备协议还有额外差异，所以你不能把它当成“论文原始流程”的逐字镜像。GitHub+1
    

* * *

# 你现在该做什么

按顺序执行下面这些：

-    `pip install subword-nmt`
    
-    创建 `wmt14_bpe_model` 和 `wmt14_bpe_en_de`
    
-    运行 `learn-joint-bpe-and-vocab`
    
-    运行 6 条 `apply-bpe`
    
-    检查 6 个文件行数
    
-    统计 `train` 的 shared union vocab size
    

然后把下面这些结果发给我：

1.  `data/wmt14_bpe_en_de/train.en` 和 `train.de` 的行数
    
2.  `valid.en / valid.de` 的行数
    
3.  `test.en / test.de` 的行数
    
4.  `shared union vocab size` 的统计结果
    

等你把这些结果发给我，下一步我就继续带你做：

## 第五步：把 BPE 文本接入你当前 Transformer 工程的数据加载器与词表系统

  

我执行了4.4，现在已经生成了这些： data/wmt14\_bpe\_model/codes.bpe data/wmt14\_bpe\_model/vocab.en data/wmt14\_bpe\_model/vocab.de 我有个疑问，我目前还没有进行对val做一致性检查，不删行的那种，需要在BPE之前做吗？

  

## 结论

**建议做，但不是“必须卡住流程的大步骤”。**

更准确地说：

-   你前面已经做过一次 **行数一致性检查**：
    
    -   `valid.en = 3000`
        
    -   `valid.de = 3000`
        
    -   `test.en = 3003`
        
    -   `test.de = 3003`
        

这已经是**最核心的一致性检查**了。  
所以从“能不能继续做 BPE”这个角度看，**可以继续**。

但是如果你问我：

> 在尽量严谨、尽量贴近论文复现的工程习惯下，要不要在 BPE 之前再做一次 `valid/test` 的轻量一致性检查？

我的答案是：

> **要，建议现在做。**

原因不是为了删句子，而是为了确认：

-   没有空行
    
-   没有异常长行
    
-   没有明显的文本损坏
    
-   英德两边逐行仍然是“一一对应”的可用样本
    

* * *

# 一、你现在到底处于什么状态

你已经完成了：

-   `raw` 导出
    
-   `Moses tokenize`
    
-   `train clean`
    
-   `joint BPE rules` 学习
    

也就是说，现在你手里已经有：

```
data/wmt14_bpe_model/├── codes.bpe├── vocab.en└── vocab.de
```

这说明**第四步的前半段**已经完成了。  
你现在要做的是第四步的后半段：

-   把 BPE 应用到：
    
    -   clean train
        
    -   tokenized valid
        
    -   tokenized test
        

在这个节点上，问一句“val/test 要不要先做轻量一致性检查”，这是对的。

* * *

# 二、答案为什么是“建议做，但不需要做重清洗”

## 1\. 为什么建议做

因为 `valid/test` 后面是评估集。  
你不应该像 `train` 那样去做 aggressive cleaning，但你应该知道：

-   里面有没有空行
    
-   有没有明显坏样本
    
-   有没有极端离谱的 token 长度
    
-   有没有行数对齐以外的明显异常
    

也就是说：

> **不是为了删，而是为了看。**

这和 `train clean` 的思想完全不同。

* * *

## 2\. 为什么不做正式 clean

因为你当前的目标是：

-   尽量贴近原始论文评估口径
    
-   不想擅自改动验证集和测试集样本集合
    

Transformer 原论文英德实验的最终报告是固定开发/测试口径下的 BLEU，英德任务使用 shared source-target BPE，词表约 37000。

所以对于 `valid/test`：

-   **看可以**
    
-   **统计可以**
    
-   **检查可以**
    
-   **删句子不建议**
    

* * *

# 三、这一步应该放在 BPE 前还是 BPE 后

## 推荐：放在 BPE 前

也就是：

-   现在检查 `wmt14_tok_en_de/valid.*`
    
-   现在检查 `wmt14_tok_en_de/test.*`
    
-   如果没问题，再 apply BPE
    

原因很简单：

### 1\. 更早发现问题

如果 `tok` 文件里就有：

-   空行
    
-   奇怪 token
    
-   明显编码损坏
    

你应该在 BPE 之前就知道。

### 2\. 避免把问题传递到下一层

BPE 只是把 token 再切成 subword，不会“修复”异常文本。  
所以如果前面有问题，后面只是把问题继续传播。

### 3\. 工程定位更清晰

你以后如果发现 dev BLEU 异常，能更清楚地区分：

-   是 raw 问题
    
-   是 Moses 问题
    
-   是 BPE 问题
    
-   还是 DataLoader/模型问题
    

* * *

# 四、但你要注意：这里的“一致性检查”到底检查什么

不是做 `clean-corpus-n.perl`。  
不是删数据。  
而是做 **轻量检查**。

* * *

## 推荐你现在检查 4 件事

## 检查 1：行数一致

这个你已经做过了，而且已经通过。

所以这一步：

-   `valid.en == valid.de`
    
-   `test.en == test.de`
    

已经没问题。

* * *

## 检查 2：是否存在空行

这一步很建议做。  
因为即使行数对齐，仍然可能出现：

-   某一行英文是空的
    
-   某一行德文是空的
    
-   两边都空
    

这些情况不会破坏“行数一致”，但会污染后面的评估输入。

### 检查脚本

```
def count_empty_lines(path: str) -> int:    cnt = 0    with open(path, "r", encoding="utf-8") as f:        for line in f:            if not line.strip():                cnt += 1    return cntfiles = [    "data/wmt14_tok_en_de/valid.en",    "data/wmt14_tok_en_de/valid.de",    "data/wmt14_tok_en_de/test.en",    "data/wmt14_tok_en_de/test.de",]for path in files:    n = count_empty_lines(path)    print(path, "空行数 =", n)
```

### 怎么判断

-   最理想：都是 `0`
    
-   如果不是 `0`，先不要删，先告诉我具体数量
    

* * *

## 检查 3：长度统计是否正常

这里不是为了过滤，而是为了观察。

你可以统计：

-   每行 token 数的最小值
    
-   最大值
    
-   平均值
    

### 检查脚本

```
def line_token_stats(path: str):    lengths = []    with open(path, "r", encoding="utf-8") as f:        for line in f:            lengths.append(len(line.strip().split()))    print(path)    print("  行数:", len(lengths))    print("  最短:", min(lengths))    print("  最长:", max(lengths))    print("  平均:", sum(lengths) / len(lengths))files = [    "data/wmt14_tok_en_de/valid.en",    "data/wmt14_tok_en_de/valid.de",    "data/wmt14_tok_en_de/test.en",    "data/wmt14_tok_en_de/test.de",]for path in files:    line_token_stats(path)
```

### 你关注什么

不是要求某个精确值，而是看有没有离谱现象：

-   平均长度很正常，但突然最长一行几千 token
    
-   大量 0 token 行
    
-   某一侧明显异常短或异常长
    

* * *

## 检查 4：随机抽样看几对句子

这个很朴素，但很有用。

### 检查脚本

```
import randomdef sample_parallel(en_path: str, de_path: str, k: int = 5, seed: int = 42):    with open(en_path, "r", encoding="utf-8") as f:        en_lines = [line.rstrip("\n") for line in f]    with open(de_path, "r", encoding="utf-8") as f:        de_lines = [line.rstrip("\n") for line in f]    assert len(en_lines) == len(de_lines)    random.seed(seed)    idxs = random.sample(range(len(en_lines)), k)    for idx in idxs:        print("=" * 80)        print(f"idx = {idx}")        print("[EN]", en_lines[idx])        print("[DE]", de_lines[idx])sample_parallel("data/wmt14_tok_en_de/valid.en", "data/wmt14_tok_en_de/valid.de", k=5)sample_parallel("data/wmt14_tok_en_de/test.en", "data/wmt14_tok_en_de/test.de", k=5)
```

### 你看什么

看是否存在：

-   明显乱码
    
-   一边是空行
    
-   一边很短一边超长
    
-   看起来完全不像翻译对应关系
    

* * *

# 五、如果这些轻量检查通过，下一步就直接 apply BPE

也就是说，你现在的流程应该是：

-    学到 `codes.bpe`
    
-    检查 `tok valid/test`
    
-    若检查正常，apply BPE 到：
    
    -   `clean train`
        
    -   `tok valid`
        
    -   `tok test`
        

* * *

# 六、你的问题本质上可以这样回答

你问的是：

> 我还没有对 val 做一致性检查，不删行的那种，需要在 BPE 之前做吗？

最简洁但准确的回答是：

> **建议现在做，而且做的是“检查”，不是“清洗”。**
> 
> 如果你前面已经确认了行数一致，那么这一步不会阻止你继续；它只是帮助你在 BPE 之前确认 `valid/test` 没有明显异常。

* * *

# 七、我给你的明确建议

你现在先不要立刻 apply BPE。  
先跑我上面给你的 4 类轻量检查，尤其是这两个最重要：

-    空行数检查
    
-    长度统计检查
    

如果结果看起来正常，就继续第四步后半段，也就是 apply BPE。

* * *

# 八、你现在最该做的事

你先把下面这两类结果发给我：

1.  `valid.en / valid.de / test.en / test.de` 的空行数
    
2.  `valid.en / valid.de / test.en / test.de` 的长度统计（最短、最长、平均）
    

拿到这两个结果后，我就直接继续带你做：

## 第四步后半段：apply BPE 到 train / valid / test，并检查最终行数与共享词表规模

  

Astudy\\python\_base\\pytorch\\my\_github\_workspace\\transformer-pytorch-main\\script\\dataset\_part\\05\_check\_val\_empty.py" data/wmt14\_tok\_en\_de/valid.en 空行数 = 0 data/wmt14\_tok\_en\_de/valid.de 空行数 = 0 data/wmt14\_tok\_en\_de/test.en 空行数 = 0 data/wmt14\_tok\_en\_de/test.de 空行数 = 0 data/wmt14\_tok\_en\_de/valid.en 行数: 3000 最短: 1 最长: 106 平均: 21.868 data/wmt14\_tok\_en\_de/valid.de 行数: 3000 最短: 1 最长: 103 平均: 21.419666666666668 data/wmt14\_tok\_en\_de/test.en 行数: 3003 最短: 2 最长: 91 平均: 22.95937395937396 data/wmt14\_tok\_en\_de/test.de 行数: 3003 最短: 1 最长: 75 平均: 21.479853479853478

  

我先核对一下 `subword-nmt` 当前官方 README 里的 joint BPE 与 apply-bpe 命令形式，然后我直接给你第四步后半段的可执行命令与检查项。

## 结论

你现在这组 `valid/test` 检查结果是正常的，可以继续做 **第四步后半段：apply BPE**。

你已经确认了：

-   `valid/test` 双语行数严格一致；
    
-   空行数都是 `0`；
    
-   长度统计也很健康，没有出现极端异常值。
    

在这种情况下，`valid/test` 已经满足我们这里定义的“轻量一致性检查，不删行”的要求，所以**不需要再额外插入别的检查步骤**，直接进入 BPE 应用阶段即可。

另外，`subword-nmt` 官方 README 也明确给出了 joint BPE 的推荐工作流：先用 `learn-joint-bpe-and-vocab` 学 joint BPE 与双语词表，再对 train/dev/test 使用相同的 `apply-bpe` 参数，并建议配合 `--vocabulary` 与 `--vocabulary-threshold` 保持分词一致性。Transformer 原论文在英德任务上也明确使用了 **shared source-target vocabulary**，规模约 **37000**。GitHub+1

* * *

# 第四步后半段：把 BPE 应用到 train / valid / test

## 这一步是干什么的

这一步的目标是：

1.  把你刚刚学到的 `codes.bpe` 应用到所有后续要用的数据上；
    
2.  训练集使用 **清洗后的 train**；
    
3.  验证集和测试集使用 **tokenized 但不删行的 valid/test**；
    
4.  最终得到真正给 Transformer 用的 **subword 级文本数据**。
    

你可以把这一步理解成：

> 从“词级文本”进一步变成“子词级文本”。

到这一步结束之后，你就会拿到后面 DataLoader 可以直接读取的 `.en/.de` 文本。

* * *

## 这一阶段的输入和输出

### 输入

训练集输入：

```
data/wmt14_clean_en_de/train.endata/wmt14_clean_en_de/train.de
```

验证/测试输入：

```
data/wmt14_tok_en_de/valid.endata/wmt14_tok_en_de/valid.dedata/wmt14_tok_en_de/test.endata/wmt14_tok_en_de/test.de
```

BPE 模型输入：

```
data/wmt14_bpe_model/codes.bpedata/wmt14_bpe_model/vocab.endata/wmt14_bpe_model/vocab.de
```

### 输出

你将生成：

```
data/wmt14_bpe_en_de/├── train.en├── train.de├── valid.en├── valid.de├── test.en└── test.de
```

* * *

## 第 4.5 步：先创建输出目录

在 PowerShell 里执行：

```
mkdir data\wmt14_bpe_en_de
```

如果目录已经存在，也没关系。

* * *

## 第 4.6 步：把 BPE 应用到训练集

这里训练集必须使用 **clean 后的 train**，因为你前面已经完成了训练集清洗。

### 英文训练集

```
cmd /c "subword-nmt apply-bpe -c data/wmt14_bpe_model/codes.bpe --vocabulary data/wmt14_bpe_model/vocab.en --vocabulary-threshold 50 < data/wmt14_clean_en_de/train.en > data/wmt14_bpe_en_de/train.en"
```

### 德文训练集

```
cmd /c "subword-nmt apply-bpe -c data/wmt14_bpe_model/codes.bpe --vocabulary data/wmt14_bpe_model/vocab.de --vocabulary-threshold 50 < data/wmt14_clean_en_de/train.de > data/wmt14_bpe_en_de/train.de"
```

这些命令就是 `subword-nmt` 官方推荐的 vocabulary-filter 形式：用 joint BPE codes，再分别配合各语言的 vocabulary 与 `--vocabulary-threshold 50` 重新应用 BPE。GitHub+1

* * *

## 第 4.7 步：把 BPE 应用到验证集

这里验证集使用的是 **tokenized 但未删行的 valid**。

### 英文验证集

```
cmd /c "subword-nmt apply-bpe -c data/wmt14_bpe_model/codes.bpe --vocabulary data/wmt14_bpe_model/vocab.en --vocabulary-threshold 50 < data/wmt14_tok_en_de/valid.en > data/wmt14_bpe_en_de/valid.en"
```

### 德文验证集

```
cmd /c "subword-nmt apply-bpe -c data/wmt14_bpe_model/codes.bpe --vocabulary data/wmt14_bpe_model/vocab.de --vocabulary-threshold 50 < data/wmt14_tok_en_de/valid.de > data/wmt14_bpe_en_de/valid.de"
```

`subword-nmt` README 也明确写到，对 dev/test 数据应复用同样的 BPE 规则和 vocabulary-filter 选项，以保持一致性。GitHub

* * *

## 第 4.8 步：把 BPE 应用到测试集

这里测试集同样使用 **tokenized 但未删行的 test**。

### 英文测试集

```
cmd /c "subword-nmt apply-bpe -c data/wmt14_bpe_model/codes.bpe --vocabulary data/wmt14_bpe_model/vocab.en --vocabulary-threshold 50 < data/wmt14_tok_en_de/test.en > data/wmt14_bpe_en_de/test.en"
```

### 德文测试集

```
cmd /c "subword-nmt apply-bpe -c data/wmt14_bpe_model/codes.bpe --vocabulary data/wmt14_bpe_model/vocab.de --vocabulary-threshold 50 < data/wmt14_tok_en_de/test.de > data/wmt14_bpe_en_de/test.de"
```

* * *

# 第四步后半段完成后，要做哪些检查

这一步不要跳。BPE 应用完成后，至少做下面 3 类检查。

* * *

## 检查 1：6 个文件都生成了

你应该看到：

```
data/wmt14_bpe_en_de/train.endata/wmt14_bpe_en_de/train.dedata/wmt14_bpe_en_de/valid.endata/wmt14_bpe_en_de/valid.dedata/wmt14_bpe_en_de/test.endata/wmt14_bpe_en_de/test.de
```

* * *

## 检查 2：行数必须保持不变

BPE 只是逐行切分，不应该改动文件行数。

### 预期行数

你现在应该看到：

-   `train.en = train.de = 3927488`
    
-   `valid.en = valid.de = 3000`
    
-   `test.en = test.de = 3003`
    

### 检查脚本

```
def count_lines(path: str) -> int:    with open(path, "r", encoding="utf-8") as f:        return sum(1 for _ in f)pairs = [    ("data/wmt14_bpe_en_de/train.en", "data/wmt14_bpe_en_de/train.de"),    ("data/wmt14_bpe_en_de/valid.en", "data/wmt14_bpe_en_de/valid.de"),    ("data/wmt14_bpe_en_de/test.en", "data/wmt14_bpe_en_de/test.de"),]for en_path, de_path in pairs:    en_n = count_lines(en_path)    de_n = count_lines(de_path)    print(en_path, en_n)    print(de_path, de_n)    assert en_n == de_n, f"行数不一致: {en_path} vs {de_path}"
```

* * *

## 检查 3：统计最终共享词表规模

因为你的目标是尽量贴近论文，而论文英德任务写的是 **shared source-target vocabulary about 37000**，所以建议你在 BPE 应用完之后，统计一下训练集 BPE 文本的共享词表规模。

### 统计脚本

```
def collect_vocab(path: str) -> set:    vocab = set()    with open(path, "r", encoding="utf-8") as f:        for line in f:            vocab.update(line.strip().split())    return vocabv_en = collect_vocab("data/wmt14_bpe_en_de/train.en")v_de = collect_vocab("data/wmt14_bpe_en_de/train.de")v_union = v_en | v_deprint(f"train.en vocab size: {len(v_en)}")print(f"train.de vocab size: {len(v_de)}")print(f"shared union vocab size: {len(v_union)}")
```

### 你怎么理解这个数

这里统计出来的是 **BPE 后训练文本里实际出现过的 token 并集大小**。  
它不一定严格等于你 `-s 37000` 的那个数，因为 `-s` 表示的是 BPE merges 数，而论文说的是最终 shared vocabulary 的量级。这两者不是一个概念。`subword-nmt` 官方也明确区分了 `learn-bpe -s {num_operations}` 与后续 vocabulary 的提取和过滤。GitHub+1

所以你现在只需要看：

-   如果 union vocab size 在 **3 万多、接近 37000** 的量级，那第一版就很正常；
    
-   如果明显偏离很多，再考虑后面微调 `-s`。
    

* * *

# 一个你现在需要明确的工程结论

你前面问的那句：

> 我目前还没有进行对 val 做一致性检查，不删行的那种，需要在 BPE 之前做吗？

现在答案已经很明确了：

-   你已经做了；
    
-   而且结果是正常的；
    
-   所以现在不需要再停下来补额外检查；
    
-   可以直接进入 apply BPE。
    

* * *

# 你现在就按这个顺序执行

-    创建 `data/wmt14_bpe_en_de`
    
-    apply BPE 到 clean train
    
-    apply BPE 到 tok valid
    
-    apply BPE 到 tok test
    
-    检查 6 个文件行数
    
-    统计 `shared union vocab size`
    

把下面这些结果发给我：

1.  `data/wmt14_bpe_en_de/train.en` 和 `train.de` 的行数
    
2.  `valid.en / valid.de` 的行数
    
3.  `test.en / test.de` 的行数
    
4.  `shared union vocab size`
    

拿到这些结果后，我下一步就继续带你做：

## 第五步：把这批 BPE 文本接到你当前 Transformer 工程的数据加载器和词表系统里

  

data/wmt14\_bpe\_en\_de/train.en 3927488 data/wmt14\_bpe\_en\_de/train.de 3927488 data/wmt14\_bpe\_en\_de/valid.en 3000 data/wmt14\_bpe\_en\_de/valid.de 3000 data/wmt14\_bpe\_en\_de/test.en 3003 data/wmt14\_bpe\_en\_de/test.de 3003 train.en vocab size: 24973 train.de vocab size: 33500 shared union vocab size: 40232

  

我先把你这个 `40232` 和论文里的 `about 37000` 对一下，再把下一步的数据加载与词表系统按你当前项目结构落地。

## 结论

你的结果：

-   `shared union vocab size = 40232`
    

**可以接受，不需要现在回炉重做 BPE。**

原因很直接：

1.  论文写的是 **shared source-target vocabulary of about 37000 tokens**，关键词是 **about**，不是一个硬性精确值。
    
2.  你现在统计的是**清洗后 train 上实际出现过的 BPE token 并集大小**；而论文里的表述是实验设置层面的共享词表量级，这两个口径不是完全相同的。
    
3.  你的 `40232` 相比 `37000` 高了大约 `8.7%`，属于**同一量级**，先继续往下走是合理的。
    

所以现在最正确的动作不是重做 BPE，而是进入：

# 第五步：构建 shared vocabulary / token-id 系统

这一步做完之后，你的数据链才真正从：

$BPE 文本\text{BPE 文本}BPE 文本$

进入：

$BPE token→id→可喂给 Transformer 的张量\text{BPE token} \rightarrow \text{id} \rightarrow \text{可喂给 Transformer 的张量}BPE token→id→可喂给 Transformer 的张量$

* * *

# 为什么第五步应该先做“shared vocabulary”，而不是立刻写训练循环

因为你的当前模型代码虽然已经有 Transformer 主干，但从项目结构上看，`make_model(src_vocab, tgt_vocab, ...)` 仍然是分开传源词表和目标词表大小，`src_embed`、`tgt_embed`、`generator` 也是分别构造的；`EncoderDecoder.decode()` 也还是直接吃 `tgt` 张量。也就是说，你现在还没有把“共享词表 + 数据侧 token-id 体系”真正接进模型工程。

而论文在 WMT14 English-German 上明确使用的是：

-   **shared source-target vocabulary**
    
-   大约 **37000 tokens**
    
-   目标端是 shifted right 的自回归输入。
    

所以第五步应该先把“词表系统”立起来，再做 DataLoader 和训练。

* * *

# 第五步的目标

这一步只做一件事：

> 从你已经准备好的 **BPE 训练文本** 中，构建一套**共享源/目标词表**，并为它分配稳定的 token-id。

* * *

## 这一步做完后，你应该得到什么

建议输出到：

```
data/wmt14_vocab/├── vocab.json├── vocab.txt└── meta.json
```

含义如下：

-   `vocab.json`：`token -> id`
    
-   `vocab.txt`：按 id 排序的 token 列表，方便肉眼检查
    
-   `meta.json`：记录特殊 token、词表大小等元信息
    

* * *

# 这一步的输入

只使用：

```
data/wmt14_bpe_en_de/train.endata/wmt14_bpe_en_de/train.de
```

注意：

-   **只用 train**
    
-   不用 valid/test
    

理由和前面一致：  
任何“从数据统计中学习规则”的步骤，都不应该把验证集和测试集混进去。

* * *

# 这一步的设计原则

## 1\. 使用 shared vocabulary

因为论文就是 shared source-target vocabulary。

所以我们不分别构建：

-   英文一套 id
    
-   德文一套 id
    

而是：

-   把 `train.en` 和 `train.de` 的 BPE token 合起来
    
-   建一套统一的 `token -> id`
    

* * *

## 2\. 加入 4 个特殊符号

建议固定这 4 个：

-   `<pad>`
    
-   `<bos>`
    
-   `<eos>`
    
-   `<unk>`
    

建议 id 顺序固定为：

```
<pad> = 0<bos> = 1<eos> = 2<unk> = 3
```

这样后面：

-   padding mask
    
-   target shifted right
    
-   推理起始 token
    
-   未登录词 fallback
    

都会统一很多。

* * *

## 3\. 训练集里出现过的 token 全保留

你现在 `train` 的 shared union vocab 是 `40232`。这个规模并不大，完全没必要再人为裁词表。

所以这一版建议：

-   `min_freq = 1`
    
-   训练集里出现过的 BPE token 全保留
    

这样最稳，最贴近你当前 BPE 输出。

* * *

# 第五步具体怎么做

下面我直接给你可落地代码。

* * *

## 文件 1：`data/shared_vocab.py`

```
from __future__ import annotations"""功能：1. 从 BPE 文本训练集构建 shared vocabulary。2. 提供 token -> id / id -> token 的双向映射。3. 保存和加载 vocab 文件。4. 为后续 DataLoader、target shifted right、mask 构造提供统一词表接口。设计原则：1. 只用 train 构建词表，不让 valid/test 泄露统计信息。2. 源语言和目标语言共享一套 token-id 体系。3. 固定特殊符号顺序，确保 pad/bos/eos/unk 的 id 稳定。"""import jsonfrom collections import Counterfrom dataclasses import dataclassfrom typing import Dict, Iterable, List, Sequence@dataclassclass SpecialTokens:    pad: str = "<pad>"    bos: str = "<bos>"    eos: str = "<eos>"    unk: str = "<unk>"    def as_list(self) -> List[str]:        return [self.pad, self.bos, self.eos, self.unk]class SharedVocab:    """    shared source-target vocabulary    入口：        - 通过 build_from_files 从 train.en/train.de 构建        - 或通过 load 从已保存文件加载    出口：        - token_to_id / id_to_token        - encode / decode        - 各特殊 token 的 id    """    def __init__(        self,        token_to_id: Dict[str, int],        special_tokens: SpecialTokens,    ) -> None:        self.token_to_id = token_to_id        self.id_to_token = {idx: tok for tok, idx in token_to_id.items()}        self.special_tokens = special_tokens        self.pad_id = self.token_to_id[self.special_tokens.pad]        self.bos_id = self.token_to_id[self.special_tokens.bos]        self.eos_id = self.token_to_id[self.special_tokens.eos]        self.unk_id = self.token_to_id[self.special_tokens.unk]    def __len__(self) -> int:        return len(self.token_to_id)    def token2id(self, token: str) -> int:        return self.token_to_id.get(token, self.unk_id)    def id2token(self, idx: int) -> str:        return self.id_to_token[idx]    def encode(self, tokens: Sequence[str]) -> List[int]:        """        将 token 序列编码成 id 序列。        """        return [self.token2id(tok) for tok in tokens]    def decode(self, ids: Sequence[int]) -> List[str]:        """        将 id 序列解码成 token 序列。        """        return [self.id2token(idx) for idx in ids]    @classmethod    def build_from_files(        cls,        file_paths: Sequence[str],        min_freq: int = 1,        special_tokens: SpecialTokens | None = None,    ) -> "SharedVocab":        """        从多个训练文本文件中构建共享词表。        参数：            file_paths: 参与构建词表的文件路径列表，通常是 train.en 和 train.de            min_freq: 最小词频阈值            special_tokens: 特殊 token 配置        返回：            SharedVocab 实例        """        if special_tokens is None:            special_tokens = SpecialTokens()        counter: Counter = Counter()        for path in file_paths:            with open(path, "r", encoding="utf-8") as f:                for line in f:                    tokens = line.strip().split()                    counter.update(tokens)        # 先放特殊 token，确保 id 稳定        token_to_id: Dict[str, int] = {}        for tok in special_tokens.as_list():            token_to_id[tok] = len(token_to_id)        # 再放普通 token        # 排序策略：        # 1. 先按词频降序        # 2. 词频相同按字典序        # 这样可以保证构建结果稳定、可复现        normal_tokens = [            (tok, freq)            for tok, freq in counter.items()            if freq >= min_freq and tok not in token_to_id        ]        normal_tokens.sort(key=lambda x: (-x[1], x[0]))        for tok, _ in normal_tokens:            token_to_id[tok] = len(token_to_id)        return cls(token_to_id=token_to_id, special_tokens=special_tokens)    def save(self, vocab_json_path: str, vocab_txt_path: str, meta_json_path: str) -> None:        """        保存词表到磁盘。        """        with open(vocab_json_path, "w", encoding="utf-8") as f:            json.dump(self.token_to_id, f, ensure_ascii=False, indent=2)        with open(vocab_txt_path, "w", encoding="utf-8") as f:            for idx in range(len(self)):                f.write(self.id_to_token[idx] + "\n")        meta = {            "vocab_size": len(self),            "pad_token": self.special_tokens.pad,            "bos_token": self.special_tokens.bos,            "eos_token": self.special_tokens.eos,            "unk_token": self.special_tokens.unk,            "pad_id": self.pad_id,            "bos_id": self.bos_id,            "eos_id": self.eos_id,            "unk_id": self.unk_id,        }        with open(meta_json_path, "w", encoding="utf-8") as f:            json.dump(meta, f, ensure_ascii=False, indent=2)    @classmethod    def load(        cls,        vocab_json_path: str,        special_tokens: SpecialTokens | None = None,    ) -> "SharedVocab":        """        从 vocab.json 加载词表。        """        if special_tokens is None:            special_tokens = SpecialTokens()        with open(vocab_json_path, "r", encoding="utf-8") as f:            token_to_id = json.load(f)        return cls(token_to_id=token_to_id, special_tokens=special_tokens)
```

* * *

## 文件 2：`script/dataset_part/06_build_shared_vocab.py`

```
from __future__ import annotations"""功能：1. 从 BPE 后的 train.en 和 train.de 构建 shared vocabulary2. 保存 vocab.json / vocab.txt / meta.json3. 打印词表规模和特殊 token id"""import osfrom data.shared_vocab import SharedVocab, SpecialTokensdef main() -> None:    train_en = "data/wmt14_bpe_en_de/train.en"    train_de = "data/wmt14_bpe_en_de/train.de"    out_dir = "data/wmt14_vocab"    os.makedirs(out_dir, exist_ok=True)    vocab = SharedVocab.build_from_files(        file_paths=[train_en, train_de],        min_freq=1,        special_tokens=SpecialTokens(            pad="<pad>",            bos="<bos>",            eos="<eos>",            unk="<unk>",        ),    )    vocab.save(        vocab_json_path=os.path.join(out_dir, "vocab.json"),        vocab_txt_path=os.path.join(out_dir, "vocab.txt"),        meta_json_path=os.path.join(out_dir, "meta.json"),    )    print("共享词表构建完成。")    print(f"vocab_size = {len(vocab)}")    print(f"pad_id = {vocab.pad_id}")    print(f"bos_id = {vocab.bos_id}")    print(f"eos_id = {vocab.eos_id}")    print(f"unk_id = {vocab.unk_id}")if __name__ == "__main__":    main()
```

* * *

# 这一步运行后，你应该看到什么

如果你当前统计到的 `shared union vocab size = 40232`，并且 4 个特殊 token 不在训练文本里，那么这一步之后：

$vocab_size≈40232+4=40236\text{vocab\_size} \approx 40232 + 4 = 40236vocab_size≈40232+4=40236$

这是我们当前最期望看到的结果。

* * *

# 为什么这一步是正确的

## 1\. 它和论文目标一致

论文明确说的是 shared source-target vocabulary，大约 37000。你现在虽然观测到的是 `40232`，但仍然是在**共享词表**这条正确路线里。

## 2\. 它和你当前 BPE 结果一致

你不是从 raw 文本建词表，而是从 **BPE 后的 train 文本** 建词表。  
这才符合你的后续模型输入形式。

## 3\. 它和后面的权重共享是一一对应的

你后面要做：

-   `src_vocab = tgt_vocab = len(vocab)`
    
-   `share_embeddings=True`
    
-   `src_embed/tgt_embed/generator` 共享权重
    

这要求你**先有一套真正统一的 token-id 体系**。  
而你当前项目里的 `make_model(src_vocab, tgt_vocab, ...)` 还只是接口层面支持两边分别传大小，数据侧还没真正打通。

* * *

# 这一阶段先不要做什么

先不要急着：

-   写 loss
    
-   写 train loop
    
-   写 beam search
    

因为现在最缺的不是这些，而是：

> **BPE 文本到 token-id 的统一桥梁。**

如果这层没搭好，后面的 shifted right、mask、共享 embedding 都会接得很乱。

* * *

# 这一步完成后的下一步是什么

这一步完成后，下一步就是：

# 第五步的后半段：把 `vocab + BPE 文本` 接成真正的 `Dataset / DataLoader`

也就是：

1.  逐行读取 `train.en / train.de`
    
2.  `split()` 成 BPE token
    
3.  用 `SharedVocab.encode()` 变成 id
    
4.  对 target 加 `<bos>/<eos>`
    
5.  构造 `src_mask / tgt_mask`
    
6.  输出 `Seq2SeqBatch`
    

* * *

# 你现在该做什么

你先运行：

```
python script/dataset_part/06_build_shared_vocab.py
```

然后把下面这些结果发给我：

1.  `vocab_size`
    
2.  `pad_id / bos_id / eos_id / unk_id`
    
3.  `data/wmt14_vocab/` 目录是否成功生成这 3 个文件：
    
    -   `vocab.json`
        
    -   `vocab.txt`
        
    -   `meta.json`
        

等你把这个结果贴出来，我们就进入：

## 第五步后半段：BPE 文本 -> token id -> `Dataset / DataLoader / Seq2SeqBatch`

  

我运行了脚本，这是结果： 共享词表构建完成。python\_base\\pytorch\\my\_github\_workspace\\transformer-pytorch-main> vocab\_size = 40236 pad\_id = 0 bos\_id = 1 eos\_id = 2 unk\_id = 3

  

## 很好，第五步前半段完成了

你现在已经有了：

-   `data/wmt14_bpe_en_de/train|valid|test.{en,de}`
    
-   `data/wmt14_vocab/vocab.json`
    
-   `data/wmt14_vocab/vocab.txt`
    
-   `data/wmt14_vocab/meta.json`
    
-   `vocab_size = 40236`
    
-   `pad_id=0, bos_id=1, eos_id=2, unk_id=3`
    

这一步已经把“共享词表”立住了。  
而且这和论文里英德任务使用 **shared source-target vocabulary** 的方向是一致的。论文里说的是大约 `37000`，你的 `40236` 仍然在同一量级，可以先继续，不必现在回炉重做 BPE。

* * *

# 第五步后半段：把 BPE 文本接到 `Dataset / DataLoader / Seq2SeqBatch`

## 这一步是干什么的

这一步的目标是把你现在的文本数据链：

$BPE 文本\text{BPE 文本}BPE 文本$

真正变成模型可吃的张量链：

$BPE 文本→token 列表→id 序列→padding→src_mask/tgt_mask\text{BPE 文本} \rightarrow \text{token 列表} \rightarrow \text{id 序列} \rightarrow \text{padding} \rightarrow \text{src\_mask/tgt\_mask}BPE 文本→token 列表→id 序列→padding→src_mask/tgt_mask$

也就是说，这一步做完后，你就可以真正跑出：

-   `batch.src`
    
-   `batch.tgt_input`
    
-   `batch.tgt_y`
    
-   `batch.src_mask`
    
-   `batch.tgt_mask`
    

然后把它们喂给你当前的 Transformer。

* * *

## 先说一个设计决策

你现在的训练集有 `3927488` 行。  
如果你直接写一个 `Dataset`，把整个 `train.en` 和 `train.de` 全部读进内存，Windows 下会非常吃内存，不划算。

所以我建议你这一版先走：

-   **IterableDataset**
    
-   **按行流式读取**
    
-   **先追求正确性和链路打通**
    
-   `shuffle / bucket / token-based batching` 等训练细节我们下一阶段再升级
    

这是一个非常合理的阶段性切分。

* * *

# 你现在需要新增 3 个文件

我建议你新增这 3 个文件：

```
data/├── batch.py├── wmt14_bpe_dataset.pyscript/dataset_part/├── 07_test_bpe_dataloader.py└── 08_test_real_batch_forward.py
```

下面我逐个给你。

* * *

# 文件 1：`data/batch.py`

## 这个文件是干什么的

它负责统一封装一个 Seq2Seq batch，并且集中处理：

-   target shifted right
    
-   `src_mask`
    
-   `tgt_mask = padding mask + causal mask`
    

也就是你前面一直在修的那几个关键点，都放到这里。

* * *

```
from __future__ import annotations"""功能：1. 构造 Transformer 训练阶段所需的 src_mask。2. 构造同时包含 padding mask + causal mask 的 tgt_mask。3. 将完整 target 序列切分为：   - tgt_input: 送入 Decoder 的输入（右移后的序列）   - tgt_y:     监督标签4. 用 Seq2SeqBatch 统一封装 batch，方便后续训练代码直接调用。重要约定：1. 传入的 tgt 必须已经包含 <bos> 和 <eos>。2. 这里的 shifted right 采用最稳妥的切片写法：   tgt_input = tgt[:, :-1]   tgt_y     = tgt[:, 1:]3. 当前实现与你前面的 MultiHeadedAttention 兼容：   - src_mask 形状: (B, 1, S)   - tgt_mask 形状: (B, T, T)"""from dataclasses import dataclassfrom typing import Tupleimport torchdef subsequent_mask(size: int, device: torch.device) -> torch.Tensor:    """    生成下三角因果 Mask。    输入：        size: 目标序列长度 T        device: 张量所在设备    输出：        mask: 形状为 (1, T, T) 的 bool Tensor              True  表示允许关注              False 表示禁止关注    """    return torch.tril(        torch.ones((1, size, size), dtype=torch.bool, device=device)    )def make_src_mask(src: torch.Tensor, pad_idx: int) -> torch.Tensor:    """    根据源序列构造 src_mask。    输入：        src: 形状 (B, S) 的源序列 token id        pad_idx: PAD 的词表 id    输出：        src_mask: 形状 (B, 1, S) 的 bool Tensor    """    return (src != pad_idx).unsqueeze(1)def shift_right(tgt: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:    """    将完整 target 序列切分为 Decoder 输入和监督标签。    输入：        tgt: 形状 (B, T_full) 的完整目标序列             要求已经包含 <bos> 和 <eos>    输出：        tgt_input: 形状 (B, T_full - 1)                   送入 Decoder 的输入        tgt_y:     形状 (B, T_full - 1)                   训练监督标签    """    if tgt.dim() != 2:        raise ValueError(f"tgt 必须是 2 维张量，当前维度为: {tgt.dim()}")    if tgt.size(1) < 2:        raise ValueError(            "tgt 的序列长度至少要 >= 2，因为必须能够切分出 tgt_input 和 tgt_y。"        )    tgt_input = tgt[:, :-1].contiguous()    tgt_y = tgt[:, 1:].contiguous()    return tgt_input, tgt_ydef make_tgt_mask(tgt_input: torch.Tensor, pad_idx: int) -> torch.Tensor:    """    构造 Decoder 自注意力所需的 tgt_mask。    输入：        tgt_input: 形状 (B, T) 的 Decoder 输入序列        pad_idx: PAD 的词表 id    输出：        tgt_mask: 形状 (B, T, T) 的 bool Tensor    构造逻辑：        1. padding mask: 屏蔽掉 PAD 位置，形状 (B, 1, T)        2. causal mask:  只允许看当前位置及之前位置，形状 (1, T, T)        3. 两者按位与，广播后得到 (B, T, T)    """    if tgt_input.dim() != 2:        raise ValueError(            f"tgt_input 必须是 2 维张量，当前维度为: {tgt_input.dim()}"        )    pad_mask = (tgt_input != pad_idx).unsqueeze(1)  # (B, 1, T)    causal_mask = subsequent_mask(tgt_input.size(1), tgt_input.device)  # (1, T, T)    tgt_mask = pad_mask & causal_mask    return tgt_maskdef pad_sequences(sequences, pad_idx: int) -> torch.Tensor:    """    将变长 id 序列 padding 成统一长度的 LongTensor。    输入：        sequences: List[List[int]]        pad_idx: PAD 的词表 id    输出：        tensor: 形状 (B, L_max)    """    if len(sequences) == 0:        raise ValueError("sequences 不能为空。")    max_len = max(len(seq) for seq in sequences)    batch_size = len(sequences)    out = torch.full((batch_size, max_len), pad_idx, dtype=torch.long)    for i, seq in enumerate(sequences):        out[i, :len(seq)] = torch.tensor(seq, dtype=torch.long)    return out@dataclassclass Seq2SeqBatch:    """    统一封装 Transformer 训练所需的一个 batch。    字段说明：        src:       源序列，形状 (B, S)        tgt_input: Decoder 输入，形状 (B, T)        tgt_y:     监督标签，形状 (B, T)        src_mask:  源序列 mask，形状 (B, 1, S)        tgt_mask:  目标序列 mask，形状 (B, T, T)        ntokens:   tgt_y 中非 PAD token 的数量    """    src: torch.Tensor    tgt_input: torch.Tensor    tgt_y: torch.Tensor    src_mask: torch.Tensor    tgt_mask: torch.Tensor    ntokens: int    @classmethod    def from_tensors(        cls,        src: torch.Tensor,        tgt: torch.Tensor,        pad_idx: int,    ) -> "Seq2SeqBatch":        """        从 src / tgt 原始张量构造训练 batch。        输入：            src: 形状 (B, S)            tgt: 形状 (B, T_full)，要求已包含 <bos> 和 <eos>            pad_idx: PAD 的词表 id        """        src_mask = make_src_mask(src, pad_idx)        tgt_input, tgt_y = shift_right(tgt)        tgt_mask = make_tgt_mask(tgt_input, pad_idx)        ntokens = int((tgt_y != pad_idx).sum().item())        return cls(            src=src,            tgt_input=tgt_input,            tgt_y=tgt_y,            src_mask=src_mask,            tgt_mask=tgt_mask,            ntokens=ntokens,        )    def to(self, device: torch.device) -> "Seq2SeqBatch":        """        将 batch 移动到指定设备。        """        return Seq2SeqBatch(            src=self.src.to(device, non_blocking=True),            tgt_input=self.tgt_input.to(device, non_blocking=True),            tgt_y=self.tgt_y.to(device, non_blocking=True),            src_mask=self.src_mask.to(device, non_blocking=True),            tgt_mask=self.tgt_mask.to(device, non_blocking=True),            ntokens=self.ntokens,        )
```

* * *

# 文件 2：`data/wmt14_bpe_dataset.py`

## 这个文件是干什么的

它负责：

1.  逐行读取 BPE 后的英德文本；
    
2.  用 `SharedVocab` 做编码；
    
3.  source/target 共享同一套 token-id；
    
4.  target 自动加 `<bos>` 和 `<eos>`；
    
5.  collate 时构造成 `Seq2SeqBatch`。
    

### 这里我用 `IterableDataset`

原因很直接：

-   你的训练集 392 万行；
    
-   这一版先打通链路，不把全量数据一次性读进内存；
    
-   训练时更复杂的 `shuffle / bucket / token-based batching`，我们下一阶段再做。
    

* * *

```
from __future__ import annotations"""功能：1. 逐行读取 BPE 后的平行语料。2. 使用 SharedVocab 将 token 编码为 id。3. 在 collate_fn 中完成：   - source EOS 追加（可选）   - target BOS/EOS 追加   - padding   - target shifted right   - src_mask / tgt_mask 构造4. 返回 Seq2SeqBatch，供 Transformer 直接使用。设计说明：1. 当前版本采用 IterableDataset，优先保证大规模训练集在 Windows 下能稳定跑通。2. 当前版本不做 shuffle / bucket / token-based batching。   这些属于训练阶段优化，后面再专门升级。"""from itertools import zip_longestfrom typing import Callable, Iterable, Iterator, List, Optional, Sequence, Tupleimport torchfrom torch.utils.data import DataLoader, IterableDataset, get_worker_infofrom data.batch import Seq2SeqBatch, pad_sequencesfrom data.shared_vocab import SharedVocabclass ParallelBPEIterableDataset(IterableDataset):    """    逐行流式读取平行 BPE 文本。    每个样本输出：        (src_tokens, tgt_tokens)    其中：        src_tokens: List[str]        tgt_tokens: List[str]    """    def __init__(        self,        src_path: str,        tgt_path: str,        skip_empty: bool = False,    ) -> None:        super().__init__()        self.src_path = src_path        self.tgt_path = tgt_path        self.skip_empty = skip_empty    def __iter__(self) -> Iterator[Tuple[List[str], List[str]]]:        worker_info = get_worker_info()        worker_id = 0 if worker_info is None else worker_info.id        num_workers = 1 if worker_info is None else worker_info.num_workers        with open(self.src_path, "r", encoding="utf-8") as f_src, \             open(self.tgt_path, "r", encoding="utf-8") as f_tgt:            for line_idx, pair in enumerate(zip_longest(f_src, f_tgt, fillvalue=None)):                src_line, tgt_line = pair                if src_line is None or tgt_line is None:                    raise RuntimeError(                        "检测到源文件和目标文件在迭代时长度不一致。"                    )                # 多 worker 下做简单按行切片，避免重复读同一批样本                if (line_idx % num_workers) != worker_id:                    continue                src_tokens = src_line.strip().split()                tgt_tokens = tgt_line.strip().split()                if self.skip_empty and (len(src_tokens) == 0 or len(tgt_tokens) == 0):                    continue                yield src_tokens, tgt_tokensdef build_bpe_collate_fn(    vocab: SharedVocab,    max_src_len: Optional[int] = None,    max_tgt_len: Optional[int] = None,    add_src_eos: bool = True,) -> Callable[[Sequence[Tuple[List[str], List[str]]]], Seq2SeqBatch]:    """    构造 DataLoader 所需的 collate_fn。    输入：        vocab: 共享词表        max_src_len: 源序列最大长度（截断前，不含额外追加的 EOS）        max_tgt_len: 目标序列最大长度（截断前，不含 BOS/EOS）        add_src_eos: 是否给源序列末尾追加 EOS    输出：        collate_fn，返回 Seq2SeqBatch    """    def collate_fn(batch: Sequence[Tuple[List[str], List[str]]]) -> Seq2SeqBatch:        src_id_list: List[List[int]] = []        tgt_id_list: List[List[int]] = []        for src_tokens, tgt_tokens in batch:            if max_src_len is not None:                src_tokens = src_tokens[:max_src_len]            if max_tgt_len is not None:                tgt_tokens = tgt_tokens[:max_tgt_len]            src_ids = vocab.encode(src_tokens)            tgt_ids = vocab.encode(tgt_tokens)            if add_src_eos:                src_ids = src_ids + [vocab.eos_id]            # 目标序列必须显式带 BOS / EOS，后续才能做 shifted right            tgt_ids = [vocab.bos_id] + tgt_ids + [vocab.eos_id]            src_id_list.append(src_ids)            tgt_id_list.append(tgt_ids)        src_tensor = pad_sequences(src_id_list, pad_idx=vocab.pad_id)        tgt_tensor = pad_sequences(tgt_id_list, pad_idx=vocab.pad_id)        batch_obj = Seq2SeqBatch.from_tensors(            src=src_tensor,            tgt=tgt_tensor,            pad_idx=vocab.pad_id,        )        return batch_obj    return collate_fndef build_bpe_dataloader(    src_path: str,    tgt_path: str,    vocab: SharedVocab,    batch_size: int,    num_workers: int = 0,    pin_memory: bool = True,    max_src_len: Optional[int] = None,    max_tgt_len: Optional[int] = None,    add_src_eos: bool = True,    skip_empty: bool = False,) -> DataLoader:    """    构建平行 BPE 文本的 DataLoader。    入口：        src_path / tgt_path: BPE 文本文件路径        vocab: 共享词表        batch_size: batch 大小        num_workers: DataLoader worker 数量        pin_memory: 是否启用 pin_memory        max_src_len: 源序列截断长度        max_tgt_len: 目标序列截断长度        add_src_eos: 是否在源序列末尾追加 EOS        skip_empty: 是否跳过空样本    出口：        DataLoader，每次迭代返回一个 Seq2SeqBatch    注意：        当前版本不支持 shuffle。        后面讨论训练策略时，再升级为 bucket sampler / token-based batching。    """    dataset = ParallelBPEIterableDataset(        src_path=src_path,        tgt_path=tgt_path,        skip_empty=skip_empty,    )    collate_fn = build_bpe_collate_fn(        vocab=vocab,        max_src_len=max_src_len,        max_tgt_len=max_tgt_len,        add_src_eos=add_src_eos,    )    loader = DataLoader(        dataset=dataset,        batch_size=batch_size,        num_workers=num_workers,        pin_memory=pin_memory,        collate_fn=collate_fn,    )    return loader
```

* * *

# 文件 3：`script/dataset_part/07_test_bpe_dataloader.py`

## 这个脚本是干什么的

它用于做**数据侧通路测试**：

-   词表能不能正常加载
    
-   DataLoader 能不能正常产出 batch
    
-   `src / tgt_input / tgt_y / src_mask / tgt_mask` 的 shape 对不对
    
-   `<pad>/<bos>/<eos>` 这些 id 有没有正常进入张量
    

* * *

```
from __future__ import annotations"""功能：1. 测试 SharedVocab 是否能正常加载。2. 测试 BPE DataLoader 是否能正常产出 Seq2SeqBatch。3. 打印一个 batch 的关键维度信息。"""from data.shared_vocab import SharedVocabfrom data.wmt14_bpe_dataset import build_bpe_dataloaderdef main() -> None:    vocab = SharedVocab.load("data/wmt14_vocab/vocab.json")    print("共享词表加载成功。")    print(f"vocab_size = {len(vocab)}")    print(f"pad_id = {vocab.pad_id}")    print(f"bos_id = {vocab.bos_id}")    print(f"eos_id = {vocab.eos_id}")    print(f"unk_id = {vocab.unk_id}")    print("-" * 60)    loader = build_bpe_dataloader(        src_path="data/wmt14_bpe_en_de/train.en",        tgt_path="data/wmt14_bpe_en_de/train.de",        vocab=vocab,        batch_size=4,        num_workers=0,        pin_memory=False,        max_src_len=128,        max_tgt_len=128,        add_src_eos=True,        skip_empty=False,    )    batch = next(iter(loader))    print("成功取到一个 batch。")    print(f"src.shape       = {tuple(batch.src.shape)}")    print(f"tgt_input.shape = {tuple(batch.tgt_input.shape)}")    print(f"tgt_y.shape     = {tuple(batch.tgt_y.shape)}")    print(f"src_mask.shape  = {tuple(batch.src_mask.shape)}")    print(f"tgt_mask.shape  = {tuple(batch.tgt_mask.shape)}")    print(f"ntokens         = {batch.ntokens}")    print("-" * 60)    print("观察第一个样本前若干个 id：")    print("src[0, :20]      =", batch.src[0, :20].tolist())    print("tgt_input[0,:20] =", batch.tgt_input[0, :20].tolist())    print("tgt_y[0, :20]    =", batch.tgt_y[0, :20].tolist())    assert batch.src.dim() == 2    assert batch.tgt_input.dim() == 2    assert batch.tgt_y.dim() == 2    assert batch.src_mask.dim() == 3    assert batch.tgt_mask.dim() == 3    print("\nDataLoader 测试通过。")if __name__ == "__main__":    main()
```

* * *

# 文件 4：`script/dataset_part/08_test_real_batch_forward.py`

## 这个脚本是干什么的

它用于做**模型侧通路测试**：

-   用真实的 BPE batch
    
-   喂给你当前的 Transformer
    
-   看 `forward` 和 `generator` 能不能跑通
    

这一步很关键，因为你之前的模型测试大多是随机张量。  
现在这一步能证明：

> 你的真实数据链已经可以进入模型了。

* * *

```
from __future__ import annotations"""功能：1. 使用真实的 WMT14 BPE batch 做一次前向传播测试。2. 验证：   - shared vocab 的大小能否正确传入 make_model   - 真实 batch 是否能喂给 Transformer   - generator 输出 logits 维度是否正确"""import torchfrom data.shared_vocab import SharedVocabfrom data.wmt14_bpe_dataset import build_bpe_dataloaderfrom nets.build_transformer import make_modeldef main() -> None:    vocab = SharedVocab.load("data/wmt14_vocab/vocab.json")    vocab_size = len(vocab)    loader = build_bpe_dataloader(        src_path="data/wmt14_bpe_en_de/train.en",        tgt_path="data/wmt14_bpe_en_de/train.de",        vocab=vocab,        batch_size=2,        num_workers=0,        pin_memory=False,        max_src_len=64,        max_tgt_len=64,        add_src_eos=True,        skip_empty=False,    )    batch = next(iter(loader))    print("成功取到真实 batch。")    print(f"src.shape       = {tuple(batch.src.shape)}")    print(f"tgt_input.shape = {tuple(batch.tgt_input.shape)}")    print(f"tgt_y.shape     = {tuple(batch.tgt_y.shape)}")    print(f"src_mask.shape  = {tuple(batch.src_mask.shape)}")    print(f"tgt_mask.shape  = {tuple(batch.tgt_mask.shape)}")    print("-" * 60)    # 这里先用一个微型模型做通路测试，避免显存压力    model = make_model(        src_vocab=vocab_size,        tgt_vocab=vocab_size,        N=2,        d_model=256,        d_ff=1024,        h=4,        dropout=0.1,        share_embeddings=True,    )    model.eval()    with torch.no_grad():        hidden_states = model(            batch.src,            batch.tgt_input,            batch.src_mask,            batch.tgt_mask,        )        logits = model.generator(hidden_states)    print(f"hidden_states.shape = {tuple(hidden_states.shape)}")    print(f"logits.shape        = {tuple(logits.shape)}")    assert hidden_states.shape[0] == batch.src.shape[0]    assert hidden_states.shape[1] == batch.tgt_input.shape[1]    assert hidden_states.shape[2] == 256    assert logits.shape[0] == batch.src.shape[0]    assert logits.shape[1] == batch.tgt_input.shape[1]    assert logits.shape[2] == vocab_size    print("\n真实 batch 前向传播测试通过。")if __name__ == "__main__":    main()
```

* * *

# 运行顺序

你现在按这个顺序做：

-    把 `data/batch.py` 放到项目里
    
-    把 `data/wmt14_bpe_dataset.py` 放到项目里
    
-    运行 `python script/dataset_part/07_test_bpe_dataloader.py`
    
-    如果通过，再运行 `python script/dataset_part/08_test_real_batch_forward.py`
    

* * *

# 预期结果

## 第一个脚本应该看到

类似：

-   `vocab_size = 40236`
    
-   `pad_id = 0`
    
-   `bos_id = 1`
    
-   `eos_id = 2`
    
-   `unk_id = 3`
    
-   `src.shape = (...)`
    
-   `tgt_input.shape = (...)`
    
-   `tgt_y.shape = (...)`
    
-   `src_mask.shape = (...)`
    
-   `tgt_mask.shape = (...)`
    

并且没有断言报错。

* * *

## 第二个脚本应该看到

类似：

-   `hidden_states.shape = (B, T, 256)`
    
-   `logits.shape = (B, T, 40236)`
    

并且没有报错。

* * *

# 这一步完成后意味着什么

如果这两个脚本都通过，说明你已经完成了下面这整条链：

$WMT14 raw→Moses→clean train→joint BPE→shared vocab→BPE 文本编码成 id→Seq2SeqBatch→真实 batch 喂给 Transformer\text{WMT14 raw} \rightarrow \text{Moses} \rightarrow \text{clean train} \rightarrow \text{joint BPE} \rightarrow \text{shared vocab} \rightarrow \text{BPE 文本编码成 id} \rightarrow \text{Seq2SeqBatch} \rightarrow \text{真实 batch 喂给 Transformer}WMT14 raw→Moses→clean train→joint BPE→shared vocab→BPE 文本编码成 id→Seq2SeqBatch→真实 batch 喂给 Transformer$

这时候你才真正有资格进入下一阶段：

# 第六步：训练循环、损失函数、label smoothing、Noam lr、验证逻辑

* * *

# 一个必须提醒你的点

上面这套代码默认你前面已经把 `make_model(..., share_embeddings=True)` 和权重共享逻辑接进去了。  
如果你当前 `nets/build_transformer.py` 还是最原始版本，没有 `share_embeddings` 参数，那 `08_test_real_batch_forward.py` 会报错。

也就是说：

-   如果你已经按我前面给你的版本改过 `make_model()`，那直接跑；
    
-   如果你还没改，我下一条就先帮你把 `build_transformer.py` 和 `Generator.py` 对齐到 shared embedding 版本。
    

* * *

你现在先运行这两个测试脚本。  
把下面这些结果发给我：

1.  `07_test_bpe_dataloader.py` 的输出
    
2.  `08_test_real_batch_forward.py` 是否通过
    
3.  如果报错，把完整报错贴出来
    

  

PS D:\\1AAAAAstudy\\python\_base\\pytorch\\my\_github\_workspace\\transformer-pytorch-main> python -m script.dataset\_part.08\_test\_bpe\_dataloader 共享词表加载成功。 vocab\_size = 40236 pad\_id = 0 bos\_id = 1 eos\_id = 2 unk\_id = 3 ------------------------------------------------------------ 成功取到一个 batch。 src.shape = (4, 43) tgt\_input.shape = (4, 38) tgt\_y.shape = (4, 38) src\_mask.shape = (4, 1, 43) tgt\_mask.shape = (4, 38, 38) ntokens = 93 ------------------------------------------------------------ 观察第一个样本前若干个 id： src\[0, :20\] = \[4859, 580, 25696, 8, 6, 3675, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0\] tgt\_input\[0,:20\] = \[1, 20529, 10, 18687, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0\] tgt\_y\[0, :20\] = \[20529, 10, 18687, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0\] DataLoader 测试通过。 PS D:\\1AAAAAstudy\\python\_base\\pytorch\\my\_github\_workspace\\transformer-pytorch-main> python -m script.dataset\_part.09\_test\_real\_batch\_forward 成功取到真实 batch。 src.shape = (2, 43) tgt\_input.shape = (2, 38) tgt\_y.shape = (2, 38) src\_mask.shape = (2, 1, 43) tgt\_mask.shape = (2, 38, 38) ------------------------------------------------------------ hidden\_states.shape = (2, 38, 256) logits.shape = (2, 38, 40236) 真实 batch 前向传播测试通过。 这是两个脚本的结果。我的设备可以训练imagenet1000，是不是可以支持一次性加载全部的数据呢？

  

## 结论

**不建议你现在“一次性把全部训练数据加载到内存里”。**

你的设备**能训练 ImageNet-1000**，只能说明：

-   你的 **GPU 显存** 和训练吞吐大概率不错；
    
-   你可以承受比较重的模型训练。
    

但这**不等价于**：

-   你的 **CPU 内存（RAM）** 适合把 `3927488` 对英德句子全部预加载成 Python 对象；
    
-   你的 Windows 进程在 DataLoader/多进程下适合把整个 WMT14 BPE 数据常驻内存。
    

这两件事是两套资源系统：

-   图像训练更看 **显存、磁盘吞吐、解码吞吐**
    
-   大规模文本预加载更看 **RAM、Python 对象开销、DataLoader 复制开销**
    

所以，基于你当前这个项目阶段，我的建议很明确：

> **继续保持你现在的“流式读取 + on-the-fly 编码 + collate 构 batch”方案。**
> 
> 先不要改成“全量一次性加载”。

* * *

## 一、为什么“能训 ImageNet-1000”不代表“应该全量加载 WMT14 文本”

这个问题很容易混淆，我给你拆开说。

### 1\. 图像训练通常不是把全部图片像素一次性塞进内存

你训练 ImageNet-1000 时，常见做法其实是：

-   磁盘上存图片文件
    
-   DataLoader 每次按 batch 读取
    
-   decode / augment / collate
    
-   再送进 GPU
    

也就是说，**你能训 ImageNet-1000**，往往代表：

-   你的机器能高效做“按需读取”
    
-   不是代表“把 1000 类所有图片像素全都 preload 到 RAM”
    

所以这个经验不能直接类比成：

> “WMT14 我也应该一股脑全读进内存。”

* * *

### 2\. 文本全量加载最大的敌人不是“数据本体”，而是 **Python 对象开销**

如果只是看“磁盘文件大小”，WMT14 英德 BPE 文本不算天文数字。  
但一旦你把它变成：

-   Python `list[str]`
    
-   Python `list[list[int]]`
    
-   或者更糟糕的 `[(src_ids, tgt_ids), ...]`
    

内存会膨胀得非常厉害。

### 一个直观类比

同样是一堆内容：

-   存在 `.txt` 文件里，像“压缩好的书”
    
-   存成 Python 列表对象，像“每个单词都单独装盒子，还给每个盒子贴标签”
    

**真正占内存的往往是“盒子和标签”，不是书本内容本身。**

* * *

## 二、你当前这套方案为什么是对的

你现在已经验证通过了两件关键事情：

1.  `DataLoader -> Seq2SeqBatch` 正常
    
2.  `真实 batch -> Transformer forward` 正常
    

这说明你当前的设计已经满足：

-   正确性
    
-   可运行性
    
-   后续可扩展性
    

而且你现在用的是：

-   `IterableDataset`
    
-   逐行读取
    
-   collate 时再 padding / mask / shifted right
    

这正适合你当前阶段：

> **先把整条训练链打通，再做性能优化。**

* * *

## 三、如果你“硬要”一次性全量加载，真正会遇到什么问题

### 1\. RAM 占用可能远大于你直觉

你现在的训练集是：

-   `3927488` 对句子
    

即使每条样本平均长度不算长，只要你存成 Python 结构：

-   每个字符串对象有额外开销
    
-   每个 list 有额外开销
    
-   每个 int 也不是裸 4 字节数组，而是 Python 对象
    

所以“看上去几 GB 的文本”，进 Python 后可能膨胀到很不舒服的量级。

* * *

### 2\. Windows 下多 worker 会更容易踩坑

在 Windows 下，`DataLoader(num_workers>0)` 默认是 **spawn** 风格，不像 Linux 的 `fork` 那么“轻”。

这意味着如果你把一个超大的 Python 数据集对象 preload 到内存，再开多个 worker，很容易出现：

-   启动慢
    
-   内存重复占用
    
-   worker 初始化成本高
    
-   稳定性差
    

这也是为什么我前面让你优先用 **流式 `IterableDataset`**，这是一个很实际的工程选择。

* * *

### 3\. 你现在还没进入真正需要“极致吞吐优化”的阶段

你当前更重要的是：

-   损失函数
    
-   label smoothing
    
-   Noam lr
    
-   valid 评估
    
-   checkpoint 保存
    
-   greedy/beam decode
    

这些东西对“能不能复现 Transformer”更关键。

如果你现在过早把精力投入到：

-   全量预加载
    
-   内存映射
    
-   复杂缓存系统
    

很容易在“训练主线”还没走通前，把工程复杂度拉高。

* * *

## 四、我给你的明确建议：当前阶段怎么选

## 推荐方案：继续用你现在的流式方案

### 原因

-   你已经验证通了；
    
-   内存更稳；
    
-   Windows 下更安全；
    
-   后续更容易逐步升级。
    

这就是我当前最推荐的路线。

* * *

## 不推荐方案：现在就改成全量 preload 的 Python 列表

### 原因

-   容易把 RAM 吃爆；
    
-   Windows 多 worker 容易复制对象；
    
-   工程复杂度增加；
    
-   对当前阶段收益不大。
    

* * *

## 可选优化方案：做“轻量缓存”，但不是全量 Python preload

如果你后面觉得流式读取太慢，可以做 **中间形态优化**，而不是一步跳到“全量全读”。

比如：

### 方案 A：只对 `valid/test` 全量加载

因为：

-   valid 只有 `3000`
    
-   test 只有 `3003`
    

这两个很小，完全可以直接 preload 到内存，简化评估逻辑。

### 方案 B：训练集仍然流式，但加本地缓存

例如：

-   每个 epoch 首次读取文本并编码
    
-   存成更紧凑的二进制格式
    
-   下次直接读 token-id
    

### 方案 C：训练集做 shard

例如把训练集分成多个块：

-   `train_000.bin`
    
-   `train_001.bin`
    
-   ...
    

每次只打开一个 shard 流式读。

这类方案都比“全量 Python preload”更稳。

* * *

## 五、什么时候才值得考虑“一次性加载全部数据”

只有当你同时满足下面几个条件时，才值得认真考虑：

-    训练主链已经完全跑通
    
-    你确认瓶颈确实在文本 I/O，而不是模型计算
    
-    你机器 RAM 很大，而且你知道 Windows 多 worker 的行为
    
-    你准备把数据存成**紧凑数值格式**，而不是 Python 对象
    
-    你愿意为数据管线写额外的缓存/索引系统
    

也就是说：

> 不是“我机器强，所以应该 preload”
> 
> 而是“我已经确认 I/O 是瓶颈，所以值得为此做工程优化”

这是两个完全不同的决策逻辑。

* * *

## 六、如果以后真的要优化，我建议的正确方向是什么

### 第一优先级：先把训练代码跑起来，再测瓶颈

你应该先完成：

-   train loop
    
-   loss
    
-   lr scheduler
    
-   valid loop
    

然后观察：

-   GPU 利用率
    
-   DataLoader 是否成为瓶颈
    
-   每 step 时间分布
    

如果 GPU 常年吃不满，而 CPU/I/O 忙，那时再优化数据管线才有依据。

* * *

### 第二优先级：优先做“紧凑缓存”，不是“Python 全量对象”

如果以后真要升级，我建议你优先考虑：

-   把 BPE 文本预编码成 `uint32/int32` 序列
    
-   配一个 offsets 索引
    
-   用 `mmap` 或二进制 shard 读
    

而不是：

-   `all_data = [(src_ids, tgt_ids), ...]`
    

后者从工程角度讲很不优雅，也不适合大规模训练。

* * *

### 第三优先级：可以把 `valid/test` 先全量加载

这个很值得做，因为体量小：

-   `3000`
    
-   `3003`
    

这两个集合全量加载进内存很轻松，而且评估时更简单。

所以你后面可以这样分策略：

-   `train`：流式
    
-   `valid/test`：内存加载
    

这是一个很合理的折中方案。

* * *

## 七、结合你当前项目，我给出最实际的建议

### 现在就这么定

-   **训练集**：保持 `IterableDataset` 流式读取
    
-   **验证集/测试集**：后面评估阶段可以改成一次性加载
    
-   **不要现在改成全量 preload 训练集**
    

这是当前最合适的工程决策。

* * *

## 八、你现在已经完成了什么，下一步最该做什么

你现在已经完成：

-   raw 数据导出
    
-   Moses 预处理
    
-   train 清洗
    
-   joint BPE
    
-   shared vocab
    
-   BPE DataLoader
    
-   真实 batch 前向测试
    

这已经非常完整了。

### 所以下一步最该做的是：

> **训练闭环，而不是数据 preload 优化。**

也就是进入：

-   损失函数
    
-   label smoothing
    
-   训练一个 epoch
    
-   Noam lr
    
-   valid 评估逻辑
    

* * *

## 九、最终结论

**你的设备强，不代表现在应该一次性加载全部训练数据。**

在你当前这个项目阶段：

-   从“正确性”
    
-   从“Windows 稳定性”
    
-   从“工程收益比”
    
-   从“论文复现主线优先级”
    

四个角度看，最佳选择都是：

> **训练集继续流式读取。**
> 
> **valid/test 可以后面改成内存加载。**
> 
> **先把训练和验证逻辑搭起来。**

下一条我建议直接进入：

## 第六步：损失函数、label smoothing、训练 step 和 Noam 学习率调度

我会按你现在这套 `SharedVocab + Seq2SeqBatch + make_model` 的工程结构，直接给你接上。

  

如果我期望继续采用流式读取，我现在的代码还需要改吗？如果需要改的话，你直接告诉我，我会按照要求去改。 如果不需要改的话，我们就继续讨论后面的损失函数等等的内容

  

## 结论

**不需要做结构性大改。**

你现在这套“流式读取”方案，从**正确性**上已经可以继续往后走了。  
也就是说：

-   你**不需要**为了继续采用流式读取，而把整个数据系统推翻重写；
    
-   你**可以继续**进入后面的损失函数、label smoothing、训练循环、Noam 学习率等内容。
    

但是，如果你问的是：

> “我现在这套流式读取代码，拿去做**正式训练**之前，要不要做一点修正？”

我的答案是：

> **要，建议改 1 个关键点，外加 2 个小注意事项。**

* * *

# 一、必须改的 1 个关键点

## 给训练集加入“流式 shuffle buffer”

你当前的 `IterableDataset` 方案有一个典型问题：

-   `DataLoader` 对 `IterableDataset` 不能直接 `shuffle=True`
    
-   所以如果你不自己做 shuffle
    
-   那么每个 epoch 训练集顺序几乎固定
    

对于机器翻译训练来说，这不是致命 bug，但它会带来两个问题：

1.  每个 epoch 数据顺序几乎一样，随机性不够；
    
2.  训练稳定性和泛化通常不如有 shuffle 的版本。
    

所以：

> **如果你继续采用流式读取，最值得改的一处，就是给 train dataset 加一个 buffer shuffle。**

这不是为了“代码好看”，而是为了**训练行为更合理**。

* * *

# 二、不需要改的数据主干

下面这些你现在**不用动**：

-   `SharedVocab`
    
-   `Seq2SeqBatch`
    
-   `src_mask`
    
-   `tgt_mask`
    
-   `target shifted right`
    
-   `BPE 文本 -> id -> padding`
    
-   `真实 batch 喂给 Transformer`
    

这些链路你已经验证通过了，说明主干是对的。

* * *

# 三、两个很重要但不是“改代码主干”的注意事项

## 注意事项 1：测试脚本里的截断长度不能直接拿去正式训练

你前面测试时用过：

-   `max_src_len=64`
    
-   `max_tgt_len=64`
    
-   或 `128`
    

这些值**只是 smoke test 用的**，不能直接拿去正式训练。

正式训练时，建议先这样：

-   `max_src_len=None`
    
-   `max_tgt_len=None`
    

也就是：

> **先不要静默截断。**

因为你现在已经做过训练集清洗了，再加上 BPE 之后如果还用很小的截断值，很容易无声地砍掉大量句子尾部信息。

* * *

## 注意事项 2：Windows 下先保持 `num_workers=0`

你当前环境是 Windows。  
在 Windows 下，`DataLoader` 的多进程行为比 Linux 更容易出各种额外问题。

所以我建议：

-   **现在先固定 `num_workers=0`**
    
-   等训练主链彻底打通，再考虑提到 `2` 或 `4`
    

当前阶段优先级是：

> **先稳定跑通训练闭环**
> 
> 不是先榨 DataLoader 性能。

* * *

# 四、你现在建议修改的唯一文件

我建议你**只替换一个文件**：

# `data/wmt14_bpe_dataset.py`

这次修改的目标只有两个：

1.  给 train 支持 **buffer shuffle**
    
2.  给 dataset 增加 `set_epoch()`，方便你后面每个 epoch 更换随机顺序
    

* * *

## 你直接把现在的 `data/wmt14_bpe_dataset.py` 替换成下面这个版本

```
from __future__ import annotations"""功能：1. 逐行读取 BPE 后的平行语料。2. 使用 SharedVocab 将 token 编码为 id。3. 在 collate_fn 中完成：   - source EOS 追加（可选）   - target BOS/EOS 追加   - padding   - target shifted right   - src_mask / tgt_mask 构造4. 返回 Seq2SeqBatch，供 Transformer 直接使用。本版本相较于前一版的关键增强：1. 支持流式 buffer shuffle（仅建议在 train 上启用）2. 支持 set_epoch(epoch)，便于每个 epoch 改变 shuffle 顺序3. 支持可选的 num_samples，用于日志统计和 __len__设计说明：1. 当前版本采用 IterableDataset，优先保证大规模训练集在 Windows 下能稳定跑通。2. 当前版本不做 bucket batching / token-based batching。   这些属于训练阶段优化，后面再专门升级。"""import randomfrom itertools import zip_longestfrom typing import Callable, Iterator, List, Optional, Sequence, Tupleimport torchfrom torch.utils.data import DataLoader, IterableDataset, get_worker_infofrom data.batch import Seq2SeqBatch, pad_sequencesfrom data.shared_vocab import SharedVocabclass ParallelBPEIterableDataset(IterableDataset):    """    逐行流式读取平行 BPE 文本。    每个样本输出：        (src_tokens, tgt_tokens)    其中：        src_tokens: List[str]        tgt_tokens: List[str]    入口参数：        src_path: BPE 英文/源语言文件路径        tgt_path: BPE 德文/目标语言文件路径        skip_empty: 是否跳过空样本        shuffle_buffer_size: 流式 shuffle 的缓冲区大小                             0 或 1 表示不做 shuffle        seed: 随机种子基值        num_samples: 可选，总样本数，用于 __len__    """    def __init__(        self,        src_path: str,        tgt_path: str,        skip_empty: bool = False,        shuffle_buffer_size: int = 0,        seed: int = 42,        num_samples: Optional[int] = None,    ) -> None:        super().__init__()        self.src_path = src_path        self.tgt_path = tgt_path        self.skip_empty = skip_empty        self.shuffle_buffer_size = shuffle_buffer_size        self.seed = seed        self.epoch = 0        self.num_samples = num_samples    def set_epoch(self, epoch: int) -> None:        """        为当前 dataset 设置 epoch 编号。        作用：            当启用 buffer shuffle 时，不同 epoch 可以产生不同的数据顺序。        """        self.epoch = epoch    def __len__(self) -> int:        """        如果提供了 num_samples，则返回样本数。        否则抛出 TypeError，提醒调用方这是流式数据集。        """        if self.num_samples is None:            raise TypeError("当前 IterableDataset 未提供 num_samples，无法可靠返回长度。")        return self.num_samples    def _line_iterator(self) -> Iterator[Tuple[List[str], List[str]]]:        """        最底层逐行读取器，不做 shuffle，只负责：        1. 打开双语文件        2. 保证逐行严格对齐        3. 在多 worker 下按行号切片，避免重复读取        """        worker_info = get_worker_info()        worker_id = 0 if worker_info is None else worker_info.id        num_workers = 1 if worker_info is None else worker_info.num_workers        with open(self.src_path, "r", encoding="utf-8") as f_src, \             open(self.tgt_path, "r", encoding="utf-8") as f_tgt:            for line_idx, pair in enumerate(zip_longest(f_src, f_tgt, fillvalue=None)):                src_line, tgt_line = pair                if src_line is None or tgt_line is None:                    raise RuntimeError(                        "检测到源文件和目标文件在迭代时长度不一致。"                    )                # 多 worker 下按行切片                if (line_idx % num_workers) != worker_id:                    continue                src_tokens = src_line.strip().split()                tgt_tokens = tgt_line.strip().split()                if self.skip_empty and (len(src_tokens) == 0 or len(tgt_tokens) == 0):                    continue                yield src_tokens, tgt_tokens    def _buffer_shuffle_iterator(self) -> Iterator[Tuple[List[str], List[str]]]:        """        基于固定大小缓冲区的流式 shuffle。        核心思想：            1. 先把样本不断塞进 buffer            2. 当 buffer 满后，每来一个新样本，就随机替换并吐出一个旧样本            3. 文件结束后，再把 buffer 剩余内容随机打乱后依次吐出        这样无法做到“全局完全随机打乱”，        但对大规模流式训练已经是非常实用的近似 shuffle。        """        worker_info = get_worker_info()        worker_id = 0 if worker_info is None else worker_info.id        rng = random.Random(self.seed + self.epoch * 100003 + worker_id)        buffer: List[Tuple[List[str], List[str]]] = []        for sample in self._line_iterator():            if len(buffer) < self.shuffle_buffer_size:                buffer.append(sample)            else:                idx = rng.randrange(len(buffer))                yield buffer[idx]                buffer[idx] = sample        rng.shuffle(buffer)        for sample in buffer:            yield sample    def __iter__(self) -> Iterator[Tuple[List[str], List[str]]]:        """        对外统一迭代入口。        """        if self.shuffle_buffer_size is None or self.shuffle_buffer_size <= 1:            yield from self._line_iterator()        else:            yield from self._buffer_shuffle_iterator()def build_bpe_collate_fn(    vocab: SharedVocab,    max_src_len: Optional[int] = None,    max_tgt_len: Optional[int] = None,    add_src_eos: bool = True,) -> Callable[[Sequence[Tuple[List[str], List[str]]]], Seq2SeqBatch]:    """    构造 DataLoader 所需的 collate_fn。    输入：        vocab: 共享词表        max_src_len: 源序列最大长度（截断前，不含额外追加的 EOS）        max_tgt_len: 目标序列最大长度（截断前，不含 BOS/EOS）        add_src_eos: 是否给源序列末尾追加 EOS    输出：        collate_fn，返回 Seq2SeqBatch    """    def collate_fn(batch: Sequence[Tuple[List[str], List[str]]]) -> Seq2SeqBatch:        src_id_list: List[List[int]] = []        tgt_id_list: List[List[int]] = []        for src_tokens, tgt_tokens in batch:            if max_src_len is not None:                src_tokens = src_tokens[:max_src_len]            if max_tgt_len is not None:                tgt_tokens = tgt_tokens[:max_tgt_len]            src_ids = vocab.encode(src_tokens)            tgt_ids = vocab.encode(tgt_tokens)            if add_src_eos:                src_ids = src_ids + [vocab.eos_id]            # 目标序列必须显式带 BOS / EOS，后续才能做 shifted right            tgt_ids = [vocab.bos_id] + tgt_ids + [vocab.eos_id]            src_id_list.append(src_ids)            tgt_id_list.append(tgt_ids)        src_tensor = pad_sequences(src_id_list, pad_idx=vocab.pad_id)        tgt_tensor = pad_sequences(tgt_id_list, pad_idx=vocab.pad_id)        batch_obj = Seq2SeqBatch.from_tensors(            src=src_tensor,            tgt=tgt_tensor,            pad_idx=vocab.pad_id,        )        return batch_obj    return collate_fndef build_bpe_dataloader(    src_path: str,    tgt_path: str,    vocab: SharedVocab,    batch_size: int,    num_workers: int = 0,    pin_memory: bool = True,    max_src_len: Optional[int] = None,    max_tgt_len: Optional[int] = None,    add_src_eos: bool = True,    skip_empty: bool = False,    shuffle_buffer_size: int = 0,    seed: int = 42,    num_samples: Optional[int] = None,) -> DataLoader:    """    构建平行 BPE 文本的 DataLoader。    入口：        src_path / tgt_path: BPE 文本文件路径        vocab: 共享词表        batch_size: batch 大小        num_workers: DataLoader worker 数量        pin_memory: 是否启用 pin_memory        max_src_len: 源序列截断长度        max_tgt_len: 目标序列截断长度        add_src_eos: 是否在源序列末尾追加 EOS        skip_empty: 是否跳过空样本        shuffle_buffer_size: 流式 shuffle 的缓冲区大小        seed: 随机种子        num_samples: 样本数，用于 __len__ 和日志统计    出口：        DataLoader，每次迭代返回一个 Seq2SeqBatch    注意：        1. 对 train，建议设置 shuffle_buffer_size 为一个正数，例如 10000 或 20000。        2. 对 valid/test，建议设置 shuffle_buffer_size=0。        3. Windows 下建议先固定 num_workers=0，等训练主链稳定后再调优。    """    dataset = ParallelBPEIterableDataset(        src_path=src_path,        tgt_path=tgt_path,        skip_empty=skip_empty,        shuffle_buffer_size=shuffle_buffer_size,        seed=seed,        num_samples=num_samples,    )    collate_fn = build_bpe_collate_fn(        vocab=vocab,        max_src_len=max_src_len,        max_tgt_len=max_tgt_len,        add_src_eos=add_src_eos,    )    loader = DataLoader(        dataset=dataset,        batch_size=batch_size,        num_workers=num_workers,        pin_memory=pin_memory,        collate_fn=collate_fn,    )    return loader
```

* * *

# 五、你现在训练阶段应该怎么用

## 训练集 DataLoader 推荐这样配

```
train_loader = build_bpe_dataloader(    src_path="data/wmt14_bpe_en_de/train.en",    tgt_path="data/wmt14_bpe_en_de/train.de",    vocab=vocab,    batch_size=64,                 # 先随便举例，后面会结合显存再定    num_workers=0,                 # Windows 先固定 0    pin_memory=True,    max_src_len=None,              # 正式训练先不要静默截断    max_tgt_len=None,              # 正式训练先不要静默截断    add_src_eos=True,    skip_empty=False,    shuffle_buffer_size=10000,     # 训练集启用 buffer shuffle    seed=42,    num_samples=3927488,)
```

* * *

## 验证集 DataLoader 推荐这样配

```
valid_loader = build_bpe_dataloader(    src_path="data/wmt14_bpe_en_de/valid.en",    tgt_path="data/wmt14_bpe_en_de/valid.de",    vocab=vocab,    batch_size=64,    num_workers=0,    pin_memory=True,    max_src_len=None,    max_tgt_len=None,    add_src_eos=True,    skip_empty=False,    shuffle_buffer_size=0,         # valid/test 不 shuffle    seed=42,    num_samples=3000,)
```

* * *

# 六、训练循环里还要多做一件事

如果你想让每个 epoch 的 train 顺序不同，那么在每个 epoch 开始前，要做：

```
if hasattr(train_loader.dataset, "set_epoch"):    train_loader.dataset.set_epoch(epoch)
```

这个很重要。

否则即使你有 buffer shuffle，  
如果随机种子一直不变，那么每个 epoch 的顺序仍然会重复。

* * *

# 七、所以最终回答你的问题

## 如果继续采用流式读取，我现在的代码还需要改吗？

### 精确回答：

-   **不需要大改**
    
-   **建议改 1 个关键点**
    
-   **其余主干不用动**
    

### 这个关键点就是：

> **把 `data/wmt14_bpe_dataset.py` 替换成我上面这个支持 buffer shuffle 的版本。**

除此以外：

-   `SharedVocab` 不用改
    
-   `Seq2SeqBatch` 不用改
    
-   `Transformer` 主干不用因为“流式读取”再改结构
    
-   你可以继续进入后面的损失函数和训练逻辑
    

* * *

# 八、我的建议

你现在就做两件事：

-    替换 `data/wmt14_bpe_dataset.py`
    
-    保持 `num_workers=0`，训练集开启 `shuffle_buffer_size`
    

然后我们就继续：

# 第六步：损失函数、label smoothing、训练 step、Noam 学习率调度

这一步我会直接按你当前这套：

-   `SharedVocab`
    
-   `Seq2SeqBatch`
    
-   `build_bpe_dataloader`
    
-   `make_model`
    

来接，不再绕回数据系统。

  

我期望可以正常的开启num\_workers和shuffle，总不能训练的时候num\_workers=0去训练吧，那也太慢了

  

## 可以开 `num_workers`，也可以有 `shuffle`，但要分清楚是哪一种 `shuffle`

你这句话是对的：

> 训练时长期 `num_workers=0` 不合适。

我前面让你先固定 `0`，本质上只是为了**先把整条训练链打通**。  
现在你已经把：

-   原始数据
    
-   Moses
    
-   clean
    
-   BPE
    
-   shared vocab
    
-   DataLoader
    
-   真实 batch 前向
    

全部走通了。

所以现在进入正式训练准备阶段，**完全可以讨论 `num_workers>0` 和随机打乱**。

* * *

# 一、先给你一个直接结论

## 1\. `num_workers`

**可以开。**

而且你当前这套 `IterableDataset` 代码，本身已经写了多 worker 下的分片逻辑：

```
if (line_idx % num_workers) != worker_id:    continue
```

这意味着：

-   每个 worker 读同一对文件
    
-   但只消费属于自己那一部分行
    
-   不会重复读取同一条样本
    

所以从**正确性**上说，你当前数据集已经支持 `num_workers>0`。

* * *

## 2\. `shuffle`

这里要区分两种含义。

### 含义 A：`DataLoader(..., shuffle=True)`

**这个对 `IterableDataset` 不成立。**

这是 PyTorch 的接口层规定，不是你代码写得好不好。  
`shuffle=True` 是给 **map-style dataset** 用的，也就是那种支持：

-   `__len__`
    
-   `__getitem__(idx)`
    

的数据集。

而你现在是：

-   `IterableDataset`
    
-   逐行流式读取
    

所以：

> **你不能指望 `DataLoader` 帮你做全局随机打乱。**

* * *

### 含义 B：训练数据在 epoch 内具有足够随机性

**这个是可以做到的。**

做法就是：

> **保留 `IterableDataset`，自己在数据流里实现 buffer shuffle。**

这就是我前面让你改的方向。

所以更准确地说：

-   **你可以正常开 `num_workers`**
    
-   **你也可以正常有 shuffle 效果**
    
-   只是这个 shuffle 不是 `DataLoader(shuffle=True)` 那种接口形式
    
-   而是 **流式近似 shuffle**
    

* * *

# 二、你真正想要的，其实不是 `shuffle=True` 这个参数，而是“训练时顺序足够随机”

这个点非常重要。

很多人会把：

-   “我想让数据打乱”
    
-   和
    
-   “我必须用 `DataLoader(shuffle=True)`”
    

混为一谈。

其实你真正需要的是前者，不是后者。

* * *

## 类比理解

### `DataLoader(shuffle=True)` 是什么

它像是：

-   你先把全班同学名单编号
    
-   然后每次训练前，把编号整体洗牌
    
-   再按编号顺序取人
    

这要求你能：

-   精确随机访问第 `i` 个样本
    

所以它天然适合 `__getitem__` 型数据集。

* * *

### `buffer shuffle` 是什么

它像是：

-   你有一条不断流动的人群
    
-   你手里有一个容量为 `K` 的等候区
    
-   人进来以后先放进等候区
    
-   然后每次从等候区随机放一个出来
    

这不是“全局完全随机排列”，但对大规模流式训练已经非常实用。

* * *

# 三、所以现在你的正确工程选择是什么

## 方案 A：继续用 `IterableDataset`，开启 `num_workers` + `buffer shuffle`

这是我现在最推荐你的方案。

### 优点

-   保持你现在整条链不推翻
    
-   支持大训练集
    
-   支持多 worker
    
-   支持足够强的近似随机化
    
-   Windows 下也能工作
    

### 缺点

-   不是严格意义上的“全局完全 shuffle”
    
-   不能直接写 `shuffle=True`
    

* * *

## 方案 B：改成 `map-style dataset`，支持真正的 `shuffle=True`

如果你特别执着于：

-   必须 `DataLoader(..., shuffle=True)`
    
-   必须每个 epoch 做更接近全局的严格随机采样
    

那就要改成：

-   `__len__`
    
-   `__getitem__(idx)`
    

这种 indexable 数据集。

但这会带来新的工程工作：

-   预先建立行偏移索引（offsets）
    
-   支持按 idx 快速 seek 到对应行
    
-   双语文件要同步索引
    
-   Windows 下还要考虑多 worker 打开文件句柄
    

这不是不能做，但它明显比你当前阶段更复杂。

* * *

# 四、我给你的明确建议

## 现在不要切到 map-style dataset

因为你当前主线目标是：

> 把 Transformer 训练闭环做出来。

不是：

> 先把数据系统做到最“优雅”的随机访问框架。

所以现阶段最优解是：

> **保留 `IterableDataset`，但把它升级成正式训练可用版本：**
> 
> -   `num_workers > 0`
>     
> -   `shuffle_buffer_size > 0`
>     
> -   `persistent_workers=True`
>     
> -   每个 epoch 调 `set_epoch(epoch)`
>     

这已经足够进入训练。

* * *

# 五、你现在的代码还需要怎么改

## 结论：需要小改，但不是重写

你不需要推翻数据系统。  
你只需要把 `build_bpe_dataloader()` 再增强一下，让它支持：

-   `persistent_workers`
    
-   `prefetch_factor`
    
-   训练/验证模式区分
    
-   `num_workers>0` 时更自然地配置
    

* * *

# 六、我建议你把 `data/wmt14_bpe_dataset.py` 再替换成下面这个版本

这个版本是在我前面给你的 buffer shuffle 版本上，进一步把正式训练需要的 DataLoader 参数补齐。

* * *

```
from __future__ import annotations"""功能：1. 逐行读取 BPE 后的平行语料。2. 使用 SharedVocab 将 token 编码为 id。3. 在 collate_fn 中完成：   - source EOS 追加（可选）   - target BOS/EOS 追加   - padding   - target shifted right   - src_mask / tgt_mask 构造4. 返回 Seq2SeqBatch，供 Transformer 直接使用。本版本面向正式训练准备，支持：1. IterableDataset + 多 worker 分片2. 流式 buffer shuffle3. set_epoch(epoch)4. DataLoader 的 persistent_workers / prefetch_factor"""import randomfrom itertools import zip_longestfrom typing import Callable, Iterator, List, Optional, Sequence, Tupleimport torchfrom torch.utils.data import DataLoader, IterableDataset, get_worker_infofrom data.batch import Seq2SeqBatch, pad_sequencesfrom data.shared_vocab import SharedVocabclass ParallelBPEIterableDataset(IterableDataset):    """    逐行流式读取平行 BPE 文本。    每个样本输出：        (src_tokens, tgt_tokens)    参数：        src_path: BPE 源语言文件路径        tgt_path: BPE 目标语言文件路径        skip_empty: 是否跳过空样本        shuffle_buffer_size: 流式 shuffle 缓冲区大小        seed: 随机种子基值        num_samples: 可选，总样本数，用于 __len__    """    def __init__(        self,        src_path: str,        tgt_path: str,        skip_empty: bool = False,        shuffle_buffer_size: int = 0,        seed: int = 42,        num_samples: Optional[int] = None,    ) -> None:        super().__init__()        self.src_path = src_path        self.tgt_path = tgt_path        self.skip_empty = skip_empty        self.shuffle_buffer_size = shuffle_buffer_size        self.seed = seed        self.epoch = 0        self.num_samples = num_samples    def set_epoch(self, epoch: int) -> None:        """        设置当前 epoch 编号。        作用：            配合 buffer shuffle，让每个 epoch 的样本顺序不同。        """        self.epoch = epoch    def __len__(self) -> int:        if self.num_samples is None:            raise TypeError("当前 IterableDataset 未提供 num_samples，无法可靠返回长度。")        return self.num_samples    def _line_iterator(self) -> Iterator[Tuple[List[str], List[str]]]:        """        最底层逐行读取器，不做 shuffle，只负责：        1. 打开双语文件        2. 保证逐行严格对齐        3. 在多 worker 下按行号切片        """        worker_info = get_worker_info()        worker_id = 0 if worker_info is None else worker_info.id        num_workers = 1 if worker_info is None else worker_info.num_workers        with open(self.src_path, "r", encoding="utf-8") as f_src, \             open(self.tgt_path, "r", encoding="utf-8") as f_tgt:            for line_idx, pair in enumerate(zip_longest(f_src, f_tgt, fillvalue=None)):                src_line, tgt_line = pair                if src_line is None or tgt_line is None:                    raise RuntimeError("检测到源文件和目标文件在迭代时长度不一致。")                # 多 worker 下按行切分                if (line_idx % num_workers) != worker_id:                    continue                src_tokens = src_line.strip().split()                tgt_tokens = tgt_line.strip().split()                if self.skip_empty and (len(src_tokens) == 0 or len(tgt_tokens) == 0):                    continue                yield src_tokens, tgt_tokens    def _buffer_shuffle_iterator(self) -> Iterator[Tuple[List[str], List[str]]]:        """        基于固定大小缓冲区的流式 shuffle。        注意：            这不是严格全局打乱，而是近似 shuffle。            对大规模流式训练是一个非常实用的折中方案。        """        worker_info = get_worker_info()        worker_id = 0 if worker_info is None else worker_info.id        rng = random.Random(self.seed + self.epoch * 100003 + worker_id)        buffer: List[Tuple[List[str], List[str]]] = []        for sample in self._line_iterator():            if len(buffer) < self.shuffle_buffer_size:                buffer.append(sample)            else:                idx = rng.randrange(len(buffer))                yield buffer[idx]                buffer[idx] = sample        rng.shuffle(buffer)        for sample in buffer:            yield sample    def __iter__(self) -> Iterator[Tuple[List[str], List[str]]]:        if self.shuffle_buffer_size is None or self.shuffle_buffer_size <= 1:            yield from self._line_iterator()        else:            yield from self._buffer_shuffle_iterator()def build_bpe_collate_fn(    vocab: SharedVocab,    max_src_len: Optional[int] = None,    max_tgt_len: Optional[int] = None,    add_src_eos: bool = True,) -> Callable[[Sequence[Tuple[List[str], List[str]]]], Seq2SeqBatch]:    """    构造 DataLoader 的 collate_fn。    """    def collate_fn(batch: Sequence[Tuple[List[str], List[str]]]) -> Seq2SeqBatch:        src_id_list: List[List[int]] = []        tgt_id_list: List[List[int]] = []        for src_tokens, tgt_tokens in batch:            if max_src_len is not None:                src_tokens = src_tokens[:max_src_len]            if max_tgt_len is not None:                tgt_tokens = tgt_tokens[:max_tgt_len]            src_ids = vocab.encode(src_tokens)            tgt_ids = vocab.encode(tgt_tokens)            if add_src_eos:                src_ids = src_ids + [vocab.eos_id]            tgt_ids = [vocab.bos_id] + tgt_ids + [vocab.eos_id]            src_id_list.append(src_ids)            tgt_id_list.append(tgt_ids)        src_tensor = pad_sequences(src_id_list, pad_idx=vocab.pad_id)        tgt_tensor = pad_sequences(tgt_id_list, pad_idx=vocab.pad_id)        return Seq2SeqBatch.from_tensors(            src=src_tensor,            tgt=tgt_tensor,            pad_idx=vocab.pad_id,        )    return collate_fndef build_bpe_dataloader(    src_path: str,    tgt_path: str,    vocab: SharedVocab,    batch_size: int,    num_workers: int = 0,    pin_memory: bool = True,    max_src_len: Optional[int] = None,    max_tgt_len: Optional[int] = None,    add_src_eos: bool = True,    skip_empty: bool = False,    shuffle_buffer_size: int = 0,    seed: int = 42,    num_samples: Optional[int] = None,    persistent_workers: bool = False,    prefetch_factor: Optional[int] = None,) -> DataLoader:    """    构建平行 BPE 文本 DataLoader。    训练推荐：        - train:            shuffle_buffer_size = 10000 ~ 50000            num_workers = 2 / 4 / 8（视机器而定）            persistent_workers = True        - valid/test:            shuffle_buffer_size = 0            num_workers = 0 或 2            persistent_workers = False    """    dataset = ParallelBPEIterableDataset(        src_path=src_path,        tgt_path=tgt_path,        skip_empty=skip_empty,        shuffle_buffer_size=shuffle_buffer_size,        seed=seed,        num_samples=num_samples,    )    collate_fn = build_bpe_collate_fn(        vocab=vocab,        max_src_len=max_src_len,        max_tgt_len=max_tgt_len,        add_src_eos=add_src_eos,    )    dataloader_kwargs = dict(        dataset=dataset,        batch_size=batch_size,        num_workers=num_workers,        pin_memory=pin_memory,        collate_fn=collate_fn,        persistent_workers=(persistent_workers and num_workers > 0),    )    if num_workers > 0 and prefetch_factor is not None:        dataloader_kwargs["prefetch_factor"] = prefetch_factor    loader = DataLoader(**dataloader_kwargs)    return loader
```

* * *

# 七、你训练时应该怎么配

## 训练集推荐第一版配置

先别一上来就 8 worker。  
建议从这个版本开始：

```
train_loader = build_bpe_dataloader(    src_path="data/wmt14_bpe_en_de/train.en",    tgt_path="data/wmt14_bpe_en_de/train.de",    vocab=vocab,    batch_size=64,    num_workers=2,    pin_memory=True,    max_src_len=None,    max_tgt_len=None,    add_src_eos=True,    skip_empty=False,    shuffle_buffer_size=10000,    seed=42,    num_samples=3927488,    persistent_workers=True,    prefetch_factor=2,)
```

### 为什么先从 `num_workers=2` 开始

因为你是 Windows：

-   `spawn` 成本更高
    
-   进程间初始化更慢
    
-   先用 2 看吞吐和稳定性
    
-   再逐步试 4
    
-   没必要直接冲很大
    

* * *

## 验证集推荐配置

```
valid_loader = build_bpe_dataloader(    src_path="data/wmt14_bpe_en_de/valid.en",    tgt_path="data/wmt14_bpe_en_de/valid.de",    vocab=vocab,    batch_size=64,    num_workers=0,    pin_memory=True,    max_src_len=None,    max_tgt_len=None,    add_src_eos=True,    skip_empty=False,    shuffle_buffer_size=0,    seed=42,    num_samples=3000,    persistent_workers=False,    prefetch_factor=None,)
```

### 为什么 valid 先不搞多 worker

因为 valid 很小：

-   3000 条
    
-   瓶颈一般不在这里
    
-   简单稳妥更重要
    

* * *

# 八、训练循环里必须加这一句

每个 epoch 开始前：

```
if hasattr(train_loader.dataset, "set_epoch"):    train_loader.dataset.set_epoch(epoch)
```

这一句不能忘。  
否则你即使设了 `shuffle_buffer_size`，每个 epoch 仍可能按同一个伪随机顺序出样本。

* * *

# 九、所以最后给你的明确答复

## 你现在的代码还需要改吗？

### 需要，但只需要做这件事：

> **把 `data/wmt14_bpe_dataset.py` 替换成我上面这个支持多 worker + buffer shuffle + persistent\_workers 的版本。**

### 不需要改的部分：

-   不需要推翻流式读取
    
-   不需要改成全量 preload
    
-   不需要改成 map-style dataset
    
-   不需要因为要开 `num_workers` 而去重写词表、batch、mask、model
    

* * *

# 十、下一步

你把这个文件替换好之后，我们就可以正式进入：

# 第六步：损失函数、label smoothing、训练 step、Noam 学习率调度

这一步我会直接按你现在已经稳定下来的这套：

-   `SharedVocab`
    
-   `Seq2SeqBatch`
    
-   `build_bpe_dataloader`
    
-   `make_model`
    

来继续接。

  

我已经替换好了。下面我们讨论损失函数和学习率调度等等内容，我提前和你说一下我偏好的训练日志记录风格： 在为我写深度学习训练代码时，请保持以下日志习惯（变量名可变，逻辑需一致）： 双轨记录：本地存配置（json）、追加核心指标（csv）和 matplotlib 趋势图（png）；动态监控全走 TensorBoard。 细粒度 Loss：必须将 Loss 拆解为具体成分（如坐标、类别、置信度），在 TensorBoard 中按 Iteration(Step) 级别记录，并在一张多线图中对比，拒绝只记录 Epoch 级的 Total Loss。 克制记录 Histogram：TensorBoard 直方图只记录最具代表性的层（首层看梯度消失、颈层看特征、尾层看爆炸）以及核心预测输出（如预测置信度分布），拒绝无脑全开。 直观可视化：验证阶段抽少量图像画上预测结果（如 Bounding Box），拼成 Grid 传到 TensorBoard 进行视觉 Debug。 直白保存权重：保留 best 和 last 模型，定期保存的权重名必须带上 Epoch 和核心评估指标（例：model\_epoch\_50\_valmap50\_0.45.pth）。

  

我先对齐两件事：一是把你这组训练日志偏好固定下来，二是用你上传的 Transformer 论文和当前工程代码把损失函数、学习率调度、训练循环的设计边界收紧。

## 结论

流式读取这一侧，**现在不用再做结构性修改了**。  
你已经把最关键的几条链全部打通了：

-   BPE 文本读取
    
-   shared vocab 编码
    
-   `target shifted right`
    
-   `src_mask / tgt_mask`
    
-   真实 batch 前向传播
    

这说明数据侧已经具备进入训练阶段的条件。  
原论文训练部分真正关键的，不再是“怎么读文件”，而是：

-   **batch 组织方式**
    
-   **label smoothing**
    
-   **Adam + Noam 学习率**
    
-   **step 级训练日志**
    
-   **验证与 checkpoint 策略**
    

所以我们现在可以正式把注意力切到**训练系统**上。

* * *

# 一、先把总体训练策略定下来

我建议你现在按下面这个思路实现第一版训练系统：

## 第一版目标

-   先做 **论文主干对齐版**
    
-   不急着做最复杂的 bucket/token-based batch sampler
    
-   先把：
    
    -   `loss`
        
    -   `label smoothing`
        
    -   `optimizer`
        
    -   `Noam scheduler`
        
    -   `train loop`
        
    -   `valid loop`
        
    -   `日志系统`
        
    -   `best/last 保存`
        

完整打通

## 第二版再优化

等第一版稳定跑起来后，再升级：

-   approximate-length batching
    
-   token budget batching
    
-   greedy decode / beam search
    
-   BLEU
    
-   checkpoint averaging
    

因为论文里训练 batching 是：

> sentence pairs were batched together by approximate sequence length，  
> each batch contained approximately 25000 source tokens and 25000 target tokens。

你现在还没做到这一步，但这不妨碍你先把训练主链跑通。

* * *

# 二、损失函数应该怎么设计

## 1\. 不要生造“检测式多成分 loss”

这是最先要说清楚的一点。

你平时做目标检测，loss 很自然会拆成：

-   box loss
    
-   cls loss
    
-   obj/conf loss
    

但 **Transformer 机器翻译不是这个范式**。  
这里真正自然、合理的损失分解不是“坐标/类别/置信度”，而是：

-   **total\_loss**
    
-   **nll\_loss**（硬标签负对数似然分量）
    
-   **smooth\_loss**（label smoothing 带来的平滑分量）
    
-   **token\_acc**
    
-   **ppl**
    

也就是说：

> 你的“细粒度 loss”偏好，在这个任务里要翻译成  
> **`总损失 + NLL 分量 + smoothing 分量 + token 级指标`**  
> 而不是强行拆成一些并不存在的训练项。

* * *

## 2\. 论文明确用了 label smoothing

原论文第 5.4 节明确写了：

-   使用 **label smoothing**
    
-   平滑系数是 $ \\epsilon\_{ls} = 0.1 $
    
-   它会让 **perplexity 变差**
    
-   但会提升 **accuracy 和 BLEU**。
    

所以你的第一版 loss，应该就是：

$Ltotal=Llabel_smoothing\mathcal{L}_{total} = \mathcal{L}_{label\_smoothing}Ltotal​=Llabel_smoothing​$

如果把它拆开看，可以写成两部分理解：

$Ltotal=(1−ϵ)⋅Lnll+ϵ⋅Lsmooth\mathcal{L}_{total} = (1-\epsilon)\cdot \mathcal{L}_{nll} + \epsilon \cdot \mathcal{L}_{smooth}Ltotal​=(1−ϵ)⋅Lnll​+ϵ⋅Lsmooth​$

其中：

-   $ \\mathcal{L}\_{nll} $：目标 token 的标准负对数似然
    
-   $ \\mathcal{L}\_{smooth} $：把一部分概率质量分给非目标类后的平滑项
    
-   $ \\epsilon = 0.1 $
    

注意这里这个分解更多是**日志解释层面的分解**，具体实现时通常会写成：

-   `KLDivLoss`
    
-   或者手工构造 smoothed target distribution
    

* * *

## 3\. 我建议你记录的训练指标

结合你的日志偏好，这个任务里我建议每个 **iteration(step)** 记录：

### 训练阶段 step 级 TensorBoard 标量

-   `train_step/total_loss`
    
-   `train_step/nll_loss`
    
-   `train_step/smooth_loss`
    
-   `train_step/token_acc`
    
-   `train_step/ppl`
    
-   `train_step/lr`
    
-   `train_step/grad_norm`
    
-   `train_step/ntokens`
    
-   `train_step/tokens_per_sec`
    

### epoch 级 CSV / PNG / JSON 指标

-   `epoch`
    
-   `global_step`
    
-   `train_loss`
    
-   `train_nll_loss`
    
-   `train_smooth_loss`
    
-   `train_token_acc`
    
-   `train_ppl`
    
-   `valid_loss`
    
-   `valid_token_acc`
    
-   `valid_ppl`
    
-   `lr_last`
    
-   `avg_grad_norm`
    
-   `tokens_per_sec`
    

这套分解和你的偏好是一致的，而且不会伪造不存在的 loss 成分。

* * *

# 三、学习率调度应该怎么设计

## 1\. 论文这里必须按 step 调度，不要改成 epoch 调度

这个点非常关键。

原论文第 5.3 节给出的学习率是：

$lrate=dmodel−0.5⋅min⁡(step_num−0.5,  step_num⋅warmup_steps−1.5)\mathrm{lrate} = d_{model}^{-0.5} \cdot \min\left( step\_num^{-0.5}, \; step\_num \cdot warmup\_steps^{-1.5} \right)lrate=dmodel−0.5​⋅min(step_num−0.5,step_num⋅warmup_steps−1.5)$

并且明确说：

-   优化器是 **Adam**
    
-   $ \\beta\_1 = 0.9 $
    
-   $ \\beta\_2 = 0.98 $
    
-   $ \\epsilon = 10^{-9} $
    
-   `warmup_steps = 4000`。
    

这说明：

> **Noam scheduler 天生就是 step-based scheduler。**

所以这次和你过去某些 CV 项目不一样：

-   这里**不建议**改成 epoch-based
    
-   否则就已经明显偏离论文训练协议了
    

* * *

## 2\. 你应该怎么理解 Noam schedule

它可以分成两个阶段：

### warmup 阶段

$lrate∝step_num⋅warmup−1.5\mathrm{lrate} \propto step\_num \cdot warmup^{-1.5}lrate∝step_num⋅warmup−1.5$

也就是：

-   前 4000 step 线性升温
    
-   让模型先稳定起来，不要一开始梯度太猛
    

### 衰减阶段

$lrate∝step_num−0.5\mathrm{lrate} \propto step\_num^{-0.5}lrate∝step_num−0.5$

也就是：

-   过了 warmup 以后按反平方根衰减
    
-   随 step 增大逐步降下来
    

### 再乘上模型维度缩放

$dmodel−0.5d_{model}^{-0.5}dmodel−0.5​$

它的作用是：

-   不同 `d_model` 下，学习率量级自动做一个尺度归一
    

* * *

## 3\. 我建议你日志里怎么记录学习率

因为这个调度是 step-based，所以你应该：

-   **每个 iteration 记录当前 lr**
    
-   并且后面画出一张 **lr-step 曲线图**
    

这张图对排查训练非常有用。  
Transformer 这类模型，如果学习率曲线不对，loss 往往会直接表现得很奇怪。

* * *

# 四、日志系统怎么按你的偏好落地

你给的偏好很清楚，我直接把它翻译成这次机器翻译项目的版本。

* * *

## 1\. 双轨记录：完全保留

这个我建议原封不动执行。

### 本地文件

-   `config.json`
    
-   `train_metrics.csv`
    
-   `curves_loss.png`
    
-   `curves_lr.png`
    
-   `curves_token_acc.png`
    
-   `curves_ppl.png`
    

### TensorBoard

-   step 级 scalars
    
-   少量 histogram
    
-   文本样例 / 注意力可视化
    

这个方向完全正确。

* * *

## 2\. 细粒度 loss：翻译成 NLP 版本

这次不记录“坐标/类别/置信度”，而记录：

-   `total_loss`
    
-   `nll_loss`
    
-   `smooth_loss`
    

并且全部在 **step 级** 写入 TensorBoard。

这个非常适合你。  
因为 Transformer 的训练最需要观察的就是：

-   总损失是否下降
    
-   NLL 分量和 smoothing 分量比例是否正常
    
-   label smoothing 是否让模型过于保守
    

* * *

## 3\. Histogram：要克制，但记录“有诊断价值”的层

这个我非常赞同，而且这次任务我建议只记录下面几类：

### 首层：看梯度和 embedding 分布

-   `src_embed.lut.weight`
    
-   `tgt_embed.lut.weight`（如果权重共享，其实就是同一份）
    

### 中层：看 FFN 或 attention 投影

例如：

-   `encoder.layers[0].self_attn.linears[0].weight`
    
-   `encoder.layers[mid].feed_forward.w_1.weight`
    

### 尾层：看 generator 输出头

-   `generator.proj.weight`
    
-   以及 **logits/max\_prob 分布**
    

这比“全层全开 histogram”有价值高得多，也不会把 TensorBoard 搞得又慢又乱。

* * *

## 4\. 视觉 debug：在 NLP 里应该换成“文本样例 + 注意力图”

你之前偏好里提到：

-   验证阶段抽图像
    
-   画预测
    
-   拼 Grid 上 TensorBoard
    

这在检测任务里很自然。  
但在翻译任务里，等价物应该是：

### 必做：文本样例

每个验证 epoch 记录少量样本：

-   source
    
-   reference
    
-   greedy prediction
    

写到 TensorBoard text 面板。

### 可选：注意力热力图

从一个固定样本里抽：

-   encoder-decoder attention
    
-   或 decoder self-attention
    

画成 heatmap，再写入 TensorBoard image。

这才是这个任务里真正有诊断价值的“直观可视化”。

* * *

## 5\. 权重保存：完全按你的风格来

这个我建议直接这样定：

### 始终保留

-   `last.pth`
    
-   `best.pth`
    

### 定期保存

命名格式类似：

```
model_epoch_05_valppl_4.92.pthmodel_epoch_10_valppl_4.61.pth
```

等你后面接了 BLEU，再进一步变成：

```
model_epoch_12_valbleu_25.84.pth
```

### 额外建议

因为原论文 base 模型在推理时用了：

-   **最后 5 个 checkpoint 平均**，而且这些 checkpoint 是 **每 10 分钟保存一次**。
    

所以你后面可以额外留一个机制：

-   定时 / 定步保存最近若干 checkpoint
    
-   供后续做 checkpoint averaging
    

但这一步不是现在立刻必须实现的第一优先级。

* * *

# 五、训练系统我建议拆成哪些模块

为了配合你的日志偏好，我建议这次训练模块拆成下面这些：

```
utils/├── label_smoothing.py├── noam_scheduler.py├── tb_log.py├── csv_logger.py├── plot_metrics.pytrain/├── train_one_epoch.py├── validate_one_epoch.py└── fit.py
```

* * *

## 1\. `label_smoothing.py`

负责：

-   构造 smoothed target distribution
    
-   返回：
    
    -   `total_loss`
        
    -   `nll_loss`
        
    -   `smooth_loss`
        

这里就是你“细粒度 loss 记录”的核心入口。

* * *

## 2\. `noam_scheduler.py`

负责：

-   按论文公式做 step-based lr 更新
    
-   每个 optimizer step 后更新一次
    

这个模块应该是独立的，不要把 lr 逻辑揉进 train loop 里。

* * *

## 3\. `tb_log.py`

负责 TensorBoard：

-   step scalars
    
-   少量 histograms
    
-   文本样例
    
-   注意力 heatmap（后续）
    

* * *

## 4\. `csv_logger.py`

负责：

-   每个 epoch 或固定 step 追加写 CSV
    
-   字段固定，方便你后面用 matplotlib 画图
    

* * *

## 5\. `plot_metrics.py`

负责从 CSV 画：

-   loss 曲线
    
-   lr 曲线
    
-   token\_acc 曲线
    
-   ppl 曲线
    

保存成本地 png。

* * *

## 6\. `train_one_epoch.py`

负责：

-   一个 epoch 内训练
    
-   forward
    
-   loss
    
-   backward
    
-   grad clip
    
-   optimizer step
    
-   scheduler step
    
-   step 级日志
    

* * *

## 7\. `validate_one_epoch.py`

负责：

-   验证集 forward
    
-   loss / token\_acc / ppl
    
-   记录少量 source/ref/pred 样例
    

注意：  
**第一版验证先不跑 beam search，不算 BLEU。**

先用：

-   valid loss
    
-   valid ppl
    
-   token\_acc
    

把训练链跑通。  
BLEU 和 beam search 我建议下一阶段再接。

* * *

## 8\. `fit.py`

负责总控：

-   epoch loop
    
-   `set_epoch(epoch)`
    
-   train / valid
    
-   保存 best / last / periodic
    
-   写 config.json
    
-   更新 CSV
    
-   画 PNG
    

* * *

# 六、我建议的第一版训练指标

## 训练时监控哪些指标最关键

### train step 级

-   `total_loss`
    
-   `nll_loss`
    
-   `smooth_loss`
    
-   `token_acc`
    
-   `ppl`
    
-   `lr`
    
-   `grad_norm`
    
-   `tokens_per_sec`
    

### valid epoch 级

-   `valid_loss`
    
-   `valid_token_acc`
    
-   `valid_ppl`
    

### 后面第二版再加

-   `valid_bleu`
    
-   `beam_size`
    
-   `length_penalty`
    
-   checkpoint averaging metrics
    

因为原论文最终解码结果依赖：

-   beam size = 4
    
-   length penalty $ \\alpha = 0.6 $
    
-   max output length = input length + 50
    
-   checkpoint averaging。
    

这些会影响最终 BLEU，但不影响你现在先把训练主链跑通。

* * *

# 七、还有一个必须说的点：batching 目前还不是论文口径

你现在的数据加载器还是：

-   固定 `batch_size`
    
-   逐条流式拼 batch
    

而论文训练 batching 是：

-   approximate sequence length
    
-   约 25k source tokens + 25k target tokens / batch。
    

所以我建议：

## 第一版

先用固定 `batch_size` 跑通训练。

## 第二版

再升级成：

-   bucket by length
    
-   token budget batching
    

这个顺序最稳。  
否则你现在一口气把数据采样器也做复杂，很容易把训练主线拖住。

* * *

# 八、现在最值得先实现的训练版本

我建议第一版训练配置就这样：

-   **criterion**：Label Smoothing, $ \\epsilon = 0.1 $
    
-   **optimizer**：Adam
    
    -   $ \\beta\_1 = 0.9 $
        
    -   $ \\beta\_2 = 0.98 $
        
    -   $ \\epsilon = 10^{-9} $
        
-   **scheduler**：Noam，`warmup_steps=4000`
    
-   **dropout**：0.1
    
-   **train monitor**：step 级记录 loss/lr/acc/ppl
    
-   **valid metric**：loss + ppl + token\_acc
    
-   **checkpoint**：best + last + periodic
    

这套就是最接近论文主干、同时又便于你工程调试的一版。

* * *

# 九、最终回答

你的日志偏好非常适合这次项目，但要做一个任务适配：

-   在检测任务里，“细粒度 loss”是 box/cls/conf
    
-   在翻译任务里，“细粒度 loss”应该是 total/nll/smooth
    

除此之外，你给的训练日志要求基本都可以原样保留，而且会让这套 Transformer 训练系统很清楚、很可诊断。

* * *

# 十、下一步怎么走

现在最合理的顺序是：

-    先实现 `label_smoothing.py`
    
-    再实现 `noam_scheduler.py`
    
-    然后实现 `train_one_epoch.py`
    
-    再实现 `validate_one_epoch.py`
    
-    最后接 `fit.py + tb_log + csv_logger + plot_metrics`
    

我建议下一条直接进入：

## `label_smoothing.py + noam_scheduler.py`

先把这两个最核心的“数学模块”定下来。

  

好，进一步实现吧。只需要名字和具体的代码即可，输出到你一次对话的极限，然后我会让你继续输出的

  

## `utils/label_smoothing.py`

```
from __future__ import annotations"""功能：1. 实现适用于 Transformer 机器翻译的 Label Smoothing 损失。2. 返回可回传梯度的总损失，以及用于日志记录的细粒度分量：   - total_loss   - nll_loss   - smooth_loss3. 提供 token-level accuracy 和 perplexity 的辅助函数。说明：1. 本实现面向共享词表场景，要求显式提供 pad_idx。2. smoothing 的平滑质量会分配到“除目标类和 pad 类之外”的其他类别。3. 返回的 loss/nll_loss/smooth_loss 都是“按有效 token 取平均”的标量。"""from dataclasses import dataclassfrom typing import Dictimport mathimport torchimport torch.nn as nnimport torch.nn.functional as F@dataclassclass LabelSmoothingLossOutput:    """    统一封装一次前向计算得到的损失结果。    """    loss: torch.Tensor    nll_loss: torch.Tensor    smooth_loss: torch.Tensor    num_tokens: int    def as_dict(self) -> Dict[str, float]:        """        将结果转换为便于日志记录的普通字典。        """        return {            "loss": float(self.loss.detach().item()),            "nll_loss": float(self.nll_loss.detach().item()),            "smooth_loss": float(self.smooth_loss.detach().item()),            "num_tokens": int(self.num_tokens),        }class LabelSmoothingLoss(nn.Module):    """    Transformer 机器翻译任务的 Label Smoothing 损失。    入口：        logits: 形状 (B, T, V)        target: 形状 (B, T)    出口：        LabelSmoothingLossOutput            - loss:        总损失（可回传梯度）            - nll_loss:    硬标签 NLL 分量（日志观察）            - smooth_loss: 平滑分量（日志观察）            - num_tokens:  有效 token 数（排除 pad）    参数说明：        vocab_size: 词表大小        pad_idx: PAD 的词表 id        smoothing: label smoothing 系数，论文默认 0.1    """    def __init__(        self,        vocab_size: int,        pad_idx: int,        smoothing: float = 0.1,    ) -> None:        super().__init__()        if vocab_size <= 2:            raise ValueError("vocab_size 必须大于 2。")        if not (0.0 <= smoothing < 1.0):            raise ValueError("smoothing 必须满足 0 <= smoothing < 1。")        self.vocab_size = vocab_size        self.pad_idx = pad_idx        self.smoothing = smoothing        self.confidence = 1.0 - smoothing    def forward(        self,        logits: torch.Tensor,        target: torch.Tensor,    ) -> LabelSmoothingLossOutput:        """        前向计算。        输入：            logits: (B, T, V)            target: (B, T)        输出：            LabelSmoothingLossOutput        """        if logits.dim() != 3:            raise ValueError(f"logits 必须是 3 维张量，当前维度为 {logits.dim()}。")        if target.dim() != 2:            raise ValueError(f"target 必须是 2 维张量，当前维度为 {target.dim()}。")        bsz, tgt_len, vocab_size = logits.shape        if vocab_size != self.vocab_size:            raise ValueError(                f"logits 最后一维 vocab_size={vocab_size} 与初始化时的 {self.vocab_size} 不一致。"            )        if target.shape[0] != bsz or target.shape[1] != tgt_len:            raise ValueError(                f"target 形状 {tuple(target.shape)} 与 logits 前两维 {(bsz, tgt_len)} 不匹配。"            )        logits_flat = logits.reshape(-1, self.vocab_size)   # (N, V)        target_flat = target.reshape(-1)                    # (N,)        valid_mask = target_flat.ne(self.pad_idx)        if valid_mask.sum().item() == 0:            raise ValueError("当前 batch 中有效 token 数为 0，无法计算损失。")        logits_valid = logits_flat[valid_mask]              # (N_valid, V)        target_valid = target_flat[valid_mask]              # (N_valid,)        log_probs = F.log_softmax(logits_valid, dim=-1)     # (N_valid, V)        # -------------------------        # 1. 硬标签 NLL 分量        # -------------------------        nll_per_token = -log_probs.gather(dim=1, index=target_valid.unsqueeze(1)).squeeze(1)        # -------------------------        # 2. 平滑分量        #    平滑概率质量分配到：        #    - 除 pad 类外        #    - 除真实目标类外        # -------------------------        if self.smoothing > 0.0:            if self.vocab_size <= 2:                raise ValueError("当 smoothing > 0 时，vocab_size 必须大于 2。")            smooth_target = torch.full_like(                log_probs,                fill_value=1.0 / (self.vocab_size - 2),            )            smooth_target[:, self.pad_idx] = 0.0            smooth_target.scatter_(1, target_valid.unsqueeze(1), 0.0)            # 保险归一化，避免数值误差            smooth_target = smooth_target / smooth_target.sum(dim=1, keepdim=True)            smooth_per_token = -(smooth_target * log_probs).sum(dim=-1)            total_per_token = self.confidence * nll_per_token + self.smoothing * smooth_per_token        else:            smooth_per_token = torch.zeros_like(nll_per_token)            total_per_token = nll_per_token        num_tokens = int(target_valid.numel())        loss = total_per_token.mean()        nll_loss = nll_per_token.mean()        smooth_loss = smooth_per_token.mean()        return LabelSmoothingLossOutput(            loss=loss,            nll_loss=nll_loss.detach(),            smooth_loss=smooth_loss.detach(),            num_tokens=num_tokens,        )@torch.no_grad()def compute_token_accuracy(    logits: torch.Tensor,    target: torch.Tensor,    pad_idx: int,) -> Dict[str, float]:    """    计算 token-level accuracy。    输入：        logits: (B, T, V)        target: (B, T)        pad_idx: PAD 的词表 id    输出：        dict:            - correct_tokens            - total_tokens            - token_acc    """    if logits.dim() != 3:        raise ValueError(f"logits 必须是 3 维张量，当前维度为 {logits.dim()}。")    if target.dim() != 2:        raise ValueError(f"target 必须是 2 维张量，当前维度为 {target.dim()}。")    pred = logits.argmax(dim=-1)                       # (B, T)    valid_mask = target.ne(pad_idx)                    # (B, T)    total_tokens = int(valid_mask.sum().item())    if total_tokens == 0:        return {            "correct_tokens": 0,            "total_tokens": 0,            "token_acc": 0.0,        }    correct_tokens = int(((pred == target) & valid_mask).sum().item())    token_acc = correct_tokens / total_tokens    return {        "correct_tokens": correct_tokens,        "total_tokens": total_tokens,        "token_acc": float(token_acc),    }def compute_perplexity_from_loss(loss_value: float) -> float:    """    根据平均 loss 计算 perplexity。    说明：        1. 通常建议对 nll_loss 计算 ppl，更直观。        2. 当 loss 非常大时，exp 可能上溢，因此这里做一个简单裁剪。    """    safe_loss = min(float(loss_value), 50.0)    return float(math.exp(safe_loss))
```

* * *

## `utils/noam_scheduler.py`

```
from __future__ import annotations"""功能：1. 实现论文中的 Noam 学习率调度器（step-based）。2. 提供 Adam + Noam 的标准构建入口。3. 支持 state_dict / load_state_dict，便于断点续训。注意：1. Noam 调度本质上是按 optimizer update step 更新，而不是按 epoch 更新。2. 推荐调用顺序：   - backward   - grad clip   - scheduler.step()   # 先把本次 update 的 lr 设好   - optimizer.step()"""from dataclasses import dataclassfrom typing import Dict, Tupleimport torchfrom torch import nn@dataclassclass NoamConfig:    """    Noam 调度器配置。    """    d_model: int    warmup_steps: int = 4000    factor: float = 1.0    beta1: float = 0.9    beta2: float = 0.98    eps: float = 1e-9    weight_decay: float = 0.0class NoamLRScheduler:    """    论文风格 Noam 学习率调度器。    数学形式：        lrate = factor * (d_model ^ -0.5) *                min(step_num ^ -0.5, step_num * warmup_steps ^ -1.5)    入口：        optimizer: PyTorch Optimizer        d_model: 模型维度        warmup_steps: 预热步数        factor: 缩放因子        init_step: 初始 step（断点恢复时使用）    出口：        - step(): 更新内部 step，并把当前 lr 写回 optimizer        - rate(step): 查询指定 step 的 lr        - get_last_lr(): 获取最近一次 lr        - state_dict/load_state_dict(): 状态保存与恢复    """    def __init__(        self,        optimizer: torch.optim.Optimizer,        d_model: int,        warmup_steps: int = 4000,        factor: float = 1.0,        init_step: int = 0,    ) -> None:        if d_model <= 0:            raise ValueError("d_model 必须大于 0。")        if warmup_steps <= 0:            raise ValueError("warmup_steps 必须大于 0。")        if factor <= 0:            raise ValueError("factor 必须大于 0。")        self.optimizer = optimizer        self.d_model = d_model        self.warmup_steps = warmup_steps        self.factor = factor        self._step = int(init_step)        self._rate = 0.0        # 初始化时先把 param group 的 lr 置 0        for group in self.optimizer.param_groups:            group["lr"] = 0.0    @property    def step_num(self) -> int:        return self._step    def rate(self, step: int | None = None) -> float:        """        计算某个 step 对应的学习率。        """        if step is None:            step = self._step        step = max(int(step), 1)        return self.factor * (self.d_model ** -0.5) * min(            step ** -0.5,            step * (self.warmup_steps ** -1.5),        )    def step(self) -> float:        """        推进一步，并将当前 lr 写入 optimizer。        返回：            当前 step 对应的 lr        """        self._step += 1        lr = self.rate(self._step)        for group in self.optimizer.param_groups:            group["lr"] = lr        self._rate = lr        return lr    def get_last_lr(self) -> list:        """        返回最近一次设置的 lr，接口风格与 PyTorch scheduler 对齐。        """        return [group["lr"] for group in self.optimizer.param_groups]    def state_dict(self) -> Dict:        """        保存调度器状态。        """        return {            "d_model": self.d_model,            "warmup_steps": self.warmup_steps,            "factor": self.factor,            "_step": self._step,            "_rate": self._rate,        }    def load_state_dict(self, state_dict: Dict) -> None:        """        恢复调度器状态。        """        self.d_model = int(state_dict["d_model"])        self.warmup_steps = int(state_dict["warmup_steps"])        self.factor = float(state_dict["factor"])        self._step = int(state_dict["_step"])        self._rate = float(state_dict["_rate"])        # 恢复后同步写回 optimizer        for group in self.optimizer.param_groups:            group["lr"] = self._ratedef build_transformer_optimizer(    model: nn.Module,    beta1: float = 0.9,    beta2: float = 0.98,    eps: float = 1e-9,    weight_decay: float = 0.0,) -> torch.optim.Adam:    """    构建论文风格 Adam 优化器。    说明：        1. 这里不做复杂的 param group 拆分。        2. 第一版优先复现论文主干，使用统一 Adam 即可。    """    optimizer = torch.optim.Adam(        params=model.parameters(),        lr=0.0,  # 实际 lr 由 NoamLRScheduler 在每次 update 前写入        betas=(beta1, beta2),        eps=eps,        weight_decay=weight_decay,    )    return optimizerdef build_transformer_optimizer_and_scheduler(    model: nn.Module,    d_model: int,    warmup_steps: int = 4000,    factor: float = 1.0,    beta1: float = 0.9,    beta2: float = 0.98,    eps: float = 1e-9,    weight_decay: float = 0.0,) -> Tuple[torch.optim.Adam, NoamLRScheduler]:    """    一次性构建 Adam + Noam 调度器。    """    optimizer = build_transformer_optimizer(        model=model,        beta1=beta1,        beta2=beta2,        eps=eps,        weight_decay=weight_decay,    )    scheduler = NoamLRScheduler(        optimizer=optimizer,        d_model=d_model,        warmup_steps=warmup_steps,        factor=factor,    )    return optimizer, scheduler
```

* * *

## `utils/csv_logger.py`

```
from __future__ import annotations"""功能：1. 以追加方式写入训练/验证核心指标到 CSV。2. 自动处理表头写入。3. 保持字段稳定，便于后续画图和分析。"""import csvimport osfrom typing import Dict, Iterable, List, Optionalclass CSVMetricLogger:    """    简单可靠的 CSV 指标记录器。    入口：        csv_path: 输出 csv 路径        fieldnames: 列名列表，建议训练前固定好    出口：        append_row(row_dict): 追加一行    """    def __init__(        self,        csv_path: str,        fieldnames: List[str],    ) -> None:        self.csv_path = csv_path        self.fieldnames = fieldnames        os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)        if not os.path.exists(self.csv_path):            with open(self.csv_path, "w", encoding="utf-8", newline="") as f:                writer = csv.DictWriter(f, fieldnames=self.fieldnames)                writer.writeheader()    def append_row(self, row: Dict) -> None:        """        追加一行数据。        """        safe_row = {}        for key in self.fieldnames:            safe_row[key] = row.get(key, "")        with open(self.csv_path, "a", encoding="utf-8", newline="") as f:            writer = csv.DictWriter(f, fieldnames=self.fieldnames)            writer.writerow(safe_row)
```

* * *

## `utils/plot_metrics.py`

```
from __future__ import annotations"""功能：1. 从 CSV 中读取训练日志。2. 生成本地 matplotlib 曲线图（png）。3. 作为 TensorBoard 之外的离线趋势观察工具。说明：1. 每张图只画一个坐标系，不做 subplot。2. 不手工指定颜色，保持默认 matplotlib 配置。"""import csvimport osfrom typing import Dict, Listimport matplotlib.pyplot as pltdef _read_csv_as_dict_list(csv_path: str) -> List[Dict[str, str]]:    rows: List[Dict[str, str]] = []    with open(csv_path, "r", encoding="utf-8") as f:        reader = csv.DictReader(f)        for row in reader:            rows.append(row)    return rowsdef _to_float_list(rows: List[Dict[str, str]], key: str) -> List[float]:    values: List[float] = []    for row in rows:        value = row.get(key, "")        if value == "" or value is None:            values.append(float("nan"))        else:            values.append(float(value))    return valuesdef plot_single_curve(    csv_path: str,    x_key: str,    y_keys: List[str],    out_png_path: str,    title: str,    xlabel: str,    ylabel: str,) -> None:    """    从 CSV 中读取多个 y 序列，画到同一张图上。    """    rows = _read_csv_as_dict_list(csv_path)    if len(rows) == 0:        return    x_values = _to_float_list(rows, x_key)    plt.figure(figsize=(10, 6))    for y_key in y_keys:        y_values = _to_float_list(rows, y_key)        plt.plot(x_values, y_values, label=y_key)    plt.title(title)    plt.xlabel(xlabel)    plt.ylabel(ylabel)    plt.grid(True)    if len(y_keys) > 1:        plt.legend()    os.makedirs(os.path.dirname(out_png_path), exist_ok=True)    plt.tight_layout()    plt.savefig(out_png_path, dpi=150)    plt.close()def plot_default_transformer_curves(    csv_path: str,    out_dir: str,) -> None:    """    为 Transformer 训练日志画一组默认曲线图。    """    os.makedirs(out_dir, exist_ok=True)    plot_single_curve(        csv_path=csv_path,        x_key="epoch",        y_keys=["train_loss", "valid_loss"],        out_png_path=os.path.join(out_dir, "curve_loss.png"),        title="Loss vs Epoch",        xlabel="Epoch",        ylabel="Loss",    )    plot_single_curve(        csv_path=csv_path,        x_key="epoch",        y_keys=["train_nll_loss", "train_smooth_loss"],        out_png_path=os.path.join(out_dir, "curve_train_loss_components.png"),        title="Train Loss Components vs Epoch",        xlabel="Epoch",        ylabel="Loss",    )    plot_single_curve(        csv_path=csv_path,        x_key="epoch",        y_keys=["train_token_acc", "valid_token_acc"],        out_png_path=os.path.join(out_dir, "curve_token_acc.png"),        title="Token Accuracy vs Epoch",        xlabel="Epoch",        ylabel="Accuracy",    )    plot_single_curve(        csv_path=csv_path,        x_key="epoch",        y_keys=["train_ppl", "valid_ppl"],        out_png_path=os.path.join(out_dir, "curve_ppl.png"),        title="Perplexity vs Epoch",        xlabel="Epoch",        ylabel="PPL",    )    plot_single_curve(        csv_path=csv_path,        x_key="epoch",        y_keys=["lr_last"],        out_png_path=os.path.join(out_dir, "curve_lr.png"),        title="Learning Rate vs Epoch",        xlabel="Epoch",        ylabel="LR",    )
```

* * *

## `utils/tb_log.py`

```
from __future__ import annotations"""功能：1. TensorBoard 日志记录。2. 记录 step 级训练标量。3. 记录 epoch 级训练/验证指标。4. 记录少量代表性 histogram。5. 记录验证文本样例。注意：1. 直方图记录要克制，只记录少量最有诊断价值的层。2. 文本样例建议只记录少量固定样本，避免 TensorBoard 过重。"""import osfrom typing import Dict, List, Optionalimport torchfrom torch.utils.tensorboard import SummaryWriterclass TransformerTBLogger:    """    Transformer 训练 TensorBoard 日志器。    """    def __init__(self, log_dir: str) -> None:        os.makedirs(log_dir, exist_ok=True)        self.writer = SummaryWriter(log_dir=log_dir)    def log_train_step(        self,        global_step: int,        loss_dict: Dict[str, float],        token_acc: float,        ppl: float,        lr: float,        grad_norm: float,        ntokens: int,        tokens_per_sec: float,    ) -> None:        """        记录训练 step 级标量。        """        self.writer.add_scalar("train_step/total_loss", loss_dict["loss"], global_step)        self.writer.add_scalar("train_step/nll_loss", loss_dict["nll_loss"], global_step)        self.writer.add_scalar("train_step/smooth_loss", loss_dict["smooth_loss"], global_step)        self.writer.add_scalar("train_step/token_acc", token_acc, global_step)        self.writer.add_scalar("train_step/ppl", ppl, global_step)        self.writer.add_scalar("train_step/lr", lr, global_step)        self.writer.add_scalar("train_step/grad_norm", grad_norm, global_step)        self.writer.add_scalar("train_step/ntokens", ntokens, global_step)        self.writer.add_scalar("train_step/tokens_per_sec", tokens_per_sec, global_step)    def log_train_epoch(self, epoch: int, stats: Dict[str, float]) -> None:        """        记录训练 epoch 级标量。        """        for key, value in stats.items():            self.writer.add_scalar(f"train_epoch/{key}", value, epoch)    def log_valid_epoch(self, epoch: int, stats: Dict[str, float]) -> None:        """        记录验证 epoch 级标量。        """        for key, value in stats.items():            self.writer.add_scalar(f"valid_epoch/{key}", value, epoch)    def log_representative_histograms(        self,        model: torch.nn.Module,        global_step: int,        prediction_confidence: Optional[torch.Tensor] = None,    ) -> None:        """        记录少量代表性 histogram。        默认策略：            1. embedding / 首层            2. 编码器中层 FFN            3. 输出头            4. 预测最大置信度分布（可选）        """        try:            if hasattr(model, "src_embed") and hasattr(model.src_embed, "lut"):                self.writer.add_histogram(                    "hist/src_embed_lut_weight",                    model.src_embed.lut.weight.detach().cpu(),                    global_step,                )        except Exception:            pass        try:            if hasattr(model, "encoder") and hasattr(model.encoder, "layers"):                layers = model.encoder.layers                if len(layers) > 0:                    self.writer.add_histogram(                        "hist/encoder_layer0_q_proj_weight",                        layers[0].self_attn.linears[0].weight.detach().cpu(),                        global_step,                    )                mid_idx = len(layers) // 2                self.writer.add_histogram(                    "hist/encoder_mid_ffn_w1_weight",                    layers[mid_idx].feed_forward.w_1.weight.detach().cpu(),                    global_step,                )        except Exception:            pass        try:            if hasattr(model, "generator") and hasattr(model.generator, "proj"):                self.writer.add_histogram(                    "hist/generator_proj_weight",                    model.generator.proj.weight.detach().cpu(),                    global_step,                )        except Exception:            pass        if prediction_confidence is not None:            try:                self.writer.add_histogram(                    "hist/prediction_confidence",                    prediction_confidence.detach().float().cpu(),                    global_step,                )            except Exception:                pass    def log_text_samples(        self,        epoch: int,        samples: List[Dict[str, str]],        tag: str = "valid_samples",    ) -> None:        """        记录少量验证文本样例。        每个 sample 约定字段：            - source            - reference            - prediction        """        text_blocks: List[str] = []        for idx, sample in enumerate(samples):            block = (                f"### Sample {idx}\n\n"                f"- Source: {sample.get('source', '')}\n"                f"- Reference: {sample.get('reference', '')}\n"                f"- Prediction: {sample.get('prediction', '')}\n"            )            text_blocks.append(block)        final_text = "\n\n---\n\n".join(text_blocks)        self.writer.add_text(tag, final_text, epoch)    def flush(self) -> None:        self.writer.flush()    def close(self) -> None:        self.writer.flush()        self.writer.close()
```

* * *

## `utils/checkpoint_manager.py`

```
from __future__ import annotations"""功能：1. 管理 Transformer 训练过程中的 checkpoint 保存。2. 固定保留：   - best.pth   - last.pth3. 周期性保存：   - model_epoch_{epoch}_valppl_{metric}.pth4. 支持恢复：   - model   - optimizer   - scheduler   - scaler"""import osimport shutilfrom typing import Dict, Optionalimport torchclass CheckpointManager:    """    checkpoint 管理器。    当前默认监控指标：        valid_ppl（越小越好）    """    def __init__(self, save_dir: str, monitor_key: str = "valid_ppl", mode: str = "min") -> None:        self.save_dir = save_dir        self.monitor_key = monitor_key        self.mode = mode        if self.mode not in {"min", "max"}:            raise ValueError("mode 只能是 'min' 或 'max'。")        os.makedirs(self.save_dir, exist_ok=True)        self.best_metric: Optional[float] = None    def _is_better(self, metric: float) -> bool:        if self.best_metric is None:            return True        if self.mode == "min":            return metric < self.best_metric        return metric > self.best_metric    def _build_state(        self,        epoch: int,        global_step: int,        model: torch.nn.Module,        optimizer: Optional[torch.optim.Optimizer],        scheduler,        scaler,        train_stats: Dict,        valid_stats: Dict,        config: Dict,    ) -> Dict:        state = {            "epoch": epoch,            "global_step": global_step,            "model_state_dict": model.state_dict(),            "train_stats": train_stats,            "valid_stats": valid_stats,            "config": config,        }        if optimizer is not None:            state["optimizer_state_dict"] = optimizer.state_dict()        if scheduler is not None and hasattr(scheduler, "state_dict"):            state["scheduler_state_dict"] = scheduler.state_dict()        if scaler is not None:            state["scaler_state_dict"] = scaler.state_dict()        return state    def save_last(        self,        epoch: int,        global_step: int,        model: torch.nn.Module,        optimizer,        scheduler,        scaler,        train_stats: Dict,        valid_stats: Dict,        config: Dict,    ) -> str:        """        保存 last.pth        """        state = self._build_state(            epoch=epoch,            global_step=global_step,            model=model,            optimizer=optimizer,            scheduler=scheduler,            scaler=scaler,            train_stats=train_stats,            valid_stats=valid_stats,            config=config,        )        out_path = os.path.join(self.save_dir, "last.pth")        torch.save(state, out_path)        return out_path    def save_best_if_needed(        self,        epoch: int,        global_step: int,        model: torch.nn.Module,        optimizer,        scheduler,        scaler,        train_stats: Dict,        valid_stats: Dict,        config: Dict,    ) -> Optional[str]:        """        若当前验证指标优于历史 best，则保存 best.pth        """        if self.monitor_key not in valid_stats:            raise KeyError(f"valid_stats 中缺少监控字段 {self.monitor_key}。")        metric = float(valid_stats[self.monitor_key])        if not self._is_better(metric):            return None        self.best_metric = metric        state = self._build_state(            epoch=epoch,            global_step=global_step,            model=model,            optimizer=optimizer,            scheduler=scheduler,            scaler=scaler,            train_stats=train_stats,            valid_stats=valid_stats,            config=config,        )        out_path = os.path.join(self.save_dir, "best.pth")        torch.save(state, out_path)        return out_path    def save_periodic(        self,        epoch: int,        global_step: int,        model: torch.nn.Module,        optimizer,        scheduler,        scaler,        train_stats: Dict,        valid_stats: Dict,        config: Dict,    ) -> str:        """        按用户偏好保存周期性权重文件，文件名携带 epoch 和核心评估指标。        """        metric = float(valid_stats.get(self.monitor_key, float("nan")))        filename = f"model_epoch_{epoch:03d}_{self.monitor_key}_{metric:.4f}.pth"        out_path = os.path.join(self.save_dir, filename)        state = self._build_state(            epoch=epoch,            global_step=global_step,            model=model,            optimizer=optimizer,            scheduler=scheduler,            scaler=scaler,            train_stats=train_stats,            valid_stats=valid_stats,            config=config,        )        torch.save(state, out_path)        return out_path
```

* * *

## `train/train_one_epoch.py`

```
from __future__ import annotations"""功能：1. 执行一个 epoch 的训练。2. 支持：   - Label Smoothing 损失   - step-based Noam 学习率   - grad clip   - step 级 TensorBoard 标量   - 少量代表性 histogram3. 返回 epoch 级聚合统计。注意：1. scheduler.step() 必须在 optimizer.step() 之前调用，   这样本次参数更新使用的就是当前 step 对应的 Noam lr。2. 当前版本默认支持 AMP，但不强制启用。"""import mathimport timefrom typing import Dict, Optional, Tupleimport torchfrom torch.nn.utils import clip_grad_norm_from utils.label_smoothing import (    LabelSmoothingLoss,    compute_perplexity_from_loss,    compute_token_accuracy,)def train_one_epoch(    model: torch.nn.Module,    train_loader,    criterion: LabelSmoothingLoss,    optimizer: torch.optim.Optimizer,    scheduler,    device: torch.device,    epoch: int,    global_step: int,    tb_logger=None,    scaler: Optional[torch.cuda.amp.GradScaler] = None,    use_amp: bool = False,    grad_clip_norm: Optional[float] = 1.0,    log_interval: int = 100,    histogram_interval: int = 1000,    max_steps_per_epoch: Optional[int] = None,) -> Tuple[Dict[str, float], int]:    """    训练一个 epoch。    入口：        model: Transformer 模型        train_loader: 训练 DataLoader        criterion: LabelSmoothingLoss        optimizer: 优化器        scheduler: NoamLRScheduler        device: 训练设备        epoch: 当前 epoch 编号（从 1 开始）        global_step: 全局 update step        tb_logger: TensorBoard 日志器，可为 None        scaler: AMP GradScaler，可为 None        use_amp: 是否启用 autocast        grad_clip_norm: 梯度裁剪阈值        log_interval: 每隔多少 step 记录一次 TensorBoard 标量        histogram_interval: 每隔多少 step 记录一次 histogram        max_steps_per_epoch: 调试时限制每个 epoch 的最大 step 数    出口：        stats: epoch 级聚合统计        global_step: 更新后的全局 step    """    model.train()    total_loss_sum = 0.0    total_nll_sum = 0.0    total_smooth_sum = 0.0    total_correct_tokens = 0    total_tokens = 0    total_grad_norm = 0.0    total_tokens_per_sec = 0.0    total_steps = 0    epoch_start_time = time.time()    for step_idx, batch in enumerate(train_loader, start=1):        if max_steps_per_epoch is not None and step_idx > max_steps_per_epoch:            break        iter_start_time = time.time()        batch = batch.to(device)        optimizer.zero_grad(set_to_none=True)        with torch.cuda.amp.autocast(enabled=(use_amp and device.type == "cuda")):            hidden_states = model(                batch.src,                batch.tgt_input,                batch.src_mask,                batch.tgt_mask,            )            logits = model.generator(hidden_states)            loss_output = criterion(logits, batch.tgt_y)            acc_output = compute_token_accuracy(logits, batch.tgt_y, criterion.pad_idx)        # -------------------------        # backward + grad clip        # -------------------------        if scaler is not None and use_amp and device.type == "cuda":            scaler.scale(loss_output.loss).backward()            scaler.unscale_(optimizer)            if grad_clip_norm is not None and grad_clip_norm > 0:                grad_norm = clip_grad_norm_(model.parameters(), grad_clip_norm)            else:                grad_norm = torch.tensor(0.0, device=device)            current_lr = scheduler.step()            scaler.step(optimizer)            scaler.update()        else:            loss_output.loss.backward()            if grad_clip_norm is not None and grad_clip_norm > 0:                grad_norm = clip_grad_norm_(model.parameters(), grad_clip_norm)            else:                grad_norm = torch.tensor(0.0, device=device)            current_lr = scheduler.step()            optimizer.step()        global_step += 1        total_steps += 1        batch_tokens = loss_output.num_tokens        iter_time = max(time.time() - iter_start_time, 1e-8)        tokens_per_sec = batch_tokens / iter_time        total_loss_sum += float(loss_output.loss.detach().item()) * batch_tokens        total_nll_sum += float(loss_output.nll_loss.detach().item()) * batch_tokens        total_smooth_sum += float(loss_output.smooth_loss.detach().item()) * batch_tokens        total_correct_tokens += int(acc_output["correct_tokens"])        total_tokens += int(acc_output["total_tokens"])        total_grad_norm += float(grad_norm.detach().item() if torch.is_tensor(grad_norm) else grad_norm)        total_tokens_per_sec += tokens_per_sec        # -------------------------        # step 级 TensorBoard 记录        # -------------------------        if tb_logger is not None and (global_step % log_interval == 0):            train_loss_dict = loss_output.as_dict()            train_ppl = compute_perplexity_from_loss(train_loss_dict["nll_loss"])            tb_logger.log_train_step(                global_step=global_step,                loss_dict=train_loss_dict,                token_acc=float(acc_output["token_acc"]),                ppl=train_ppl,                lr=float(current_lr),                grad_norm=float(grad_norm.detach().item() if torch.is_tensor(grad_norm) else grad_norm),                ntokens=int(batch_tokens),                tokens_per_sec=float(tokens_per_sec),            )        # -------------------------        # histogram 记录（克制）        # -------------------------        if tb_logger is not None and (global_step % histogram_interval == 0):            with torch.no_grad():                prediction_confidence = torch.softmax(logits.detach(), dim=-1).amax(dim=-1)            tb_logger.log_representative_histograms(                model=model,                global_step=global_step,                prediction_confidence=prediction_confidence,            )    # -------------------------    # epoch 聚合    # -------------------------    if total_tokens == 0:        raise ValueError("当前 epoch 没有有效 token，无法汇总训练指标。")    avg_loss = total_loss_sum / total_tokens    avg_nll_loss = total_nll_sum / total_tokens    avg_smooth_loss = total_smooth_sum / total_tokens    avg_token_acc = total_correct_tokens / total_tokens    avg_ppl = compute_perplexity_from_loss(avg_nll_loss)    avg_grad_norm = total_grad_norm / max(total_steps, 1)    avg_tokens_per_sec = total_tokens_per_sec / max(total_steps, 1)    epoch_time_sec = time.time() - epoch_start_time    stats = {        "epoch": float(epoch),        "train_loss": float(avg_loss),        "train_nll_loss": float(avg_nll_loss),        "train_smooth_loss": float(avg_smooth_loss),        "train_token_acc": float(avg_token_acc),        "train_ppl": float(avg_ppl),        "avg_grad_norm": float(avg_grad_norm),        "avg_tokens_per_sec": float(avg_tokens_per_sec),        "epoch_time_sec": float(epoch_time_sec),        "lr_last": float(scheduler.get_last_lr()[0]),        "global_step": float(global_step),    }    if tb_logger is not None:        tb_logger.log_train_epoch(epoch=epoch, stats=stats)    return stats, global_step
```

* * *

## `train/validate_one_epoch.py`

```
from __future__ import annotations"""功能：1. 执行一个 epoch 的验证。2. 计算：   - valid_loss   - valid_nll_loss   - valid_smooth_loss   - valid_token_acc   - valid_ppl3. 可选记录少量文本样例（source/reference/prediction）。说明：1. 第一版验证以 loss / ppl / token_acc 为主。2. 文本样例默认使用 greedy decode。3. 这里不引入 beam search，后面再单独加。"""from typing import Dict, List, Optionalimport torchfrom data.batch import make_tgt_maskfrom utils.label_smoothing import (    LabelSmoothingLoss,    compute_perplexity_from_loss,    compute_token_accuracy,)def _ids_to_bpe_string(    ids: List[int],    id_to_token_func,    pad_id: int,    bos_id: int,    eos_id: int,) -> str:    """    将 id 序列恢复成便于人读的 BPE 文本。    """    tokens: List[str] = []    for idx in ids:        if idx == pad_id:            continue        if idx == bos_id:            continue        if idx == eos_id:            break        tokens.append(id_to_token_func(idx))    # subword-nmt 的 @@ 拼接还原    text = " ".join(tokens).replace("@@ ", "")    return text.strip()@torch.no_grad()def greedy_decode(    model: torch.nn.Module,    src: torch.Tensor,    src_mask: torch.Tensor,    bos_id: int,    eos_id: int,    pad_id: int,    max_len: int,) -> torch.Tensor:    """    Greedy 解码。    输入：        src: (B, S)        src_mask: (B, 1, S)        bos_id / eos_id / pad_id: 特殊 token id        max_len: 生成最大长度    输出：        ys: (B, T_pred)，包含 BOS 开头，后续逐步生成    """    memory = model.encode(src, src_mask)    batch_size = src.size(0)    ys = torch.full(        (batch_size, 1),        fill_value=bos_id,        dtype=torch.long,        device=src.device,    )    finished = torch.zeros(batch_size, dtype=torch.bool, device=src.device)    for _ in range(max_len - 1):        tgt_mask = make_tgt_mask(ys, pad_idx=pad_id)        out = model.decode(memory, src_mask, ys, tgt_mask)        logits = model.generator(out[:, -1, :])  # (B, V)        next_token = logits.argmax(dim=-1, keepdim=True)  # (B, 1)        ys = torch.cat([ys, next_token], dim=1)        finished = finished | next_token.squeeze(1).eq(eos_id)        if finished.all():            break    return ys@torch.no_grad()def validate_one_epoch(    model: torch.nn.Module,    valid_loader,    criterion: LabelSmoothingLoss,    device: torch.device,    epoch: int,    tb_logger=None,    vocab=None,    num_text_samples: int = 3,    max_decode_extra_len: int = 50,    max_steps_per_epoch: Optional[int] = None,) -> Dict[str, float]:    """    验证一个 epoch。    入口：        model: Transformer 模型        valid_loader: 验证集 DataLoader        criterion: LabelSmoothingLoss        device: 设备        epoch: 当前 epoch        tb_logger: TensorBoard 日志器        vocab: SharedVocab，可选；若提供则可记录文本样例        num_text_samples: 记录多少个文本样例        max_decode_extra_len: greedy decode 时的最大额外长度        max_steps_per_epoch: 调试时限制验证 step 数    出口：        stats: 验证 epoch 聚合指标    """    model.eval()    total_loss_sum = 0.0    total_nll_sum = 0.0    total_smooth_sum = 0.0    total_correct_tokens = 0    total_tokens = 0    sample_batch = None    for step_idx, batch in enumerate(valid_loader, start=1):        if max_steps_per_epoch is not None and step_idx > max_steps_per_epoch:            break        batch = batch.to(device)        if sample_batch is None and vocab is not None and num_text_samples > 0:            sample_batch = batch        hidden_states = model(            batch.src,            batch.tgt_input,            batch.src_mask,            batch.tgt_mask,        )        logits = model.generator(hidden_states)        loss_output = criterion(logits, batch.tgt_y)        acc_output = compute_token_accuracy(logits, batch.tgt_y, criterion.pad_idx)        batch_tokens = loss_output.num_tokens        total_loss_sum += float(loss_output.loss.detach().item()) * batch_tokens        total_nll_sum += float(loss_output.nll_loss.detach().item()) * batch_tokens        total_smooth_sum += float(loss_output.smooth_loss.detach().item()) * batch_tokens        total_correct_tokens += int(acc_output["correct_tokens"])        total_tokens += int(acc_output["total_tokens"])    if total_tokens == 0:        raise ValueError("验证阶段有效 token 数为 0，无法汇总指标。")    avg_loss = total_loss_sum / total_tokens    avg_nll_loss = total_nll_sum / total_tokens    avg_smooth_loss = total_smooth_sum / total_tokens    avg_token_acc = total_correct_tokens / total_tokens    avg_ppl = compute_perplexity_from_loss(avg_nll_loss)    stats = {        "epoch": float(epoch),        "valid_loss": float(avg_loss),        "valid_nll_loss": float(avg_nll_loss),        "valid_smooth_loss": float(avg_smooth_loss),        "valid_token_acc": float(avg_token_acc),        "valid_ppl": float(avg_ppl),    }    if tb_logger is not None:        tb_logger.log_valid_epoch(epoch=epoch, stats=stats)    # -------------------------    # 记录少量文本样例    # -------------------------    if tb_logger is not None and vocab is not None and sample_batch is not None and num_text_samples > 0:        sample_src = sample_batch.src[:num_text_samples]        sample_src_mask = sample_batch.src_mask[:num_text_samples]        sample_tgt_y = sample_batch.tgt_y[:num_text_samples]        max_len = int(sample_src.size(1) + max_decode_extra_len)        pred_ids = greedy_decode(            model=model,            src=sample_src,            src_mask=sample_src_mask,            bos_id=vocab.bos_id,            eos_id=vocab.eos_id,            pad_id=vocab.pad_id,            max_len=max_len,        )        samples: List[Dict[str, str]] = []        for i in range(sample_src.size(0)):            source_text = _ids_to_bpe_string(                ids=sample_src[i].tolist(),                id_to_token_func=vocab.id2token,                pad_id=vocab.pad_id,                bos_id=vocab.bos_id,                eos_id=vocab.eos_id,            )            reference_text = _ids_to_bpe_string(                ids=sample_tgt_y[i].tolist(),                id_to_token_func=vocab.id2token,                pad_id=vocab.pad_id,                bos_id=vocab.bos_id,                eos_id=vocab.eos_id,            )            prediction_text = _ids_to_bpe_string(                ids=pred_ids[i].tolist(),                id_to_token_func=vocab.id2token,                pad_id=vocab.pad_id,                bos_id=vocab.bos_id,                eos_id=vocab.eos_id,            )            samples.append(                {                    "source": source_text,                    "reference": reference_text,                    "prediction": prediction_text,                }            )        tb_logger.log_text_samples(            epoch=epoch,            samples=samples,            tag="valid_samples_greedy",        )    return stats
```

* * *

## `train/fit.py`

```
from __future__ import annotations"""功能：1. 组织 Transformer 的完整训练流程。2. 保存：   - config.json   - metrics.csv   - matplotlib 曲线 png   - TensorBoard 日志   - best / last / periodic checkpoint3. 调用：   - train_one_epoch   - validate_one_epoch说明：1. 第一版优先做论文主干对齐：   - Label Smoothing   - Adam + Noam   - step 级训练日志   - valid loss/ppl/token_acc2. BLEU、beam search、checkpoint averaging 后续再接。"""import jsonimport osfrom typing import Dict, Optionalimport torchfrom train.train_one_epoch import train_one_epochfrom train.validate_one_epoch import validate_one_epochfrom utils.checkpoint_manager import CheckpointManagerfrom utils.csv_logger import CSVMetricLoggerfrom utils.plot_metrics import plot_default_transformer_curvesfrom utils.tb_log import TransformerTBLoggerdef _make_json_safe(obj):    """    将配置对象转成可 JSON 序列化的形式。    """    if isinstance(obj, dict):        return {k: _make_json_safe(v) for k, v in obj.items()}    if isinstance(obj, (list, tuple)):        return [_make_json_safe(v) for v in obj]    if isinstance(obj, torch.device):        return str(obj)    return objdef fit(    model: torch.nn.Module,    train_loader,    valid_loader,    criterion,    optimizer,    scheduler,    device: torch.device,    num_epochs: int,    output_dir: str,    config: Dict,    vocab=None,    scaler: Optional[torch.cuda.amp.GradScaler] = None,    use_amp: bool = False,    grad_clip_norm: Optional[float] = 1.0,    train_log_interval: int = 100,    histogram_interval: int = 1000,    save_every_epochs: int = 1,    valid_num_text_samples: int = 3,    max_train_steps_per_epoch: Optional[int] = None,    max_valid_steps_per_epoch: Optional[int] = None,) -> None:    """    顶层训练入口。    入口：        model: Transformer 模型        train_loader: 训练 DataLoader        valid_loader: 验证 DataLoader        criterion: LabelSmoothingLoss        optimizer: Adam        scheduler: NoamLRScheduler        device: 训练设备        num_epochs: 训练总 epoch 数        output_dir: 输出目录        config: 实验配置字典        vocab: SharedVocab，可选，用于验证阶段文本样例        scaler: AMP GradScaler        use_amp: 是否启用 AMP        grad_clip_norm: 梯度裁剪阈值        train_log_interval: train step 标量日志间隔        histogram_interval: histogram 记录间隔        save_every_epochs: 周期保存频率        valid_num_text_samples: 验证时记录多少个文本样例        max_train_steps_per_epoch / max_valid_steps_per_epoch:            调试时限制每个 epoch 的最大 step 数    出口：        无。所有结果都落到 output_dir。    """    os.makedirs(output_dir, exist_ok=True)    tb_dir = os.path.join(output_dir, "tb")    ckpt_dir = os.path.join(output_dir, "checkpoints")    plot_dir = os.path.join(output_dir, "plots")    csv_path = os.path.join(output_dir, "metrics.csv")    config_path = os.path.join(output_dir, "config.json")    os.makedirs(tb_dir, exist_ok=True)    os.makedirs(ckpt_dir, exist_ok=True)    os.makedirs(plot_dir, exist_ok=True)    # -------------------------    # 保存配置    # -------------------------    safe_config = _make_json_safe(config)    with open(config_path, "w", encoding="utf-8") as f:        json.dump(safe_config, f, ensure_ascii=False, indent=2)    # -------------------------    # 初始化日志系统    # -------------------------    tb_logger = TransformerTBLogger(log_dir=tb_dir)    csv_fieldnames = [        "epoch",        "global_step",        "train_loss",        "train_nll_loss",        "train_smooth_loss",        "train_token_acc",        "train_ppl",        "valid_loss",        "valid_nll_loss",        "valid_smooth_loss",        "valid_token_acc",        "valid_ppl",        "avg_grad_norm",        "avg_tokens_per_sec",        "epoch_time_sec",        "lr_last",    ]    csv_logger = CSVMetricLogger(        csv_path=csv_path,        fieldnames=csv_fieldnames,    )    ckpt_manager = CheckpointManager(        save_dir=ckpt_dir,        monitor_key="valid_ppl",        mode="min",    )    global_step = 0    model = model.to(device)    for epoch in range(1, num_epochs + 1):        # -------------------------        # 训练集 epoch 控制（用于 buffer shuffle）        # -------------------------        if hasattr(train_loader, "dataset") and hasattr(train_loader.dataset, "set_epoch"):            train_loader.dataset.set_epoch(epoch)        # -------------------------        # train        # -------------------------        train_stats, global_step = train_one_epoch(            model=model,            train_loader=train_loader,            criterion=criterion,            optimizer=optimizer,            scheduler=scheduler,            device=device,            epoch=epoch,            global_step=global_step,            tb_logger=tb_logger,            scaler=scaler,            use_amp=use_amp,            grad_clip_norm=grad_clip_norm,            log_interval=train_log_interval,            histogram_interval=histogram_interval,            max_steps_per_epoch=max_train_steps_per_epoch,        )        # -------------------------        # valid        # -------------------------        valid_stats = validate_one_epoch(            model=model,            valid_loader=valid_loader,            criterion=criterion,            device=device,            epoch=epoch,            tb_logger=tb_logger,            vocab=vocab,            num_text_samples=valid_num_text_samples,            max_decode_extra_len=50,            max_steps_per_epoch=max_valid_steps_per_epoch,        )        # -------------------------        # 合并并写 CSV        # -------------------------        row = {            "epoch": epoch,            "global_step": int(global_step),            "train_loss": train_stats["train_loss"],            "train_nll_loss": train_stats["train_nll_loss"],            "train_smooth_loss": train_stats["train_smooth_loss"],            "train_token_acc": train_stats["train_token_acc"],            "train_ppl": train_stats["train_ppl"],            "valid_loss": valid_stats["valid_loss"],            "valid_nll_loss": valid_stats["valid_nll_loss"],            "valid_smooth_loss": valid_stats["valid_smooth_loss"],            "valid_token_acc": valid_stats["valid_token_acc"],            "valid_ppl": valid_stats["valid_ppl"],            "avg_grad_norm": train_stats["avg_grad_norm"],            "avg_tokens_per_sec": train_stats["avg_tokens_per_sec"],            "epoch_time_sec": train_stats["epoch_time_sec"],            "lr_last": train_stats["lr_last"],        }        csv_logger.append_row(row)        # -------------------------        # 本地趋势图        # -------------------------        plot_default_transformer_curves(            csv_path=csv_path,            out_dir=plot_dir,        )        # -------------------------        # checkpoint        # -------------------------        ckpt_manager.save_last(            epoch=epoch,            global_step=global_step,            model=model,            optimizer=optimizer,            scheduler=scheduler,            scaler=scaler,            train_stats=train_stats,            valid_stats=valid_stats,            config=safe_config,        )        ckpt_manager.save_best_if_needed(            epoch=epoch,            global_step=global_step,            model=model,            optimizer=optimizer,            scheduler=scheduler,            scaler=scaler,            train_stats=train_stats,            valid_stats=valid_stats,            config=safe_config,        )        if save_every_epochs > 0 and (epoch % save_every_epochs == 0):            ckpt_manager.save_periodic(                epoch=epoch,                global_step=global_step,                model=model,                optimizer=optimizer,                scheduler=scheduler,                scaler=scaler,                train_stats=train_stats,                valid_stats=valid_stats,                config=safe_config,            )        tb_logger.flush()    tb_logger.close()
```

* * *

## `script/train_part/01_smoke_test_loss_scheduler.py`

```
from __future__ import annotations"""功能：1. 测试 LabelSmoothingLoss 是否能正常前向。2. 测试 Adam + Noam 调度器是否能正常更新学习率。3. 作为训练系统最底层数学模块的 smoke test。"""import torchfrom utils.label_smoothing import LabelSmoothingLoss, compute_token_accuracy, compute_perplexity_from_lossfrom utils.noam_scheduler import build_transformer_optimizer_and_schedulerclass DummyModel(torch.nn.Module):    def __init__(self, vocab_size: int, d_model: int = 32) -> None:        super().__init__()        self.emb = torch.nn.Embedding(vocab_size, d_model)        self.proj = torch.nn.Linear(d_model, vocab_size)    def forward(self, x: torch.Tensor) -> torch.Tensor:        h = self.emb(x)        return self.proj(h)def main() -> None:    vocab_size = 100    pad_idx = 0    model = DummyModel(vocab_size=vocab_size, d_model=32)    criterion = LabelSmoothingLoss(        vocab_size=vocab_size,        pad_idx=pad_idx,        smoothing=0.1,    )    optimizer, scheduler = build_transformer_optimizer_and_scheduler(        model=model,        d_model=32,        warmup_steps=10,    )    x = torch.tensor([        [5, 6, 7, 8],        [9, 10, 0, 0],    ], dtype=torch.long)    target = torch.tensor([        [6, 7, 8, 2],        [10, 2, 0, 0],    ], dtype=torch.long)    logits = model(x)    loss_out = criterion(logits, target)    acc_out = compute_token_accuracy(logits, target, pad_idx=pad_idx)    print("loss_out =", loss_out.as_dict())    print("acc_out  =", acc_out)    print("ppl      =", compute_perplexity_from_loss(loss_out.as_dict()["nll_loss"]))    optimizer.zero_grad(set_to_none=True)    loss_out.loss.backward()    lr = scheduler.step()    optimizer.step()    print("当前 lr =", lr)    print("scheduler step_num =", scheduler.step_num)if __name__ == "__main__":    main()
```

* * *

## `script/train_part/02_smoke_test_train_loop.py`

```
from __future__ import annotations"""功能：1. 用真实的 WMT14 BPE 数据和真实 Transformer 模型，   跑一个极小规模的训练/验证 smoke test。2. 验证：   - Label Smoothing   - Noam 调度   - train_one_epoch   - validate_one_epoch   - CSV / TensorBoard / checkpoint是否能协同工作。"""import osimport torchfrom data.shared_vocab import SharedVocabfrom data.wmt14_bpe_dataset import build_bpe_dataloaderfrom nets.build_transformer import make_modelfrom train.fit import fitfrom utils.label_smoothing import LabelSmoothingLossfrom utils.noam_scheduler import build_transformer_optimizer_and_schedulerdef main() -> None:    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    vocab = SharedVocab.load("data/wmt14_vocab/vocab.json")    vocab_size = len(vocab)    train_loader = build_bpe_dataloader(        src_path="data/wmt14_bpe_en_de/train.en",        tgt_path="data/wmt14_bpe_en_de/train.de",        vocab=vocab,        batch_size=8,        num_workers=0,        pin_memory=False,        max_src_len=64,        max_tgt_len=64,        add_src_eos=True,        skip_empty=False,        shuffle_buffer_size=1000,        seed=42,        num_samples=3927488,        persistent_workers=False,        prefetch_factor=None,    )    valid_loader = build_bpe_dataloader(        src_path="data/wmt14_bpe_en_de/valid.en",        tgt_path="data/wmt14_bpe_en_de/valid.de",        vocab=vocab,        batch_size=8,        num_workers=0,        pin_memory=False,        max_src_len=64,        max_tgt_len=64,        add_src_eos=True,        skip_empty=False,        shuffle_buffer_size=0,        seed=42,        num_samples=3000,        persistent_workers=False,        prefetch_factor=None,    )    model = make_model(        src_vocab=vocab_size,        tgt_vocab=vocab_size,        N=2,        d_model=256,        d_ff=1024,        h=4,        dropout=0.1,        share_embeddings=True,    )    criterion = LabelSmoothingLoss(        vocab_size=vocab_size,        pad_idx=vocab.pad_id,        smoothing=0.1,    )    optimizer, scheduler = build_transformer_optimizer_and_scheduler(        model=model,        d_model=256,        warmup_steps=4000,        factor=1.0,        beta1=0.9,        beta2=0.98,        eps=1e-9,        weight_decay=0.0,    )    config = {        "exp_name": "transformer_smoke_test",        "device": str(device),        "vocab_size": vocab_size,        "pad_id": vocab.pad_id,        "bos_id": vocab.bos_id,        "eos_id": vocab.eos_id,        "model": {            "N": 2,            "d_model": 256,            "d_ff": 1024,            "h": 4,            "dropout": 0.1,            "share_embeddings": True,        },        "criterion": {            "type": "LabelSmoothingLoss",            "smoothing": 0.1,        },        "optimizer": {            "type": "Adam",            "beta1": 0.9,            "beta2": 0.98,            "eps": 1e-9,            "weight_decay": 0.0,        },        "scheduler": {            "type": "NoamLRScheduler",            "warmup_steps": 4000,            "factor": 1.0,        },        "dataloader": {            "train_batch_size": 8,            "valid_batch_size": 8,            "train_num_workers": 0,            "valid_num_workers": 0,            "train_shuffle_buffer_size": 1000,        },    }    fit(        model=model,        train_loader=train_loader,        valid_loader=valid_loader,        criterion=criterion,        optimizer=optimizer,        scheduler=scheduler,        device=device,        num_epochs=2,        output_dir=os.path.join("experiments", "transformer_smoke_test"),        config=config,        vocab=vocab,        scaler=None,        use_amp=False,        grad_clip_norm=1.0,        train_log_interval=10,        histogram_interval=50,        save_every_epochs=1,        valid_num_text_samples=2,        max_train_steps_per_epoch=20,        max_valid_steps_per_epoch=10,    )if __name__ == "__main__":    main()
```

* * *

## `script/train_part/03_build_real_train_objects.py`

```
from __future__ import annotations"""功能：1. 给正式训练阶段提供一个“构建真实训练对象”的参考脚本。2. 不直接开训，只负责把：   - vocab   - train_loader   - valid_loader   - model   - criterion   - optimizer   - scheduler全部构建出来，便于单独调试。"""import torchfrom data.shared_vocab import SharedVocabfrom data.wmt14_bpe_dataset import build_bpe_dataloaderfrom nets.build_transformer import make_modelfrom utils.label_smoothing import LabelSmoothingLossfrom utils.noam_scheduler import build_transformer_optimizer_and_schedulerdef main() -> None:    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    vocab = SharedVocab.load("data/wmt14_vocab/vocab.json")    vocab_size = len(vocab)    train_loader = build_bpe_dataloader(        src_path="data/wmt14_bpe_en_de/train.en",        tgt_path="data/wmt14_bpe_en_de/train.de",        vocab=vocab,        batch_size=64,        num_workers=2,        pin_memory=True,        max_src_len=None,        max_tgt_len=None,        add_src_eos=True,        skip_empty=False,        shuffle_buffer_size=10000,        seed=42,        num_samples=3927488,        persistent_workers=True,        prefetch_factor=2,    )    valid_loader = build_bpe_dataloader(        src_path="data/wmt14_bpe_en_de/valid.en",        tgt_path="data/wmt14_bpe_en_de/valid.de",        vocab=vocab,        batch_size=64,        num_workers=0,        pin_memory=True,        max_src_len=None,        max_tgt_len=None,        add_src_eos=True,        skip_empty=False,        shuffle_buffer_size=0,        seed=42,        num_samples=3000,        persistent_workers=False,        prefetch_factor=None,    )    model = make_model(        src_vocab=vocab_size,        tgt_vocab=vocab_size,        N=6,        d_model=512,        d_ff=2048,        h=8,        dropout=0.1,        share_embeddings=True,    ).to(device)    criterion = LabelSmoothingLoss(        vocab_size=vocab_size,        pad_idx=vocab.pad_id,        smoothing=0.1,    )    optimizer, scheduler = build_transformer_optimizer_and_scheduler(        model=model,        d_model=512,        warmup_steps=4000,        factor=1.0,        beta1=0.9,        beta2=0.98,        eps=1e-9,        weight_decay=0.0,    )    print("对象构建成功。")    print(f"device = {device}")    print(f"vocab_size = {vocab_size}")    print(f"pad_id = {vocab.pad_id}, bos_id = {vocab.bos_id}, eos_id = {vocab.eos_id}")    print("train_loader / valid_loader / model / criterion / optimizer / scheduler 均已构建完成。")if __name__ == "__main__":    main()
```

  

我有一个疑问，为什么要手动实现labelsmoothing？不能让模型最后的linear输出logits，然后调用celoss的时候指定labelsmooth参数吗？

  

## 可以，而且很多情况下这是更简洁的做法

你这个疑问是对的。  
**如果你只是想要“标准的交叉熵 + label smoothing”**，那么完全可以让模型最后输出 `logits`，然后直接用：

```
nn.CrossEntropyLoss(    ignore_index=pad_idx,    label_smoothing=0.1,)
```

PyTorch 官方文档明确说明了：

-   `CrossEntropyLoss` 的输入就是 **未归一化的 logits**
    
-   支持 `ignore_index`
    
-   支持 `label_smoothing`
    
-   `label_smoothing` 的含义是：把目标分布变成“原始 one-hot 标签”和“均匀分布”的混合。PyTorch 文档+1
    

所以从**功能上**说：

> **能用内置就能用，不是必须手写。**

* * *

# 但是，为什么我前面还是给你手写版

因为这不是“能不能做”的问题，而是“你当前这个项目更适合哪种实现”的问题。

* * *

## 一、什么时候内置 `CrossEntropyLoss(label_smoothing=...)` 就够了

如果你满足下面这些条件，内置版本就很合适：

-   你只需要一个标准总损失；
    
-   你不需要把 loss 精细拆成多个成分；
    
-   你不在意 smoothing 质量如何在类别上精确分配；
    
-   你只想要一个稳定、简洁、少代码的实现；
    
-   你只需要 `ignore_index=pad_idx` 去忽略 PAD 位置。PyTorch 文档+1
    

在这种情况下，最常见写法就是：

```
criterion = torch.nn.CrossEntropyLoss(    ignore_index=pad_idx,    label_smoothing=0.1,    reduction="mean",)
```

然后训练时：

```
logits = model.generator(hidden_states)     # (B, T, V)loss = criterion(    logits.reshape(-1, logits.size(-1)),    batch.tgt_y.reshape(-1),)
```

这完全没问题。

* * *

## 二、那为什么我仍然建议你当前项目保留手写版

核心原因只有一个：

> **你要求的日志粒度，比内置 CE 直接返回一个标量更细。**

你前面明确要求：

-   step 级细粒度 loss 记录
    
-   不接受只记一个 total loss
    
-   要做 TensorBoard 多线对比
    
-   要把训练过程看得比较“解剖式”
    

而内置 `CrossEntropyLoss(label_smoothing=...)` 的直接输出只有一个：

-   `loss`
    

它不会额外把下面这些拆给你：

-   `nll_loss`
    
-   `smooth_loss`
    
-   `num_tokens`
    

所以如果你直接用内置版本，你后面会遇到一个问题：

> **总损失你有了，但你很难优雅地把“硬标签分量”和“平滑分量”拆开记录。**

而你现在这个项目，恰恰很适合监控这两个量：

-   `nll_loss`：看模型是否真正学会预测正确 token
    
-   `smooth_loss`：看平滑项是否占比异常
    
-   `total_loss`：看最终训练目标
    

也就是说：

### 用内置版

优点：

-   简洁
    
-   可靠
    
-   少写代码
    

缺点：

-   日志可解释性差
    
-   不方便做你想要的“loss 成分拆解”
    

### 用手写版

优点：

-   能直接返回：
    
    -   `loss`
        
    -   `nll_loss`
        
    -   `smooth_loss`
        
    -   `num_tokens`
        
-   很适合你当前日志体系
    

缺点：

-   代码更长
    
-   你要自己保证实现没 bug
    

* * *

## 三、还有一个更细的点：PAD 类怎么处理

PyTorch 文档明确说明：

-   `ignore_index` 可以让某些 target 位置不参与 loss 计算；
    
-   `label_smoothing` 会把目标分布变成原始目标和均匀分布的混合。PyTorch 文档+1
    

但是在 seq2seq 任务里，我们经常还会进一步关心一个实现细节：

> 对于**非 PAD 的有效位置**，平滑概率质量是否还要分给 **PAD 这个类别本身**？

这是一个“实现口径”问题。

我前面给你的手写版，做的是一种更显式、更可控的选择：

-   **忽略 target 为 PAD 的位置**
    
-   并且在平滑分配时，**不把 PAD 类放进被分配的类别池里**
    

这不是因为内置版一定错，而是因为：

> **手写版让这个设计决策是透明的、可审计的。**

对于机器翻译这种有 `<pad>`、`<bos>`、`<eos>` 的任务，这种透明性是有价值的。

* * *

## 四、所以现在最准确的回答是

### 你的想法对不对？

**对。**

你完全可以改成：

-   模型输出 logits
    
-   直接 `CrossEntropyLoss(ignore_index=pad_idx, label_smoothing=0.1)`
    

这在 PyTorch 里是标准支持的。PyTorch 文档+1

* * *

### 为什么我之前没有直接这么写？

因为你这个项目有两个额外诉求：

1.  **你要细粒度训练日志**
    
2.  **你要对 loss 行为有更强可解释性和可控性**
    

所以我才给了你手写的 `LabelSmoothingLoss`。

* * *

# 五、我给你的实战建议

## 方案 A：你偏好代码简洁

那就直接换成内置版。

适合你如果现在想：

-   尽快缩短代码
    
-   优先保证训练主线跑起来
    
-   暂时接受 TensorBoard 里只记：
    
    -   total loss
        
    -   token\_acc
        
    -   ppl
        

### 这种写法最简洁

```
criterion = torch.nn.CrossEntropyLoss(    ignore_index=pad_idx,    label_smoothing=0.1,)
```

* * *

## 方案 B：你偏好训练日志可解释性

那就继续保留我前面写的手动实现。

这更适合你当前风格，因为你明确偏好：

-   loss 拆解
    
-   step 级多线记录
    
-   更强的训练可观测性
    

我认为以你的训练日志偏好，**手写版更适合你当前项目**。

* * *

# 六、如果你问我“我个人建议你选哪个”

我给你的建议是：

> **第一版先保留手写版。**

理由很简单：

-   你现在不是在写一个“最短 demo”
    
-   你是在搭一个“可诊断、可复现实验系统”
    
-   你明确希望 loss 拆解、TensorBoard 多线图、本地 csv/png 全都清楚
    

在这种目标下，手写版的价值是明显的。

* * *

# 七、但是我也给你一个折中方案

如果你想兼顾：

-   内置 CE 的简洁
    
-   以及更细的日志
    

那也不是不行。  
你可以：

1.  主损失用内置 `CrossEntropyLoss(label_smoothing=0.1)`
    
2.  额外再单独计算一个：
    
    -   `nll_loss`（不带 smoothing 的标准 CE）
        
3.  然后用：
    
    -   `smooth_gap = total_loss - nll_loss`  
        做一个近似日志分解
        

但这个分解不是最严格的“平滑项数学本体”，只是一个工程近似。  
如果你想日志解释严格，还是手写版更干净。

* * *

# 八、最终一句话总结

**能用内置，而且功能上完全成立；但就你当前这个项目的日志要求和可解释性要求来看，手写版更合适。**

* * *

你要是愿意，我下一条可以直接给你两套版本并行：

1.  **保留手写 `LabelSmoothingLoss` 的训练系统**
    
2.  **改成内置 `CrossEntropyLoss(label_smoothing=0.1)` 的精简版训练系统**
    

你选其中一个继续往下接。

  

好，那我采用方案B，保留你之前生成的代码。那下面我们进行哪一步呢？请你生成代码吧

  

## 下一步

下面直接进入：

-   正式训练入口
    
-   断点续训
    
-   Python 3.9 兼容修正
    
-   训练环境工具
    

你先把下面这些文件落到项目里。

* * *

## `data/shared_vocab.py`（替换为 Python 3.9 兼容版）

```
from __future__ import annotations"""功能：1. 从 BPE 文本训练集构建 shared vocabulary。2. 提供 token -> id / id -> token 的双向映射。3. 保存和加载 vocab 文件。4. 为后续 DataLoader、target shifted right、mask 构造提供统一词表接口。设计原则：1. 只用 train 构建词表，不让 valid/test 泄露统计信息。2. 源语言和目标语言共享一套 token-id 体系。3. 固定特殊符号顺序，确保 pad/bos/eos/unk 的 id 稳定。"""import jsonfrom collections import Counterfrom dataclasses import dataclassfrom typing import Dict, Iterable, List, Optional, Sequence@dataclassclass SpecialTokens:    pad: str = "<pad>"    bos: str = "<bos>"    eos: str = "<eos>"    unk: str = "<unk>"    def as_list(self) -> List[str]:        return [self.pad, self.bos, self.eos, self.unk]class SharedVocab:    """    shared source-target vocabulary    入口：        - 通过 build_from_files 从 train.en/train.de 构建        - 或通过 load 从已保存文件加载    出口：        - token_to_id / id_to_token        - encode / decode        - 各特殊 token 的 id    """    def __init__(        self,        token_to_id: Dict[str, int],        special_tokens: SpecialTokens,    ) -> None:        self.token_to_id = token_to_id        self.id_to_token = {idx: tok for tok, idx in token_to_id.items()}        self.special_tokens = special_tokens        self.pad_id = self.token_to_id[self.special_tokens.pad]        self.bos_id = self.token_to_id[self.special_tokens.bos]        self.eos_id = self.token_to_id[self.special_tokens.eos]        self.unk_id = self.token_to_id[self.special_tokens.unk]    def __len__(self) -> int:        return len(self.token_to_id)    def token2id(self, token: str) -> int:        return self.token_to_id.get(token, self.unk_id)    def id2token(self, idx: int) -> str:        return self.id_to_token[idx]    def encode(self, tokens: Sequence[str]) -> List[int]:        """        将 token 序列编码成 id 序列。        """        return [self.token2id(tok) for tok in tokens]    def decode(self, ids: Sequence[int]) -> List[str]:        """        将 id 序列解码成 token 序列。        """        return [self.id2token(idx) for idx in ids]    @classmethod    def build_from_files(        cls,        file_paths: Sequence[str],        min_freq: int = 1,        special_tokens: Optional[SpecialTokens] = None,    ) -> "SharedVocab":        """        从多个训练文本文件中构建共享词表。        参数：            file_paths: 参与构建词表的文件路径列表，通常是 train.en 和 train.de            min_freq: 最小词频阈值            special_tokens: 特殊 token 配置        返回：            SharedVocab 实例        """        if special_tokens is None:            special_tokens = SpecialTokens()        counter: Counter = Counter()        for path in file_paths:            with open(path, "r", encoding="utf-8") as f:                for line in f:                    tokens = line.strip().split()                    counter.update(tokens)        # 先放特殊 token，确保 id 稳定        token_to_id: Dict[str, int] = {}        for tok in special_tokens.as_list():            token_to_id[tok] = len(token_to_id)        # 再放普通 token        # 排序策略：        # 1. 先按词频降序        # 2. 词频相同按字典序        normal_tokens = [            (tok, freq)            for tok, freq in counter.items()            if freq >= min_freq and tok not in token_to_id        ]        normal_tokens.sort(key=lambda x: (-x[1], x[0]))        for tok, _ in normal_tokens:            token_to_id[tok] = len(token_to_id)        return cls(token_to_id=token_to_id, special_tokens=special_tokens)    def save(self, vocab_json_path: str, vocab_txt_path: str, meta_json_path: str) -> None:        """        保存词表到磁盘。        """        with open(vocab_json_path, "w", encoding="utf-8") as f:            json.dump(self.token_to_id, f, ensure_ascii=False, indent=2)        with open(vocab_txt_path, "w", encoding="utf-8") as f:            for idx in range(len(self)):                f.write(self.id_to_token[idx] + "\n")        meta = {            "vocab_size": len(self),            "pad_token": self.special_tokens.pad,            "bos_token": self.special_tokens.bos,            "eos_token": self.special_tokens.eos,            "unk_token": self.special_tokens.unk,            "pad_id": self.pad_id,            "bos_id": self.bos_id,            "eos_id": self.eos_id,            "unk_id": self.unk_id,        }        with open(meta_json_path, "w", encoding="utf-8") as f:            json.dump(meta, f, ensure_ascii=False, indent=2)    @classmethod    def load(        cls,        vocab_json_path: str,        special_tokens: Optional[SpecialTokens] = None,    ) -> "SharedVocab":        """        从 vocab.json 加载词表。        """        if special_tokens is None:            special_tokens = SpecialTokens()        with open(vocab_json_path, "r", encoding="utf-8") as f:            token_to_id = json.load(f)        return cls(token_to_id=token_to_id, special_tokens=special_tokens)
```

* * *

## `utils/noam_scheduler.py`（替换为 Python 3.9 兼容版）

```
from __future__ import annotations"""功能：1. 实现论文中的 Noam 学习率调度器（step-based）。2. 提供 Adam + Noam 的标准构建入口。3. 支持 state_dict / load_state_dict，便于断点续训。注意：1. Noam 调度本质上是按 optimizer update step 更新，而不是按 epoch 更新。2. 推荐调用顺序：   - backward   - grad clip   - scheduler.step()   # 先把本次 update 的 lr 设好   - optimizer.step()"""from dataclasses import dataclassfrom typing import Dict, Optional, Tupleimport torchfrom torch import nn@dataclassclass NoamConfig:    """    Noam 调度器配置。    """    d_model: int    warmup_steps: int = 4000    factor: float = 1.0    beta1: float = 0.9    beta2: float = 0.98    eps: float = 1e-9    weight_decay: float = 0.0class NoamLRScheduler:    """    论文风格 Noam 学习率调度器。    数学形式：        lrate = factor * (d_model ^ -0.5) *                min(step_num ^ -0.5, step_num * warmup_steps ^ -1.5)    入口：        optimizer: PyTorch Optimizer        d_model: 模型维度        warmup_steps: 预热步数        factor: 缩放因子        init_step: 初始 step（断点恢复时使用）    出口：        - step(): 更新内部 step，并把当前 lr 写回 optimizer        - rate(step): 查询指定 step 的 lr        - get_last_lr(): 获取最近一次 lr        - state_dict/load_state_dict(): 状态保存与恢复    """    def __init__(        self,        optimizer: torch.optim.Optimizer,        d_model: int,        warmup_steps: int = 4000,        factor: float = 1.0,        init_step: int = 0,    ) -> None:        if d_model <= 0:            raise ValueError("d_model 必须大于 0。")        if warmup_steps <= 0:            raise ValueError("warmup_steps 必须大于 0。")        if factor <= 0:            raise ValueError("factor 必须大于 0。")        self.optimizer = optimizer        self.d_model = d_model        self.warmup_steps = warmup_steps        self.factor = factor        self._step = int(init_step)        self._rate = 0.0        for group in self.optimizer.param_groups:            group["lr"] = 0.0    @property    def step_num(self) -> int:        return self._step    def rate(self, step: Optional[int] = None) -> float:        """        计算某个 step 对应的学习率。        """        if step is None:            step = self._step        step = max(int(step), 1)        return self.factor * (self.d_model ** -0.5) * min(            step ** -0.5,            step * (self.warmup_steps ** -1.5),        )    def step(self) -> float:        """        推进一步，并将当前 lr 写入 optimizer。        返回：            当前 step 对应的 lr        """        self._step += 1        lr = self.rate(self._step)        for group in self.optimizer.param_groups:            group["lr"] = lr        self._rate = lr        return lr    def get_last_lr(self) -> list:        """        返回最近一次设置的 lr，接口风格与 PyTorch scheduler 对齐。        """        return [group["lr"] for group in self.optimizer.param_groups]    def state_dict(self) -> Dict:        """        保存调度器状态。        """        return {            "d_model": self.d_model,            "warmup_steps": self.warmup_steps,            "factor": self.factor,            "_step": self._step,            "_rate": self._rate,        }    def load_state_dict(self, state_dict: Dict) -> None:        """        恢复调度器状态。        """        self.d_model = int(state_dict["d_model"])        self.warmup_steps = int(state_dict["warmup_steps"])        self.factor = float(state_dict["factor"])        self._step = int(state_dict["_step"])        self._rate = float(state_dict["_rate"])        for group in self.optimizer.param_groups:            group["lr"] = self._ratedef build_transformer_optimizer(    model: nn.Module,    beta1: float = 0.9,    beta2: float = 0.98,    eps: float = 1e-9,    weight_decay: float = 0.0,) -> torch.optim.Adam:    """    构建论文风格 Adam 优化器。    """    optimizer = torch.optim.Adam(        params=model.parameters(),        lr=0.0,        betas=(beta1, beta2),        eps=eps,        weight_decay=weight_decay,    )    return optimizerdef build_transformer_optimizer_and_scheduler(    model: nn.Module,    d_model: int,    warmup_steps: int = 4000,    factor: float = 1.0,    beta1: float = 0.9,    beta2: float = 0.98,    eps: float = 1e-9,    weight_decay: float = 0.0,) -> Tuple[torch.optim.Adam, NoamLRScheduler]:    """    一次性构建 Adam + Noam 调度器。    """    optimizer = build_transformer_optimizer(        model=model,        beta1=beta1,        beta2=beta2,        eps=eps,        weight_decay=weight_decay,    )    scheduler = NoamLRScheduler(        optimizer=optimizer,        d_model=d_model,        warmup_steps=warmup_steps,        factor=factor,    )    return optimizer, scheduler
```

* * *

## `utils/train_env.py`（新建）

```
from __future__ import annotations"""功能：1. 设置随机种子。2. 获取训练设备。3. 统计模型参数量。4. 生成实验时间戳。"""import osimport randomfrom datetime import datetimefrom typing import Dictimport numpy as npimport torchdef seed_everything(seed: int = 42, deterministic: bool = False) -> None:    """    设置全局随机种子。    参数：        seed: 随机种子        deterministic: 是否开启更强的确定性模式    """    random.seed(seed)    np.random.seed(seed)    torch.manual_seed(seed)    torch.cuda.manual_seed_all(seed)    if deterministic:        torch.backends.cudnn.deterministic = True        torch.backends.cudnn.benchmark = False    else:        torch.backends.cudnn.deterministic = False        torch.backends.cudnn.benchmark = Truedef get_device() -> torch.device:    """    获取当前推荐训练设备。    """    return torch.device("cuda" if torch.cuda.is_available() else "cpu")def count_trainable_parameters(model: torch.nn.Module) -> int:    """    统计模型可训练参数量。    """    return sum(p.numel() for p in model.parameters() if p.requires_grad)def get_timestamp_str() -> str:    """    获取实验时间戳字符串。    """    return datetime.now().strftime("%Y%m%d_%H%M%S")
```

* * *

## `utils/resume_utils.py`（新建）

```
from __future__ import annotations"""功能：1. 从 checkpoint 恢复模型、优化器、scheduler、scaler。2. 返回恢复训练所需的起始 epoch、global_step、best_metric 等信息。"""from typing import Any, Dict, Optionalimport torchdef load_checkpoint_for_resume(    checkpoint_path: str,    model: torch.nn.Module,    optimizer: Optional[torch.optim.Optimizer] = None,    scheduler: Optional[Any] = None,    scaler: Optional[torch.cuda.amp.GradScaler] = None,    map_location: str = "cpu",) -> Dict[str, Any]:    """    加载 checkpoint 并恢复训练状态。    返回字段：        - epoch        - global_step        - best_metric        - train_stats        - valid_stats        - config    """    checkpoint = torch.load(checkpoint_path, map_location=map_location)    model.load_state_dict(checkpoint["model_state_dict"])    if optimizer is not None and "optimizer_state_dict" in checkpoint:        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])    if scheduler is not None and "scheduler_state_dict" in checkpoint:        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])    if scaler is not None and "scaler_state_dict" in checkpoint:        scaler.load_state_dict(checkpoint["scaler_state_dict"])    valid_stats = checkpoint.get("valid_stats", {})    best_metric = None    if isinstance(valid_stats, dict) and "valid_ppl" in valid_stats:        best_metric = float(valid_stats["valid_ppl"])    return {        "epoch": int(checkpoint.get("epoch", 0)),        "global_step": int(checkpoint.get("global_step", 0)),        "best_metric": best_metric,        "train_stats": checkpoint.get("train_stats", {}),        "valid_stats": checkpoint.get("valid_stats", {}),        "config": checkpoint.get("config", {}),    }
```

* * *

## `train/fit.py`（替换为增强版，支持 resume）

```
from __future__ import annotations"""功能：1. 组织 Transformer 的完整训练流程。2. 保存：   - config.json   - metrics.csv   - matplotlib 曲线 png   - TensorBoard 日志   - best / last / periodic checkpoint3. 调用：   - train_one_epoch   - validate_one_epoch4. 支持 resume：   - start_epoch   - global_step_init   - best_metric_init说明：1. 第一版优先做论文主干对齐：   - Label Smoothing   - Adam + Noam   - step 级训练日志   - valid loss/ppl/token_acc2. BLEU、beam search、checkpoint averaging 后续再接。"""import jsonimport osfrom typing import Dict, Optionalimport torchfrom train.train_one_epoch import train_one_epochfrom train.validate_one_epoch import validate_one_epochfrom utils.checkpoint_manager import CheckpointManagerfrom utils.csv_logger import CSVMetricLoggerfrom utils.plot_metrics import plot_default_transformer_curvesfrom utils.tb_log import TransformerTBLoggerdef _make_json_safe(obj):    """    将配置对象转成可 JSON 序列化的形式。    """    if isinstance(obj, dict):        return {k: _make_json_safe(v) for k, v in obj.items()}    if isinstance(obj, (list, tuple)):        return [_make_json_safe(v) for v in obj]    if isinstance(obj, torch.device):        return str(obj)    return objdef fit(    model: torch.nn.Module,    train_loader,    valid_loader,    criterion,    optimizer,    scheduler,    device: torch.device,    num_epochs: int,    output_dir: str,    config: Dict,    vocab=None,    scaler: Optional[torch.cuda.amp.GradScaler] = None,    use_amp: bool = False,    grad_clip_norm: Optional[float] = 1.0,    train_log_interval: int = 100,    histogram_interval: int = 1000,    save_every_epochs: int = 1,    valid_num_text_samples: int = 3,    max_train_steps_per_epoch: Optional[int] = None,    max_valid_steps_per_epoch: Optional[int] = None,    start_epoch: int = 1,    global_step_init: int = 0,    best_metric_init: Optional[float] = None,) -> None:    """    顶层训练入口。    """    os.makedirs(output_dir, exist_ok=True)    tb_dir = os.path.join(output_dir, "tb")    ckpt_dir = os.path.join(output_dir, "checkpoints")    plot_dir = os.path.join(output_dir, "plots")    csv_path = os.path.join(output_dir, "metrics.csv")    config_path = os.path.join(output_dir, "config.json")    os.makedirs(tb_dir, exist_ok=True)    os.makedirs(ckpt_dir, exist_ok=True)    os.makedirs(plot_dir, exist_ok=True)    safe_config = _make_json_safe(config)    with open(config_path, "w", encoding="utf-8") as f:        json.dump(safe_config, f, ensure_ascii=False, indent=2)    tb_logger = TransformerTBLogger(log_dir=tb_dir)    csv_fieldnames = [        "epoch",        "global_step",        "train_loss",        "train_nll_loss",        "train_smooth_loss",        "train_token_acc",        "train_ppl",        "valid_loss",        "valid_nll_loss",        "valid_smooth_loss",        "valid_token_acc",        "valid_ppl",        "avg_grad_norm",        "avg_tokens_per_sec",        "epoch_time_sec",        "lr_last",    ]    csv_logger = CSVMetricLogger(        csv_path=csv_path,        fieldnames=csv_fieldnames,    )    ckpt_manager = CheckpointManager(        save_dir=ckpt_dir,        monitor_key="valid_ppl",        mode="min",    )    ckpt_manager.best_metric = best_metric_init    global_step = int(global_step_init)    model = model.to(device)    for epoch in range(start_epoch, num_epochs + 1):        if hasattr(train_loader, "dataset") and hasattr(train_loader.dataset, "set_epoch"):            train_loader.dataset.set_epoch(epoch)        print("=" * 100)        print(f"[Epoch {epoch}/{num_epochs}] 开始训练")        print("=" * 100)        train_stats, global_step = train_one_epoch(            model=model,            train_loader=train_loader,            criterion=criterion,            optimizer=optimizer,            scheduler=scheduler,            device=device,            epoch=epoch,            global_step=global_step,            tb_logger=tb_logger,            scaler=scaler,            use_amp=use_amp,            grad_clip_norm=grad_clip_norm,            log_interval=train_log_interval,            histogram_interval=histogram_interval,            max_steps_per_epoch=max_train_steps_per_epoch,        )        valid_stats = validate_one_epoch(            model=model,            valid_loader=valid_loader,            criterion=criterion,            device=device,            epoch=epoch,            tb_logger=tb_logger,            vocab=vocab,            num_text_samples=valid_num_text_samples,            max_decode_extra_len=50,            max_steps_per_epoch=max_valid_steps_per_epoch,        )        row = {            "epoch": epoch,            "global_step": int(global_step),            "train_loss": train_stats["train_loss"],            "train_nll_loss": train_stats["train_nll_loss"],            "train_smooth_loss": train_stats["train_smooth_loss"],            "train_token_acc": train_stats["train_token_acc"],            "train_ppl": train_stats["train_ppl"],            "valid_loss": valid_stats["valid_loss"],            "valid_nll_loss": valid_stats["valid_nll_loss"],            "valid_smooth_loss": valid_stats["valid_smooth_loss"],            "valid_token_acc": valid_stats["valid_token_acc"],            "valid_ppl": valid_stats["valid_ppl"],            "avg_grad_norm": train_stats["avg_grad_norm"],            "avg_tokens_per_sec": train_stats["avg_tokens_per_sec"],            "epoch_time_sec": train_stats["epoch_time_sec"],            "lr_last": train_stats["lr_last"],        }        csv_logger.append_row(row)        plot_default_transformer_curves(            csv_path=csv_path,            out_dir=plot_dir,        )        last_path = ckpt_manager.save_last(            epoch=epoch,            global_step=global_step,            model=model,            optimizer=optimizer,            scheduler=scheduler,            scaler=scaler,            train_stats=train_stats,            valid_stats=valid_stats,            config=safe_config,        )        best_path = ckpt_manager.save_best_if_needed(            epoch=epoch,            global_step=global_step,            model=model,            optimizer=optimizer,            scheduler=scheduler,            scaler=scaler,            train_stats=train_stats,            valid_stats=valid_stats,            config=safe_config,        )        periodic_path = None        if save_every_epochs > 0 and (epoch % save_every_epochs == 0):            periodic_path = ckpt_manager.save_periodic(                epoch=epoch,                global_step=global_step,                model=model,                optimizer=optimizer,                scheduler=scheduler,                scaler=scaler,                train_stats=train_stats,                valid_stats=valid_stats,                config=safe_config,            )        print("-" * 100)        print(            f"[Epoch {epoch}] "            f"train_loss={train_stats['train_loss']:.6f}, "            f"train_ppl={train_stats['train_ppl']:.4f}, "            f"train_token_acc={train_stats['train_token_acc']:.4f}, "            f"valid_loss={valid_stats['valid_loss']:.6f}, "            f"valid_ppl={valid_stats['valid_ppl']:.4f}, "            f"valid_token_acc={valid_stats['valid_token_acc']:.4f}, "            f"lr={train_stats['lr_last']:.8f}"        )        print(f"last checkpoint: {last_path}")        if best_path is not None:            print(f"best checkpoint: {best_path}")        if periodic_path is not None:            print(f"periodic checkpoint: {periodic_path}")        print("-" * 100)        tb_logger.flush()    tb_logger.close()
```

* * *

## `script/train_part/04_train_transformer_base.py`（新建，正式训练入口）

```
from __future__ import annotations"""功能：1. 构建正式训练所需的全部对象：   - vocab   - train_loader   - valid_loader   - model   - criterion   - optimizer   - scheduler   - scaler2. 启动 Transformer WMT14 英德训练。说明：1. 当前版本采用固定 batch_size + 流式 buffer shuffle。2. 这是“第一版正式训练入口”。3. 若后续要更贴论文 batching，可再升级成近似长度分桶 + token budget。"""import osimport torchfrom data.shared_vocab import SharedVocabfrom data.wmt14_bpe_dataset import build_bpe_dataloaderfrom nets.build_transformer import make_modelfrom train.fit import fitfrom utils.label_smoothing import LabelSmoothingLossfrom utils.noam_scheduler import build_transformer_optimizer_and_schedulerfrom utils.train_env import count_trainable_parameters, get_device, get_timestamp_str, seed_everythingdef build_config(device: torch.device, vocab_size: int) -> dict:    """    构建实验配置。    """    use_amp = device.type == "cuda"    config = {        "exp_name": "transformer_wmt14_en_de_base",        "seed": 42,        "device": str(device),        "use_amp": use_amp,        "data": {            "train_src": "data/wmt14_bpe_en_de/train.en",            "train_tgt": "data/wmt14_bpe_en_de/train.de",            "valid_src": "data/wmt14_bpe_en_de/valid.en",            "valid_tgt": "data/wmt14_bpe_en_de/valid.de",            "train_num_samples": 3927488,            "valid_num_samples": 3000,        },        "vocab": {            "vocab_json": "data/wmt14_vocab/vocab.json",            "vocab_size": vocab_size,            "pad_id": 0,            "bos_id": 1,            "eos_id": 2,            "unk_id": 3,        },        "model": {            "N": 6,            "d_model": 512,            "d_ff": 2048,            "h": 8,            "dropout": 0.1,            "share_embeddings": True,        },        "criterion": {            "type": "LabelSmoothingLoss",            "smoothing": 0.1,        },        "optimizer": {            "type": "Adam",            "beta1": 0.9,            "beta2": 0.98,            "eps": 1e-9,            "weight_decay": 0.0,        },        "scheduler": {            "type": "NoamLRScheduler",            "warmup_steps": 4000,            "factor": 1.0,        },        "train_loader": {            "batch_size": 16,            "num_workers": 2,            "pin_memory": device.type == "cuda",            "max_src_len": None,            "max_tgt_len": None,            "add_src_eos": True,            "skip_empty": False,            "shuffle_buffer_size": 10000,            "seed": 42,            "persistent_workers": True,            "prefetch_factor": 2,        },        "valid_loader": {            "batch_size": 16,            "num_workers": 0,            "pin_memory": device.type == "cuda",            "max_src_len": None,            "max_tgt_len": None,            "add_src_eos": True,            "skip_empty": False,            "shuffle_buffer_size": 0,            "seed": 42,            "persistent_workers": False,            "prefetch_factor": None,        },        "fit": {            "num_epochs": 30,            "grad_clip_norm": 1.0,            "train_log_interval": 100,            "histogram_interval": 1000,            "save_every_epochs": 1,            "valid_num_text_samples": 3,            "max_train_steps_per_epoch": None,            "max_valid_steps_per_epoch": None,        },    }    return configdef main() -> None:    device = get_device()    seed_everything(seed=42, deterministic=False)    vocab = SharedVocab.load("data/wmt14_vocab/vocab.json")    vocab_size = len(vocab)    config = build_config(device=device, vocab_size=vocab_size)    train_loader = build_bpe_dataloader(        src_path=config["data"]["train_src"],        tgt_path=config["data"]["train_tgt"],        vocab=vocab,        batch_size=config["train_loader"]["batch_size"],        num_workers=config["train_loader"]["num_workers"],        pin_memory=config["train_loader"]["pin_memory"],        max_src_len=config["train_loader"]["max_src_len"],        max_tgt_len=config["train_loader"]["max_tgt_len"],        add_src_eos=config["train_loader"]["add_src_eos"],        skip_empty=config["train_loader"]["skip_empty"],        shuffle_buffer_size=config["train_loader"]["shuffle_buffer_size"],        seed=config["train_loader"]["seed"],        num_samples=config["data"]["train_num_samples"],        persistent_workers=config["train_loader"]["persistent_workers"],        prefetch_factor=config["train_loader"]["prefetch_factor"],    )    valid_loader = build_bpe_dataloader(        src_path=config["data"]["valid_src"],        tgt_path=config["data"]["valid_tgt"],        vocab=vocab,        batch_size=config["valid_loader"]["batch_size"],        num_workers=config["valid_loader"]["num_workers"],        pin_memory=config["valid_loader"]["pin_memory"],        max_src_len=config["valid_loader"]["max_src_len"],        max_tgt_len=config["valid_loader"]["max_tgt_len"],        add_src_eos=config["valid_loader"]["add_src_eos"],        skip_empty=config["valid_loader"]["skip_empty"],        shuffle_buffer_size=config["valid_loader"]["shuffle_buffer_size"],        seed=config["valid_loader"]["seed"],        num_samples=config["data"]["valid_num_samples"],        persistent_workers=config["valid_loader"]["persistent_workers"],        prefetch_factor=config["valid_loader"]["prefetch_factor"],    )    model = make_model(        src_vocab=vocab_size,        tgt_vocab=vocab_size,        N=config["model"]["N"],        d_model=config["model"]["d_model"],        d_ff=config["model"]["d_ff"],        h=config["model"]["h"],        dropout=config["model"]["dropout"],        share_embeddings=config["model"]["share_embeddings"],    )    param_count = count_trainable_parameters(model)    print(f"模型可训练参数量: {param_count / 1e6:.2f} M")    criterion = LabelSmoothingLoss(        vocab_size=vocab_size,        pad_idx=vocab.pad_id,        smoothing=config["criterion"]["smoothing"],    )    optimizer, scheduler = build_transformer_optimizer_and_scheduler(        model=model,        d_model=config["model"]["d_model"],        warmup_steps=config["scheduler"]["warmup_steps"],        factor=config["scheduler"]["factor"],        beta1=config["optimizer"]["beta1"],        beta2=config["optimizer"]["beta2"],        eps=config["optimizer"]["eps"],        weight_decay=config["optimizer"]["weight_decay"],    )    use_amp = config["use_amp"]    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)    exp_dir = os.path.join(        "experiments",        "{0}_{1}".format(config["exp_name"], get_timestamp_str()),    )    fit(        model=model,        train_loader=train_loader,        valid_loader=valid_loader,        criterion=criterion,        optimizer=optimizer,        scheduler=scheduler,        device=device,        num_epochs=config["fit"]["num_epochs"],        output_dir=exp_dir,        config=config,        vocab=vocab,        scaler=scaler,        use_amp=use_amp,        grad_clip_norm=config["fit"]["grad_clip_norm"],        train_log_interval=config["fit"]["train_log_interval"],        histogram_interval=config["fit"]["histogram_interval"],        save_every_epochs=config["fit"]["save_every_epochs"],        valid_num_text_samples=config["fit"]["valid_num_text_samples"],        max_train_steps_per_epoch=config["fit"]["max_train_steps_per_epoch"],        max_valid_steps_per_epoch=config["fit"]["max_valid_steps_per_epoch"],        start_epoch=1,        global_step_init=0,        best_metric_init=None,    )if __name__ == "__main__":    main()
```

* * *

## `script/train_part/05_resume_train_transformer.py`（新建，断点续训入口）

```
from __future__ import annotations"""功能：1. 从已有 checkpoint 恢复训练。2. 恢复：   - model   - optimizer   - scheduler   - scaler   - epoch/global_step/best_metric3. 继续调用 fit() 完成后续训练。说明：1. 当前版本默认从 last.pth 恢复。2. 你可以手动改 resume_path。"""import osimport torchfrom data.shared_vocab import SharedVocabfrom data.wmt14_bpe_dataset import build_bpe_dataloaderfrom nets.build_transformer import make_modelfrom train.fit import fitfrom utils.label_smoothing import LabelSmoothingLossfrom utils.noam_scheduler import build_transformer_optimizer_and_schedulerfrom utils.resume_utils import load_checkpoint_for_resumefrom utils.train_env import count_trainable_parameters, get_device, seed_everythingdef main() -> None:    resume_path = "experiments/your_exp_name/checkpoints/last.pth"    device = get_device()    seed_everything(seed=42, deterministic=False)    checkpoint = torch.load(resume_path, map_location="cpu")    config = checkpoint["config"]    vocab = SharedVocab.load(config["vocab"]["vocab_json"])    vocab_size = len(vocab)    train_loader = build_bpe_dataloader(        src_path=config["data"]["train_src"],        tgt_path=config["data"]["train_tgt"],        vocab=vocab,        batch_size=config["train_loader"]["batch_size"],        num_workers=config["train_loader"]["num_workers"],        pin_memory=config["train_loader"]["pin_memory"],        max_src_len=config["train_loader"]["max_src_len"],        max_tgt_len=config["train_loader"]["max_tgt_len"],        add_src_eos=config["train_loader"]["add_src_eos"],        skip_empty=config["train_loader"]["skip_empty"],        shuffle_buffer_size=config["train_loader"]["shuffle_buffer_size"],        seed=config["train_loader"]["seed"],        num_samples=config["data"]["train_num_samples"],        persistent_workers=config["train_loader"]["persistent_workers"],        prefetch_factor=config["train_loader"]["prefetch_factor"],    )    valid_loader = build_bpe_dataloader(        src_path=config["data"]["valid_src"],        tgt_path=config["data"]["valid_tgt"],        vocab=vocab,        batch_size=config["valid_loader"]["batch_size"],        num_workers=config["valid_loader"]["num_workers"],        pin_memory=config["valid_loader"]["pin_memory"],        max_src_len=config["valid_loader"]["max_src_len"],        max_tgt_len=config["valid_loader"]["max_tgt_len"],        add_src_eos=config["valid_loader"]["add_src_eos"],        skip_empty=config["valid_loader"]["skip_empty"],        shuffle_buffer_size=config["valid_loader"]["shuffle_buffer_size"],        seed=config["valid_loader"]["seed"],        num_samples=config["data"]["valid_num_samples"],        persistent_workers=config["valid_loader"]["persistent_workers"],        prefetch_factor=config["valid_loader"]["prefetch_factor"],    )    model = make_model(        src_vocab=vocab_size,        tgt_vocab=vocab_size,        N=config["model"]["N"],        d_model=config["model"]["d_model"],        d_ff=config["model"]["d_ff"],        h=config["model"]["h"],        dropout=config["model"]["dropout"],        share_embeddings=config["model"]["share_embeddings"],    )    param_count = count_trainable_parameters(model)    print(f"模型可训练参数量: {param_count / 1e6:.2f} M")    criterion = LabelSmoothingLoss(        vocab_size=vocab_size,        pad_idx=vocab.pad_id,        smoothing=config["criterion"]["smoothing"],    )    optimizer, scheduler = build_transformer_optimizer_and_scheduler(        model=model,        d_model=config["model"]["d_model"],        warmup_steps=config["scheduler"]["warmup_steps"],        factor=config["scheduler"]["factor"],        beta1=config["optimizer"]["beta1"],        beta2=config["optimizer"]["beta2"],        eps=config["optimizer"]["eps"],        weight_decay=config["optimizer"]["weight_decay"],    )    use_amp = config["use_amp"]    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)    resume_info = load_checkpoint_for_resume(        checkpoint_path=resume_path,        model=model,        optimizer=optimizer,        scheduler=scheduler,        scaler=scaler,        map_location="cpu",    )    print("断点恢复成功。")    print(f"恢复 epoch = {resume_info['epoch']}")    print(f"恢复 global_step = {resume_info['global_step']}")    print(f"恢复 best_metric(valid_ppl) = {resume_info['best_metric']}")    fit(        model=model,        train_loader=train_loader,        valid_loader=valid_loader,        criterion=criterion,        optimizer=optimizer,        scheduler=scheduler,        device=device,        num_epochs=config["fit"]["num_epochs"],        output_dir=os.path.dirname(os.path.dirname(resume_path)),        config=config,        vocab=vocab,        scaler=scaler,        use_amp=use_amp,        grad_clip_norm=config["fit"]["grad_clip_norm"],        train_log_interval=config["fit"]["train_log_interval"],        histogram_interval=config["fit"]["histogram_interval"],        save_every_epochs=config["fit"]["save_every_epochs"],        valid_num_text_samples=config["fit"]["valid_num_text_samples"],        max_train_steps_per_epoch=config["fit"]["max_train_steps_per_epoch"],        max_valid_steps_per_epoch=config["fit"]["max_valid_steps_per_epoch"],        start_epoch=resume_info["epoch"] + 1,        global_step_init=resume_info["global_step"],        best_metric_init=resume_info["best_metric"],    )if __name__ == "__main__":    main()
```

* * *

## `script/train_part/06_test_train_pipeline_smoke.py`（新建，先做完整训练闭环 smoke test）

```
from __future__ import annotations"""功能：1. 用真实的 WMT14 BPE 数据 + 真实训练组件，   跑一个极小规模的训练闭环 smoke test。2. 验证：   - train_one_epoch   - validate_one_epoch   - fit   - csv/png/tb/checkpoint整体是否能协同工作。"""import osimport torchfrom data.shared_vocab import SharedVocabfrom data.wmt14_bpe_dataset import build_bpe_dataloaderfrom nets.build_transformer import make_modelfrom train.fit import fitfrom utils.label_smoothing import LabelSmoothingLossfrom utils.noam_scheduler import build_transformer_optimizer_and_schedulerfrom utils.train_env import get_device, seed_everythingdef main() -> None:    device = get_device()    seed_everything(seed=42, deterministic=False)    vocab = SharedVocab.load("data/wmt14_vocab/vocab.json")    vocab_size = len(vocab)    train_loader = build_bpe_dataloader(        src_path="data/wmt14_bpe_en_de/train.en",        tgt_path="data/wmt14_bpe_en_de/train.de",        vocab=vocab,        batch_size=8,        num_workers=0,        pin_memory=False,        max_src_len=64,        max_tgt_len=64,        add_src_eos=True,        skip_empty=False,        shuffle_buffer_size=1000,        seed=42,        num_samples=3927488,        persistent_workers=False,        prefetch_factor=None,    )    valid_loader = build_bpe_dataloader(        src_path="data/wmt14_bpe_en_de/valid.en",        tgt_path="data/wmt14_bpe_en_de/valid.de",        vocab=vocab,        batch_size=8,        num_workers=0,        pin_memory=False,        max_src_len=64,        max_tgt_len=64,        add_src_eos=True,        skip_empty=False,        shuffle_buffer_size=0,        seed=42,        num_samples=3000,        persistent_workers=False,        prefetch_factor=None,    )    model = make_model(        src_vocab=vocab_size,        tgt_vocab=vocab_size,        N=2,        d_model=256,        d_ff=1024,        h=4,        dropout=0.1,        share_embeddings=True,    )    criterion = LabelSmoothingLoss(        vocab_size=vocab_size,        pad_idx=vocab.pad_id,        smoothing=0.1,    )    optimizer, scheduler = build_transformer_optimizer_and_scheduler(        model=model,        d_model=256,        warmup_steps=4000,        factor=1.0,        beta1=0.9,        beta2=0.98,        eps=1e-9,        weight_decay=0.0,    )    scaler = torch.cuda.amp.GradScaler(enabled=False)    config = {        "exp_name": "transformer_train_pipeline_smoke",        "device": str(device),        "use_amp": False,        "model": {            "N": 2,            "d_model": 256,            "d_ff": 1024,            "h": 4,            "dropout": 0.1,            "share_embeddings": True,        },        "criterion": {            "type": "LabelSmoothingLoss",            "smoothing": 0.1,        },        "optimizer": {            "type": "Adam",            "beta1": 0.9,            "beta2": 0.98,            "eps": 1e-9,            "weight_decay": 0.0,        },        "scheduler": {            "type": "NoamLRScheduler",            "warmup_steps": 4000,            "factor": 1.0,        },        "vocab": {            "vocab_json": "data/wmt14_vocab/vocab.json",            "vocab_size": vocab_size,            "pad_id": vocab.pad_id,            "bos_id": vocab.bos_id,            "eos_id": vocab.eos_id,            "unk_id": vocab.unk_id,        },        "data": {            "train_src": "data/wmt14_bpe_en_de/train.en",            "train_tgt": "data/wmt14_bpe_en_de/train.de",            "valid_src": "data/wmt14_bpe_en_de/valid.en",            "valid_tgt": "data/wmt14_bpe_en_de/valid.de",            "train_num_samples": 3927488,            "valid_num_samples": 3000,        },    }    fit(        model=model,        train_loader=train_loader,        valid_loader=valid_loader,        criterion=criterion,        optimizer=optimizer,        scheduler=scheduler,        device=device,        num_epochs=2,        output_dir=os.path.join("experiments", "transformer_train_pipeline_smoke"),        config=config,        vocab=vocab,        scaler=scaler,        use_amp=False,        grad_clip_norm=1.0,        train_log_interval=10,        histogram_interval=50,        save_every_epochs=1,        valid_num_text_samples=2,        max_train_steps_per_epoch=20,        max_valid_steps_per_epoch=10,        start_epoch=1,        global_step_init=0,        best_metric_init=None,    )if __name__ == "__main__":    main()
```

* * *

## 你现在先做这几步

先不要直接开正式大训练，先按这个顺序来：

1.  先替换：
    
    -   `data/shared_vocab.py`
        
    -   `utils/noam_scheduler.py`
        
    -   `train/fit.py`
        
2.  新建：
    
    -   `utils/train_env.py`
        
    -   `utils/resume_utils.py`
        
    -   `script/train_part/04_train_transformer_base.py`
        
    -   `script/train_part/05_resume_train_transformer.py`
        
    -   `script/train_part/06_test_train_pipeline_smoke.py`
        
3.  先运行：
    

```
python -m script.train_part.06_test_train_pipeline_smoke
```

* * *

## 你跑完 `06_test_train_pipeline_smoke.py` 之后，把结果贴给我

我下一条继续给你：

-   如果报错，我帮你逐项修
    
-   如果 smoke test 通过，我继续给你：
    
    -   正式训练前的配置调整建议
        
    -   `build_transformer.py` / `Generator.py` 的最终核对
        
    -   以及后续 `BLEU / beam search / checkpoint averaging` 代码
        

  

这个是结果： PS D:\\1AAAAAstudy\\python\_base\\pytorch\\my\_github\_workspace\\transformer-pytorch-main> python -m script.train\_part.04\_test\_train\_pipeline\_smoke ==================================================================================================== \[Epoch 1/2\] 开始训练 ==================================================================================================== qt.qpa.window: SetProcessDpiAwarenessContext() failed: 操作成功完成。 Qt's default DPI awareness context is DPI\_AWARENESS\_CONTEXT\_PER\_MONITOR\_AWARE\_V2. If you know what you are doing, you can overwrite this default using qt.conf (https://doc.qt.io/qt-6/highdpi.html#configuring-windows). ---------------------------------------------------------------------------------------------------- \[Epoch 1\] train\_loss=10.601345, train\_ppl=40155.3703, train\_token\_acc=0.0000, valid\_loss=10.583060, valid\_ppl=39347.6965, valid\_token\_acc=0.0000, lr=0.00000494 last checkpoint: experiments\\transformer\_train\_pipeline\_smoke\\checkpoints\\last.pth best checkpoint: experiments\\transformer\_train\_pipeline\_smoke\\checkpoints\\best.pth periodic checkpoint: experiments\\transformer\_train\_pipeline\_smoke\\checkpoints\\model\_epoch\_001\_valid\_ppl\_39347.6965.pth ---------------------------------------------------------------------------------------------------- ==================================================================================================== \[Epoch 2/2\] 开始训练 ==================================================================================================== ---------------------------------------------------------------------------------------------------- \[Epoch 2\] train\_loss=10.566135, train\_ppl=38614.6956, train\_token\_acc=0.0002, valid\_loss=10.533642, valid\_ppl=37245.2548, valid\_token\_acc=0.0060, lr=0.00000988 last checkpoint: experiments\\transformer\_train\_pipeline\_smoke\\checkpoints\\last.pth best checkpoint: experiments\\transformer\_train\_pipeline\_smoke\\checkpoints\\best.pth periodic checkpoint: experiments\\transformer\_train\_pipeline\_smoke\\checkpoints\\model\_epoch\_002\_valid\_ppl\_37245.2548.pth ----------------------------------------------------------------------------------------------------

  

## `nets/utils/Generator.py`

```
from __future__ import annotations"""功能：1. Transformer 输出头。2. 将 Decoder 输出的隐状态映射到词表 logits。3. 为 shared embedding / generator weight tying 提供干净接口。说明：1. 默认 bias=False，更适合与 embedding.weight 做权重共享。2. forward 返回未归一化 logits，后续交给损失函数处理。"""import torchimport torch.nn as nnclass Generator(nn.Module):    """    Transformer 词表投影头。    入口：        x: (B, T, d_model)    出口：        logits: (B, T, vocab_size)    """    def __init__(self, d_model: int, vocab_size: int, bias: bool = False) -> None:        super().__init__()        self.proj = nn.Linear(d_model, vocab_size, bias=bias)    def forward(self, x: torch.Tensor) -> torch.Tensor:        """        前向传播。        参数：            x: 解码器最后一层隐状态，形状 (B, T, d_model)        返回：            logits: 形状 (B, T, vocab_size)        """        return self.proj(x)
```

* * *

## `nets/build_transformer.py`

```
from __future__ import annotations"""功能：1. 构建完整 Transformer 模型。2. 支持 shared source-target vocabulary 下的 embedding / generator 权重共享。3. 提供用于真实训练和 smoke test 的统一入口。重要说明：1. 当前保留 Pre-LN 风格，不改你的主干设计。2. 当 share_embeddings=True 时，要求 src_vocab == tgt_vocab。3. 训练阶段：   - model(src, tgt_input, src_mask, tgt_mask) 返回 decoder hidden states   - 再通过 model.generator(hidden_states) 得到 logits"""import copyimport torchimport torch.nn as nnfrom nets.utils.attention import MultiHeadedAttentionfrom nets.utils.PositionwiseFeedForward import PositionwiseFeedForwardfrom nets.utils.Generator import Generatorfrom nets.utils.encoder_decoder import (    Encoder,    Decoder,    EncoderLayer,    DecoderLayer,    EncoderDecoder,    Embeddings,)def tie_transformer_weights(model: EncoderDecoder) -> None:    """    将 Transformer 的三处权重绑定为同一组参数。    绑定对象：        1. src_embed.lut.weight        2. tgt_embed.lut.weight        3. generator.proj.weight    注意：        仅适用于 shared source-target vocabulary。    """    shared_weight = model.tgt_embed.lut.weight    model.src_embed.lut.weight = shared_weight    model.generator.proj.weight = shared_weightdef make_model(    src_vocab: int,    tgt_vocab: int,    N: int = 6,    d_model: int = 512,    d_ff: int = 2048,    h: int = 8,    dropout: float = 0.1,    share_embeddings: bool = True,) -> EncoderDecoder:    """    构建完整 Transformer 模型。    参数：        src_vocab: 源语言词表大小        tgt_vocab: 目标语言词表大小        N: Encoder / Decoder 层数        d_model: 模型维度        d_ff: FFN 中间层维度        h: 多头注意力头数        dropout: dropout        share_embeddings: 是否共享 src/tgt embedding 以及 generator 权重    返回：        EncoderDecoder 实例    """    if share_embeddings and src_vocab != tgt_vocab:        raise ValueError(            "当 share_embeddings=True 时，src_vocab 和 tgt_vocab 必须一致。"        )    c = copy.deepcopy    attn = MultiHeadedAttention(h, d_model, dropout)    ff = PositionwiseFeedForward(d_model, d_ff, dropout)    model = EncoderDecoder(        encoder=Encoder(            EncoderLayer(d_model, c(attn), c(ff), dropout),            N,        ),        decoder=Decoder(            DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout),            N,        ),        src_embed=Embeddings(d_model, src_vocab, dropout),        tgt_embed=Embeddings(d_model, tgt_vocab, dropout),        generator=Generator(d_model, tgt_vocab, bias=False),    )    if share_embeddings:        tie_transformer_weights(model)    for p in model.parameters():        if p.dim() > 1:            nn.init.xavier_uniform_(p)    return modeldef test_factory_method() -> None:    """    微型模型构建测试。    """    from data.batch import Seq2SeqBatch    src_vocab_size = 5000    tgt_vocab_size = 5000    batch_size = 3    pad_idx = 0    bos_idx = 1    eos_idx = 2    model = make_model(        src_vocab=src_vocab_size,        tgt_vocab=tgt_vocab_size,        N=2,        d_model=256,        d_ff=1024,        h=4,        dropout=0.1,        share_embeddings=True,    )    src_tokens = torch.tensor([        [11, 12, 13, 14, 15, pad_idx, pad_idx],        [21, 22, 23, 24, pad_idx, pad_idx, pad_idx],        [31, 32, 33, 34, 35, 36, 37],    ], dtype=torch.long)    tgt_tokens = torch.tensor([        [bos_idx, 101, 102, 103, eos_idx, pad_idx, pad_idx],        [bos_idx, 201, 202, eos_idx, pad_idx, pad_idx, pad_idx],        [bos_idx, 301, 302, 303, 304, eos_idx, pad_idx],    ], dtype=torch.long)    batch = Seq2SeqBatch.from_tensors(        src=src_tokens,        tgt=tgt_tokens,        pad_idx=pad_idx,    )    src_ptr = model.src_embed.lut.weight.data_ptr()    tgt_ptr = model.tgt_embed.lut.weight.data_ptr()    gen_ptr = model.generator.proj.weight.data_ptr()    assert src_ptr == tgt_ptr == gen_ptr, "权重共享失败。"    hidden_states = model(        batch.src,        batch.tgt_input,        batch.src_mask,        batch.tgt_mask,    )    logits = model.generator(hidden_states)    assert hidden_states.shape == (batch_size, batch.tgt_input.size(1), 256)    assert logits.shape == (batch_size, batch.tgt_input.size(1), tgt_vocab_size)    print("make_model 测试通过。")if __name__ == "__main__":    test_factory_method()
```

* * *

## `infer/beam_search.py`

```
from __future__ import annotations"""功能：1. 提供 greedy decode。2. 提供 beam search decode。3. 提供 BPE 文本恢复辅助函数。4. 用于验证阶段文本样例和后续 BLEU 评估。说明：1. 当前 beam search 以“单样本 beam search + 外层 batch 循环”为主，代码更清晰。2. 第一版优先保证正确性与可读性，不强行追求最极致推理吞吐。3. 长度惩罚使用论文风格 alpha，可设为 0.6。"""import mathfrom dataclasses import dataclassfrom typing import Callable, Dict, List, Optional, Sequence, Tupleimport torchfrom data.batch import make_tgt_mask@dataclassclass BeamHypothesis:    """    单条 beam 假设。    """    tokens: List[int]    logprob_sum: float    finished: bool    def score(self, alpha: float) -> float:        """        带长度惩罚的打分。        length penalty:            lp = ((5 + len) / 6) ^ alpha            score = logprob_sum / lp        """        length = max(len(self.tokens), 1)        lp = ((5.0 + float(length)) / 6.0) ** alpha        return self.logprob_sum / lpdef ids_to_bpe_string(    ids: Sequence[int],    id_to_token_func: Callable[[int], str],    pad_id: int,    bos_id: int,    eos_id: int,) -> str:    """    将 id 序列恢复成 BPE 文本，并去掉特殊 token。    """    tokens: List[str] = []    for idx in ids:        if idx == pad_id:            continue        if idx == bos_id:            continue        if idx == eos_id:            break        tokens.append(id_to_token_func(idx))    return " ".join(tokens).strip()def bpe_to_text(bpe_text: str) -> str:    """    将 subword-nmt 风格的 BPE 文本还原成普通 token 文本。    """    return bpe_text.replace("@@ ", "").strip()@torch.no_grad()def greedy_decode_batch(    model: torch.nn.Module,    src: torch.Tensor,    src_mask: torch.Tensor,    bos_id: int,    eos_id: int,    pad_id: int,    max_len: int,) -> torch.Tensor:    """    batch greedy decode。    输入：        src: (B, S)        src_mask: (B, 1, S)        bos_id/eos_id/pad_id: 特殊 token id        max_len: 最大解码长度    输出：        ys: (B, T_pred)    """    memory = model.encode(src, src_mask)    batch_size = src.size(0)    ys = torch.full(        (batch_size, 1),        fill_value=bos_id,        dtype=torch.long,        device=src.device,    )    finished = torch.zeros(batch_size, dtype=torch.bool, device=src.device)    for _ in range(max_len - 1):        tgt_mask = make_tgt_mask(ys, pad_idx=pad_id)        out = model.decode(memory, src_mask, ys, tgt_mask)        logits = model.generator(out[:, -1, :])  # (B, V)        next_token = logits.argmax(dim=-1, keepdim=True)  # (B, 1)        ys = torch.cat([ys, next_token], dim=1)        finished = finished | next_token.squeeze(1).eq(eos_id)        if finished.all():            break    return ys@torch.no_grad()def beam_search_single(    model: torch.nn.Module,    src: torch.Tensor,    src_mask: torch.Tensor,    bos_id: int,    eos_id: int,    pad_id: int,    beam_size: int = 4,    max_len: int = 100,    alpha: float = 0.6,) -> List[int]:    """    单样本 beam search。    输入：        src: (1, S)        src_mask: (1, 1, S)        beam_size: beam 宽度        max_len: 最大长度        alpha: 长度惩罚系数    输出：        best hypothesis 的 token id 列表（含 BOS，通常也含 EOS）    """    if src.size(0) != 1:        raise ValueError("beam_search_single 要求 src 的 batch_size 必须为 1。")    memory = model.encode(src, src_mask)    beams: List[BeamHypothesis] = [        BeamHypothesis(tokens=[bos_id], logprob_sum=0.0, finished=False)    ]    for _ in range(max_len - 1):        all_candidates: List[BeamHypothesis] = []        all_finished = all(b.finished for b in beams)        if all_finished:            break        for beam in beams:            if beam.finished:                all_candidates.append(beam)                continue            ys = torch.tensor(                [beam.tokens],                dtype=torch.long,                device=src.device,            )  # (1, T)            tgt_mask = make_tgt_mask(ys, pad_idx=pad_id)            out = model.decode(memory, src_mask, ys, tgt_mask)            logits = model.generator(out[:, -1, :])       # (1, V)            log_probs = torch.log_softmax(logits, dim=-1) # (1, V)            topk_log_probs, topk_ids = torch.topk(log_probs, k=beam_size, dim=-1)            for k in range(beam_size):                token_id = int(topk_ids[0, k].item())                token_logprob = float(topk_log_probs[0, k].item())                new_tokens = beam.tokens + [token_id]                new_logprob_sum = beam.logprob_sum + token_logprob                new_finished = (token_id == eos_id)                all_candidates.append(                    BeamHypothesis(                        tokens=new_tokens,                        logprob_sum=new_logprob_sum,                        finished=new_finished,                    )                )        all_candidates.sort(key=lambda x: x.score(alpha), reverse=True)        beams = all_candidates[:beam_size]    best_beam = max(beams, key=lambda x: x.score(alpha))    return best_beam.tokens@torch.no_grad()def beam_search_batch(    model: torch.nn.Module,    src: torch.Tensor,    src_mask: torch.Tensor,    bos_id: int,    eos_id: int,    pad_id: int,    beam_size: int = 4,    max_len: int = 100,    alpha: float = 0.6,) -> List[List[int]]:    """    batch 外层循环版 beam search。    输入：        src: (B, S)        src_mask: (B, 1, S)    输出：        长度为 B 的列表，每个元素是一条解码结果 token 序列    """    batch_size = src.size(0)    outputs: List[List[int]] = []    for i in range(batch_size):        one_src = src[i:i + 1]        one_src_mask = src_mask[i:i + 1]        pred_ids = beam_search_single(            model=model,            src=one_src,            src_mask=one_src_mask,            bos_id=bos_id,            eos_id=eos_id,            pad_id=pad_id,            beam_size=beam_size,            max_len=max_len,            alpha=alpha,        )        outputs.append(pred_ids)    return outputs
```

* * *

## `eval/bleu_eval.py`

```
from __future__ import annotations"""功能：1. 计算机器翻译 BLEU。2. 支持 greedy / beam 两种解码方式。3. 支持：   - 直接从 valid/test DataLoader 评估   - 记录少量样例4. 尽量兼容没有安装 sacrebleu 的环境：   - 优先使用 sacrebleu   - 若未安装，则抛出明确提示说明：1. 推荐后续正式报告 BLEU 时使用 sacrebleu。2. 输入参考文本与预测文本均先做 BPE 去除：   bpe_text.replace("@@ ", "")"""from typing import Dict, List, Optional, Tupleimport torchfrom infer.beam_search import (    bpe_to_text,    greedy_decode_batch,    beam_search_batch,    ids_to_bpe_string,)def _require_sacrebleu():    try:        import sacrebleu  # type: ignore        return sacrebleu    except ImportError as e:        raise ImportError(            "未安装 sacrebleu。请先执行：pip install sacrebleu"        ) from e@torch.no_grad()def evaluate_bleu(    model: torch.nn.Module,    data_loader,    vocab,    device: torch.device,    decode_mode: str = "beam",    beam_size: int = 4,    alpha: float = 0.6,    max_decode_extra_len: int = 50,    max_steps: Optional[int] = None,    return_examples: int = 5,) -> Dict:    """    在一个数据集上评估 BLEU。    参数：        model: Transformer 模型        data_loader: valid/test DataLoader        vocab: SharedVocab        device: 设备        decode_mode: "greedy" 或 "beam"        beam_size: beam 宽度        alpha: 长度惩罚系数        max_decode_extra_len: 最大额外生成长度        max_steps: 仅用于调试，限制评估步数        return_examples: 返回多少条文本样例    返回：        dict:            - bleu            - decode_mode            - beam_size            - alpha            - num_sentences            - examples    """    sacrebleu = _require_sacrebleu()    if decode_mode not in {"greedy", "beam"}:        raise ValueError("decode_mode 只能是 'greedy' 或 'beam'。")    model.eval()    predictions: List[str] = []    references: List[str] = []    examples: List[Dict[str, str]] = []    for step_idx, batch in enumerate(data_loader, start=1):        if max_steps is not None and step_idx > max_steps:            break        batch = batch.to(device)        max_len = int(batch.src.size(1) + max_decode_extra_len)        if decode_mode == "greedy":            pred_ids_batch = greedy_decode_batch(                model=model,                src=batch.src,                src_mask=batch.src_mask,                bos_id=vocab.bos_id,                eos_id=vocab.eos_id,                pad_id=vocab.pad_id,                max_len=max_len,            )            pred_ids_list = [pred_ids_batch[i].tolist() for i in range(pred_ids_batch.size(0))]        else:            pred_ids_list = beam_search_batch(                model=model,                src=batch.src,                src_mask=batch.src_mask,                bos_id=vocab.bos_id,                eos_id=vocab.eos_id,                pad_id=vocab.pad_id,                beam_size=beam_size,                max_len=max_len,                alpha=alpha,            )        for i in range(batch.src.size(0)):            src_bpe = ids_to_bpe_string(                ids=batch.src[i].tolist(),                id_to_token_func=vocab.id2token,                pad_id=vocab.pad_id,                bos_id=vocab.bos_id,                eos_id=vocab.eos_id,            )            ref_bpe = ids_to_bpe_string(                ids=batch.tgt_y[i].tolist(),                id_to_token_func=vocab.id2token,                pad_id=vocab.pad_id,                bos_id=vocab.bos_id,                eos_id=vocab.eos_id,            )            pred_bpe = ids_to_bpe_string(                ids=pred_ids_list[i],                id_to_token_func=vocab.id2token,                pad_id=vocab.pad_id,                bos_id=vocab.bos_id,                eos_id=vocab.eos_id,            )            src_text = bpe_to_text(src_bpe)            ref_text = bpe_to_text(ref_bpe)            pred_text = bpe_to_text(pred_bpe)            predictions.append(pred_text)            references.append(ref_text)            if len(examples) < return_examples:                examples.append(                    {                        "source": src_text,                        "reference": ref_text,                        "prediction": pred_text,                    }                )    bleu = sacrebleu.corpus_bleu(predictions, [references])    return {        "bleu": float(bleu.score),        "decode_mode": decode_mode,        "beam_size": int(beam_size),        "alpha": float(alpha),        "num_sentences": len(predictions),        "examples": examples,    }
```

* * *

## `utils/checkpoint_average.py`

```
from __future__ import annotations"""功能：1. 对多个 checkpoint 的 model_state_dict 做参数平均。2. 生成一个新的 averaged checkpoint。3. 用于后续贴近原论文的 checkpoint averaging。说明：1. 当前只平均 model_state_dict。2. 对非浮点 tensor，直接使用第一个 checkpoint 的值。3. 适合 average 最近 K 个周期性 checkpoint。"""import osfrom collections import OrderedDictfrom typing import Dict, Listimport torchdef average_checkpoints(    checkpoint_paths: List[str],    out_path: str,    map_location: str = "cpu",) -> str:    """    平均多个 checkpoint 的 model_state_dict。    参数：        checkpoint_paths: 待平均的 checkpoint 路径列表        out_path: 输出路径        map_location: 加载设备    返回：        out_path    """    if len(checkpoint_paths) == 0:        raise ValueError("checkpoint_paths 不能为空。")    checkpoints = [torch.load(p, map_location=map_location) for p in checkpoint_paths]    model_state_dicts = [ckpt["model_state_dict"] for ckpt in checkpoints]    avg_state_dict = OrderedDict()    first_state = model_state_dicts[0]    for key in first_state.keys():        tensors = [state[key] for state in model_state_dicts]        if torch.is_floating_point(tensors[0]):            avg_tensor = tensors[0].clone().float()            for t in tensors[1:]:                avg_tensor += t.float()            avg_tensor /= float(len(tensors))            avg_state_dict[key] = avg_tensor.to(dtype=tensors[0].dtype)        else:            avg_state_dict[key] = tensors[0].clone()    out_checkpoint = dict(checkpoints[0])    out_checkpoint["model_state_dict"] = avg_state_dict    out_checkpoint["averaged_from"] = checkpoint_paths    os.makedirs(os.path.dirname(out_path), exist_ok=True)    torch.save(out_checkpoint, out_path)    return out_path
```

* * *

## `script/eval_part/01_average_checkpoints.py`

```
from __future__ import annotations"""功能：1. 对若干周期性 checkpoint 做平均。2. 生成 averaged.pth。使用方式：1. 手动修改 checkpoint_paths2. 运行脚本"""from utils.checkpoint_average import average_checkpointsdef main() -> None:    checkpoint_paths = [        "experiments/your_exp_name/checkpoints/model_epoch_026_valid_ppl_4.8123.pth",        "experiments/your_exp_name/checkpoints/model_epoch_027_valid_ppl_4.7510.pth",        "experiments/your_exp_name/checkpoints/model_epoch_028_valid_ppl_4.7021.pth",        "experiments/your_exp_name/checkpoints/model_epoch_029_valid_ppl_4.6880.pth",        "experiments/your_exp_name/checkpoints/model_epoch_030_valid_ppl_4.6505.pth",    ]    out_path = "experiments/your_exp_name/checkpoints/averaged_last5.pth"    avg_path = average_checkpoints(        checkpoint_paths=checkpoint_paths,        out_path=out_path,        map_location="cpu",    )    print("平均 checkpoint 完成：", avg_path)if __name__ == "__main__":    main()
```

* * *

## `script/eval_part/02_eval_bleu_greedy_or_beam.py`

```
from __future__ import annotations"""功能：1. 加载某个 checkpoint。2. 在 valid 或 test 上评估 BLEU。3. 支持 greedy / beam 两种模式。"""import torchfrom data.shared_vocab import SharedVocabfrom data.wmt14_bpe_dataset import build_bpe_dataloaderfrom eval.bleu_eval import evaluate_bleufrom nets.build_transformer import make_modelfrom utils.train_env import get_devicedef main() -> None:    checkpoint_path = "experiments/your_exp_name/checkpoints/best.pth"    split = "valid"   # 可改为 "test"    decode_mode = "beam"  # 可改为 "greedy"    device = get_device()    checkpoint = torch.load(checkpoint_path, map_location="cpu")    config = checkpoint["config"]    vocab = SharedVocab.load(config["vocab"]["vocab_json"])    vocab_size = len(vocab)    if split == "valid":        src_path = config["data"]["valid_src"]        tgt_path = config["data"]["valid_tgt"]        num_samples = config["data"]["valid_num_samples"]    else:        src_path = "data/wmt14_bpe_en_de/test.en"        tgt_path = "data/wmt14_bpe_en_de/test.de"        num_samples = 3003    data_loader = build_bpe_dataloader(        src_path=src_path,        tgt_path=tgt_path,        vocab=vocab,        batch_size=16,        num_workers=0,        pin_memory=(device.type == "cuda"),        max_src_len=None,        max_tgt_len=None,        add_src_eos=True,        skip_empty=False,        shuffle_buffer_size=0,        seed=42,        num_samples=num_samples,        persistent_workers=False,        prefetch_factor=None,    )    model = make_model(        src_vocab=vocab_size,        tgt_vocab=vocab_size,        N=config["model"]["N"],        d_model=config["model"]["d_model"],        d_ff=config["model"]["d_ff"],        h=config["model"]["h"],        dropout=config["model"]["dropout"],        share_embeddings=config["model"]["share_embeddings"],    )    model.load_state_dict(checkpoint["model_state_dict"])    model = model.to(device)    model.eval()    result = evaluate_bleu(        model=model,        data_loader=data_loader,        vocab=vocab,        device=device,        decode_mode=decode_mode,        beam_size=4,        alpha=0.6,        max_decode_extra_len=50,        max_steps=None,        return_examples=5,    )    print("=" * 100)    print("BLEU 评估结果")    print("=" * 100)    print("split =", split)    print("decode_mode =", result["decode_mode"])    print("beam_size =", result["beam_size"])    print("alpha =", result["alpha"])    print("num_sentences =", result["num_sentences"])    print("BLEU =", result["bleu"])    print("-" * 100)    for i, sample in enumerate(result["examples"]):        print(f"[Sample {i}]")        print("SRC :", sample["source"])        print("REF :", sample["reference"])        print("PRED:", sample["prediction"])        print("-" * 100)if __name__ == "__main__":    main()
```

* * *

## 你现在继续按这个顺序做

先别碰 BLEU 和 checkpoint averaging，先验证训练主链增强版代码没问题。

先运行：

```
python -m script.train_part.06_test_train_pipeline_smoke
```

如果这个通过，再运行：

```
python -m script.train_part.04_train_transformer_base
```

先用小一点的 `batch_size` 和 `num_workers` 跑，确认正式训练入口也稳定。

把 `06_test_train_pipeline_smoke` 的结果贴给我，我下一条继续输出。

  

metrics

epoch

global\_step

train\_loss

train\_nll\_loss

1

1000

7.8543169893784945

7.425019796429375

2

2000

6.294133743420147

5.623216126865627

3

3000

5.595103376191846

4.826069420194222

4

4000

5.081653450518462

4.235591340202109

5

5000

4.640978661342726

3.7265085824202977

这是我全量训练，在3080上，我设定的参数为： "src\_token\_budget": 2048, "tgt\_token\_budget": 2048, 请评估一下当前的结果

  

我先读一下你上传的 `metrics.csv`，看训练/验证曲线、学习率、ppl 和 token\_acc 的变化，再给你一个偏“复现论文视角”的判断。

我先把 `metrics.csv` 读出来，重点看一下 loss、ppl、token\_acc、学习率随 epoch 的变化，再结合你现在的 `2048/2048 token budget` 给你一个更具体的判断。

## 结论先说

你这次训练结果，从“**训练链是否正确**”这个角度看，**是正常的，而且说明你的训练系统已经真正跑起来了**。

但从“**当前这一版结果是否已经达到一个健康的长期训练状态**”来看，我的判断是：

> **前 10~13 个 epoch 是有效学习阶段，之后出现了明显的泛化退化。**
> 
> 更具体地说：**best checkpoint 应该看第 11 个 epoch 附近，而不是最后一个 epoch。**

* * *

# 一、我先直接解读你当前的核心现象

你这份 `metrics.csv` 最关键的几个信号是：

## 1\. 训练是正常收敛的

你的训练集指标一路变好：

-   `train_loss` 从大约 `7.85` 持续降到 `2.13`
    
-   `train_ppl` 从大约 `1677` 持续降到 `2.23`
    
-   `train_token_acc` 从接近 `0` 持续升到 `0.8277`
    

这说明：

-   模型前向、反向、loss、optimizer、scheduler 这整条训练链没有明显 bug
    
-   label smoothing、Noam lr、shared vocab、mask 这些关键组件大体是工作的
    
-   不是“根本训不动”的状态
    

也就是说：

> **你的工程链路已经不是 demo 级别了，而是真正在学习。**

* * *

## 2\. 验证集在第 11 个 epoch 左右达到最佳

你当前最关键的验证指标里：

-   `best valid_ppl = 152.68`
    
-   `best valid_loss = 5.8455`
    
-   最好点都出现在 **epoch 11**
    

这非常重要。

这说明：

> **模型在前 11 个 epoch 左右确实学到了最有泛化价值的参数区域。**

* * *

## 3\. 从第 12 个 epoch 以后，出现了明显的“训练继续变好，但验证变差”

这是你当前结果里最核心的现象。

### 训练侧

继续稳步变好：

-   `train_loss` 一路下降
    
-   `train_ppl` 一路下降
    
-   `train_token_acc` 一路上升
    

### 验证侧

在 epoch 11 之后：

-   `valid_ppl` 逐步变差  
    从 `152.68` 一路走到最后的 `241.23`
    
-   `valid_loss` 也整体变差
    
-   `valid_token_acc` 还在很小幅度上升或波动
    

这说明：

> **模型越来越会“记住训练集”，但在验证集上的概率分布质量变差了。**

这是很典型的：

-   过拟合
    
-   或者更准确地说：**校准变差 + 泛化退化**
    

* * *

# 二、为什么会出现“valid token\_acc 还在涨，但 valid ppl 却在恶化”

这个现象你要特别理解一下，因为它在 NLP 里很常见。

* * *

## 1\. `token_acc` 和 `ppl` 看的不是同一件事

### `token_acc` 看的是：

“预测对没对”。

它只关心：

-   argmax 选出来的 token 对不对
    

### `ppl` / `nll_loss` 看的是：

“模型给真实 token 的概率到底给得好不好”。

它不只关心对错，还关心：

-   你是不是把正确答案压到了很高概率
    
-   你是不是对错误答案变得过度自信
    

* * *

## 2\. 所以会出现这种情况

模型后期可能会变成：

-   预测出来的 token **略微更常对**
    
-   但一旦错，就错得**特别自信**
    
-   或者即便对了，分布也变得更尖、更不稳定
    

于是：

-   `token_acc` 还可能缓慢上涨
    
-   但 `nll / ppl` 已经开始恶化
    

这就是你现在看到的现象。

* * *

## 3\. 这对当前阶段意味着什么

对于 Transformer 翻译训练的第一阶段来说，我会更看重：

-   `valid_ppl`
    
-   `valid_loss`
    

而不是只看 `valid_token_acc`

因为：

> **你最终要的是翻译质量，而不是单纯 token argmax 命中率。**

尤其后面你还要做：

-   greedy decode
    
-   beam search
    
-   BLEU
    

这些都更依赖模型的概率分布质量，而不只是 token\_acc。

* * *

# 三、关于你设定的 `src_token_budget=2048, tgt_token_budget=2048`，我怎么评价

## 我的判断：**能训，但偏保守，而且和论文口径差得比较远**

论文原文在 WMT14 英德任务里写的是：

-   训练 batch 按近似长度组织
    
-   每个 batch 大约包含：
    
    -   `25000` source tokens
        
    -   `25000` target tokens
        

你现在设的是：

-   `2048`
    
-   `2048`
    

这相当于论文 token 预算的：

$204825000≈0.082\frac{2048}{25000} \approx 0.082250002048​≈0.082$

也就是大约只有论文单侧 token budget 的 **8.2%**。

* * *

## 这意味着什么

### 好处

-   更稳
    
-   更容易在 3080 上先跑通
    
-   显存压力更可控
    
-   调试阶段非常合适
    

### 代价

-   每个 step 看到的 token 太少
    
-   梯度噪声更大
    
-   优化会更抖
    
-   单位 epoch 的统计意义更弱
    
-   对 Noam schedule 来说，你的 “step 数语义” 和论文也更不一样
    

* * *

## 一个更直观的类比

论文里的 batch 更像：

-   每次开会，来 25000 个英文 token 和 25000 个德文 token，一次讨论很多内容
    

你现在的 batch 更像：

-   每次只来 2048 + 2048 个 token，小会开得很频繁，但单次信息量明显更少
    

这样当然可以学，但：

> **每一步的梯度代表性会更弱。**

* * *

# 四、所以我对你当前结果的整体评价

## 可以分成三层看

* * *

## 第一层：工程正确性

**通过。**

这是一个很重要的结论。

你已经证明：

-   数据准备没问题
    
-   DataLoader 没问题
    
-   词表系统没问题
    
-   target shifted right 没问题
    
-   mask 没问题
    
-   label smoothing 没问题
    
-   Noam lr 没问题
    
-   训练循环没问题
    
-   checkpoint / 日志系统也在工作
    

这一步价值非常大。

* * *

## 第二层：训练行为

**前半段健康，后半段明显过拟合。**

最健康的区间在：

-   **epoch 8 ~ 13**
    

其中最好的验证 PPL 在：

-   **epoch 11**
    

所以现在最直接的训练判断是：

> **不要拿最后一个 epoch 当结果。**
> 
> 你现在真正该看的，是 `best.pth`，而且最好围绕 epoch 9~13 那一段去做进一步评估。

* * *

## 第三层：论文复现程度

**还不能说接近论文结果。**

原因不是你代码坏，而是因为现在还有几个差距：

1.  你的 token budget 明显小于论文设定
    
2.  你还没做真正的：
    
    -   beam search 正式评估
        
    -   BLEU
        
    -   checkpoint averaging
        
3.  你现在的 batching 还不是论文那种“按近似长度 + 25k token budget”
    
4.  还没有真正拿 `newstest2014` 上的 BLEU 去对齐论文表格
    

所以：

> 你现在最准确的定位是：
> 
> **“训练系统已经成立，模型已经有效学习，但还处在论文复现的中间阶段。”**

* * *

# 五、你现在最应该做的，不是继续盲目加 epoch

这一点我要说得很明确。

从你的曲线看：

-   第 11 epoch 已经是 valid ppl 最优
    
-   后面越训越差
    

所以再继续沿着当前配置训更多 epoch，**大概率收益不大，甚至更差**。

* * *

## 你现在更合理的动作有三个

### 1\. 立刻以 `best.pth` 为主，不要盯着 `last.pth`

这个是当前最直接的结论。

* * *

### 2\. 开始做 BLEU 评估

你现在最缺的不是继续看 loss，而是：

-   `best.pth`
    
-   `epoch 9~13` 附近的几个 checkpoint
    
-   在 valid / test 上做：
    
    -   greedy
        
    -   beam size = 4
        
    -   length penalty = 0.6
        

因为论文最终报告看的是 `newstest2014` BLEU，而且推理策略明确包括 beam search 和长度惩罚。

* * *

### 3\. 下一轮训练先优先尝试“增大有效 token batch”

我更建议你做的是：

-   **不要先加 epoch**
    
-   **先加大每次更新的有效 token 数**
    

* * *

# 六、关于 `2048/2048`，我下一轮会怎么调

## 我建议你下一轮优先尝试这两种路线之一

* * *

## 路线 A：直接把 token budget 往上提

例如尝试：

-   `3072 / 3072`
    
-   或 `4096 / 4096`
    

如果 3080 能扛住，就先提。

### 为什么

因为你当前结果更像是：

-   模型会学
    
-   但 batch 太小，训练噪声偏大
    
-   很早就开始泛化退化
    

更大的 token budget 往往会让：

-   梯度更稳定
    
-   验证曲线更平滑
    
-   最优点来得更晚一些
    

* * *

## 路线 B：保持 `2048/2048`，但做 gradient accumulation

例如：

-   每 4 个 mini-step 再更新一次参数
    

那等效有效 token 数就更大。

### 为什么这个方案很值得考虑

因为：

-   不一定需要马上吃更多显存
    
-   但可以让优化器看到更大的“等效 batch”
    

这对 Transformer 这类模型通常是有帮助的。

* * *

# 七、我对你当前这版训练结果的明确判断

我用一句话总结：

> **这是一版“健康的第一轮正式训练结果”。**
> 
> 它成功证明了训练系统没问题，但同时也很清楚地告诉你：  
> **当前配置下，第 11 个 epoch 左右已经是泛化峰值，后面在过拟合。**

所以：

-   从工程角度，这是成功的
    
-   从论文复现角度，这是中间结果
    
-   从下一步实验设计角度，你现在最该做的是：
    
    -   `best checkpoint` 评估 BLEU
        
    -   checkpoint averaging
        
    -   增大有效 token batch
        

* * *

# 八、我给你的下一步建议，按优先级排序

## 第一优先级

**马上评估 `best.pth` 的 BLEU**

不是继续看最后 epoch 的 loss。

* * *

## 第二优先级

**对 epoch 9~13 附近的 checkpoint 做平均**

比如：

-   9,10,11,12,13
    
-   或者 10,11,12
    

因为你当前最优 valid ppl 区间就在这里。  
不要平均最后 5 个，因为你最后 5 个已经明显退化了。

* * *

## 第三优先级

**下一轮训练提高有效 token batch**

优先尝试：

-   更高 token budget
    
-   或 gradient accumulation
    

* * *

# 九、最后给你一个非常直接的结论

## 当前结果值不值得继续往后做？

**值得。**

## 当前结果是不是已经说明“训练系统正确”？

**是。**

## 当前结果是不是说明“继续训更多 epoch 就会更好”？

**不是。**

## 当前最该看的 checkpoint 是哪个？

**epoch 11 附近的 best checkpoint。**

* * *

下一条最合理的动作是：  
**我直接继续给你补全 BLEU 评估和 checkpoint averaging 的使用方式，并告诉你如何基于你当前这批 checkpoint 选平均区间。**

  

是不是用5090 32G做实验会更合适？甚至用pro6000 96G？

  

我先核对一下 5090 32G 和 RTX PRO 6000 96G 的官方规格，然后按你现在这个 Transformer 实验场景给结论。

## 直接结论

**是，5090 32G 会更合适。**  
**PRO 6000 96G 当然更强，但对你当前这个 WMT14 Transformer 实验来说，大概率属于“能用，但明显过配”。**

我把判断拆开说。

* * *

## 1\. 对你当前这个实验，5090 32G 是很合理的升级目标

你现在做的是：

-   WMT14 英德翻译
    
-   Transformer base 路线
    
-   目前 `src_token_budget=2048, tgt_token_budget=2048`
    
-   并且从你前面的训练结果看，训练链已经成立，但 token budget 还偏小，离论文里“每个 batch 大约 25000 source tokens + 25000 target tokens”的量级差得很远。
    

RTX 5090 官方规格里明确写了：

-   **32 GB GDDR7**
    
-   **1792 GB/s** 内存带宽
    
-   **21760 CUDA cores**。NVIDIA+1
    

这对你当前场景的意义非常直接：

-   **显存从 3080 跳到 32GB**，你可以明显提高 token budget，或者加大 batch/accumulation。NVIDIA+1
    
-   **带宽也很高**，对 Transformer 这类 attention + FFN 密集计算比较友好。NVIDIA+1
    

所以如果你的目标是：

-   把 `2048/2048` 往上推
    
-   更接近论文的有效 batch 语义
    
-   做更多 smoke test / 正式训练 / beam search / BLEU 评估
    
-   同时还兼顾你其他 CV 任务
    

那么：

> **5090 32G 是很合适的主力卡。**

* * *

## 2\. 5090 对你最大的价值，不只是“更快”，而是“更容易把训练设定调到合理区间”

你当前最主要的问题，不是“模型不会学”，而是：

-   训练系统已经会学了
    
-   但有效 token batch 偏小
    
-   结果表现出较早的泛化退化
    

这时候更大的显存，真正带来的收益通常是：

-   可以把 `src/tgt token budget` 提高
    
-   或者保留 token budget，同时提高 batch size / accumulation
    
-   或者允许你在更长句子、更多 workers、更多实验并行下更从容地跑
    

也就是说，**5090 对你最有价值的是“实验空间更大”**，而不是单纯 benchmark 更高。

* * *

## 3\. 但 5090 也有明显的硬件门槛

这个你不能忽视。

NVIDIA 的 5090 用户手册写得很明确：

-   **建议最小电源 1000W**
    
-   需要满足高功率供电条件。NVIDIA
    

所以如果你准备上 5090，你要同时确认：

-   电源够不够
    
-   机箱散热行不行
    
-   主板/供电/空间是否匹配
    

否则“卡买了，但平台不稳”，那就不值。

* * *

## 4\. PRO 6000 96G 更像“工作站级/研究站级方案”

NVIDIA 官方给 RTX PRO 6000 Blackwell Workstation Edition 的规格是：

-   **96 GB GDDR7 with ECC**
    
-   **1792 GB/s** 带宽
    
-   **600W** 最大功耗。NVIDIA+1
    

它的核心优势不是“比 5090 稍微快一点”，而是：

-   **显存是 96GB**
    
-   **带 ECC**
    
-   面向工作站/专业场景。NVIDIA+2NVIDIA+2
    

这意味着它更适合：

-   更大的模型
    
-   更长上下文
    
-   更大的 token budget
    
-   多模型并存
    
-   更重的本地微调/多任务负载
    
-   对稳定性、ECC、工作站属性要求高的场景。NVIDIA+1
    

* * *

## 5\. 但对你当前这个实验，PRO 6000 96G 大概率是过配

你现在的任务是：

-   单卡本地做 Transformer 英德翻译复现
    
-   模型还是 base 级别
    
-   当前瓶颈主要是有效 batch 不够大，而不是“模型根本放不下”
    

在这种情况下：

-   **5090 32G**：很可能已经能把你当前实验推到更舒服的区间
    
-   **PRO 6000 96G**：当然更强，但你未必能立刻把那 96G 变成等比例的实验收益
    

也就是说：

> **PRO 6000 96G 更适合“我要做更大模型/更大上下文/更重多任务”的未来路线。**
> 
> **对你眼前这组 WMT14 实验，它不是最优性价比选择。**

* * *

## 6\. 如果只针对“你现在这个项目”，我的推荐顺序是

### 第一选择：5090 32G

适合你现在最实际的目标：

-   提高 token budget
    
-   减少过小 batch 带来的训练噪声
    
-   更从容地跑正式训练、验证、beam search
    
-   还能兼顾你原本的 CV 任务。NVIDIA+1
    

### 第二选择：继续用 3080，但配合 gradient accumulation

如果你暂时不换卡，这也是合理路线。  
只是实验空间会小很多。

### 第三选择：PRO 6000 96G

只有在你已经明确想做下面这些事情时，我才会更推荐它：

-   更大的 Transformer / LLM 类实验
    
-   更大的 token budget
    
-   更长序列
    
-   多模型并行
    
-   更重的工作站级本地研发
    
-   对 ECC 和专业卡稳定性有明确需求。NVIDIA+2NVIDIA+2
    

* * *

## 7\. 我给你的非常直接的判断

### 只看你当前这个 WMT14 Transformer 实验

**5090 32G 更合适。**

### 如果你在问“最强能不能一步到位”

**PRO 6000 96G 当然更强，但对当前项目明显属于超前配置。**

### 如果你在问“值不值得”

-   **5090：值得认真考虑**
    
-   **PRO 6000：除非你后面明确要做更大规模 AI 训练/微调，否则现在不优先**
    

* * *

## 8\. 最后一句实用建议

如果你准备升级硬件，我建议你的决策顺序是：

1.  先确认你平台能不能稳定上 5090（尤其电源，官方建议 1000W）。NVIDIA
    
2.  如果能上，**优先 5090 32G**。
    
3.  上了之后，先把你当前实验的：
    
    -   `token budget`
        
    -   `batch size`
        
    -   `gradient accumulation`  
        做系统扫描。
        
4.  等你真的进入“32G 也明显不够”的阶段，再考虑 96G 工作站卡。
    

下一条我建议直接做这个：  
**我按你现在这套训练代码，给你规划一份“3080 / 5090 / 96G 工作站卡”三档 token budget 与 batch 配置建议表。**

  

这是预测的指标： (base) root@autodl-container-19b1118252-bcdca59c:~/autodl-tmp/github\_workspace/transformer-pytorch-main# python evaluate\_transformer\_bleu.py \\ --experiment-dir experiments/transformer\_wmt14\_en\_de\_base\_20260323\_052605 \\ --output-path predictions/test\_translations.txt Using checkpoints: experiments/transformer\_wmt14\_en\_de\_base\_20260323\_052605/checkpoints/model\_epoch\_049\_valid\_ppl\_248.3947.pth experiments/transformer\_wmt14\_en\_de\_base\_20260323\_052605/checkpoints/model\_epoch\_050\_valid\_ppl\_246.4773.pth experiments/transformer\_wmt14\_en\_de\_base\_20260323\_052605/checkpoints/model\_epoch\_051\_valid\_ppl\_250.2109.pth experiments/transformer\_wmt14\_en\_de\_base\_20260323\_052605/checkpoints/model\_epoch\_052\_valid\_ppl\_248.7295.pth experiments/transformer\_wmt14\_en\_de\_base\_20260323\_052605/checkpoints/model\_epoch\_053\_valid\_ppl\_250.0877.pth decoded 100 sentences decoded 200 sentences decoded 300 sentences decoded 400 sentences decoded 500 sentences decoded 600 sentences decoded 700 sentences decoded 800 sentences decoded 900 sentences decoded 1000 sentences decoded 1100 sentences decoded 1200 sentences decoded 1300 sentences decoded 1400 sentences decoded 1500 sentences decoded 1600 sentences decoded 1700 sentences decoded 1800 sentences decoded 1900 sentences decoded 2000 sentences decoded 2100 sentences decoded 2200 sentences decoded 2300 sentences decoded 2400 sentences decoded 2500 sentences decoded 2600 sentences decoded 2700 sentences decoded 2800 sentences decoded 2900 sentences decoded 3000 sentences That's 100 lines that end in a tokenized period ('.') It looks like you forgot to detokenize your test data, which may hurt your score. If you insist your data is detokenized, or don't care, you can suppress this message with the `force` parameter. BLEU = 5.49 predictions saved to predictions/test\_translations.txt

  

我先对齐一下论文里英德任务的 BLEU、beam search 和 checkpoint averaging 设定，然后我会直接判断你这个 `BLEU = 5.49` 到底意味着什么、最可能的问题在哪。

## 结论

**`BLEU = 5.49` 很低，当前结果不能认为接近论文复现。**

如果拿论文做参照，Transformer 在 **WMT14 English-to-German newstest2014** 上，base 模型是通过 **平均最后 5 个 checkpoint**、并使用 **beam size = 4**、**length penalty = 0.6** 得到最终结果的；论文报告的是一个远高于 5.49 的 BLEU 水平。论文同时明确写了英德任务使用了约 **4.5M** 句对、**shared source-target BPE vocabulary ~37000**、以及每 batch 约 **25000 source tokens + 25000 target tokens**。

所以，**5.49 不是“差一点”，而是“还明显不对”**。

* * *

# 一、先判断：这个 BLEU 低到什么程度

从你前面的训练日志来看：

-   你的训练系统是能学的；
    
-   但最好的 `valid_ppl` 出现在 **epoch 11 左右**；
    
-   你这次 BLEU 评估，却平均的是：
    

```
epoch 49 ~ 53valid_ppl ≈ 246 ~ 250
```

这两个信息放在一起，问题就非常明显了：

> **你现在拿去平均和评估的，不是训练最好的 checkpoint 区间，而是已经明显退化后的 checkpoint 区间。**

这基本上就是当前 BLEU 极低的**第一嫌疑项**。

* * *

# 二、当前最可能的问题，按优先级排序

* * *

## 问题 1：你平均错了 checkpoint 区间

你这次用的是：

-   `model_epoch_049_valid_ppl_248.3947.pth`
    
-   `model_epoch_050_valid_ppl_246.4773.pth`
    
-   `model_epoch_051_valid_ppl_250.2109.pth`
    
-   `model_epoch_052_valid_ppl_248.7295.pth`
    
-   `model_epoch_053_valid_ppl_250.0877.pth`
    

而你前面训练结果里，最好的验证表现是：

-   **epoch 11 左右**
    
-   `best valid_ppl ≈ 152.68`
    

这说明：

-   49~53 并不是“最后 5 个且仍在好区域”
    
-   而是“最后 5 个但已经严重退化的 checkpoint”
    

论文里说“平均最后 5 个 checkpoint”，它成立的前提是：

> **训练末段仍然处在一个高质量平台区间。**

你现在不是这个情况。  
你现在是：

> **训练后半段已经明显过拟合/退化。**

所以继续照搬“平均最后 5 个”这个字面做法，反而会把结果拉低。论文里 base 模型的 checkpoint averaging 是建立在其训练协议和收敛行为之上的，不是一个脱离上下文的机械规则。

### 这一步的直接修正建议

你下一轮应该评估这几类模型：

-   `best.pth`
    
-   **epoch 9 ~ 13** 附近的单点 checkpoint
    
-   **epoch 9~13** 这段里的 3~5 个 checkpoint 平均结果
    

而不是继续用 49~53。

* * *

## 问题 2：你现在的 BLEU 评估输入还没有 detokenize

你的评估程序已经明确提示：

```
It looks like you forgot to detokenize your test data, which may hurt your score.
```

这说明当前送进 BLEU 计算的文本，至少存在：

-   tokenized punctuation
    
-   例如句号前有空格这种 Moses tokenized 风格
    

这会**压低分数**。

但我需要非常明确地说：

> **这个问题会伤分，但通常不足以把一个本该 20+ 的结果直接打到 5.49。**

也就是说：

-   **没 detokenize 是问题**
    
-   但它更像是“次要伤害项”
    
-   不是当前最核心的问题
    

### 为什么这么说

因为 tokenized BLEU 通常会有几分到十来分的偏差，但你现在这个 `5.49` 低得太多了，更像是：

-   模型本身翻译质量就不行
    
-   再叠加 tokenization 评估口径不对
    

所以：

> **detokenize 要修，但修完也不太可能从 5.49 直接跳到论文级结果。**

* * *

## 问题 3：你的训练配置离论文 batching 还差很远

论文训练批次写得很明确：

-   按近似长度组 batch
    
-   每 batch 约：
    
    -   `25000 source tokens`
        
    -   `25000 target tokens`
        

而你之前自己说过，你当前实验用的是：

-   `src_token_budget = 2048`
    
-   `tgt_token_budget = 2048`
    

这和论文的 token 预算不是一个量级。

### 这会带来什么

-   每 step 看到的信息量太小
    
-   梯度噪声更大
    
-   收敛区间更不稳
    
-   更容易出现“训练会学，但最后翻译质量起不来”
    

所以即使你的 loss/acc 训练链看起来成立，也不代表你已经接近论文训练条件。

* * *

## 问题 4：你当前最好的验证指标本身也不够好

你前面贴过：

-   `best valid_ppl ≈ 152.68`
    

这个数本身就说明：

> **模型还远没有进入一个高质量翻译模型应有的分布状态。**

论文在 Table 3 上给的是 **development set newstest2013** 上的 per-wordpiece perplexity，并且 base 模型是在合理收敛状态下再去做 test BLEU 的。

虽然我们不能机械拿你的 `valid_ppl` 和论文表格做一一等号比较，但可以肯定的是：

-   `valid_ppl` 还在百级以上
    
-   一般不可能对应一个高质量 BLEU
    

也就是说，从验证损失侧，你这个 BLEU 低并不意外。

* * *

# 三、所以当前 `BLEU = 5.49` 应该怎么定性

我给你一个非常直接的判断：

> **这不是“评估细节没调好”那么简单。**
> 
> 它说明当前模型质量本身就不够，再叠加 checkpoint 选择错误和 detokenize 缺失，最后把分数进一步压低了。

更具体地说：

-   **根因层**：当前训练质量还不够
    
-   **直接导火索**：你选错了要平均的 checkpoint 区间
    
-   **评估层附加伤害**：没有 detokenize
    

* * *

# 四、你现在最应该怎么做

下面我按“收益最大”的顺序给你。

* * *

## 第一步：立刻重做 BLEU 评估，但只评估 `best.pth`

这一步最重要。

你现在不要再先平均 49~53。  
先直接拿：

-   `best.pth`
    

做一次：

-   greedy BLEU
    
-   beam BLEU（beam=4, alpha=0.6）
    

因为论文推理时明确使用：

-   beam size = 4
    
-   length penalty $ \\alpha = 0.6 $
    
-   max output length = input length + 50。
    

### 为什么先做这个

因为这一步能最快回答一个核心问题：

> 你现在 BLEU 低，到底有多少是因为 checkpoint 选错了？

* * *

## 第二步：把 epoch 9~13 附近做一轮 checkpoint averaging

我建议你平均的是：

-   epoch 9,10,11,12,13  
    或者更保守一点：
    
-   epoch 10,11,12
    

### 为什么是这段

因为你前面的训练结果显示：

-   最好的验证 ppl 在 epoch 11 左右
    
-   9~13 是最接近峰值的平台区
    

如果要做 averaging，就应该平均这个平台区，而不是训练后期已经坏掉的区间。

* * *

## 第三步：在 BLEU 评估前做 detokenize

你现在的评估脚本至少要确保：

-   BPE 已经去掉
    
-   Moses tokenization 也已经 detokenize
    

也就是说，最终送入 BLEU 的文本应该是：

-   正常句子
    
-   而不是 `This is a test .`
    

这种 tokenized 形式

### 原因

你当前日志已经直接提示了这个问题。  
虽然它不是主因，但一定要修。

* * *

# 五、关于 detokenize，我给你一个非常实际的判断

你现在的 `predictions/test_translations.txt` 很可能是：

-   已经去掉了 BPE 的 `@@`
    
-   但还保留了 Moses tokenization 风格
    

也就是类似：

```
I am here .
```

而不是：

```
I am here.
```

这就是 sacrebleu 给你提示的原因。

所以后续正确流程应该是：

$预测 id→BPE token→去掉 @@→Moses detokenize→BLEU\text{预测 id} \rightarrow \text{BPE token} \rightarrow \text{去掉 @@} \rightarrow \text{Moses detokenize} \rightarrow \text{BLEU}预测 id→BPE token→去掉 @@→Moses detokenize→BLEU$

不是仅仅去掉 BPE 就结束。

* * *

# 六、如果你问我“现在最可能的一刀见血修复点是什么”

答案是：

> **先不要再看 epoch 49~53。**
> 
> **先评估 best checkpoint。**

因为你这次拿来平均的 checkpoint，验证 ppl 都在 246~250 左右，这和你自己前面最好的验证点差了太多。这个差距已经足够解释“为什么 BLEU 会非常差”。

* * *

# 七、你当前结果和论文还差在哪里

我帮你用最短的话总结一下：

### 论文侧

-   WMT14 En-De
    
-   shared BPE ~37k
    
-   batch ~25k src + 25k tgt tokens
    
-   Adam + Noam + warmup 4000
    
-   averaged last 5 checkpoints
    
-   beam=4, alpha=0.6
    
-   最终在 newstest2014 报告高 BLEU。
    

### 你当前侧

-   数据链基本成立
    
-   训练链基本成立
    
-   但有效 token batch 小很多
    
-   最优 checkpoint 区间已经很早出现
    
-   你这次又选了后期退化 checkpoint 去平均
    
-   BLEU 评估还没 detokenize
    

所以现在最准确的阶段判断是：

> **你还在“训练系统成立，但结果还没拉起来”的阶段。**

* * *

# 八、我给你的下一步顺序

你现在最应该按这个顺序做：

-    用 `best.pth` 直接评估一次 BLEU
    
-    用 epoch 9~13 做 averaging，再评一次 BLEU
    
-    在评估脚本里加 detokenize
    
-    对比：
    
    -   best single checkpoint
        
    -   best-window averaged checkpoint
        
    -   greedy vs beam
        

这四步做完，你才能真正判断：

-   当前训练配置的真实可达翻译质量
    
-   问题主要在训练还是主要在评估
    

* * *

# 九、最终结论

**`BLEU = 5.49` 说明当前结果明显不对，但最可能不是单一原因。**

我给你的优先级判断是：

1.  **checkpoint 选错区间** —— 主因
    
2.  **模型训练质量本身还不够** —— 根因
    
3.  **没有 detokenize** —— 次要但确实存在的伤分项
    

* * *

下一条最合适的动作是：  
**我直接给你改造 `evaluate_transformer_bleu.py`，让它支持：**

-   评估 `best.pth`
    
-   指定任意 checkpoint 列表做 averaging
    
-   自动 Moses detokenize
    
-   同时输出 greedy / beam 两种 BLEU
    

这样你马上就能定位问题。