import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


def attention(query, key, value, mask=None, dropout=None):
    ''' 
    缩放点积注意力
    query: (bs, num_heads, seq_len, d_k)
    key: (bs, num_heads, seq_len, d_k)
    value: (bs, num_heads, seq_len, d_v)
    对于 torch.matmul 来说，只要张量维度大于等于 2，
    它永远只会把最后两个维度当成真正的“矩阵”去相乘，
    而把前面所有的维度都统统视为“Batch（批次）
    '''
    d_k = query.size(-1) 
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    p_attn = F.softmax(scores, dim=-1)

    if dropout is not None:
        p_attn = dropout(p_attn)

    return torch.matmul(p_attn, value), p_attn
    

class MultiHeadedAttention(nn.Module):
    def __init__(self, num_heads, d_model, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.p_attn = None
        self.num_heads = num_heads
        self.d_model = d_model

        self.d_k = d_model // num_heads
        self.dropout_layer = nn.Dropout(dropout)
        # 从效率上，QKV不能真的用8个linear，所以我们把8个linear合并，也就是num_heads个d_k = d_model
        self.linears = nn.ModuleList(
            [nn.Linear(d_model, d_model), # query 的 Linear
            nn.Linear(d_model, d_model), # key 的 Linear
            nn.Linear(d_model, d_model), # value 的 Linear
            nn.Linear(d_model, d_model)]
        )

    
    def forward(self, query, key, value, mask=None):
        ''' 
        query --> (batch_size, query_len, d_model)
        '''
        batch_size = query.size(0)
        projected_query = self.linears[0](query) # (batch_size, query_len, d_model)
        projected_key = self.linears[1](key) # (batch_size, key_len, d_model)
        projected_value = self.linears[2](value) # (batch_size, value_len, d_model)

        # 多头切分
        viewed_query = projected_query.reshape(batch_size, -1, self.num_heads, self.d_k) # (batch_size, query_len, num_heads, d_k)
        viewed_key = projected_key.reshape(batch_size, -1, self.num_heads, self.d_k)
        viewed_value = projected_value.reshape(batch_size, -1, self.num_heads, self.d_k)

        # 调整维度，使得每个头独立
        transdim_query = viewed_query.transpose(1, 2) # (batch_size, num_heads, query_len, d_k)
        transdim_key = viewed_key.transpose(1, 2) # (batch_size, num_heads, key_len, d_k)
        transdim_value = viewed_value.transpose(1, 2) # (batch_size, num_heads, value_len, d_k)

        # 计算att
        att, self.p_attn = attention(transdim_query, transdim_key, transdim_value, mask=mask, dropout=self.dropout_layer)

        # 多头重组，分开的多个头再合并回去，提高效率
        transdim_att = att.transpose(1, 2) # (batch_size, query_len, num_heads, d_k)
        transdim_att = transdim_att.contiguous()

        merge_att = transdim_att.reshape(batch_size, -1, self.d_model) # (batch_size, query_len, d_model)
        projected_att = self.linears[3](merge_att) # (batch_size, query_len, d_model)

        return projected_att


class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.w_1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.w_2(x)
        return x
    

class PositionalEncoding(nn.Module):
    """
    固定的正弦/余弦位置编码。

    与论文一致：
    1. 不引入可学习参数。
    2. 对输入 `(B, T, d_model)` 直接按位置相加。
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
        # `pe` 是 buffer，不参与梯度更新。
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class Encoder(nn.Module):
    def __init__(self, d_model, self_attn, feed_forward, dropout, num_layers=6):
        super().__init__()
        self.self_attn = nn.ModuleList(self_attn)
        self.feed_forward = nn.ModuleList(feed_forward)

        self.norm = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])
        self.norm_ff = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])
        self.final_norm = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

        # parameters
        self.num_layers = num_layers

    def forward(self, x, mask):
        # EncoderLayer
        for i in range(self.num_layers): 
            norm_x = self.norm[i](x)
            x = x + self.dropout(self.self_attn[i](
                query=norm_x,
                key=norm_x,
                value=norm_x,
                mask=mask
            ))

            x = x + self.dropout(self.feed_forward[i](self.norm_ff[i](x)))
        
        x = self.final_norm(x)
        return x


class Decoder(nn.Module):
    def __init__(self, d_model, self_attn, cross_attn, feed_forward, dropout, num_layers=6):
        super().__init__()
        self.self_attn = nn.ModuleList(self_attn)
        self.cross_attn = nn.ModuleList(cross_attn)
        self.feed_forward = nn.ModuleList(feed_forward)

        self.norm_attn = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])
        self.norm_cross_attn = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])
        self.norm_ff = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])
        self.final_norm = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

        # parameters
        self.num_layers = num_layers

    def forward(self, x, memory, src_mask, tgt_mask):
        # EncoderLayer
        for i in range(self.num_layers): 
            norm_attn_x = self.norm_attn[i](x)
            x = x + self.dropout(self.self_attn[i](
                query=norm_attn_x,
                key=norm_attn_x,
                value=norm_attn_x,
                mask=tgt_mask
            ))

            norm_cross_attn_x = self.norm_cross_attn[i](x)
            x = x + self.dropout(self.cross_attn[i](
                query=norm_cross_attn_x,
                key=memory,
                value=memory,
                mask=src_mask
            ))
            
            x = x + self.dropout(self.feed_forward[i](self.norm_ff[i](x)))
        
        x = self.final_norm(x)
        return x


class Embeddings(nn.Module):
    """
    token embedding + sinusoidal positional encoding。
    """
    def __init__(self, d_model, vocab_size, dropout=0.1):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model
        self.pe = PositionalEncoding(d_model, dropout)

    def forward(self, x):
        # 与论文一致：embedding 在叠加位置编码前先乘上 sqrt(d_model)。
        x = self.lut(x) * math.sqrt(self.d_model)
        return self.pe(x)

class Generator(nn.Module):
    """
    Generator / 词表投影头

    功能：
        将 Decoder 最后一层输出的隐状态映射到词表维度。

    说明：
         bias=False，要共享权重，但是embedding没有bias，所以投影头也不能有bias
        即让 generator.proj.weight 与 embedding.weight 共享同一组参数。
        这里返回的是 logits，不在模块内做 softmax；概率归一化交给损失函数或解码逻辑。
    """
    def __init__(self, d_model: int, vocab_size: int, bias: bool = False):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入：
            x: (B, T, d_model)

        输出：
            logits: (B, T, vocab_size)
        """
        return self.proj(x)


def tie_transformer_weights(model):
    """
    将 Transformer 的三处权重绑定为同一组参数。

    绑定对象：
        1. src_embed.lut.weight
        2. tgt_embed.lut.weight
        3. generator.proj.weight

    说明：
        这要求 src/tgt 使用同一个 joint vocabulary，
        也就是同一套 token id 体系。
        在当前项目中，这和论文里共享 BPE 词表、共享 embedding / generator 权重的做法一致。
    """
    shared_weight = model.tgt_embed.lut.weight

    # 源 embedding 与目标 embedding 共享
    model.src_embed.lut.weight = shared_weight

    # Generator 输出投影与 embedding 共享
    model.generator.proj.weight = shared_weight

class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        memory = self.encoder(self.src_embed(src), src_mask)
        output = self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)
        # 不要加Generator，而是在模型的外部手动Generator，因为后续还会涉及到推理的beam search,需要原始的输出。
        return output




def build_transformer(share_embeddings=True):
    vocab_size = 32000   # 真实场景下的典型词表大小 (如 BPE 分词)
    d_model = 512        # 隐藏层维度 
    d_ff = 2048          # FFN 膨胀维度
    num_heads = 8        # 注意力头数
    num_layers = 6       # 编码器层数
    dropout = 0.1
    batch_size = 4

    # attn = MultiHeadedAttention(num_heads=num_heads, d_model=d_model, dropout=dropout)
    # ff = PositionWiseFeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout)

    attn_list = [MultiHeadedAttention(num_heads=num_heads, d_model=d_model, dropout=dropout) for _ in range(num_layers)]
    ff_list = [PositionWiseFeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout) for _ in range(num_layers)]
    encoder = Encoder(d_model=d_model, self_attn=attn_list, feed_forward=ff_list, dropout=dropout, num_layers=num_layers)
    
    decoder_attn_list = [MultiHeadedAttention(num_heads=num_heads, d_model=d_model, dropout=dropout) for _ in range(num_layers)]
    decoder_cross_attn_list = [MultiHeadedAttention(num_heads=num_heads, d_model=d_model, dropout=dropout) for _ in range(num_layers)]
    decoder_ff_list = [PositionWiseFeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout) for _ in range(num_layers)]
    decoder = Decoder(d_model=d_model, self_attn=decoder_attn_list, cross_attn=decoder_cross_attn_list, feed_forward=decoder_ff_list, dropout=dropout, num_layers=num_layers)

    # embeddings
    src_embed = Embeddings(d_model, vocab_size, dropout)
    tgt_embed = Embeddings(d_model, vocab_size, dropout)

    # generator
    generator = Generator(d_model, vocab_size)

    model = EncoderDecoder(encoder, decoder, src_embed, tgt_embed, generator)

    if share_embeddings:
        tie_transformer_weights(model)

    # Xavier / Glorot 初始化
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return model




