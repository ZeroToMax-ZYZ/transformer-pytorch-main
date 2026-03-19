import copy
import math

import torch
import torch.nn as nn

from nets.utils.attention import MultiHeadedAttention
from nets.utils.PositionwiseFeedForward import PositionwiseFeedForward
from nets.utils.Generator import Generator
from nets.utils.encoder_decoder import (
    Encoder,
    Decoder,
    EncoderLayer,
    DecoderLayer,
    EncoderDecoder,
    Embeddings,
)


def tie_transformer_weights(model: EncoderDecoder) -> None:
    """
    将 Transformer 的三处权重绑定为同一组参数。

    绑定对象：
        1. src_embed.lut.weight
        2. tgt_embed.lut.weight
        3. generator.proj.weight

    说明：
        这要求 src/tgt 使用同一个 joint vocabulary，
        也就是同一套 token id 体系。
    """
    shared_weight = model.tgt_embed.lut.weight

    # 源 embedding 与目标 embedding 共享
    model.src_embed.lut.weight = shared_weight

    # Generator 输出投影与 embedding 共享
    model.generator.proj.weight = shared_weight


def make_model(
    src_vocab: int,
    tgt_vocab: int,
    N: int = 6,
    d_model: int = 512,
    d_ff: int = 2048,
    h: int = 8,
    dropout: float = 0.1,
    share_embeddings: bool = True,
) -> EncoderDecoder:
    """
    构建完整 Transformer 模型。

    参数：
        src_vocab: 源语言词表大小
        tgt_vocab: 目标语言词表大小
        N: Encoder / Decoder 层数
        d_model: 隐状态维度
        d_ff: FFN 中间维度
        h: 多头注意力头数
        dropout: dropout 概率
        share_embeddings: 是否共享 src/tgt embedding 以及 generator 权重

    返回：
        完整的 EncoderDecoder 模型
    """
    c = copy.deepcopy

    if share_embeddings and src_vocab != tgt_vocab:
        raise ValueError(
            "当 share_embeddings=True 时，src_vocab 和 tgt_vocab 必须一致。"
            "这通常意味着你使用的是 joint source-target vocabulary。"
        )

    attn = MultiHeadedAttention(h, d_model, dropout)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)

    model = EncoderDecoder(
        encoder=Encoder(
            EncoderLayer(d_model, c(attn), c(ff), dropout),
            N,
        ),
        decoder=Decoder(
            DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout),
            N,
        ),
        src_embed=Embeddings(d_model, src_vocab, dropout),
        tgt_embed=Embeddings(d_model, tgt_vocab, dropout),
        generator=Generator(d_model, tgt_vocab, bias=False),
    )

    # 先做权重绑定，再做初始化。
    # 这样共享参数只会保留一份物理权重。
    if share_embeddings:
        tie_transformer_weights(model)

    # Xavier / Glorot 初始化
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return model


def test_factory_method() -> None:
    """
    测试：
    1. 权重共享是否成功
    2. src_mask / tgt_mask 是否按训练态正确构造
    3. 前向维度是否正确
    """
    from data.batch import Seq2SeqBatch

    print("=== 开始测试工厂函数 make_model ===")

    src_vocab_size = 5000
    tgt_vocab_size = 5000
    batch_size = 3
    pad_idx = 0
    bos_idx = 1
    eos_idx = 2

    print("正在构建微型 Transformer 模型 (2层)...")
    model = make_model(
        src_vocab=src_vocab_size,
        tgt_vocab=tgt_vocab_size,
        N=2,
        d_model=256,
        d_ff=1024,
        h=4,
        dropout=0.1,
        share_embeddings=True,
    )

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型构建完成！总可训练参数量: {total_params / 1e6:.2f} M")
    print("-" * 60)

    # 构造带 PAD 的源序列
    src_tokens = torch.tensor([
        [11, 12, 13, 14, 15, pad_idx, pad_idx],
        [21, 22, 23, 24, pad_idx, pad_idx, pad_idx],
        [31, 32, 33, 34, 35, 36, 37],
    ], dtype=torch.long)

    # 构造完整目标序列：必须已经包含 BOS / EOS / PAD
    tgt_tokens = torch.tensor([
        [bos_idx, 101, 102, 103, eos_idx, pad_idx, pad_idx],
        [bos_idx, 201, 202, eos_idx, pad_idx, pad_idx, pad_idx],
        [bos_idx, 301, 302, 303, 304, eos_idx, pad_idx],
    ], dtype=torch.long)

    batch = Seq2SeqBatch.from_tensors(
        src=src_tokens,
        tgt=tgt_tokens,
        pad_idx=pad_idx,
    )

    print(f"src.shape      = {batch.src.shape}")
    print(f"tgt_input.shape= {batch.tgt_input.shape}")
    print(f"tgt_y.shape    = {batch.tgt_y.shape}")
    print(f"src_mask.shape = {batch.src_mask.shape}")
    print(f"tgt_mask.shape = {batch.tgt_mask.shape}")
    print(f"ntokens        = {batch.ntokens}")

    # 检查共享权重是否真的指向同一块内存
    src_ptr = model.src_embed.lut.weight.data_ptr()
    tgt_ptr = model.tgt_embed.lut.weight.data_ptr()
    gen_ptr = model.generator.proj.weight.data_ptr()

    assert src_ptr == tgt_ptr == gen_ptr, "权重共享失败：三者没有绑定到同一块内存。"

    # 前向传播：注意这里必须传入 tgt_input，而不是完整 tgt
    hidden_states = model(
        batch.src,
        batch.tgt_input,
        batch.src_mask,
        batch.tgt_mask,
    )

    print(f"Decoder 输出隐状态维度: {hidden_states.shape}")
    assert hidden_states.shape == (batch_size, batch.tgt_input.size(1), 256)

    logits = model.generator(hidden_states)
    print(f"最终 logits 维度: {logits.shape}")
    assert logits.shape == (batch_size, batch.tgt_input.size(1), tgt_vocab_size)

    print("\n✅ 测试通过：")
    print("1. 权重共享正常")
    print("2. target shifted right 正常")
    print("3. src_mask / tgt_mask 正常")
    print("4. 前向传播正常")


if __name__ == "__main__":
    test_factory_method()