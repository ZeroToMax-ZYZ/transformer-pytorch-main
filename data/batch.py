from __future__ import annotations

"""
功能：
1. 构造 Transformer 训练阶段所需的 src_mask。
2. 构造同时包含 padding mask + causal mask 的 tgt_mask。
3. 将完整 target 序列切分为：
   - tgt_input: 送入 Decoder 的输入（右移后的序列）
   - tgt_y:     监督标签
4. 用 Seq2SeqBatch 统一封装 batch，方便后续训练代码直接调用。

重要约定：
1. 传入的 tgt 必须已经包含 <bos> 和 <eos>。
2. 这里的 shifted right 采用最稳妥的切片写法：
   tgt_input = tgt[:, :-1]
   tgt_y     = tgt[:, 1:]
3. 当前实现与你前面的 MultiHeadedAttention 兼容：
   - src_mask 形状: (B, 1, S)
   - tgt_mask 形状: (B, T, T)
"""

from dataclasses import dataclass
from typing import Tuple

import torch


def subsequent_mask(size: int, device: torch.device) -> torch.Tensor:
    """
    生成下三角因果 Mask。

    输入：
        size: 目标序列长度 T
        device: 张量所在设备

    输出：
        mask: 形状为 (1, T, T) 的 bool Tensor
              True  表示允许关注
              False 表示禁止关注
    """
    return torch.tril(
        torch.ones((1, size, size), dtype=torch.bool, device=device)
    )


def make_src_mask(src: torch.Tensor, pad_idx: int) -> torch.Tensor:
    """
    根据源序列构造 src_mask。

    输入：
        src: 形状 (B, S) 的源序列 token id
        pad_idx: PAD 的词表 id

    输出：
        src_mask: 形状 (B, 1, S) 的 bool Tensor
    """
    return (src != pad_idx).unsqueeze(1)


def shift_right(tgt: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    将完整 target 序列切分为 Decoder 输入和监督标签。

    输入：
        tgt: 形状 (B, T_full) 的完整目标序列
             要求已经包含 <bos> 和 <eos>

    输出：
        tgt_input: 形状 (B, T_full - 1)
                   送入 Decoder 的输入
        tgt_y:     形状 (B, T_full - 1)
                   训练监督标签
    """
    if tgt.dim() != 2:
        raise ValueError(f"tgt 必须是 2 维张量，当前维度为: {tgt.dim()}")

    if tgt.size(1) < 2:
        raise ValueError(
            "tgt 的序列长度至少要 >= 2，因为必须能够切分出 tgt_input 和 tgt_y。"
        )

    tgt_input = tgt[:, :-1].contiguous()
    tgt_y = tgt[:, 1:].contiguous()
    return tgt_input, tgt_y


def make_tgt_mask(tgt_input: torch.Tensor, pad_idx: int) -> torch.Tensor:
    """
    构造 Decoder 自注意力所需的 tgt_mask。

    输入：
        tgt_input: 形状 (B, T) 的 Decoder 输入序列
        pad_idx: PAD 的词表 id

    输出：
        tgt_mask: 形状 (B, T, T) 的 bool Tensor

    构造逻辑：
        1. padding mask: 屏蔽掉 PAD 位置，形状 (B, 1, T)
        2. causal mask:  只允许看当前位置及之前位置，形状 (1, T, T)
        3. 两者按位与，广播后得到 (B, T, T)
    """
    if tgt_input.dim() != 2:
        raise ValueError(
            f"tgt_input 必须是 2 维张量，当前维度为: {tgt_input.dim()}"
        )

    pad_mask = (tgt_input != pad_idx).unsqueeze(1)  # (B, 1, T)
    causal_mask = subsequent_mask(tgt_input.size(1), tgt_input.device)  # (1, T, T)

    tgt_mask = pad_mask & causal_mask
    return tgt_mask


def pad_sequences(sequences, pad_idx: int) -> torch.Tensor:
    """
    将变长 id 序列 padding 成统一长度的 LongTensor。

    输入：
        sequences: List[List[int]]
        pad_idx: PAD 的词表 id

    输出：
        tensor: 形状 (B, L_max)
    """
    if len(sequences) == 0:
        raise ValueError("sequences 不能为空。")

    max_len = max(len(seq) for seq in sequences)
    batch_size = len(sequences)

    out = torch.full((batch_size, max_len), pad_idx, dtype=torch.long)

    for i, seq in enumerate(sequences):
        out[i, :len(seq)] = torch.tensor(seq, dtype=torch.long)

    return out


@dataclass
class Seq2SeqBatch:
    """
    统一封装 Transformer 训练所需的一个 batch。

    字段说明：
        src:       源序列，形状 (B, S)
        tgt_input: Decoder 输入，形状 (B, T)
        tgt_y:     监督标签，形状 (B, T)
        src_mask:  源序列 mask，形状 (B, 1, S)
        tgt_mask:  目标序列 mask，形状 (B, T, T)
        ntokens:   tgt_y 中非 PAD token 的数量
    """
    src: torch.Tensor
    tgt_input: torch.Tensor
    tgt_y: torch.Tensor
    src_mask: torch.Tensor
    tgt_mask: torch.Tensor
    ntokens: int

    @classmethod
    def from_tensors(
        cls,
        src: torch.Tensor,
        tgt: torch.Tensor,
        pad_idx: int,
    ) -> "Seq2SeqBatch":
        """
        从 src / tgt 原始张量构造训练 batch。

        输入：
            src: 形状 (B, S)
            tgt: 形状 (B, T_full)，要求已包含 <bos> 和 <eos>
            pad_idx: PAD 的词表 id
        """
        src_mask = make_src_mask(src, pad_idx)
        tgt_input, tgt_y = shift_right(tgt)
        tgt_mask = make_tgt_mask(tgt_input, pad_idx)
        ntokens = int((tgt_y != pad_idx).sum().item())

        return cls(
            src=src,
            tgt_input=tgt_input,
            tgt_y=tgt_y,
            src_mask=src_mask,
            tgt_mask=tgt_mask,
            ntokens=ntokens,
        )

    def to(self, device: torch.device) -> "Seq2SeqBatch":
        """
        将 batch 移动到指定设备。
        """
        return Seq2SeqBatch(
            src=self.src.to(device, non_blocking=True),
            tgt_input=self.tgt_input.to(device, non_blocking=True),
            tgt_y=self.tgt_y.to(device, non_blocking=True),
            src_mask=self.src_mask.to(device, non_blocking=True),
            tgt_mask=self.tgt_mask.to(device, non_blocking=True),
            ntokens=self.ntokens,
        )