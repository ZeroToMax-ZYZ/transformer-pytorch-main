from __future__ import annotations

"""
功能：
1. 实现适用于 Transformer 机器翻译的 Label Smoothing 损失。
2. 返回可回传梯度的总损失，以及用于日志记录的细粒度分量：
   - total_loss
   - nll_loss
   - smooth_loss
3. 提供 token-level accuracy 和 perplexity 的辅助函数。

说明：
1. 本实现面向共享词表场景，要求显式提供 pad_idx。
2. smoothing 的平滑质量会分配到“除目标类和 pad 类之外”的其他类别。
3. 返回的 loss/nll_loss/smooth_loss 都是“按有效 token 取平均”的标量。
"""

from dataclasses import dataclass
from typing import Dict

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LabelSmoothingLossOutput:
    """
    统一封装一次前向计算得到的损失结果。
    """
    loss: torch.Tensor
    nll_loss: torch.Tensor
    smooth_loss: torch.Tensor
    num_tokens: int

    def as_dict(self) -> Dict[str, float]:
        """
        将结果转换为便于日志记录的普通字典。
        """
        return {
            "loss": float(self.loss.detach().item()),
            "nll_loss": float(self.nll_loss.detach().item()),
            "smooth_loss": float(self.smooth_loss.detach().item()),
            "num_tokens": int(self.num_tokens),
        }


class LabelSmoothingLoss(nn.Module):
    """
    Transformer 机器翻译任务的 Label Smoothing 损失。

    入口：
        logits: 形状 (B, T, V)
        target: 形状 (B, T)

    出口：
        LabelSmoothingLossOutput
            - loss:        总损失（可回传梯度）
            - nll_loss:    硬标签 NLL 分量（日志观察）
            - smooth_loss: 平滑分量（日志观察）
            - num_tokens:  有效 token 数（排除 pad）

    参数说明：
        vocab_size: 词表大小
        pad_idx: PAD 的词表 id
        smoothing: label smoothing 系数，论文默认 0.1
    """

    def __init__(
        self,
        vocab_size: int,
        pad_idx: int,
        smoothing: float = 0.1,
    ) -> None:
        super().__init__()

        if vocab_size <= 2:
            raise ValueError("vocab_size 必须大于 2。")

        if not (0.0 <= smoothing < 1.0):
            raise ValueError("smoothing 必须满足 0 <= smoothing < 1。")

        self.vocab_size = vocab_size
        self.pad_idx = pad_idx
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
    ) -> LabelSmoothingLossOutput:
        """
        前向计算。

        输入：
            logits: (B, T, V)
            target: (B, T)

        输出：
            LabelSmoothingLossOutput
        """
        if logits.dim() != 3:
            raise ValueError(f"logits 必须是 3 维张量，当前维度为 {logits.dim()}。")
        if target.dim() != 2:
            raise ValueError(f"target 必须是 2 维张量，当前维度为 {target.dim()}。")

        bsz, tgt_len, vocab_size = logits.shape
        if vocab_size != self.vocab_size:
            raise ValueError(
                f"logits 最后一维 vocab_size={vocab_size} 与初始化时的 {self.vocab_size} 不一致。"
            )
        if target.shape[0] != bsz or target.shape[1] != tgt_len:
            raise ValueError(
                f"target 形状 {tuple(target.shape)} 与 logits 前两维 {(bsz, tgt_len)} 不匹配。"
            )

        logits_flat = logits.reshape(-1, self.vocab_size)   # (N, V)
        target_flat = target.reshape(-1)                    # (N,)

        valid_mask = target_flat.ne(self.pad_idx)
        if valid_mask.sum().item() == 0:
            raise ValueError("当前 batch 中有效 token 数为 0，无法计算损失。")

        logits_valid = logits_flat[valid_mask]              # (N_valid, V)
        target_valid = target_flat[valid_mask]              # (N_valid,)

        log_probs = F.log_softmax(logits_valid, dim=-1)     # (N_valid, V)

        # -------------------------
        # 1. 硬标签 NLL 分量
        # -------------------------
        nll_per_token = -log_probs.gather(dim=1, index=target_valid.unsqueeze(1)).squeeze(1)

        # -------------------------
        # 2. 平滑分量
        #    平滑概率质量分配到：
        #    - 除 pad 类外
        #    - 除真实目标类外
        # -------------------------
        if self.smoothing > 0.0:
            if self.vocab_size <= 2:
                raise ValueError("当 smoothing > 0 时，vocab_size 必须大于 2。")

            smooth_target = torch.full_like(
                log_probs,
                fill_value=1.0 / (self.vocab_size - 2),
            )
            smooth_target[:, self.pad_idx] = 0.0
            smooth_target.scatter_(1, target_valid.unsqueeze(1), 0.0)

            # 保险归一化，避免数值误差
            smooth_target = smooth_target / smooth_target.sum(dim=1, keepdim=True)

            smooth_per_token = -(smooth_target * log_probs).sum(dim=-1)
            total_per_token = self.confidence * nll_per_token + self.smoothing * smooth_per_token
        else:
            smooth_per_token = torch.zeros_like(nll_per_token)
            total_per_token = nll_per_token

        num_tokens = int(target_valid.numel())

        loss = total_per_token.mean()
        nll_loss = nll_per_token.mean()
        smooth_loss = smooth_per_token.mean()

        return LabelSmoothingLossOutput(
            loss=loss,
            nll_loss=nll_loss.detach(),
            smooth_loss=smooth_loss.detach(),
            num_tokens=num_tokens,
        )


@torch.no_grad()
def compute_token_accuracy(
    logits: torch.Tensor,
    target: torch.Tensor,
    pad_idx: int,
) -> Dict[str, float]:
    """
    计算 token-level accuracy。

    输入：
        logits: (B, T, V)
        target: (B, T)
        pad_idx: PAD 的词表 id

    输出：
        dict:
            - correct_tokens
            - total_tokens
            - token_acc
    """
    if logits.dim() != 3:
        raise ValueError(f"logits 必须是 3 维张量，当前维度为 {logits.dim()}。")
    if target.dim() != 2:
        raise ValueError(f"target 必须是 2 维张量，当前维度为 {target.dim()}。")

    pred = logits.argmax(dim=-1)                       # (B, T)
    valid_mask = target.ne(pad_idx)                    # (B, T)

    total_tokens = int(valid_mask.sum().item())
    if total_tokens == 0:
        return {
            "correct_tokens": 0,
            "total_tokens": 0,
            "token_acc": 0.0,
        }

    correct_tokens = int(((pred == target) & valid_mask).sum().item())
    token_acc = correct_tokens / total_tokens

    return {
        "correct_tokens": correct_tokens,
        "total_tokens": total_tokens,
        "token_acc": float(token_acc),
    }


def compute_perplexity_from_loss(loss_value: float) -> float:
    """
    根据平均 loss 计算 perplexity。

    说明：
        1. 通常建议对 nll_loss 计算 ppl，更直观。
        2. 当 loss 非常大时，exp 可能上溢，因此这里做一个简单裁剪。
    """
    safe_loss = min(float(loss_value), 50.0)
    return float(math.exp(safe_loss))