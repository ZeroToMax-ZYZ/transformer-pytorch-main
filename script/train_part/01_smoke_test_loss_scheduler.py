from __future__ import annotations

"""
功能：
1. 测试 LabelSmoothingLoss 是否能正常前向。
2. 测试 Adam + Noam 调度器是否能正常更新学习率。
3. 作为训练系统最底层数学模块的 smoke test。
"""

import torch

from utils.label_smoothing import LabelSmoothingLoss, compute_token_accuracy, compute_perplexity_from_loss
from utils.noam_scheduler import build_transformer_optimizer_and_scheduler


class DummyModel(torch.nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 32) -> None:
        super().__init__()
        self.emb = torch.nn.Embedding(vocab_size, d_model)
        self.proj = torch.nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.emb(x)
        return self.proj(h)


def main() -> None:
    vocab_size = 100
    pad_idx = 0

    model = DummyModel(vocab_size=vocab_size, d_model=32)
    criterion = LabelSmoothingLoss(
        vocab_size=vocab_size,
        pad_idx=pad_idx,
        smoothing=0.1,
    )
    optimizer, scheduler = build_transformer_optimizer_and_scheduler(
        model=model,
        d_model=32,
        warmup_steps=10,
    )

    x = torch.tensor([
        [5, 6, 7, 8],
        [9, 10, 0, 0],
    ], dtype=torch.long)

    target = torch.tensor([
        [6, 7, 8, 2],
        [10, 2, 0, 0],
    ], dtype=torch.long)

    logits = model(x)
    loss_out = criterion(logits, target)
    acc_out = compute_token_accuracy(logits, target, pad_idx=pad_idx)

    print("loss_out =", loss_out.as_dict())
    print("acc_out  =", acc_out)
    print("ppl      =", compute_perplexity_from_loss(loss_out.as_dict()["nll_loss"]))

    optimizer.zero_grad(set_to_none=True)
    loss_out.loss.backward()
    lr = scheduler.step()
    optimizer.step()

    print("当前 lr =", lr)
    print("scheduler step_num =", scheduler.step_num)


if __name__ == "__main__":
    main()