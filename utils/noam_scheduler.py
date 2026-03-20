from __future__ import annotations

"""
功能：
1. 实现论文中的 Noam 学习率调度器（step-based）。
2. 提供 Adam + Noam 的标准构建入口。
3. 支持 state_dict / load_state_dict，便于断点续训。

注意：
1. Noam 调度本质上是按 optimizer update step 更新，而不是按 epoch 更新。
2. 推荐调用顺序：
   - backward
   - grad clip
   - scheduler.step()   # 先把本次 update 的 lr 设好
   - optimizer.step()
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
from torch import nn


@dataclass
class NoamConfig:
    """
    Noam 调度器配置。
    """
    d_model: int
    warmup_steps: int = 4000
    factor: float = 1.0
    beta1: float = 0.9
    beta2: float = 0.98
    eps: float = 1e-9
    weight_decay: float = 0.0


class NoamLRScheduler:
    """
    论文风格 Noam 学习率调度器。

    数学形式：
        lrate = factor * (d_model ^ -0.5) *
                min(step_num ^ -0.5, step_num * warmup_steps ^ -1.5)

    入口：
        optimizer: PyTorch Optimizer
        d_model: 模型维度
        warmup_steps: 预热步数
        factor: 缩放因子
        init_step: 初始 step（断点恢复时使用）

    出口：
        - step(): 更新内部 step，并把当前 lr 写回 optimizer
        - rate(step): 查询指定 step 的 lr
        - get_last_lr(): 获取最近一次 lr
        - state_dict/load_state_dict(): 状态保存与恢复
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        d_model: int,
        warmup_steps: int = 4000,
        factor: float = 1.0,
        init_step: int = 0,
    ) -> None:
        if d_model <= 0:
            raise ValueError("d_model 必须大于 0。")
        if warmup_steps <= 0:
            raise ValueError("warmup_steps 必须大于 0。")
        if factor <= 0:
            raise ValueError("factor 必须大于 0。")

        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.factor = factor
        self._step = int(init_step)
        self._rate = 0.0

        for group in self.optimizer.param_groups:
            group["lr"] = 0.0

    @property
    def step_num(self) -> int:
        return self._step

    def rate(self, step: Optional[int] = None) -> float:
        """
        计算某个 step 对应的学习率。
        """
        if step is None:
            step = self._step

        step = max(int(step), 1)

        return self.factor * (self.d_model ** -0.5) * min(
            step ** -0.5,
            step * (self.warmup_steps ** -1.5),
        )

    def step(self) -> float:
        """
        推进一步，并将当前 lr 写入 optimizer。

        返回：
            当前 step 对应的 lr
        """
        self._step += 1
        lr = self.rate(self._step)

        for group in self.optimizer.param_groups:
            group["lr"] = lr

        self._rate = lr
        return lr

    def get_last_lr(self) -> list:
        """
        返回最近一次设置的 lr，接口风格与 PyTorch scheduler 对齐。
        """
        return [group["lr"] for group in self.optimizer.param_groups]

    def state_dict(self) -> Dict:
        """
        保存调度器状态。
        """
        return {
            "d_model": self.d_model,
            "warmup_steps": self.warmup_steps,
            "factor": self.factor,
            "_step": self._step,
            "_rate": self._rate,
        }

    def load_state_dict(self, state_dict: Dict) -> None:
        """
        恢复调度器状态。
        """
        self.d_model = int(state_dict["d_model"])
        self.warmup_steps = int(state_dict["warmup_steps"])
        self.factor = float(state_dict["factor"])
        self._step = int(state_dict["_step"])
        self._rate = float(state_dict["_rate"])

        for group in self.optimizer.param_groups:
            group["lr"] = self._rate


def build_transformer_optimizer(
    model: nn.Module,
    beta1: float = 0.9,
    beta2: float = 0.98,
    eps: float = 1e-9,
    weight_decay: float = 0.0,
) -> torch.optim.Adam:
    """
    构建论文风格 Adam 优化器。
    """
    optimizer = torch.optim.Adam(
        params=model.parameters(),
        lr=0.0,
        betas=(beta1, beta2),
        eps=eps,
        weight_decay=weight_decay,
    )
    return optimizer


def build_transformer_optimizer_and_scheduler(
    model: nn.Module,
    d_model: int,
    warmup_steps: int = 4000,
    factor: float = 1.0,
    beta1: float = 0.9,
    beta2: float = 0.98,
    eps: float = 1e-9,
    weight_decay: float = 0.0,
) -> Tuple[torch.optim.Adam, NoamLRScheduler]:
    """
    一次性构建 Adam + Noam 调度器。
    """
    optimizer = build_transformer_optimizer(
        model=model,
        beta1=beta1,
        beta2=beta2,
        eps=eps,
        weight_decay=weight_decay,
    )

    scheduler = NoamLRScheduler(
        optimizer=optimizer,
        d_model=d_model,
        warmup_steps=warmup_steps,
        factor=factor,
    )

    return optimizer, scheduler