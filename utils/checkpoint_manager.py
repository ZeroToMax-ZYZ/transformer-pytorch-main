from __future__ import annotations

"""
功能：
1. 管理 Transformer 训练过程中的 checkpoint 保存。
2. 固定保留：
   - best.pth
   - last.pth
3. 周期性保存：
   - model_epoch_{epoch}_valppl_{metric}.pth
4. 支持恢复：
   - model
   - optimizer
   - scheduler
   - scaler
"""

import os
import shutil
from typing import Dict, Optional

import torch


class CheckpointManager:
    """
    checkpoint 管理器。

    当前默认监控指标：
        valid_ppl（越小越好）
    """

    def __init__(self, save_dir: str, monitor_key: str = "valid_ppl", mode: str = "min") -> None:
        self.save_dir = save_dir
        self.monitor_key = monitor_key
        self.mode = mode

        if self.mode not in {"min", "max"}:
            raise ValueError("mode 只能是 'min' 或 'max'。")

        os.makedirs(self.save_dir, exist_ok=True)

        self.best_metric: Optional[float] = None

    def _is_better(self, metric: float) -> bool:
        if self.best_metric is None:
            return True
        if self.mode == "min":
            return metric < self.best_metric
        return metric > self.best_metric

    def _build_state(
        self,
        epoch: int,
        global_step: int,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer],
        scheduler,
        scaler,
        train_stats: Dict,
        valid_stats: Dict,
        config: Dict,
    ) -> Dict:
        state = {
            "epoch": epoch,
            "global_step": global_step,
            "model_state_dict": model.state_dict(),
            "train_stats": train_stats,
            "valid_stats": valid_stats,
            "config": config,
        }

        if optimizer is not None:
            state["optimizer_state_dict"] = optimizer.state_dict()

        if scheduler is not None and hasattr(scheduler, "state_dict"):
            state["scheduler_state_dict"] = scheduler.state_dict()

        if scaler is not None:
            state["scaler_state_dict"] = scaler.state_dict()

        return state

    def save_last(
        self,
        epoch: int,
        global_step: int,
        model: torch.nn.Module,
        optimizer,
        scheduler,
        scaler,
        train_stats: Dict,
        valid_stats: Dict,
        config: Dict,
    ) -> str:
        """
        保存 last.pth
        """
        state = self._build_state(
            epoch=epoch,
            global_step=global_step,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            train_stats=train_stats,
            valid_stats=valid_stats,
            config=config,
        )

        out_path = os.path.join(self.save_dir, "last.pth")
        torch.save(state, out_path)
        return out_path

    def save_best_if_needed(
        self,
        epoch: int,
        global_step: int,
        model: torch.nn.Module,
        optimizer,
        scheduler,
        scaler,
        train_stats: Dict,
        valid_stats: Dict,
        config: Dict,
    ) -> Optional[str]:
        """
        若当前验证指标优于历史 best，则保存 best.pth
        """
        if self.monitor_key not in valid_stats:
            raise KeyError(f"valid_stats 中缺少监控字段 {self.monitor_key}。")

        metric = float(valid_stats[self.monitor_key])

        if not self._is_better(metric):
            return None

        self.best_metric = metric

        state = self._build_state(
            epoch=epoch,
            global_step=global_step,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            train_stats=train_stats,
            valid_stats=valid_stats,
            config=config,
        )

        out_path = os.path.join(self.save_dir, "best.pth")
        torch.save(state, out_path)
        return out_path

    def save_periodic(
        self,
        epoch: int,
        global_step: int,
        model: torch.nn.Module,
        optimizer,
        scheduler,
        scaler,
        train_stats: Dict,
        valid_stats: Dict,
        config: Dict,
    ) -> str:
        """
        按用户偏好保存周期性权重文件，文件名携带 epoch 和核心评估指标。
        """
        metric = float(valid_stats.get(self.monitor_key, float("nan")))
        filename = f"model_epoch_{epoch:03d}_{self.monitor_key}_{metric:.4f}.pth"
        out_path = os.path.join(self.save_dir, filename)

        state = self._build_state(
            epoch=epoch,
            global_step=global_step,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            train_stats=train_stats,
            valid_stats=valid_stats,
            config=config,
        )

        torch.save(state, out_path)
        return out_path