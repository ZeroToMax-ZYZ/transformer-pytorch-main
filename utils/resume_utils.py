from __future__ import annotations

"""
功能：
1. 从 checkpoint 恢复模型、优化器、scheduler、scaler。
2. 返回恢复训练所需的起始 epoch、global_step、best_metric 等信息。
"""

from typing import Any, Dict, Optional

import torch


def load_checkpoint_for_resume(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    map_location: str = "cpu",
) -> Dict[str, Any]:
    """
    加载 checkpoint 并恢复训练状态。

    返回字段：
        - epoch
        - global_step
        - best_metric
        - train_stats
        - valid_stats
        - config
    """
    checkpoint = torch.load(checkpoint_path, map_location=map_location)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    if scaler is not None and "scaler_state_dict" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])

    valid_stats = checkpoint.get("valid_stats", {})
    best_metric = None
    if isinstance(valid_stats, dict) and "valid_ppl" in valid_stats:
        best_metric = float(valid_stats["valid_ppl"])

    return {
        "epoch": int(checkpoint.get("epoch", 0)),
        "global_step": int(checkpoint.get("global_step", 0)),
        "best_metric": best_metric,
        "train_stats": checkpoint.get("train_stats", {}),
        "valid_stats": checkpoint.get("valid_stats", {}),
        "config": checkpoint.get("config", {}),
    }