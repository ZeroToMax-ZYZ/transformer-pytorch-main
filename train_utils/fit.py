from __future__ import annotations

"""
功能：
1. 组织 Transformer 的完整训练流程。
2. 保存：
   - config.json
   - metrics.csv
   - matplotlib 曲线 png
   - TensorBoard 日志
   - best / last / periodic checkpoint
3. 调用：
   - train_one_epoch
   - validate_one_epoch
4. 支持 resume：
   - start_epoch
   - global_step_init
   - best_metric_init

说明：
1. 第一版优先做论文主干对齐：
   - Label Smoothing
   - Adam + Noam
   - step 级训练日志
   - valid loss/ppl/token_acc
2. BLEU、beam search、checkpoint averaging 后续再接。
"""

import json
import os
from typing import Dict, Optional

import torch

from train_utils.train_one_epoch import train_one_epoch
from train_utils.validate_one_epoch import validate_one_epoch
from utils.checkpoint_manager import CheckpointManager
from utils.csv_logger import CSVMetricLogger
from utils.plot_metrics import plot_default_transformer_curves
from utils.tb_log import TransformerTBLogger


def _make_json_safe(obj):
    """
    将配置对象转成可 JSON 序列化的形式。
    """
    if isinstance(obj, dict):
        return {k: _make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_json_safe(v) for v in obj]
    if isinstance(obj, torch.device):
        return str(obj)
    return obj


def fit(
    model: torch.nn.Module,
    train_loader,
    valid_loader,
    criterion,
    optimizer,
    scheduler,
    device: torch.device,
    num_epochs: int,
    output_dir: str,
    config: Dict,
    vocab=None,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    use_amp: bool = False,
    grad_clip_norm: Optional[float] = 1.0,
    train_log_interval: int = 100,
    histogram_interval: int = 1000,
    save_every_epochs: int = 1,
    valid_num_text_samples: int = 3,
    max_train_steps_per_epoch: Optional[int] = None,
    max_valid_steps_per_epoch: Optional[int] = None,
    start_epoch: int = 1,
    global_step_init: int = 0,
    best_metric_init: Optional[float] = None,
) -> None:
    """
    顶层训练入口。

    负责串起整轮实验管理：
    1. 保存配置、曲线和 checkpoint。
    2. 调用 `train_one_epoch` 与 `validate_one_epoch`。
    3. 维护 resume 所需的 epoch / global_step / best metric。
    """
    os.makedirs(output_dir, exist_ok=True)

    tb_dir = os.path.join(output_dir, "tb")
    ckpt_dir = os.path.join(output_dir, "checkpoints")
    plot_dir = os.path.join(output_dir, "plots")
    csv_path = os.path.join(output_dir, "metrics.csv")
    config_path = os.path.join(output_dir, "config.json")

    os.makedirs(tb_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    safe_config = _make_json_safe(config)
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(safe_config, f, ensure_ascii=False, indent=2)

    tb_logger = TransformerTBLogger(log_dir=tb_dir)

    csv_fieldnames = [
        "epoch",
        "global_step",
        "train_loss",
        "train_nll_loss",
        "train_smooth_loss",
        "train_token_acc",
        "train_ppl",
        "valid_loss",
        "valid_nll_loss",
        "valid_smooth_loss",
        "valid_token_acc",
        "valid_ppl",
        "avg_grad_norm",
        "avg_tokens_per_sec",
        "epoch_time_sec",
        "lr_last",
    ]
    csv_logger = CSVMetricLogger(
        csv_path=csv_path,
        fieldnames=csv_fieldnames,
    )

    ckpt_manager = CheckpointManager(
        save_dir=ckpt_dir,
        monitor_key="valid_ppl",
        mode="min",
    )
    ckpt_manager.best_metric = best_metric_init

    global_step = int(global_step_init)

    model = model.to(device)

    for epoch in range(start_epoch, num_epochs + 1):
        if hasattr(train_loader, "dataset") and hasattr(train_loader.dataset, "set_epoch"):
            train_loader.dataset.set_epoch(epoch)

        print("=" * 100)
        print(f"[Epoch {epoch}/{num_epochs}] 开始训练")
        print("=" * 100)

        train_stats, global_step = train_one_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            epoch=epoch,
            global_step=global_step,
            tb_logger=tb_logger,
            scaler=scaler,
            use_amp=use_amp,
            grad_clip_norm=grad_clip_norm,
            log_interval=train_log_interval,
            histogram_interval=histogram_interval,
            max_steps_per_epoch=max_train_steps_per_epoch,
        )

        valid_stats = validate_one_epoch(
            model=model,
            valid_loader=valid_loader,
            criterion=criterion,
            device=device,
            epoch=epoch,
            tb_logger=tb_logger,
            vocab=vocab,
            num_text_samples=valid_num_text_samples,
            max_decode_extra_len=50,
            max_steps_per_epoch=max_valid_steps_per_epoch,
        )

        row = {
            "epoch": epoch,
            "global_step": int(global_step),
            "train_loss": train_stats["train_loss"],
            "train_nll_loss": train_stats["train_nll_loss"],
            "train_smooth_loss": train_stats["train_smooth_loss"],
            "train_token_acc": train_stats["train_token_acc"],
            "train_ppl": train_stats["train_ppl"],
            "valid_loss": valid_stats["valid_loss"],
            "valid_nll_loss": valid_stats["valid_nll_loss"],
            "valid_smooth_loss": valid_stats["valid_smooth_loss"],
            "valid_token_acc": valid_stats["valid_token_acc"],
            "valid_ppl": valid_stats["valid_ppl"],
            "avg_grad_norm": train_stats["avg_grad_norm"],
            "avg_tokens_per_sec": train_stats["avg_tokens_per_sec"],
            "epoch_time_sec": train_stats["epoch_time_sec"],
            "lr_last": train_stats["lr_last"],
        }
        csv_logger.append_row(row)

        plot_default_transformer_curves(
            csv_path=csv_path,
            out_dir=plot_dir,
        )

        last_path = ckpt_manager.save_last(
            epoch=epoch,
            global_step=global_step,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            train_stats=train_stats,
            valid_stats=valid_stats,
            config=safe_config,
        )

        best_path = ckpt_manager.save_best_if_needed(
            epoch=epoch,
            global_step=global_step,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            train_stats=train_stats,
            valid_stats=valid_stats,
            config=safe_config,
        )

        periodic_path = None
        if save_every_epochs > 0 and (epoch % save_every_epochs == 0):
            periodic_path = ckpt_manager.save_periodic(
                epoch=epoch,
                global_step=global_step,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                train_stats=train_stats,
                valid_stats=valid_stats,
                config=safe_config,
            )

        print("-" * 100)
        print(
            f"[Epoch {epoch}] "
            f"train_loss={train_stats['train_loss']:.6f}, "
            f"train_ppl={train_stats['train_ppl']:.4f}, "
            f"train_token_acc={train_stats['train_token_acc']:.4f}, "
            f"valid_loss={valid_stats['valid_loss']:.6f}, "
            f"valid_ppl={valid_stats['valid_ppl']:.4f}, "
            f"valid_token_acc={valid_stats['valid_token_acc']:.4f}, "
            f"lr={train_stats['lr_last']:.8f}"
        )
        print(f"last checkpoint: {last_path}")
        if best_path is not None:
            print(f"best checkpoint: {best_path}")
        if periodic_path is not None:
            print(f"periodic checkpoint: {periodic_path}")
        print("-" * 100)

        tb_logger.flush()

    tb_logger.close()
