from __future__ import annotations

"""
功能：
1. 执行一个 epoch 的训练。
2. 每个 batch 对应一次参数更新，不使用梯度累计。
3. 返回训练指标，并按需写入 TensorBoard。

实现约定：
1. `scheduler.step()` 先于 `optimizer.step()`，这样当前 update 使用的就是本 step 的 Noam 学习率。
2. 训练阶段使用 teacher forcing：模型输入 `batch.tgt_input`，监督目标是 `batch.tgt_y`。
"""

import time
from typing import Dict, Optional, Tuple

import torch
from torch.nn.utils import clip_grad_norm_
from tqdm.auto import tqdm

from utils.label_smoothing import (
    LabelSmoothingLoss,
    compute_perplexity_from_loss,
    compute_token_accuracy,
)


def train_one_epoch(
    model: torch.nn.Module,
    train_loader,
    criterion: LabelSmoothingLoss,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    epoch: int,
    global_step: int,
    tb_logger=None,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    use_amp: bool = False,
    grad_clip_norm: Optional[float] = 1.0,
    log_interval: int = 100,
    histogram_interval: int = 1000,
    max_steps_per_epoch: Optional[int] = None,
) -> Tuple[Dict[str, float], int]:
    """
    训练一个 epoch。

    入口：
        model: Transformer 模型
        train_loader: 训练 DataLoader
        criterion: LabelSmoothingLoss
        optimizer: 优化器
        scheduler: NoamLRScheduler
        device: 训练设备
        epoch: 当前 epoch 编号（从 1 开始）
        global_step: 全局 update step
        tb_logger: TensorBoard 日志器，可为 None
        scaler: AMP GradScaler，可为 None
        use_amp: 是否启用 autocast
        grad_clip_norm: 梯度裁剪阈值
        log_interval: 每隔多少 step 记录一次 TensorBoard 标量
        histogram_interval: 每隔多少 step 记录一次 histogram
        max_steps_per_epoch: 调试时限制每个 epoch 的最大 step 数

    出口：
        stats: epoch 级聚合统计
        global_step: 更新后的全局 step
    """
    model.train()

    total_loss_sum = 0.0
    total_nll_sum = 0.0
    total_smooth_sum = 0.0
    total_correct_tokens = 0
    total_tokens = 0
    total_grad_norm = 0.0
    total_tokens_per_sec = 0.0
    total_steps = 0

    epoch_start_time = time.time()
    window_start_time = time.time()
    window_loss_sum = 0.0
    window_nll_sum = 0.0
    window_smooth_sum = 0.0
    window_correct_tokens = 0
    window_total_tokens = 0

    optimizer.zero_grad(set_to_none=True)
    progress_total = max_steps_per_epoch
    if progress_total is None:
        try:
            progress_total = len(train_loader)
        except TypeError:
            progress_total = None

    progress_bar = tqdm(
        total=progress_total,
        desc=f"Epoch {epoch} Train",
        dynamic_ncols=True,
        leave=True,
    )

    for batch in train_loader:
        if max_steps_per_epoch is not None and total_steps >= max_steps_per_epoch:
            break

        step_start_time = time.time()
        batch = batch.to(device)

        with torch.cuda.amp.autocast(enabled=(use_amp and device.type == "cuda")):
            hidden_states = model(
                batch.src,
                batch.tgt_input,
                batch.src_mask,
                batch.tgt_mask,
            )
            logits = model.generator(hidden_states)

            loss_output = criterion(logits, batch.tgt_y)
            acc_output = compute_token_accuracy(logits, batch.tgt_y, criterion.pad_idx)

        if scaler is not None and use_amp and device.type == "cuda":
            scaler.scale(loss_output.loss).backward()
            scaler.unscale_(optimizer)
        else:
            loss_output.loss.backward()

        if grad_clip_norm is not None and grad_clip_norm > 0:
            grad_norm = clip_grad_norm_(model.parameters(), grad_clip_norm)
        else:
            grad_norm = torch.tensor(0.0, device=device)

        current_lr = scheduler.step()
        if scaler is not None and use_amp and device.type == "cuda":
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        optimizer.zero_grad(set_to_none=True)

        batch_tokens = loss_output.num_tokens
        batch_loss = float(loss_output.loss.detach().item())
        batch_nll_loss = float(loss_output.nll_loss.detach().item())
        batch_smooth_loss = float(loss_output.smooth_loss.detach().item())
        batch_correct_tokens = int(acc_output["correct_tokens"])
        batch_total_tokens = int(acc_output["total_tokens"])

        global_step += 1
        total_steps += 1
        progress_bar.update(1)

        iter_time = max(time.time() - step_start_time, 1e-8)
        tokens_per_sec = batch_tokens / iter_time

        total_loss_sum += batch_loss * batch_tokens
        total_nll_sum += batch_nll_loss * batch_tokens
        total_smooth_sum += batch_smooth_loss * batch_tokens
        total_correct_tokens += batch_correct_tokens
        total_tokens += batch_total_tokens
        total_grad_norm += float(grad_norm.detach().item() if torch.is_tensor(grad_norm) else grad_norm)
        total_tokens_per_sec += tokens_per_sec

        window_loss_sum += batch_loss * batch_tokens
        window_nll_sum += batch_nll_loss * batch_tokens
        window_smooth_sum += batch_smooth_loss * batch_tokens
        window_correct_tokens += batch_correct_tokens
        window_total_tokens += batch_total_tokens

        progress_bar.set_postfix(
            lr=f"{float(current_lr):.2e}",
            loss=f"{batch_loss:.4f}",
            tok_s=f"{tokens_per_sec:.0f}",
        )

        # -------------------------
        # step 级 TensorBoard 记录
        # -------------------------
        if tb_logger is not None and (global_step % log_interval == 0):
            window_time = max(time.time() - window_start_time, 1e-8)
            train_loss_dict = {
                "loss": window_loss_sum / max(window_total_tokens, 1),
                "nll_loss": window_nll_sum / max(window_total_tokens, 1),
                "smooth_loss": window_smooth_sum / max(window_total_tokens, 1),
                "num_tokens": int(window_total_tokens),
            }
            train_ppl = compute_perplexity_from_loss(train_loss_dict["nll_loss"])
            window_token_acc = window_correct_tokens / max(window_total_tokens, 1)

            tb_logger.log_train_step(
                global_step=global_step,
                loss_dict=train_loss_dict,
                token_acc=float(window_token_acc),
                ppl=train_ppl,
                lr=float(current_lr),
                grad_norm=float(grad_norm.detach().item() if torch.is_tensor(grad_norm) else grad_norm),
                ntokens=int(window_total_tokens),
                tokens_per_sec=float(window_total_tokens / window_time),
            )

            window_start_time = time.time()
            window_loss_sum = 0.0
            window_nll_sum = 0.0
            window_smooth_sum = 0.0
            window_correct_tokens = 0
            window_total_tokens = 0

        # -------------------------
        # histogram 记录（克制）
        # -------------------------
        if tb_logger is not None and histogram_interval > 0 and (global_step % histogram_interval == 0):
            with torch.no_grad():
                prediction_confidence = torch.softmax(logits.detach(), dim=-1).amax(dim=-1)
            tb_logger.log_representative_histograms(
                model=model,
                global_step=global_step,
                prediction_confidence=prediction_confidence,
            )

    progress_bar.close()

    # -------------------------
    # epoch 聚合
    # -------------------------
    if total_tokens == 0:
        raise ValueError("当前 epoch 没有有效 token，无法汇总训练指标。")

    avg_loss = total_loss_sum / total_tokens
    avg_nll_loss = total_nll_sum / total_tokens
    avg_smooth_loss = total_smooth_sum / total_tokens
    avg_token_acc = total_correct_tokens / total_tokens
    avg_ppl = compute_perplexity_from_loss(avg_nll_loss)
    avg_grad_norm = total_grad_norm / max(total_steps, 1)
    avg_tokens_per_sec = total_tokens_per_sec / max(total_steps, 1)

    epoch_time_sec = time.time() - epoch_start_time

    stats = {
        "epoch": float(epoch),
        "train_loss": float(avg_loss),
        "train_nll_loss": float(avg_nll_loss),
        "train_smooth_loss": float(avg_smooth_loss),
        "train_token_acc": float(avg_token_acc),
        "train_ppl": float(avg_ppl),
        "avg_grad_norm": float(avg_grad_norm),
        "avg_tokens_per_sec": float(avg_tokens_per_sec),
        "epoch_time_sec": float(epoch_time_sec),
        "lr_last": float(scheduler.get_last_lr()[0]),
        "global_step": float(global_step),
    }

    if tb_logger is not None:
        tb_logger.log_train_epoch(epoch=epoch, stats=stats)

    return stats, global_step
