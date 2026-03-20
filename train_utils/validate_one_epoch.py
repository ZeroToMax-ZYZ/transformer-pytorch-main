from __future__ import annotations

"""
功能：
1. 执行一个 epoch 的验证。
2. 计算：
   - valid_loss
   - valid_nll_loss
   - valid_smooth_loss
   - valid_token_acc
   - valid_ppl
3. 可选记录少量文本样例（source/reference/prediction）。

说明：
1. 第一版验证以 loss / ppl / token_acc 为主。
2. 文本样例默认使用 greedy decode。
3. 这里不引入 beam search，后面再单独加。
"""

from typing import Dict, List, Optional

import torch

from data.batch import make_tgt_mask
from utils.label_smoothing import (
    LabelSmoothingLoss,
    compute_perplexity_from_loss,
    compute_token_accuracy,
)


def _ids_to_bpe_string(
    ids: List[int],
    id_to_token_func,
    pad_id: int,
    bos_id: int,
    eos_id: int,
) -> str:
    """
    将 id 序列恢复成便于人读的 BPE 文本。
    """
    tokens: List[str] = []
    for idx in ids:
        if idx == pad_id:
            continue
        if idx == bos_id:
            continue
        if idx == eos_id:
            break
        tokens.append(id_to_token_func(idx))

    # subword-nmt 的 @@ 拼接还原
    text = " ".join(tokens).replace("@@ ", "")
    return text.strip()


@torch.no_grad()
def greedy_decode(
    model: torch.nn.Module,
    src: torch.Tensor,
    src_mask: torch.Tensor,
    bos_id: int,
    eos_id: int,
    pad_id: int,
    max_len: int,
) -> torch.Tensor:
    """
    Greedy 解码。

    输入：
        src: (B, S)
        src_mask: (B, 1, S)
        bos_id / eos_id / pad_id: 特殊 token id
        max_len: 生成最大长度

    输出：
        ys: (B, T_pred)，包含 BOS 开头，后续逐步生成
    """
    memory = model.encode(src, src_mask)
    batch_size = src.size(0)

    ys = torch.full(
        (batch_size, 1),
        fill_value=bos_id,
        dtype=torch.long,
        device=src.device,
    )

    finished = torch.zeros(batch_size, dtype=torch.bool, device=src.device)

    for _ in range(max_len - 1):
        tgt_mask = make_tgt_mask(ys, pad_idx=pad_id)
        out = model.decode(memory, src_mask, ys, tgt_mask)
        logits = model.generator(out[:, -1, :])  # (B, V)
        next_token = logits.argmax(dim=-1, keepdim=True)  # (B, 1)

        ys = torch.cat([ys, next_token], dim=1)
        finished = finished | next_token.squeeze(1).eq(eos_id)

        if finished.all():
            break

    return ys


@torch.no_grad()
def validate_one_epoch(
    model: torch.nn.Module,
    valid_loader,
    criterion: LabelSmoothingLoss,
    device: torch.device,
    epoch: int,
    tb_logger=None,
    vocab=None,
    num_text_samples: int = 3,
    max_decode_extra_len: int = 50,
    max_steps_per_epoch: Optional[int] = None,
) -> Dict[str, float]:
    """
    验证一个 epoch。

    入口：
        model: Transformer 模型
        valid_loader: 验证集 DataLoader
        criterion: LabelSmoothingLoss
        device: 设备
        epoch: 当前 epoch
        tb_logger: TensorBoard 日志器
        vocab: SharedVocab，可选；若提供则可记录文本样例
        num_text_samples: 记录多少个文本样例
        max_decode_extra_len: greedy decode 时的最大额外长度
        max_steps_per_epoch: 调试时限制验证 step 数

    出口：
        stats: 验证 epoch 聚合指标
    """
    model.eval()

    total_loss_sum = 0.0
    total_nll_sum = 0.0
    total_smooth_sum = 0.0
    total_correct_tokens = 0
    total_tokens = 0

    sample_batch = None

    for step_idx, batch in enumerate(valid_loader, start=1):
        if max_steps_per_epoch is not None and step_idx > max_steps_per_epoch:
            break

        batch = batch.to(device)

        if sample_batch is None and vocab is not None and num_text_samples > 0:
            sample_batch = batch

        hidden_states = model(
            batch.src,
            batch.tgt_input,
            batch.src_mask,
            batch.tgt_mask,
        )
        logits = model.generator(hidden_states)

        loss_output = criterion(logits, batch.tgt_y)
        acc_output = compute_token_accuracy(logits, batch.tgt_y, criterion.pad_idx)

        batch_tokens = loss_output.num_tokens

        total_loss_sum += float(loss_output.loss.detach().item()) * batch_tokens
        total_nll_sum += float(loss_output.nll_loss.detach().item()) * batch_tokens
        total_smooth_sum += float(loss_output.smooth_loss.detach().item()) * batch_tokens
        total_correct_tokens += int(acc_output["correct_tokens"])
        total_tokens += int(acc_output["total_tokens"])

    if total_tokens == 0:
        raise ValueError("验证阶段有效 token 数为 0，无法汇总指标。")

    avg_loss = total_loss_sum / total_tokens
    avg_nll_loss = total_nll_sum / total_tokens
    avg_smooth_loss = total_smooth_sum / total_tokens
    avg_token_acc = total_correct_tokens / total_tokens
    avg_ppl = compute_perplexity_from_loss(avg_nll_loss)

    stats = {
        "epoch": float(epoch),
        "valid_loss": float(avg_loss),
        "valid_nll_loss": float(avg_nll_loss),
        "valid_smooth_loss": float(avg_smooth_loss),
        "valid_token_acc": float(avg_token_acc),
        "valid_ppl": float(avg_ppl),
    }

    if tb_logger is not None:
        tb_logger.log_valid_epoch(epoch=epoch, stats=stats)

    # -------------------------
    # 记录少量文本样例
    # -------------------------
    if tb_logger is not None and vocab is not None and sample_batch is not None and num_text_samples > 0:
        sample_src = sample_batch.src[:num_text_samples]
        sample_src_mask = sample_batch.src_mask[:num_text_samples]
        sample_tgt_y = sample_batch.tgt_y[:num_text_samples]

        max_len = int(sample_src.size(1) + max_decode_extra_len)

        pred_ids = greedy_decode(
            model=model,
            src=sample_src,
            src_mask=sample_src_mask,
            bos_id=vocab.bos_id,
            eos_id=vocab.eos_id,
            pad_id=vocab.pad_id,
            max_len=max_len,
        )

        samples: List[Dict[str, str]] = []
        for i in range(sample_src.size(0)):
            source_text = _ids_to_bpe_string(
                ids=sample_src[i].tolist(),
                id_to_token_func=vocab.id2token,
                pad_id=vocab.pad_id,
                bos_id=vocab.bos_id,
                eos_id=vocab.eos_id,
            )
            reference_text = _ids_to_bpe_string(
                ids=sample_tgt_y[i].tolist(),
                id_to_token_func=vocab.id2token,
                pad_id=vocab.pad_id,
                bos_id=vocab.bos_id,
                eos_id=vocab.eos_id,
            )
            prediction_text = _ids_to_bpe_string(
                ids=pred_ids[i].tolist(),
                id_to_token_func=vocab.id2token,
                pad_id=vocab.pad_id,
                bos_id=vocab.bos_id,
                eos_id=vocab.eos_id,
            )

            samples.append(
                {
                    "source": source_text,
                    "reference": reference_text,
                    "prediction": prediction_text,
                }
            )

        tb_logger.log_text_samples(
            epoch=epoch,
            samples=samples,
            tag="valid_samples_greedy",
        )

    return stats