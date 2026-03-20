from __future__ import annotations

"""
功能：
1. TensorBoard 日志记录。
2. 记录 step 级训练标量。
3. 记录 epoch 级训练/验证指标。
4. 记录少量代表性 histogram。
5. 记录验证文本样例。

注意：
1. 直方图记录要克制，只记录少量最有诊断价值的层。
2. 文本样例建议只记录少量固定样本，避免 TensorBoard 过重。
"""

import os
from typing import Dict, List, Optional

import torch
from torch.utils.tensorboard import SummaryWriter


class TransformerTBLogger:
    """
    Transformer 训练 TensorBoard 日志器。
    """

    def __init__(self, log_dir: str) -> None:
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=log_dir)

    def log_train_step(
        self,
        global_step: int,
        loss_dict: Dict[str, float],
        token_acc: float,
        ppl: float,
        lr: float,
        grad_norm: float,
        ntokens: int,
        tokens_per_sec: float,
    ) -> None:
        """
        记录训练 step 级标量。
        """
        self.writer.add_scalar("train_step/total_loss", loss_dict["loss"], global_step)
        self.writer.add_scalar("train_step/nll_loss", loss_dict["nll_loss"], global_step)
        self.writer.add_scalar("train_step/smooth_loss", loss_dict["smooth_loss"], global_step)
        self.writer.add_scalar("train_step/token_acc", token_acc, global_step)
        self.writer.add_scalar("train_step/ppl", ppl, global_step)
        self.writer.add_scalar("train_step/lr", lr, global_step)
        self.writer.add_scalar("train_step/grad_norm", grad_norm, global_step)
        self.writer.add_scalar("train_step/ntokens", ntokens, global_step)
        self.writer.add_scalar("train_step/tokens_per_sec", tokens_per_sec, global_step)

    def log_train_epoch(self, epoch: int, stats: Dict[str, float]) -> None:
        """
        记录训练 epoch 级标量。
        """
        for key, value in stats.items():
            self.writer.add_scalar(f"train_epoch/{key}", value, epoch)

    def log_valid_epoch(self, epoch: int, stats: Dict[str, float]) -> None:
        """
        记录验证 epoch 级标量。
        """
        for key, value in stats.items():
            self.writer.add_scalar(f"valid_epoch/{key}", value, epoch)

    def log_representative_histograms(
        self,
        model: torch.nn.Module,
        global_step: int,
        prediction_confidence: Optional[torch.Tensor] = None,
    ) -> None:
        """
        记录少量代表性 histogram。

        默认策略：
            1. embedding / 首层
            2. 编码器中层 FFN
            3. 输出头
            4. 预测最大置信度分布（可选）
        """
        try:
            if hasattr(model, "src_embed") and hasattr(model.src_embed, "lut"):
                self.writer.add_histogram(
                    "hist/src_embed_lut_weight",
                    model.src_embed.lut.weight.detach().cpu(),
                    global_step,
                )
        except Exception:
            pass

        try:
            if hasattr(model, "encoder") and hasattr(model.encoder, "layers"):
                layers = model.encoder.layers
                if len(layers) > 0:
                    self.writer.add_histogram(
                        "hist/encoder_layer0_q_proj_weight",
                        layers[0].self_attn.linears[0].weight.detach().cpu(),
                        global_step,
                    )

                mid_idx = len(layers) // 2
                self.writer.add_histogram(
                    "hist/encoder_mid_ffn_w1_weight",
                    layers[mid_idx].feed_forward.w_1.weight.detach().cpu(),
                    global_step,
                )
        except Exception:
            pass

        try:
            if hasattr(model, "generator") and hasattr(model.generator, "proj"):
                self.writer.add_histogram(
                    "hist/generator_proj_weight",
                    model.generator.proj.weight.detach().cpu(),
                    global_step,
                )
        except Exception:
            pass

        if prediction_confidence is not None:
            try:
                self.writer.add_histogram(
                    "hist/prediction_confidence",
                    prediction_confidence.detach().float().cpu(),
                    global_step,
                )
            except Exception:
                pass

    def log_text_samples(
        self,
        epoch: int,
        samples: List[Dict[str, str]],
        tag: str = "valid_samples",
    ) -> None:
        """
        记录少量验证文本样例。

        每个 sample 约定字段：
            - source
            - reference
            - prediction
        """
        text_blocks: List[str] = []
        for idx, sample in enumerate(samples):
            block = (
                f"### Sample {idx}\n\n"
                f"- Source: {sample.get('source', '')}\n"
                f"- Reference: {sample.get('reference', '')}\n"
                f"- Prediction: {sample.get('prediction', '')}\n"
            )
            text_blocks.append(block)

        final_text = "\n\n---\n\n".join(text_blocks)
        self.writer.add_text(tag, final_text, epoch)

    def flush(self) -> None:
        self.writer.flush()

    def close(self) -> None:
        self.writer.flush()
        self.writer.close()