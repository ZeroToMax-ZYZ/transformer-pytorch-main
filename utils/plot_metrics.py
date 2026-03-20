from __future__ import annotations

"""
功能：
1. 从 CSV 中读取训练日志。
2. 生成本地 matplotlib 曲线图（png）。
3. 作为 TensorBoard 之外的离线趋势观察工具。

说明：
1. 每张图只画一个坐标系，不做 subplot。
2. 不手工指定颜色，保持默认 matplotlib 配置。
"""

import csv
import os
from typing import Dict, List

import matplotlib.pyplot as plt


def _read_csv_as_dict_list(csv_path: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def _to_float_list(rows: List[Dict[str, str]], key: str) -> List[float]:
    values: List[float] = []
    for row in rows:
        value = row.get(key, "")
        if value == "" or value is None:
            values.append(float("nan"))
        else:
            values.append(float(value))
    return values


def plot_single_curve(
    csv_path: str,
    x_key: str,
    y_keys: List[str],
    out_png_path: str,
    title: str,
    xlabel: str,
    ylabel: str,
) -> None:
    """
    从 CSV 中读取多个 y 序列，画到同一张图上。
    """
    rows = _read_csv_as_dict_list(csv_path)
    if len(rows) == 0:
        return

    x_values = _to_float_list(rows, x_key)

    plt.figure(figsize=(10, 6))
    for y_key in y_keys:
        y_values = _to_float_list(rows, y_key)
        plt.plot(x_values, y_values, label=y_key)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    if len(y_keys) > 1:
        plt.legend()

    os.makedirs(os.path.dirname(out_png_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png_path, dpi=150)
    plt.close()


def plot_default_transformer_curves(
    csv_path: str,
    out_dir: str,
) -> None:
    """
    为 Transformer 训练日志画一组默认曲线图。
    """
    os.makedirs(out_dir, exist_ok=True)

    plot_single_curve(
        csv_path=csv_path,
        x_key="epoch",
        y_keys=["train_loss", "valid_loss"],
        out_png_path=os.path.join(out_dir, "curve_loss.png"),
        title="Loss vs Epoch",
        xlabel="Epoch",
        ylabel="Loss",
    )

    plot_single_curve(
        csv_path=csv_path,
        x_key="epoch",
        y_keys=["train_nll_loss", "train_smooth_loss"],
        out_png_path=os.path.join(out_dir, "curve_train_loss_components.png"),
        title="Train Loss Components vs Epoch",
        xlabel="Epoch",
        ylabel="Loss",
    )

    plot_single_curve(
        csv_path=csv_path,
        x_key="epoch",
        y_keys=["train_token_acc", "valid_token_acc"],
        out_png_path=os.path.join(out_dir, "curve_token_acc.png"),
        title="Token Accuracy vs Epoch",
        xlabel="Epoch",
        ylabel="Accuracy",
    )

    plot_single_curve(
        csv_path=csv_path,
        x_key="epoch",
        y_keys=["train_ppl", "valid_ppl"],
        out_png_path=os.path.join(out_dir, "curve_ppl.png"),
        title="Perplexity vs Epoch",
        xlabel="Epoch",
        ylabel="PPL",
    )

    plot_single_curve(
        csv_path=csv_path,
        x_key="epoch",
        y_keys=["lr_last"],
        out_png_path=os.path.join(out_dir, "curve_lr.png"),
        title="Learning Rate vs Epoch",
        xlabel="Epoch",
        ylabel="LR",
    )