from __future__ import annotations

import os
from typing import Dict

from datasets import load_dataset
from tqdm import tqdm


"""
功能：
1. 从 Hugging Face 加载 wmt14 / de-en 数据集。
2. 将 train / validation / test 三个 split 导出为严格对齐的 txt 文件。
3. 通过“强制单行化”保证每条样本最终只占一行，避免 \r / \n / 其他行分隔符破坏对齐。

输出目录结构：
data/wmt14_raw_en_de/
├── train.en
├── train.de
├── valid.en
├── valid.de
├── test.en
└── test.de
"""


def sanitize_to_single_line(text: str) -> str:
    """
    将任意文本强制压成单行文本。

    输入：
        text: 原始字符串

    输出：
        单行字符串

    设计说明：
        1. splitlines() 会按各种常见“换行类字符”切分，而不仅仅是 '\n'
        2. 再用一个空格拼接，保证最终严格一行一句
        3. 最后 strip() 去掉首尾空白
    """
    single_line = " ".join(text.splitlines()).strip()

    # 额外保险：如果仍然残留最常见的回车/换行字符，继续替换
    single_line = single_line.replace("\r", " ").replace("\n", " ").strip()

    return single_line


def export_split_to_text(
    dataset_split,
    out_en_path: str,
    out_de_path: str,
    desc_name: str,
) -> None:
    """
    将指定 split 导出为严格对齐的英德文本文件。
    """
    os.makedirs(os.path.dirname(out_en_path), exist_ok=True)

    print(f"\n正在导出 {desc_name}")
    print(f"  EN -> {out_en_path}")
    print(f"  DE -> {out_de_path}")

    num_pairs = 0

    # newline="\n" 用来统一输出换行风格，避免平台差异带来的额外混乱
    with open(out_en_path, "w", encoding="utf-8", newline="\n") as f_en, \
         open(out_de_path, "w", encoding="utf-8", newline="\n") as f_de:

        for item in tqdm(dataset_split, desc=f"导出 {desc_name}"):
            en_text = sanitize_to_single_line(item["translation"]["en"])
            de_text = sanitize_to_single_line(item["translation"]["de"])

            # 不在这里跳过空样本，先保持原始对齐完整
            # 后续 train 清洗阶段再统一处理
            f_en.write(en_text + "\n")
            f_de.write(de_text + "\n")

            num_pairs += 1

    print(f"完成：{desc_name}，共导出 {num_pairs} 对句子。")


def count_lines(path: str) -> int:
    """
    统计文本文件的行数。
    """
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def check_parallel_files(en_path: str, de_path: str) -> None:
    """
    检查双语文件行数是否一致。
    """
    en_n = count_lines(en_path)
    de_n = count_lines(de_path)

    print(f"{en_path}: {en_n}")
    print(f"{de_path}: {de_n}")

    if en_n != de_n:
        raise AssertionError(f"行数不一致: {en_path} vs {de_path}")

    print("行数一致，检查通过。")


def export_all_wmt14(out_dir: str = "data/wmt14_raw_en_de") -> None:
    """
    导出 Hugging Face 的 WMT14 de-en 全部 split。
    """
    print("正在加载 Hugging Face 数据集: wmt14 / de-en")
    dataset = load_dataset("wmt14", "de-en")

    export_split_to_text(
        dataset["train"],
        os.path.join(out_dir, "train.en"),
        os.path.join(out_dir, "train.de"),
        "训练集 train",
    )

    export_split_to_text(
        dataset["validation"],
        os.path.join(out_dir, "valid.en"),
        os.path.join(out_dir, "valid.de"),
        "验证集 validation",
    )

    export_split_to_text(
        dataset["test"],
        os.path.join(out_dir, "test.en"),
        os.path.join(out_dir, "test.de"),
        "测试集 test",
    )

    print("\n开始做导出后的一致性检查...\n")

    check_parallel_files(
        os.path.join(out_dir, "train.en"),
        os.path.join(out_dir, "train.de"),
    )
    check_parallel_files(
        os.path.join(out_dir, "valid.en"),
        os.path.join(out_dir, "valid.de"),
    )
    check_parallel_files(
        os.path.join(out_dir, "test.en"),
        os.path.join(out_dir, "test.de"),
    )

    print("\n全部导出并检查完成。")


if __name__ == "__main__":
    export_all_wmt14()