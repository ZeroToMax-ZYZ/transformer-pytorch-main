from __future__ import annotations

"""
功能：
1. 给正式训练阶段提供一个“构建真实训练对象”的参考脚本。
2. 不直接开训，只负责把：
   - vocab
   - train_loader
   - valid_loader
   - model
   - criterion
   - optimizer
   - scheduler
全部构建出来，便于单独调试。
"""

import torch

from data.shared_vocab import SharedVocab
from data.wmt14_bpe_dataset import build_bpe_dataloader
from nets.build_transformer import make_model
from utils.label_smoothing import LabelSmoothingLoss
from utils.noam_scheduler import build_transformer_optimizer_and_scheduler


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab = SharedVocab.load("data/wmt14_vocab/vocab.json")
    vocab_size = len(vocab)

    train_loader = build_bpe_dataloader(
        src_path="data/wmt14_bpe_en_de/train.en",
        tgt_path="data/wmt14_bpe_en_de/train.de",
        vocab=vocab,
        batch_size=64,
        num_workers=2,
        pin_memory=True,
        max_src_len=None,
        max_tgt_len=None,
        add_src_eos=True,
        skip_empty=False,
        shuffle_buffer_size=10000,
        seed=42,
        num_samples=3927488,
        persistent_workers=True,
        prefetch_factor=2,
    )

    valid_loader = build_bpe_dataloader(
        src_path="data/wmt14_bpe_en_de/valid.en",
        tgt_path="data/wmt14_bpe_en_de/valid.de",
        vocab=vocab,
        batch_size=64,
        num_workers=0,
        pin_memory=True,
        max_src_len=None,
        max_tgt_len=None,
        add_src_eos=True,
        skip_empty=False,
        shuffle_buffer_size=0,
        seed=42,
        num_samples=3000,
        persistent_workers=False,
        prefetch_factor=None,
    )

    model = make_model(
        src_vocab=vocab_size,
        tgt_vocab=vocab_size,
        N=6,
        d_model=512,
        d_ff=2048,
        h=8,
        dropout=0.1,
        share_embeddings=True,
    ).to(device)

    criterion = LabelSmoothingLoss(
        vocab_size=vocab_size,
        pad_idx=vocab.pad_id,
        smoothing=0.1,
    )

    optimizer, scheduler = build_transformer_optimizer_and_scheduler(
        model=model,
        d_model=512,
        warmup_steps=4000,
        factor=1.0,
        beta1=0.9,
        beta2=0.98,
        eps=1e-9,
        weight_decay=0.0,
    )

    print("对象构建成功。")
    print(f"device = {device}")
    print(f"vocab_size = {vocab_size}")
    print(f"pad_id = {vocab.pad_id}, bos_id = {vocab.bos_id}, eos_id = {vocab.eos_id}")
    print("train_loader / valid_loader / model / criterion / optimizer / scheduler 均已构建完成。")


if __name__ == "__main__":
    main()