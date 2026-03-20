from __future__ import annotations

"""
功能：
1. 用真实的 WMT14 BPE 数据 + 真实训练组件，
   跑一个极小规模的训练闭环 smoke test。
2. 验证：
   - train_one_epoch
   - validate_one_epoch
   - fit
   - csv/png/tb/checkpoint
整体是否能协同工作。
"""

import os
import torch

from data.shared_vocab import SharedVocab
from data.wmt_14_bpe_dataset import build_bpe_dataloader
from nets.build_transformer import make_model
from train_utils.fit import fit
from utils.label_smoothing import LabelSmoothingLoss
from utils.noam_scheduler import build_transformer_optimizer_and_scheduler
from utils.train_env import get_device, seed_everything


def main() -> None:
    device = get_device()
    seed_everything(seed=42, deterministic=False)

    vocab = SharedVocab.load("data/wmt14_vocab/vocab.json")
    vocab_size = len(vocab)

    train_loader = build_bpe_dataloader(
        src_path="data/wmt14_bpe_en_de/train.en",
        tgt_path="data/wmt14_bpe_en_de/train.de",
        vocab=vocab,
        batch_size=8,
        num_workers=0,
        pin_memory=False,
        max_src_len=64,
        max_tgt_len=64,
        add_src_eos=True,
        skip_empty=False,
        shuffle_buffer_size=1000,
        seed=42,
        num_samples=3927488,
        persistent_workers=False,
        prefetch_factor=None,
    )

    valid_loader = build_bpe_dataloader(
        src_path="data/wmt14_bpe_en_de/valid.en",
        tgt_path="data/wmt14_bpe_en_de/valid.de",
        vocab=vocab,
        batch_size=8,
        num_workers=0,
        pin_memory=False,
        max_src_len=64,
        max_tgt_len=64,
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
        N=2,
        d_model=256,
        d_ff=1024,
        h=4,
        dropout=0.1,
        share_embeddings=True,
    )

    criterion = LabelSmoothingLoss(
        vocab_size=vocab_size,
        pad_idx=vocab.pad_id,
        smoothing=0.1,
    )

    optimizer, scheduler = build_transformer_optimizer_and_scheduler(
        model=model,
        d_model=256,
        warmup_steps=4000,
        factor=1.0,
        beta1=0.9,
        beta2=0.98,
        eps=1e-9,
        weight_decay=0.0,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=False)

    config = {
        "exp_name": "transformer_train_pipeline_smoke",
        "device": str(device),
        "use_amp": False,
        "model": {
            "N": 2,
            "d_model": 256,
            "d_ff": 1024,
            "h": 4,
            "dropout": 0.1,
            "share_embeddings": True,
        },
        "criterion": {
            "type": "LabelSmoothingLoss",
            "smoothing": 0.1,
        },
        "optimizer": {
            "type": "Adam",
            "beta1": 0.9,
            "beta2": 0.98,
            "eps": 1e-9,
            "weight_decay": 0.0,
        },
        "scheduler": {
            "type": "NoamLRScheduler",
            "warmup_steps": 4000,
            "factor": 1.0,
        },
        "vocab": {
            "vocab_json": "data/wmt14_vocab/vocab.json",
            "vocab_size": vocab_size,
            "pad_id": vocab.pad_id,
            "bos_id": vocab.bos_id,
            "eos_id": vocab.eos_id,
            "unk_id": vocab.unk_id,
        },
        "data": {
            "train_src": "data/wmt14_bpe_en_de/train.en",
            "train_tgt": "data/wmt14_bpe_en_de/train.de",
            "valid_src": "data/wmt14_bpe_en_de/valid.en",
            "valid_tgt": "data/wmt14_bpe_en_de/valid.de",
            "train_num_samples": 3927488,
            "valid_num_samples": 3000,
        },
    }

    fit(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_epochs=2,
        output_dir=os.path.join("experiments", "transformer_train_pipeline_smoke"),
        config=config,
        vocab=vocab,
        scaler=scaler,
        use_amp=False,
        grad_clip_norm=1.0,
        train_log_interval=10,
        histogram_interval=50,
        save_every_epochs=1,
        valid_num_text_samples=2,
        max_train_steps_per_epoch=20,
        max_valid_steps_per_epoch=10,
        start_epoch=1,
        global_step_init=0,
        best_metric_init=None,
    )


if __name__ == "__main__":
    main()