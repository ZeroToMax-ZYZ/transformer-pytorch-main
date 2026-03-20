from __future__ import annotations

"""
功能：
1. 构建正式训练所需的全部对象：
   - vocab
   - train_loader
   - valid_loader
   - model
   - criterion
   - optimizer
   - scheduler
   - scaler
2. 启动 Transformer WMT14 英德训练。

说明：
1. 当前版本采用固定 batch_size + 流式 buffer shuffle。
2. 这是“第一版正式训练入口”。
3. 若后续要更贴论文 batching，可再升级成近似长度分桶 + token budget。
"""

import os
import torch

from data.shared_vocab import SharedVocab
from data.wmt_14_bpe_dataset import build_bpe_dataloader
from nets.build_transformer import make_model
from train_utils.fit import fit
from utils.label_smoothing import LabelSmoothingLoss
from utils.noam_scheduler import build_transformer_optimizer_and_scheduler
from utils.train_env import count_trainable_parameters, get_device, get_timestamp_str, seed_everything


def build_config(device: torch.device, vocab_size: int) -> dict:
    """
    构建实验配置。
    """
    use_amp = device.type == "cuda"

    config = {
        "exp_name": "transformer_wmt14_en_de_base",
        "seed": 42,
        "device": str(device),
        "use_amp": use_amp,

        "data": {
            "train_src": "data/wmt14_bpe_en_de/train.en",
            "train_tgt": "data/wmt14_bpe_en_de/train.de",
            "valid_src": "data/wmt14_bpe_en_de/valid.en",
            "valid_tgt": "data/wmt14_bpe_en_de/valid.de",
            "train_num_samples": 3927488,
            "valid_num_samples": 3000,
        },

        "vocab": {
            "vocab_json": "data/wmt14_vocab/vocab.json",
            "vocab_size": vocab_size,
            "pad_id": 0,
            "bos_id": 1,
            "eos_id": 2,
            "unk_id": 3,
        },

        "model": {
            "N": 6,
            "d_model": 512,
            "d_ff": 2048,
            "h": 8,
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

        "train_loader": {
            "batch_size": 16,
            "num_workers": 2,
            "pin_memory": device.type == "cuda",
            "max_src_len": None,
            "max_tgt_len": None,
            "add_src_eos": True,
            "skip_empty": False,
            "shuffle_buffer_size": 10000,
            "seed": 42,
            "persistent_workers": True,
            "prefetch_factor": 2,
        },

        "valid_loader": {
            "batch_size": 16,
            "num_workers": 0,
            "pin_memory": device.type == "cuda",
            "max_src_len": None,
            "max_tgt_len": None,
            "add_src_eos": True,
            "skip_empty": False,
            "shuffle_buffer_size": 0,
            "seed": 42,
            "persistent_workers": False,
            "prefetch_factor": None,
        },

        "fit": {
            "num_epochs": 30,
            "grad_clip_norm": 1.0,
            "train_log_interval": 100,
            "histogram_interval": 1000,
            "save_every_epochs": 1,
            "valid_num_text_samples": 3,
            "max_train_steps_per_epoch": None,
            "max_valid_steps_per_epoch": None,
        },
    }
    return config


def main() -> None:
    device = get_device()
    seed_everything(seed=42, deterministic=False)

    vocab = SharedVocab.load("data/wmt14_vocab/vocab.json")
    vocab_size = len(vocab)

    config = build_config(device=device, vocab_size=vocab_size)

    train_loader = build_bpe_dataloader(
        src_path=config["data"]["train_src"],
        tgt_path=config["data"]["train_tgt"],
        vocab=vocab,
        batch_size=config["train_loader"]["batch_size"],
        num_workers=config["train_loader"]["num_workers"],
        pin_memory=config["train_loader"]["pin_memory"],
        max_src_len=config["train_loader"]["max_src_len"],
        max_tgt_len=config["train_loader"]["max_tgt_len"],
        add_src_eos=config["train_loader"]["add_src_eos"],
        skip_empty=config["train_loader"]["skip_empty"],
        shuffle_buffer_size=config["train_loader"]["shuffle_buffer_size"],
        seed=config["train_loader"]["seed"],
        num_samples=config["data"]["train_num_samples"],
        persistent_workers=config["train_loader"]["persistent_workers"],
        prefetch_factor=config["train_loader"]["prefetch_factor"],
    )

    valid_loader = build_bpe_dataloader(
        src_path=config["data"]["valid_src"],
        tgt_path=config["data"]["valid_tgt"],
        vocab=vocab,
        batch_size=config["valid_loader"]["batch_size"],
        num_workers=config["valid_loader"]["num_workers"],
        pin_memory=config["valid_loader"]["pin_memory"],
        max_src_len=config["valid_loader"]["max_src_len"],
        max_tgt_len=config["valid_loader"]["max_tgt_len"],
        add_src_eos=config["valid_loader"]["add_src_eos"],
        skip_empty=config["valid_loader"]["skip_empty"],
        shuffle_buffer_size=config["valid_loader"]["shuffle_buffer_size"],
        seed=config["valid_loader"]["seed"],
        num_samples=config["data"]["valid_num_samples"],
        persistent_workers=config["valid_loader"]["persistent_workers"],
        prefetch_factor=config["valid_loader"]["prefetch_factor"],
    )

    model = make_model(
        src_vocab=vocab_size,
        tgt_vocab=vocab_size,
        N=config["model"]["N"],
        d_model=config["model"]["d_model"],
        d_ff=config["model"]["d_ff"],
        h=config["model"]["h"],
        dropout=config["model"]["dropout"],
        share_embeddings=config["model"]["share_embeddings"],
    )

    param_count = count_trainable_parameters(model)
    print(f"模型可训练参数量: {param_count / 1e6:.2f} M")

    criterion = LabelSmoothingLoss(
        vocab_size=vocab_size,
        pad_idx=vocab.pad_id,
        smoothing=config["criterion"]["smoothing"],
    )

    optimizer, scheduler = build_transformer_optimizer_and_scheduler(
        model=model,
        d_model=config["model"]["d_model"],
        warmup_steps=config["scheduler"]["warmup_steps"],
        factor=config["scheduler"]["factor"],
        beta1=config["optimizer"]["beta1"],
        beta2=config["optimizer"]["beta2"],
        eps=config["optimizer"]["eps"],
        weight_decay=config["optimizer"]["weight_decay"],
    )

    use_amp = config["use_amp"]
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    exp_dir = os.path.join(
        "experiments",
        "{0}_{1}".format(config["exp_name"], get_timestamp_str()),
    )

    fit(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_epochs=config["fit"]["num_epochs"],
        output_dir=exp_dir,
        config=config,
        vocab=vocab,
        scaler=scaler,
        use_amp=use_amp,
        grad_clip_norm=config["fit"]["grad_clip_norm"],
        train_log_interval=config["fit"]["train_log_interval"],
        histogram_interval=config["fit"]["histogram_interval"],
        save_every_epochs=config["fit"]["save_every_epochs"],
        valid_num_text_samples=config["fit"]["valid_num_text_samples"],
        max_train_steps_per_epoch=config["fit"]["max_train_steps_per_epoch"],
        max_valid_steps_per_epoch=config["fit"]["max_valid_steps_per_epoch"],
        start_epoch=1,
        global_step_init=0,
        best_metric_init=None,
    )


if __name__ == "__main__":
    main()