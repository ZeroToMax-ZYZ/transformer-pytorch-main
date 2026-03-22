from __future__ import annotations

"""
功能：
1. 从已有 checkpoint 恢复训练。
2. 恢复：
   - model
   - optimizer
   - scheduler
   - scaler
   - epoch/global_step/best_metric
3. 继续调用 fit() 完成后续训练。

说明：
1. 当前版本默认从 last.pth 恢复。
2. 你可以手动改 resume_path。
"""

import os
import torch

from data.shared_vocab import SharedVocab
from data.wmt_14_bpe_dataset import build_bpe_dataloader, resolve_num_samples_for_ratio
from nets.build_transformer import make_model
from train_utils.fit import fit
from utils.label_smoothing import LabelSmoothingLoss
from utils.noam_scheduler import build_transformer_optimizer_and_scheduler
from utils.resume_utils import load_checkpoint_for_resume
from utils.train_env import count_trainable_parameters, get_device, seed_everything


def main() -> None:
    resume_path = "experiments/your_exp_name/checkpoints/last.pth"

    device = get_device()
    seed_everything(seed=42, deterministic=False)

    checkpoint = torch.load(resume_path, map_location="cpu")
    config = checkpoint["config"]
    config.setdefault("data", {})
    config["data"].setdefault("train_subset_ratio", 1.0)
    fit_config = config.setdefault("fit", {})
    config["fit"] = {
        "num_epochs": fit_config.get("num_epochs", 100),
        "grad_clip_norm": fit_config.get("grad_clip_norm", 1.0),
        "train_log_interval": fit_config.get("train_log_interval", 100),
        "histogram_interval": fit_config.get("histogram_interval", 0),
        "save_every_epochs": fit_config.get("save_every_epochs", 1),
        "valid_num_text_samples": fit_config.get("valid_num_text_samples", 0),
        "max_train_steps_per_epoch": fit_config.get("max_train_steps_per_epoch"),
        "max_valid_steps_per_epoch": fit_config.get("max_valid_steps_per_epoch"),
    }

    vocab = SharedVocab.load(config["vocab"]["vocab_json"])
    vocab_size = len(vocab)
    train_subset_num_samples = resolve_num_samples_for_ratio(
        total_num_samples=config["data"]["train_num_samples"],
        subset_ratio=config["data"]["train_subset_ratio"],
    )

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
        num_samples=train_subset_num_samples,
        sample_limit=train_subset_num_samples,
        persistent_workers=config["train_loader"]["persistent_workers"],
        prefetch_factor=config["train_loader"]["prefetch_factor"],
        src_token_budget=config["train_loader"].get("src_token_budget"),
        tgt_token_budget=config["train_loader"].get("tgt_token_budget"),
        max_sentences_per_batch=config["train_loader"].get("max_sentences_per_batch"),
        batch_pool_size=config["train_loader"].get("batch_pool_size", 2048),
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
        src_token_budget=config["valid_loader"].get("src_token_budget"),
        tgt_token_budget=config["valid_loader"].get("tgt_token_budget"),
        max_sentences_per_batch=config["valid_loader"].get("max_sentences_per_batch"),
        batch_pool_size=config["valid_loader"].get("batch_pool_size", 2048),
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
    print(
        "训练集样本数: "
        f"{train_subset_num_samples} / {config['data']['train_num_samples']} "
        f"(ratio={config['data']['train_subset_ratio']:.4f})"
    )

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

    resume_info = load_checkpoint_for_resume(
        checkpoint_path=resume_path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        map_location="cpu",
    )

    print("断点恢复成功。")
    print(f"恢复 epoch = {resume_info['epoch']}")
    print(f"恢复 global_step = {resume_info['global_step']}")
    print(f"恢复 best_metric(valid_ppl) = {resume_info['best_metric']}")

    fit(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_epochs=config["fit"]["num_epochs"],
        output_dir=os.path.dirname(os.path.dirname(resume_path)),
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
        start_epoch=resume_info["epoch"] + 1,
        global_step_init=resume_info["global_step"],
        best_metric_init=resume_info["best_metric"],
    )


if __name__ == "__main__":
    main()
