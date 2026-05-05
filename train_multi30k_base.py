from __future__ import annotations

import argparse
import os

import torch

from data.shared_vocab import SharedVocab
from data.wmt_14_bpe_dataset import build_bpe_dataloader, resolve_num_samples_for_ratio
from nets.build_transformer import make_model
from train_utils.fit import fit
from utils.distributed import cleanup_distributed, setup_distributed, wrap_ddp
from utils.label_smoothing import LabelSmoothingLoss
from utils.noam_scheduler import build_transformer_optimizer_and_scheduler
from utils.train_env import count_trainable_parameters, get_timestamp_str, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Transformer base model on Multi30k En-De BPE data.")
    parser.add_argument("--num-epochs", type=int, default=None, help="Override the number of training epochs.")
    parser.add_argument("--train-batch-size", type=int, default=None, help="Override fixed train batch size.")
    parser.add_argument("--valid-batch-size", type=int, default=None, help="Override the validation batch size.")
    parser.add_argument("--train-src-token-budget", type=int, default=None, help="Override train source token budget.")
    parser.add_argument("--train-tgt-token-budget", type=int, default=None, help="Override train target token budget.")
    parser.add_argument("--train-batch-pool-size", type=int, default=None, help="Override train approximate-length pooling size.")
    parser.add_argument("--train-subset-ratio", type=float, default=None, help="Use only a prefix subset of the training split.")
    parser.add_argument("--train-num-workers", type=int, default=None, help="Override train DataLoader workers.")
    parser.add_argument("--valid-num-workers", type=int, default=None, help="Override valid DataLoader workers.")
    parser.add_argument("--max-train-steps-per-epoch", type=int, default=None, help="Limit train steps per epoch.")
    parser.add_argument("--max-valid-steps-per-epoch", type=int, default=None, help="Limit valid steps per epoch.")
    parser.add_argument("--max-src-len", type=int, default=None, help="Truncate source length for debugging.")
    parser.add_argument("--max-tgt-len", type=int, default=None, help="Truncate target length for debugging.")
    parser.add_argument("--warmup-steps", type=int, default=None, help="Override Noam warmup steps.")
    parser.add_argument("--output-dir", type=str, default=None, help="Override the experiment output directory.")
    return parser.parse_args()


def build_config(device: torch.device, vocab_size: int) -> dict:
    use_amp = device.type == "cuda"
    default_train_num_workers = 0 if os.name == "nt" else 2

    return {
        "exp_name": "transformer_multi30k_en_de_base",
        "seed": 42,
        "device": str(device),
        "use_amp": use_amp,
        "data": {
            "train_src": "data/multi30k_bpe_en_de/train.en",
            "train_tgt": "data/multi30k_bpe_en_de/train.de",
            "valid_src": "data/multi30k_bpe_en_de/valid.en",
            "valid_tgt": "data/multi30k_bpe_en_de/valid.de",
            "test_src": "data/multi30k_bpe_en_de/test.en",
            "test_tgt": "data/multi30k_bpe_en_de/test.de",
            "train_num_samples": 29000,
            "train_subset_ratio": 1.0,
            "valid_num_samples": 1014,
            "test_num_samples": 1000,
        },
        "vocab": {
            "vocab_json": "data/multi30k_vocab/vocab.json",
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
            "warmup_steps": 2000,
            "factor": 1.0,
        },
        "train_loader": {
            "batch_size": None,
            "num_workers": default_train_num_workers,
            "pin_memory": device.type == "cuda",
            "max_src_len": 100,
            "max_tgt_len": 100,
            "add_src_eos": True,
            "skip_empty": False,
            "shuffle_buffer_size": 29000,
            "seed": 42,
            "persistent_workers": default_train_num_workers > 0,
            "prefetch_factor": 2 if default_train_num_workers > 0 else None,
            "src_token_budget": 4096,
            "tgt_token_budget": 4096,
            "max_sentences_per_batch": 128,
            "batch_pool_size": 2048,
        },
        "valid_loader": {
            "batch_size": 64,
            "num_workers": 0,
            "pin_memory": device.type == "cuda",
            "max_src_len": 100,
            "max_tgt_len": 100,
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
            "train_log_interval": 50,
            "histogram_interval": 0,
            "save_every_epochs": 1,
            "valid_num_text_samples": 3,
            "max_train_steps_per_epoch": None,
            "max_valid_steps_per_epoch": None,
        },
    }


def apply_cli_overrides(config: dict, args: argparse.Namespace) -> dict:
    if args.num_epochs is not None:
        config["fit"]["num_epochs"] = args.num_epochs
    if args.train_batch_size is not None:
        config["train_loader"]["batch_size"] = args.train_batch_size
        config["train_loader"]["src_token_budget"] = None
        config["train_loader"]["tgt_token_budget"] = None
    if args.valid_batch_size is not None:
        config["valid_loader"]["batch_size"] = args.valid_batch_size
    if args.train_src_token_budget is not None:
        config["train_loader"]["src_token_budget"] = args.train_src_token_budget
    if args.train_tgt_token_budget is not None:
        config["train_loader"]["tgt_token_budget"] = args.train_tgt_token_budget
    if args.train_batch_pool_size is not None:
        config["train_loader"]["batch_pool_size"] = args.train_batch_pool_size
    if args.train_subset_ratio is not None:
        config["data"]["train_subset_ratio"] = args.train_subset_ratio
    if args.train_num_workers is not None:
        config["train_loader"]["num_workers"] = args.train_num_workers
        config["train_loader"]["persistent_workers"] = args.train_num_workers > 0
        config["train_loader"]["prefetch_factor"] = 2 if args.train_num_workers > 0 else None
    if args.valid_num_workers is not None:
        config["valid_loader"]["num_workers"] = args.valid_num_workers
        config["valid_loader"]["persistent_workers"] = args.valid_num_workers > 0
        config["valid_loader"]["prefetch_factor"] = 2 if args.valid_num_workers > 0 else None
    if args.max_train_steps_per_epoch is not None:
        config["fit"]["max_train_steps_per_epoch"] = args.max_train_steps_per_epoch
    if args.max_valid_steps_per_epoch is not None:
        config["fit"]["max_valid_steps_per_epoch"] = args.max_valid_steps_per_epoch
    if args.max_src_len is not None:
        config["train_loader"]["max_src_len"] = args.max_src_len
        config["valid_loader"]["max_src_len"] = args.max_src_len
    if args.max_tgt_len is not None:
        config["train_loader"]["max_tgt_len"] = args.max_tgt_len
        config["valid_loader"]["max_tgt_len"] = args.max_tgt_len
    if args.warmup_steps is not None:
        config["scheduler"]["warmup_steps"] = args.warmup_steps
    return config


def build_train_loader_kwargs(config: dict, vocab: SharedVocab) -> dict:
    total_train_samples = config["data"]["train_num_samples"]
    train_subset_ratio = config["data"]["train_subset_ratio"]
    train_subset_num_samples = resolve_num_samples_for_ratio(
        total_num_samples=total_train_samples,
        subset_ratio=train_subset_ratio,
    )

    return {
        "src_path": config["data"]["train_src"],
        "tgt_path": config["data"]["train_tgt"],
        "vocab": vocab,
        "batch_size": config["train_loader"]["batch_size"],
        "num_workers": config["train_loader"]["num_workers"],
        "pin_memory": config["train_loader"]["pin_memory"],
        "max_src_len": config["train_loader"]["max_src_len"],
        "max_tgt_len": config["train_loader"]["max_tgt_len"],
        "add_src_eos": config["train_loader"]["add_src_eos"],
        "skip_empty": config["train_loader"]["skip_empty"],
        "shuffle_buffer_size": config["train_loader"]["shuffle_buffer_size"],
        "seed": config["train_loader"]["seed"],
        "num_samples": train_subset_num_samples,
        "sample_limit": train_subset_num_samples,
        "persistent_workers": config["train_loader"]["persistent_workers"],
        "prefetch_factor": config["train_loader"]["prefetch_factor"],
        "src_token_budget": config["train_loader"]["src_token_budget"],
        "tgt_token_budget": config["train_loader"]["tgt_token_budget"],
        "max_sentences_per_batch": config["train_loader"]["max_sentences_per_batch"],
        "batch_pool_size": config["train_loader"]["batch_pool_size"],
    }


def main() -> None:
    args = parse_args()
    ctx = setup_distributed()

    try:
        seed_everything(seed=42, deterministic=False)

        vocab = SharedVocab.load("data/multi30k_vocab/vocab.json")
        vocab_size = len(vocab)

        config = build_config(device=ctx.device, vocab_size=vocab_size)
        config = apply_cli_overrides(config=config, args=args)
        config["distributed"] = {
            "enabled": ctx.is_distributed,
            "rank": ctx.rank,
            "world_size": ctx.world_size,
            "local_rank": ctx.local_rank,
        }

        train_loader_kwargs = build_train_loader_kwargs(config=config, vocab=vocab)
        train_loader_kwargs["rank"] = ctx.rank
        train_loader_kwargs["world_size"] = ctx.world_size
        train_loader = build_bpe_dataloader(**train_loader_kwargs)

        valid_loader = None
        if ctx.is_main_process:
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
                rank=0,
                world_size=1,
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
        if ctx.is_main_process:
            print(f"model trainable params: {param_count / 1e6:.2f} M")
            print(
                "training set size: "
                f"{train_loader_kwargs['num_samples']} / {config['data']['train_num_samples']} "
                f"(ratio={config['data']['train_subset_ratio']:.4f})"
            )

        model = model.to(ctx.device)
        model = wrap_ddp(model, ctx)

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

        exp_dir = args.output_dir
        if exp_dir is None:
            exp_dir = os.path.join("experiments", f"{config['exp_name']}_{get_timestamp_str()}")

        fit(
            model=model,
            train_loader=train_loader,
            valid_loader=valid_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=ctx.device,
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
            is_main_process=ctx.is_main_process,
        )
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
