from __future__ import annotations

import argparse
import os

import torch

from data.copy_task import build_char_copy_dataloader, build_char_copy_vocab
from nets.build_transformer import make_model
from train_utils.fit import fit
from utils.distributed import cleanup_distributed, setup_distributed, wrap_ddp
from utils.label_smoothing import LabelSmoothingLoss
from utils.train_env import count_trainable_parameters, get_timestamp_str, seed_everything


DEFAULT_TRAIN_COPY_ALPHABET = "abcdefgh"


class ConstantLRScheduler:
    def __init__(self, optimizer: torch.optim.Optimizer, lr: float) -> None:
        if lr <= 0:
            raise ValueError("learning rate must be positive.")
        self.optimizer = optimizer
        self.lr = float(lr)
        for group in self.optimizer.param_groups:
            group["lr"] = self.lr

    def step(self) -> float:
        return self.lr

    def get_last_lr(self) -> list[float]:
        return [group["lr"] for group in self.optimizer.param_groups]

    def state_dict(self) -> dict:
        return {"lr": self.lr}

    def load_state_dict(self, state_dict: dict) -> None:
        self.lr = float(state_dict["lr"])
        for group in self.optimizer.param_groups:
            group["lr"] = self.lr


def build_copy_optimizer_and_scheduler(model: torch.nn.Module, config: dict):
    optimizer = torch.optim.Adam(
        params=model.parameters(),
        lr=config["optimizer"]["lr"],
        betas=(config["optimizer"]["beta1"], config["optimizer"]["beta2"]),
        eps=config["optimizer"]["eps"],
        weight_decay=config["optimizer"]["weight_decay"],
    )
    scheduler = ConstantLRScheduler(optimizer=optimizer, lr=config["optimizer"]["lr"])
    return optimizer, scheduler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a small Transformer on a synthetic character copy task.")
    parser.add_argument("--num-epochs", type=int, default=None, help="Override the number of training epochs.")
    parser.add_argument("--train-samples", type=int, default=None, help="Override the number of train samples.")
    parser.add_argument("--valid-samples", type=int, default=None, help="Override the number of valid samples.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override the training batch size.")
    parser.add_argument("--valid-batch-size", type=int, default=None, help="Override the validation batch size.")
    parser.add_argument("--min-seq-len", type=int, default=None, help="Override the minimum sequence length.")
    parser.add_argument("--max-seq-len", type=int, default=None, help="Override the maximum sequence length.")
    parser.add_argument("--alphabet", type=str, default=None, help="Characters used by the copy task.")
    parser.add_argument("--train-num-workers", type=int, default=None, help="Override train DataLoader workers.")
    parser.add_argument("--valid-num-workers", type=int, default=None, help="Override valid DataLoader workers.")
    parser.add_argument("--learning-rate", type=float, default=None, help="Override the constant learning rate.")
    parser.add_argument("--max-train-steps-per-epoch", type=int, default=None, help="Limit train steps per epoch.")
    parser.add_argument("--max-valid-steps-per-epoch", type=int, default=None, help="Limit valid steps per epoch.")
    parser.add_argument("--output-dir", type=str, default=None, help="Override the experiment output directory.")
    return parser.parse_args()


def build_config(device: torch.device, alphabet: str) -> dict:
    use_amp = device.type == "cuda"
    default_num_workers = 0 if os.name == "nt" else 2
    vocab = build_char_copy_vocab(alphabet)

    return {
        "exp_name": "test_char_copy_base",
        "seed": 42,
        "device": str(device),
        "use_amp": use_amp,
        "data": {
            "alphabet": alphabet,
            "train_num_samples": 64,
            "valid_num_samples": 64,
            "min_seq_len": 3,
            "max_seq_len": 8,
        },
        "vocab": {
            "vocab_size": len(vocab),
            "pad_id": vocab.pad_id,
            "bos_id": vocab.bos_id,
            "eos_id": vocab.eos_id,
            "unk_id": vocab.unk_id,
        },
        "model": {
            "N": 2,
            "d_model": 128,
            "d_ff": 256,
            "h": 4,
            "dropout": 0.0,
            "share_embeddings": True,
        },
        "criterion": {
            "type": "LabelSmoothingLoss",
            "smoothing": 0.0,
        },
        "optimizer": {
            "type": "Adam",
            "lr": 1e-3,
            "beta1": 0.9,
            "beta2": 0.98,
            "eps": 1e-9,
            "weight_decay": 0.0,
        },
        "scheduler": {
            "type": "ConstantLRScheduler",
        },
        "train_loader": {
            "batch_size": 32,
            "num_workers": default_num_workers,
            "pin_memory": device.type == "cuda",
            "persistent_workers": default_num_workers > 0,
            "prefetch_factor": 2 if default_num_workers > 0 else None,
        },
        "valid_loader": {
            "batch_size": 32,
            "num_workers": 0,
            "pin_memory": device.type == "cuda",
            "persistent_workers": False,
            "prefetch_factor": None,
        },
        "fit": {
            "num_epochs": 30,
            "grad_clip_norm": 1.0,
            "train_log_interval": 5,
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
    if args.train_samples is not None:
        config["data"]["train_num_samples"] = args.train_samples
    if args.valid_samples is not None:
        config["data"]["valid_num_samples"] = args.valid_samples
    if args.batch_size is not None:
        config["train_loader"]["batch_size"] = args.batch_size
    if args.valid_batch_size is not None:
        config["valid_loader"]["batch_size"] = args.valid_batch_size
    if args.min_seq_len is not None:
        config["data"]["min_seq_len"] = args.min_seq_len
    if args.max_seq_len is not None:
        config["data"]["max_seq_len"] = args.max_seq_len
    if args.alphabet is not None:
        config["data"]["alphabet"] = args.alphabet
    if args.train_num_workers is not None:
        config["train_loader"]["num_workers"] = args.train_num_workers
        config["train_loader"]["persistent_workers"] = args.train_num_workers > 0
        config["train_loader"]["prefetch_factor"] = 2 if args.train_num_workers > 0 else None
    if args.valid_num_workers is not None:
        config["valid_loader"]["num_workers"] = args.valid_num_workers
        config["valid_loader"]["persistent_workers"] = args.valid_num_workers > 0
        config["valid_loader"]["prefetch_factor"] = 2 if args.valid_num_workers > 0 else None
    if args.learning_rate is not None:
        config["optimizer"]["lr"] = args.learning_rate
    if args.max_train_steps_per_epoch is not None:
        config["fit"]["max_train_steps_per_epoch"] = args.max_train_steps_per_epoch
    if args.max_valid_steps_per_epoch is not None:
        config["fit"]["max_valid_steps_per_epoch"] = args.max_valid_steps_per_epoch

    min_seq_len = config["data"]["min_seq_len"]
    max_seq_len = config["data"]["max_seq_len"]
    if min_seq_len <= 0:
        raise ValueError("min_seq_len must be positive.")
    if max_seq_len < min_seq_len:
        raise ValueError("max_seq_len must be >= min_seq_len.")

    return config


def main() -> None:
    args = parse_args()
    ctx = setup_distributed()

    try:
        seed_everything(seed=42, deterministic=False)

        alphabet = args.alphabet or DEFAULT_TRAIN_COPY_ALPHABET
        config = build_config(device=ctx.device, alphabet=alphabet)
        config = apply_cli_overrides(config=config, args=args)
        config["distributed"] = {
            "enabled": ctx.is_distributed,
            "rank": ctx.rank,
            "world_size": ctx.world_size,
            "local_rank": ctx.local_rank,
        }

        vocab = build_char_copy_vocab(config["data"]["alphabet"])
        vocab_size = len(vocab)
        config["vocab"]["vocab_size"] = vocab_size
        config["vocab"]["pad_id"] = vocab.pad_id
        config["vocab"]["bos_id"] = vocab.bos_id
        config["vocab"]["eos_id"] = vocab.eos_id
        config["vocab"]["unk_id"] = vocab.unk_id

        train_loader = build_char_copy_dataloader(
            num_samples=config["data"]["train_num_samples"],
            vocab=vocab,
            batch_size=config["train_loader"]["batch_size"],
            alphabet=config["data"]["alphabet"],
            min_seq_len=config["data"]["min_seq_len"],
            max_seq_len=config["data"]["max_seq_len"],
            shuffle=True,
            seed=config["seed"],
            num_workers=config["train_loader"]["num_workers"],
            pin_memory=config["train_loader"]["pin_memory"],
            persistent_workers=config["train_loader"]["persistent_workers"],
            prefetch_factor=config["train_loader"]["prefetch_factor"],
            rank=ctx.rank,
            world_size=ctx.world_size,
        )

        valid_loader = None
        if ctx.is_main_process:
            valid_loader = build_char_copy_dataloader(
                num_samples=config["data"]["valid_num_samples"],
                vocab=vocab,
                batch_size=config["valid_loader"]["batch_size"],
                alphabet=config["data"]["alphabet"],
                min_seq_len=config["data"]["min_seq_len"],
                max_seq_len=config["data"]["max_seq_len"],
                shuffle=False,
                seed=config["seed"],
                num_workers=config["valid_loader"]["num_workers"],
                pin_memory=config["valid_loader"]["pin_memory"],
                persistent_workers=config["valid_loader"]["persistent_workers"],
                prefetch_factor=config["valid_loader"]["prefetch_factor"],
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
                "copy task setup: "
                f"alphabet_size={vocab_size - 4}, "
                f"train_samples={config['data']['train_num_samples']}, "
                f"valid_samples={config['data']['valid_num_samples']}, "
                f"seq_len=[{config['data']['min_seq_len']}, {config['data']['max_seq_len']}]"
            )

        model = model.to(ctx.device)
        model = wrap_ddp(model, ctx)

        criterion = LabelSmoothingLoss(
            vocab_size=vocab_size,
            pad_idx=vocab.pad_id,
            smoothing=config["criterion"]["smoothing"],
        )

        optimizer, scheduler = build_copy_optimizer_and_scheduler(model=model, config=config)

        use_amp = config["use_amp"]
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

        exp_dir = args.output_dir
        if exp_dir is None:
            exp_dir = os.path.join(
                "experiments",
                f"{config['exp_name']}_{get_timestamp_str()}",
            )

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
