from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import sacrebleu
import torch
import torch.nn.functional as F

from data.batch import make_src_mask, make_tgt_mask
from data.shared_vocab import SharedVocab
from nets.build_transformer import make_model
from utils.train_env import get_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Multi30k BLEU with checkpoint averaging.")
    parser.add_argument("--experiment-dir", type=str, required=True, help="Experiment directory containing config and checkpoints.")
    parser.add_argument("--checkpoint-paths", type=str, nargs="*", default=None, help="Explicit checkpoint paths to average.")
    parser.add_argument("--average-last-n", type=int, default=5, help="Average the last N periodic checkpoints when checkpoint paths are not given.")
    parser.add_argument("--src-path", type=str, default="data/multi30k_bpe_en_de/test.en", help="BPE source text path.")
    parser.add_argument("--tgt-path", type=str, default="data/multi30k_bpe_en_de/test.de", help="BPE reference text path.")
    parser.add_argument("--beam-size", type=int, default=4, help="Beam size.")
    parser.add_argument("--length-penalty-alpha", type=float, default=0.6, help="Length penalty alpha.")
    parser.add_argument("--max-output-extra-len", type=int, default=30, help="Decode up to src_len + this extra length.")
    parser.add_argument("--max-sentences", type=int, default=None, help="Optional sentence limit for debugging.")
    parser.add_argument("--output-path", type=str, default=None, help="Optional predictions output path.")
    return parser.parse_args()


def load_config(experiment_dir: Path) -> Dict:
    with (experiment_dir / "config.json").open("r", encoding="utf-8") as f:
        return json.load(f)


def list_periodic_checkpoints(checkpoint_dir: Path) -> List[Path]:
    candidates = sorted(checkpoint_dir.glob("model_epoch_*.pth"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"No periodic checkpoints found in {checkpoint_dir}")
    return candidates


def resolve_checkpoint_paths(args: argparse.Namespace, experiment_dir: Path) -> List[Path]:
    if args.checkpoint_paths:
        return [Path(path) for path in args.checkpoint_paths]
    return list_periodic_checkpoints(experiment_dir / "checkpoints")[-args.average_last_n :]


def average_state_dicts(checkpoint_paths: Sequence[Path]) -> Tuple[Dict[str, torch.Tensor], Dict]:
    if not checkpoint_paths:
        raise ValueError("At least one checkpoint is required.")

    avg_state: Dict[str, torch.Tensor] = {}
    last_checkpoint = None

    for idx, path in enumerate(checkpoint_paths, start=1):
        checkpoint = torch.load(path, map_location="cpu")
        state_dict = checkpoint["model_state_dict"]
        last_checkpoint = checkpoint

        for key, value in state_dict.items():
            if not torch.is_floating_point(value):
                if idx == len(checkpoint_paths):
                    avg_state[key] = value.clone()
                continue

            value = value.detach().float()
            if key not in avg_state:
                avg_state[key] = value.clone()
            else:
                avg_state[key].add_(value)

    for key, value in list(avg_state.items()):
        if torch.is_floating_point(value):
            avg_state[key] = (value / len(checkpoint_paths)).to(dtype=last_checkpoint["model_state_dict"][key].dtype)

    return avg_state, last_checkpoint


def build_model_from_config(config: Dict, vocab_size: int) -> torch.nn.Module:
    return make_model(
        src_vocab=vocab_size,
        tgt_vocab=vocab_size,
        N=config["model"]["N"],
        d_model=config["model"]["d_model"],
        d_ff=config["model"]["d_ff"],
        h=config["model"]["h"],
        dropout=config["model"]["dropout"],
        share_embeddings=config["model"]["share_embeddings"],
    )


def encode_line(vocab: SharedVocab, text: str, add_src_eos: bool = True) -> List[int]:
    ids = vocab.encode(text.strip().split())
    if add_src_eos:
        ids.append(vocab.eos_id)
    return ids


def bpe_ids_to_text(vocab: SharedVocab, ids: Sequence[int]) -> str:
    tokens: List[str] = []
    for idx in ids:
        if idx == vocab.pad_id or idx == vocab.bos_id:
            continue
        if idx == vocab.eos_id:
            break
        tokens.append(vocab.id2token(int(idx)))
    return " ".join(tokens).replace("@@ ", "").strip()


@torch.no_grad()
def beam_search_decode(
    model: torch.nn.Module,
    src_ids: Sequence[int],
    vocab: SharedVocab,
    beam_size: int,
    alpha: float,
    max_output_extra_len: int,
    device: torch.device,
) -> List[int]:
    src = torch.tensor([src_ids], dtype=torch.long, device=device)
    src_mask = make_src_mask(src, pad_idx=vocab.pad_id)
    memory = model.encode(src, src_mask)

    max_len = len(src_ids) + max_output_extra_len
    beams: List[Tuple[List[int], float, bool]] = [([vocab.bos_id], 0.0, False)]

    def normalized_score(tokens: Sequence[int], log_prob: float) -> float:
        length_penalty = ((5.0 + len(tokens)) / 6.0) ** alpha
        return log_prob / length_penalty

    for _ in range(max_len - 1):
        candidates: List[Tuple[List[int], float, bool]] = []

        for tokens, log_prob, finished in beams:
            if finished:
                candidates.append((tokens, log_prob, finished))
                continue

            tgt = torch.tensor([tokens], dtype=torch.long, device=device)
            tgt_mask = make_tgt_mask(tgt, pad_idx=vocab.pad_id)
            decoded = model.decode(memory, src_mask, tgt, tgt_mask)
            logits = model.generator(decoded[:, -1, :])
            log_probs = F.log_softmax(logits, dim=-1)

            topk_log_probs, topk_indices = torch.topk(log_probs, k=beam_size, dim=-1)
            for next_log_prob, next_idx in zip(topk_log_probs[0].tolist(), topk_indices[0].tolist()):
                next_tokens = tokens + [int(next_idx)]
                next_finished = int(next_idx) == vocab.eos_id
                candidates.append((next_tokens, log_prob + float(next_log_prob), next_finished))

        candidates.sort(key=lambda x: normalized_score(x[0], x[1]), reverse=True)
        beams = candidates[:beam_size]
        if all(finished for _, _, finished in beams):
            break

    best_tokens, _, _ = max(beams, key=lambda x: normalized_score(x[0], x[1]))
    return best_tokens


def iter_parallel_lines(src_path: Path, tgt_path: Path) -> Iterable[Tuple[str, str]]:
    with src_path.open("r", encoding="utf-8") as f_src, tgt_path.open("r", encoding="utf-8") as f_tgt:
        for src_line, tgt_line in zip(f_src, f_tgt):
            yield src_line.rstrip("\n"), tgt_line.rstrip("\n")


def main() -> None:
    args = parse_args()

    experiment_dir = Path(args.experiment_dir)
    config = load_config(experiment_dir)
    checkpoint_paths = resolve_checkpoint_paths(args=args, experiment_dir=experiment_dir)

    print("Using checkpoints:")
    for path in checkpoint_paths:
        print(path)

    vocab = SharedVocab.load(config["vocab"]["vocab_json"])
    model = build_model_from_config(config=config, vocab_size=len(vocab))
    avg_state_dict, _ = average_state_dicts(checkpoint_paths)
    model.load_state_dict(avg_state_dict)

    device = get_device()
    model = model.to(device)
    model.eval()

    predictions: List[str] = []
    references: List[str] = []

    src_path = Path(args.src_path)
    tgt_path = Path(args.tgt_path)

    for idx, (src_line, tgt_line) in enumerate(iter_parallel_lines(src_path, tgt_path), start=1):
        if args.max_sentences is not None and idx > args.max_sentences:
            break

        src_ids = encode_line(vocab=vocab, text=src_line, add_src_eos=config["train_loader"]["add_src_eos"])
        pred_ids = beam_search_decode(
            model=model,
            src_ids=src_ids,
            vocab=vocab,
            beam_size=args.beam_size,
            alpha=args.length_penalty_alpha,
            max_output_extra_len=args.max_output_extra_len,
            device=device,
        )

        predictions.append(bpe_ids_to_text(vocab, pred_ids))
        references.append(tgt_line.replace("@@ ", "").strip())

        if idx % 100 == 0:
            print(f"decoded {idx} sentences")

    bleu = sacrebleu.corpus_bleu(predictions, [references])
    print(f"BLEU = {bleu.score:.2f}")

    if args.output_path is not None:
        output_path = Path(args.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("\n".join(predictions), encoding="utf-8")
        print(f"predictions saved to {output_path}")


if __name__ == "__main__":
    main()

''' 
experiments\transformer_multi30k_en_de_base_20260331_191821ce\transformerriments\transformer_multi30k_en_de_base_20260331_191821ce\transformerUsing checkpoints:
experiments\transformer_multi30k_en_de_base_20260331_191821\checkpoints\model_epoch_026_valid_ppl_6.9435.pth
experiments\transformer_multi30k_en_de_base_20260331_191821\checkpoints\model_epoch_027_valid_ppl_6.8146.pth
experiments\transformer_multi30k_en_de_base_20260331_191821\checkpoints\model_epoch_028_valid_ppl_7.0123.pth
experiments\transformer_multi30k_en_de_base_20260331_191821\checkpoints\model_epoch_029_valid_ppl_6.9239.pth
experiments\transformer_multi30k_en_de_base_20260331_191821\checkpoints\model_epoch_030_valid_ppl_7.1085.pth

# BLEU = 38.24

'''
