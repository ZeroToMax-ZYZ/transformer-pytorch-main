from __future__ import annotations

import argparse
import json
import shutil
import sys
import urllib.request
from pathlib import Path

from subword_nmt.apply_bpe import BPE
from subword_nmt.learn_bpe import learn_bpe

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.shared_vocab import SharedVocab, SpecialTokens


RAW_BASE_URL = "https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/tok"
SPLIT_TO_FILENAMES = {
    "train": ("train.lc.norm.tok.en", "train.lc.norm.tok.de"),
    "valid": ("val.lc.norm.tok.en", "val.lc.norm.tok.de"),
    "test": ("test_2016_flickr.lc.norm.tok.en", "test_2016_flickr.lc.norm.tok.de"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare Multi30k En-De in the same BPE-text format used by the current project."
    )
    parser.add_argument("--num-bpe-merges", type=int, default=10000, help="Number of joint BPE merge operations.")
    parser.add_argument("--bpe-min-frequency", type=int, default=2, help="Minimum pair frequency when learning BPE.")
    parser.add_argument("--raw-dir", type=str, default="data/multi30k_tok_en_de", help="Output directory for tokenized text.")
    parser.add_argument("--bpe-dir", type=str, default="data/multi30k_bpe_en_de", help="Output directory for BPE text.")
    parser.add_argument("--bpe-model-dir", type=str, default="data/multi30k_bpe_model", help="Output directory for BPE codes/meta.")
    parser.add_argument("--vocab-dir", type=str, default="data/multi30k_vocab", help="Output directory for shared vocabulary.")
    parser.add_argument("--force-download", action="store_true", help="Redownload tokenized files even if they already exist.")
    parser.add_argument("--force-bpe", action="store_true", help="Regenerate BPE text even if it already exists.")
    parser.add_argument("--force-vocab", action="store_true", help="Rebuild the shared vocabulary even if it already exists.")
    return parser.parse_args()


def sanitize_to_single_line(text: str) -> str:
    return " ".join(text.splitlines()).replace("\r", " ").replace("\n", " ").strip()


def download_file(url: str, out_path: Path, force: bool) -> None:
    if out_path.exists() and not force:
        print(f"skip existing file: {out_path}")
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"download {url}")
    with urllib.request.urlopen(url) as response, out_path.open("w", encoding="utf-8", newline="\n") as f_out:
        raw_text = response.read().decode("utf-8")
        for line in raw_text.splitlines():
            f_out.write(sanitize_to_single_line(line) + "\n")


def count_lines(path: Path) -> int:
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def check_parallel(src_path: Path, tgt_path: Path) -> None:
    src_lines = count_lines(src_path)
    tgt_lines = count_lines(tgt_path)
    print(f"check {src_path.name} vs {tgt_path.name}: {src_lines} / {tgt_lines}")
    if src_lines != tgt_lines:
        raise AssertionError(f"line count mismatch: {src_path} ({src_lines}) vs {tgt_path} ({tgt_lines})")


def export_official_tokenized_files(raw_dir: Path, force_download: bool) -> None:
    for split_name, (en_name, de_name) in SPLIT_TO_FILENAMES.items():
        split_prefix = "valid" if split_name == "valid" else split_name
        out_en = raw_dir / f"{split_prefix}.en"
        out_de = raw_dir / f"{split_prefix}.de"

        download_file(f"{RAW_BASE_URL}/{en_name}", out_en, force=force_download)
        download_file(f"{RAW_BASE_URL}/{de_name}", out_de, force=force_download)
        check_parallel(out_en, out_de)


def learn_joint_bpe_codes(
    train_en_path: Path,
    train_de_path: Path,
    codes_path: Path,
    num_merges: int,
    min_frequency: int,
    force_bpe: bool,
) -> None:
    if codes_path.exists() and not force_bpe:
        print(f"skip existing BPE codes: {codes_path}")
        return

    codes_path.parent.mkdir(parents=True, exist_ok=True)
    concat_path = codes_path.parent / "_joint_bpe_corpus.tmp"

    with concat_path.open("w", encoding="utf-8", newline="\n") as f_out:
        for src_path in (train_en_path, train_de_path):
            with src_path.open("r", encoding="utf-8") as f_in:
                shutil.copyfileobj(f_in, f_out)

    with concat_path.open("r", encoding="utf-8") as f_in, codes_path.open("w", encoding="utf-8", newline="\n") as f_out:
        learn_bpe(
            infile=f_in,
            outfile=f_out,
            num_symbols=num_merges,
            min_frequency=min_frequency,
            verbose=False,
            is_dict=False,
            total_symbols=False,
            num_workers=1,
        )

    concat_path.unlink(missing_ok=True)


def apply_bpe_file(in_path: Path, out_path: Path, bpe: BPE, force_bpe: bool) -> None:
    if out_path.exists() and not force_bpe:
        print(f"skip existing BPE file: {out_path}")
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"apply BPE: {in_path} -> {out_path}")
    with in_path.open("r", encoding="utf-8") as f_in, out_path.open("w", encoding="utf-8", newline="\n") as f_out:
        for line in f_in:
            f_out.write(bpe.process_line(line))


def build_bpe_text(raw_dir: Path, bpe_dir: Path, codes_path: Path, force_bpe: bool) -> None:
    with codes_path.open("r", encoding="utf-8") as f_codes:
        bpe = BPE(codes=f_codes, merges=-1, separator="@@", vocab=None, glossaries=None)

    for split_prefix in ("train", "valid", "test"):
        apply_bpe_file(raw_dir / f"{split_prefix}.en", bpe_dir / f"{split_prefix}.en", bpe=bpe, force_bpe=force_bpe)
        apply_bpe_file(raw_dir / f"{split_prefix}.de", bpe_dir / f"{split_prefix}.de", bpe=bpe, force_bpe=force_bpe)
        check_parallel(bpe_dir / f"{split_prefix}.en", bpe_dir / f"{split_prefix}.de")


def build_shared_vocab_from_bpe(bpe_dir: Path, vocab_dir: Path, force_vocab: bool) -> SharedVocab:
    vocab_json_path = vocab_dir / "vocab.json"
    vocab_txt_path = vocab_dir / "vocab.txt"
    meta_json_path = vocab_dir / "meta.json"

    if vocab_json_path.exists() and meta_json_path.exists() and not force_vocab:
        print(f"skip existing vocab: {vocab_json_path}")
        return SharedVocab.load(str(vocab_json_path))

    vocab_dir.mkdir(parents=True, exist_ok=True)

    vocab = SharedVocab.build_from_files(
        file_paths=[str(bpe_dir / "train.en"), str(bpe_dir / "train.de")],
        min_freq=1,
        special_tokens=SpecialTokens(),
    )
    vocab.save(
        vocab_json_path=str(vocab_json_path),
        vocab_txt_path=str(vocab_txt_path),
        meta_json_path=str(meta_json_path),
    )
    return vocab


def save_prepare_meta(
    raw_dir: Path,
    bpe_dir: Path,
    bpe_model_dir: Path,
    vocab_dir: Path,
    vocab: SharedVocab,
    num_merges: int,
    min_frequency: int,
) -> None:
    meta = {
        "dataset": "multi30k",
        "language_pair": "en-de",
        "source": "official_multi30k_task1_tokenized",
        "raw_dir": str(raw_dir).replace("\\", "/"),
        "bpe_dir": str(bpe_dir).replace("\\", "/"),
        "vocab_dir": str(vocab_dir).replace("\\", "/"),
        "num_bpe_merges": int(num_merges),
        "bpe_min_frequency": int(min_frequency),
        "train_num_samples": count_lines(bpe_dir / "train.en"),
        "valid_num_samples": count_lines(bpe_dir / "valid.en"),
        "test_num_samples": count_lines(bpe_dir / "test.en"),
        "vocab_size": len(vocab),
    }
    meta_path = bpe_model_dir / "prepare_meta.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()

    raw_dir = Path(args.raw_dir)
    bpe_dir = Path(args.bpe_dir)
    bpe_model_dir = Path(args.bpe_model_dir)
    vocab_dir = Path(args.vocab_dir)
    codes_path = bpe_model_dir / "codes.shared"

    export_official_tokenized_files(raw_dir=raw_dir, force_download=args.force_download)
    learn_joint_bpe_codes(
        train_en_path=raw_dir / "train.en",
        train_de_path=raw_dir / "train.de",
        codes_path=codes_path,
        num_merges=args.num_bpe_merges,
        min_frequency=args.bpe_min_frequency,
        force_bpe=args.force_bpe,
    )
    build_bpe_text(raw_dir=raw_dir, bpe_dir=bpe_dir, codes_path=codes_path, force_bpe=args.force_bpe)
    vocab = build_shared_vocab_from_bpe(bpe_dir=bpe_dir, vocab_dir=vocab_dir, force_vocab=args.force_vocab)
    save_prepare_meta(
        raw_dir=raw_dir,
        bpe_dir=bpe_dir,
        bpe_model_dir=bpe_model_dir,
        vocab_dir=vocab_dir,
        vocab=vocab,
        num_merges=args.num_bpe_merges,
        min_frequency=args.bpe_min_frequency,
    )

    print("Multi30k preparation finished.")
    print(f"train samples = {count_lines(bpe_dir / 'train.en')}")
    print(f"valid samples = {count_lines(bpe_dir / 'valid.en')}")
    print(f"test samples  = {count_lines(bpe_dir / 'test.en')}")
    print(f"vocab size    = {len(vocab)}")
    print(f"BPE codes     = {codes_path}")


if __name__ == "__main__":
    main()
