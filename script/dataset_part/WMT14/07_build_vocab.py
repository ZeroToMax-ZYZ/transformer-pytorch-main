from __future__ import annotations

"""
功能：
1. 从 BPE 后的 train.en 和 train.de 构建 shared vocabulary
2. 保存 vocab.json / vocab.txt / meta.json
3. 打印词表规模和特殊 token id
"""

import os

from data.shared_vocab import SharedVocab, SpecialTokens


def main() -> None:
    train_en = "data/wmt14_bpe_en_de/train.en"
    train_de = "data/wmt14_bpe_en_de/train.de"

    out_dir = "data/wmt14_vocab"
    os.makedirs(out_dir, exist_ok=True)

    vocab = SharedVocab.build_from_files(
        file_paths=[train_en, train_de],
        min_freq=1,
        special_tokens=SpecialTokens(
            pad="<pad>",
            bos="<bos>",
            eos="<eos>",
            unk="<unk>",
        ),
    )

    vocab.save(
        vocab_json_path=os.path.join(out_dir, "vocab.json"),
        vocab_txt_path=os.path.join(out_dir, "vocab.txt"),
        meta_json_path=os.path.join(out_dir, "meta.json"),
    )

    print("共享词表构建完成。")
    print(f"vocab_size = {len(vocab)}")
    print(f"pad_id = {vocab.pad_id}")
    print(f"bos_id = {vocab.bos_id}")
    print(f"eos_id = {vocab.eos_id}")
    print(f"unk_id = {vocab.unk_id}")


if __name__ == "__main__":
    main()