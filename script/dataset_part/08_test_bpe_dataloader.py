from __future__ import annotations

"""
功能：
1. 测试 SharedVocab 是否能正常加载。
2. 测试 BPE DataLoader 是否能正常产出 Seq2SeqBatch。
3. 打印一个 batch 的关键维度信息。
"""

from data.shared_vocab import SharedVocab
from data.wmt_14_bpe_dataset import build_bpe_dataloader


def main() -> None:
    vocab = SharedVocab.load("data/wmt14_vocab/vocab.json")

    print("共享词表加载成功。")
    print(f"vocab_size = {len(vocab)}")
    print(f"pad_id = {vocab.pad_id}")
    print(f"bos_id = {vocab.bos_id}")
    print(f"eos_id = {vocab.eos_id}")
    print(f"unk_id = {vocab.unk_id}")
    print("-" * 60)

    loader = build_bpe_dataloader(
        src_path="data/wmt14_bpe_en_de/train.en",
        tgt_path="data/wmt14_bpe_en_de/train.de",
        vocab=vocab,
        batch_size=4,
        num_workers=0,
        pin_memory=False,
        max_src_len=128,
        max_tgt_len=128,
        add_src_eos=True,
        skip_empty=False,
    )

    batch = next(iter(loader))

    print("成功取到一个 batch。")
    print(f"src.shape       = {tuple(batch.src.shape)}")
    print(f"tgt_input.shape = {tuple(batch.tgt_input.shape)}")
    print(f"tgt_y.shape     = {tuple(batch.tgt_y.shape)}")
    print(f"src_mask.shape  = {tuple(batch.src_mask.shape)}")
    print(f"tgt_mask.shape  = {tuple(batch.tgt_mask.shape)}")
    print(f"ntokens         = {batch.ntokens}")

    print("-" * 60)
    print("观察第一个样本前若干个 id：")
    print("src[0, :20]      =", batch.src[0, :20].tolist())
    print("tgt_input[0,:20] =", batch.tgt_input[0, :20].tolist())
    print("tgt_y[0, :20]    =", batch.tgt_y[0, :20].tolist())

    assert batch.src.dim() == 2
    assert batch.tgt_input.dim() == 2
    assert batch.tgt_y.dim() == 2
    assert batch.src_mask.dim() == 3
    assert batch.tgt_mask.dim() == 3

    print("\nDataLoader 测试通过。")


if __name__ == "__main__":
    main()