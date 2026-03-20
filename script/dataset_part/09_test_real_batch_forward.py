from __future__ import annotations

"""
功能：
1. 使用真实的 WMT14 BPE batch 做一次前向传播测试。
2. 验证：
   - shared vocab 的大小能否正确传入 make_model
   - 真实 batch 是否能喂给 Transformer
   - generator 输出 logits 维度是否正确
"""

import torch

from data.shared_vocab import SharedVocab
from data.wmt_14_bpe_dataset import build_bpe_dataloader
from nets.build_transformer import make_model


def main() -> None:
    vocab = SharedVocab.load("data/wmt14_vocab/vocab.json")
    vocab_size = len(vocab)

    loader = build_bpe_dataloader(
        src_path="data/wmt14_bpe_en_de/train.en",
        tgt_path="data/wmt14_bpe_en_de/train.de",
        vocab=vocab,
        batch_size=2,
        num_workers=0,
        pin_memory=False,
        max_src_len=64,
        max_tgt_len=64,
        add_src_eos=True,
        skip_empty=False,
    )

    batch = next(iter(loader))

    print("成功取到真实 batch。")
    print(f"src.shape       = {tuple(batch.src.shape)}")
    print(f"tgt_input.shape = {tuple(batch.tgt_input.shape)}")
    print(f"tgt_y.shape     = {tuple(batch.tgt_y.shape)}")
    print(f"src_mask.shape  = {tuple(batch.src_mask.shape)}")
    print(f"tgt_mask.shape  = {tuple(batch.tgt_mask.shape)}")
    print("-" * 60)

    # 这里先用一个微型模型做通路测试，避免显存压力
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

    model.eval()
    with torch.no_grad():
        hidden_states = model(
            batch.src,
            batch.tgt_input,
            batch.src_mask,
            batch.tgt_mask,
        )
        logits = model.generator(hidden_states)

    print(f"hidden_states.shape = {tuple(hidden_states.shape)}")
    print(f"logits.shape        = {tuple(logits.shape)}")

    assert hidden_states.shape[0] == batch.src.shape[0]
    assert hidden_states.shape[1] == batch.tgt_input.shape[1]
    assert hidden_states.shape[2] == 256

    assert logits.shape[0] == batch.src.shape[0]
    assert logits.shape[1] == batch.tgt_input.shape[1]
    assert logits.shape[2] == vocab_size

    print("\n真实 batch 前向传播测试通过。")


if __name__ == "__main__":
    main()