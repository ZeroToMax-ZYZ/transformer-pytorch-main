from data.batch import Seq2SeqBatch

pad_idx = 0

# src: (B, S)
# tgt: (B, T_full)，要求已经包含 BOS / EOS / PAD
batch = Seq2SeqBatch.from_tensors(src, tgt, pad_idx=pad_idx)

hidden = model(
    batch.src,
    batch.tgt_input,
    batch.src_mask,
    batch.tgt_mask,
)

logits = model.generator(hidden)

# 后面你的损失函数要和 batch.tgt_y 对齐
# logits: (B, T, V)
# tgt_y : (B, T)

train_loader = build_bpe_dataloader(
    src_path="data/wmt14_bpe_en_de/train.en",
    tgt_path="data/wmt14_bpe_en_de/train.de",
    vocab=vocab,
    batch_size=64,
    num_workers=2,
    pin_memory=True,
    max_src_len=None,
    max_tgt_len=None,
    add_src_eos=True,
    skip_empty=False,
    shuffle_buffer_size=10000,
    seed=42,
    num_samples=3927488,
    persistent_workers=True,
    prefetch_factor=2,
)

valid_loader = build_bpe_dataloader(
    src_path="data/wmt14_bpe_en_de/valid.en",
    tgt_path="data/wmt14_bpe_en_de/valid.de",
    vocab=vocab,
    batch_size=64,
    num_workers=0,
    pin_memory=True,
    max_src_len=None,
    max_tgt_len=None,
    add_src_eos=True,
    skip_empty=False,
    shuffle_buffer_size=0,
    seed=42,
    num_samples=3000,
    persistent_workers=False,
    prefetch_factor=None,
)