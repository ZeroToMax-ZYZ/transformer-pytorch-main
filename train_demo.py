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