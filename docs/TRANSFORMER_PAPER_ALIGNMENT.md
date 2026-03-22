# Transformer Paper Alignment

This document compares the current repository with the 2017 Transformer paper, "Attention Is All You Need".

The goal is not to say "same" or "different" at a vague level, but to explain exactly where the implementation follows the paper and where it intentionally diverges for engineering reasons.

## 1. Mostly Aligned with the Paper

### 1.1 Model scale and block structure

Code path:

- `train_transformer_base.py`
- `nets/build_transformer.py`
- `nets/utils/encoder_decoder.py`

Current default config:

- `N = 6`
- `d_model = 512`
- `d_ff = 2048`
- `h = 8`
- `dropout = 0.1`

This matches the Transformer base configuration described in the paper.

### 1.2 Scaled dot-product attention

Code path:

- `nets/utils/attention.py`

The implementation uses:

```text
softmax(Q K^T / sqrt(d_k)) V
```

This is the exact attention core from the paper.

### 1.3 Multi-head attention structure

Code path:

- `nets/utils/attention.py`

The implementation:

1. Projects to Q, K, V.
2. Splits into multiple heads.
3. Computes attention independently per head.
4. Concatenates heads and applies output projection.

This matches the paper structurally.

### 1.4 Position-wise feed-forward network

Code path:

- `nets/utils/PositionwiseFeedForward.py`

The FFN is:

```text
Linear(d_model -> d_ff) -> ReLU -> Dropout -> Linear(d_ff -> d_model)
```

This matches the paper.

### 1.5 Sinusoidal positional encoding

Code path:

- `nets/utils/PositionalEncoding.py`

The project uses fixed sine/cosine positional encodings, not learned positional embeddings. This is aligned with the paper's default formulation.

### 1.6 Embedding scaling by sqrt(d_model)

Code path:

- `nets/utils/encoder_decoder.py`

The embedding output is multiplied by `sqrt(d_model)` before adding positional encoding. This matches the paper.

### 1.7 Shared vocabulary and tied weights

Code path:

- `data/shared_vocab.py`
- `nets/build_transformer.py`

The project uses:

1. one shared source-target vocabulary
2. tied source embedding / target embedding / generator projection weights

This is consistent with the paper's shared BPE vocabulary setup and shared embedding/pre-softmax weights.

### 1.8 Training objective and optimizer choices

Code path:

- `utils/label_smoothing.py`
- `utils/noam_scheduler.py`
- `train_transformer_base.py`

Current defaults:

- label smoothing `0.1`
- Adam `beta1=0.9`, `beta2=0.98`, `eps=1e-9`
- Noam warmup `4000`

These are aligned with the paper.

### 1.9 Decoder masking and autoregressive decoding

Code path:

- `data/batch.py`
- `train_utils/validate_one_epoch.py`
- `evaluate_transformer_bleu.py`

The decoder uses causal masking during training and autoregressive decoding during inference. This is aligned with the paper.

## 2. Intentionally Different from the Paper

### 2.1 Pre-LN instead of Post-LN

Code path:

- `nets/utils/encoder_decoder.py`

Current implementation:

```text
x + sublayer(norm(x))
```

Paper-style formulation:

```text
norm(x + sublayer(x))
```

This is the largest architectural difference in the core model code.

Practical effect:

1. The project keeps the same high-level Transformer block structure.
2. But optimization behavior is different because normalization is applied before the sublayer instead of after it.

### 2.2 Training loop is epoch-based, not purely step-based

Code path:

- `train_transformer_base.py`
- `train_utils/fit.py`

The paper reports training largely in update steps. This project wraps training in epochs and optionally caps steps per epoch with:

- `max_train_steps_per_epoch`
- `max_valid_steps_per_epoch`

This is an engineering control layer, not a paper-level training loop.

### 2.3 Current default training cap is debug-friendly, not paper-faithful

Code path:

- `train_transformer_base.py`

Default config currently sets:

- `num_epochs = 100`
- `max_train_steps_per_epoch = 1000`
- `valid_num_text_samples = 0`
- `histogram_interval = 0`

This is convenient for incremental engineering work, but it is not the paper's training schedule.

### 2.4 Token-budget batching is only an approximation of the paper's batching

Code path:

- `data/wmt_14_bpe_dataset.py`
- `train_transformer_base.py`

Alignment:

1. The code keeps the same intent: group by similar lengths and control batch size by token count.

Difference:

1. The paper discusses multi-GPU training with batches sized by source/target tokens.
2. This project implements a single-machine `IterableDataset` + local pool sort + token-budget packing.
3. Current defaults are `4096 / 4096`, which are far smaller than the paper-scale setup.

So this is conceptually aligned, but not a strict reproduction of the original batching regime.

### 2.5 Data preprocessing is not a full paper-faithful reproduction

Code path:

- `script/dataset_part/01_download_dataset.py`
- `data/wmt14_raw_en_de/`
- `data/wmt14_tok_en_de/`
- `data/wmt14_clean_en_de/`
- `data/wmt14_bpe_en_de/`

The project clearly uses a raw -> tokenized -> cleaned -> BPE pipeline, which is in spirit close to the paper.

However:

1. The raw source is Hugging Face `wmt14/de-en`, not an explicit reproduction of the original data acquisition path.
2. The repository does not contain the scripts that actually perform tokenization, cleaning, and BPE generation.
3. The final shared vocab size in this workspace is `40236`, whereas the paper discusses a shared BPE vocabulary around `37k`.

This means the corpus pipeline is paper-inspired, but not fully reproducible from code in this repo and not guaranteed to match the exact original preprocessing recipe.

### 2.6 Validation focuses on loss/ppl/token accuracy during training

Code path:

- `train_utils/validate_one_epoch.py`
- `train_utils/fit.py`

The training loop monitors:

- valid loss
- valid NLL loss
- valid smooth loss
- valid token accuracy
- valid perplexity

The paper's headline evaluation is BLEU on translation outputs.

This project does include BLEU evaluation, but it is not the main in-training selection metric.

### 2.7 Greedy sample logging is not the paper's evaluation setup

Code path:

- `train_utils/validate_one_epoch.py`

Validation logs a few greedy-decoded text samples for quick inspection.

That is useful for debugging, but it is not part of the paper's final evaluation procedure.

### 2.8 Checkpoint averaging and beam search are external to training

Code path:

- `evaluate_transformer_bleu.py`

The paper's reported translation quality depends on inference-side choices such as beam search and checkpoint averaging.

This project supports both, but only in the standalone BLEU evaluation script, not in the main training loop.

### 2.9 Mixed precision is an engineering addition

Code path:

- `train_transformer_base.py`
- `train_utils/train_one_epoch.py`

AMP and `GradScaler` are modern engineering additions. They are not part of the original 2017 training recipe.

### 2.10 Training-set subset ratio is a local debugging feature

Code path:

- `train_transformer_base.py`
- `resume_train_transformer.py`
- `data/wmt_14_bpe_dataset.py`

The new `train_subset_ratio` option is intentionally outside the paper. It exists to shorten feedback loops when debugging the project.

## 3. Neutral Implementation Choices

These points are not meaningful paper deviations, but they are worth knowing:

### 3.1 Generator is separate from `model.forward(...)`

`EncoderDecoder.forward(...)` returns hidden states, and vocabulary projection is applied explicitly by `model.generator(...)`.

This is a code-organization choice, not a conceptual model change.

### 3.2 Attention masking uses dtype minimum

Code path:

- `nets/utils/attention.py`

Masked scores are filled with `torch.finfo(scores.dtype).min` rather than a handwritten constant. This is a numerical-stability choice for AMP compatibility, not a modeling change.

### 3.3 Xavier initialization is a practical default

Code path:

- `nets/build_transformer.py`

The project applies Xavier/Glorot initialization to matrix parameters. This is a conventional implementation choice and does not change the conceptual model.

## 4. Bottom Line

If you want to summarize the project in one sentence:

This repository is a solid Transformer-base implementation that follows the paper closely in architecture, objective, and optimization recipe, but differs in normalization placement, engineering-oriented training loop design, and partial reproducibility of the original preprocessing/evaluation pipeline.

If you want a stricter paper reproduction, the highest-priority gaps to address would be:

1. switch Pre-LN back to Post-LN
2. make batching/training schedule match the original step and token budget setup
3. fully script tokenization/cleaning/BPE creation inside the repo
4. integrate BLEU, beam search, and checkpoint averaging into the main experiment protocol rather than only the external evaluation script
