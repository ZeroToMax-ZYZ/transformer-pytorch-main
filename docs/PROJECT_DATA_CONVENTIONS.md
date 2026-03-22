# Project Data Conventions

## 1. Scope

This document describes the data contract used by the current repository:

1. How raw WMT14 data is exported and organized on disk.
2. What each data stage contains.
3. What `build_bpe_dataloader(...)` reads and returns.
4. What tensor shapes are expected by the model, loss, and decoding code.
5. Which conventions are fixed, and which are configurable.

The goal is that you can follow the data flow end to end without guessing hidden assumptions.

## 2. Directory-Level Data Flow

The repository currently contains the following dataset stages:

| Stage | Directory | Meaning |
| --- | --- | --- |
| Raw export | `data/wmt14_raw_en_de/` | Direct text export from Hugging Face `wmt14/de-en` |
| Tokenized text | `data/wmt14_tok_en_de/` | Tokenized parallel text |
| Cleaned text | `data/wmt14_clean_en_de/` | Cleaned training text |
| BPE text | `data/wmt14_bpe_en_de/` | Final subword text used by training/evaluation |
| BPE model files | `data/wmt14_bpe_model/` | BPE codes and auxiliary vocab files |
| Shared vocab | `data/wmt14_vocab/` | Final shared token-id mapping used by the model |

The implemented script chain that is visible in the repo is:

1. `script/dataset_part/01_download_dataset.py`
   Exports `wmt14/de-en` into aligned text files under `data/wmt14_raw_en_de/`.
2. `script/dataset_part/02_check_download.py`
   Verifies raw English/German line counts match.
3. `script/dataset_part/03_check_tok.py`
   Verifies tokenized files remain aligned.
4. `script/dataset_part/04_check_clean.py`
   Verifies cleaned training files remain aligned.
5. `script/dataset_part/05_check_val_empty.py`
   Checks empty lines and token-length stats for validation/test tokenized files.
6. `script/dataset_part/06_check_bpe_result.py`
   Verifies BPE files remain aligned and inspects vocabulary coverage.
7. `script/dataset_part/07_build_shared_vocab.py`
   Builds the final shared vocabulary from BPE `train.en` and `train.de`.

Important limitation:

The repository contains the tokenized/cleaned/BPE artifacts and the inspection scripts for them, but it does not currently contain the scripts that actually perform tokenization, cleaning, or BPE application. Those stages are therefore part of the project state, but not fully reproducible from code inside this repo alone.

## 3. Current File Counts in This Workspace

Observed from the current workspace:

| Split | Raw lines | BPE lines |
| --- | --- | --- |
| Train | 4,508,785 | 3,927,488 |
| Valid | 3,000 | 3,000 |
| Test | 3,003 | 3,003 |

Implication:

1. The training set was reduced during preprocessing before the final BPE stage.
2. Validation and test sizes remain aligned with the exported dataset splits.

## 4. Shared Vocabulary Contract

Code path:

- `data/shared_vocab.py`
- `script/dataset_part/07_build_shared_vocab.py`

The model uses one shared vocabulary for source and target.

### Fixed special tokens

The shared vocabulary reserves these ids:

| Token | ID |
| --- | --- |
| `<pad>` | 0 |
| `<bos>` | 1 |
| `<eos>` | 2 |
| `<unk>` | 3 |

Current workspace metadata from `data/wmt14_vocab/meta.json`:

- `vocab_size = 40236`

### Build rule

`SharedVocab.build_from_files(...)`:

1. Reads only BPE training files.
2. Counts token frequencies over both `train.en` and `train.de`.
3. Inserts special tokens first.
4. Inserts normal tokens sorted by:
   - frequency descending
   - token lexicographic order for ties

This means:

1. Validation/test statistics do not leak into vocab construction.
2. Source and target embeddings can share the same id space and the same weight matrix.

## 5. What the Dataloader Reads

Code path:

- `data/wmt_14_bpe_dataset.py`

The training and evaluation dataloaders read plain UTF-8 text files where:

1. Each line is one sentence.
2. English and German files must remain strictly line-aligned.
3. Tokens are already split by spaces.
4. BPE continuation marks such as `@@` are still present in file text.

Example file pair:

- `data/wmt14_bpe_en_de/train.en`
- `data/wmt14_bpe_en_de/train.de`

### Per-sample representation before collation

`ParallelBPEIterableDataset` yields:

```python
(src_tokens, tgt_tokens)
```

where both elements are Python `List[str]`.

At this stage:

1. No ids exist yet.
2. No padding exists yet.
3. No BOS/EOS have been added yet.
4. The sample is still a pair of token strings.

## 6. Training-Set Subset Ratio

Code path:

- `train_transformer_base.py`
- `resume_train_transformer.py`
- `data/wmt_14_bpe_dataset.py`

The repository now supports:

```bash
python train_transformer_base.py --train-subset-ratio 0.2
```

Meaning:

1. Only the first `20%` of the BPE training split is used.
2. Validation and test behavior do not change.
3. The ratio is stored in config as `data.train_subset_ratio`.
4. Resume training reuses the same ratio from the checkpoint config.

Implementation detail:

1. `resolve_num_samples_for_ratio(total_num_samples, subset_ratio)` converts the ratio into an integer count.
2. The resulting count is passed as both:
   - `num_samples`
   - `sample_limit`
3. `sample_limit` is enforced inside `ParallelBPEIterableDataset` before worker sharding.

Important consequence:

This is a prefix subset, not a random subset over the entire corpus.

## 7. Collation Contract

Code path:

- `BPEBatchCollator` in `data/wmt_14_bpe_dataset.py`
- `Seq2SeqBatch` in `data/batch.py`

For each mini-batch:

1. Source tokens are encoded with `SharedVocab.encode(...)`.
2. Target tokens are encoded with `SharedVocab.encode(...)`.
3. Source optionally appends `<eos>`.
4. Target always becomes:

```text
<bos> target_tokens <eos>
```

5. Variable-length sequences are padded to the longest length in the batch.
6. The padded tensors are converted into `Seq2SeqBatch`.

### Source-side convention

`src` contains:

```text
source_tokens + [<eos>] + [<pad> ...]
```

There is no `<bos>` on the source side.

### Target-side convention

Before shift-right, the full target sequence is:

```text
[<bos>] + target_tokens + [<eos>] + [<pad> ...]
```

Then:

```python
tgt_input = tgt[:, :-1]
tgt_y = tgt[:, 1:]
```

So:

1. `tgt_input` is what the decoder sees.
2. `tgt_y` is the supervision target.
3. This is the standard teacher-forcing contract used by training and validation.

## 8. Seq2SeqBatch Tensor Contract

Code path:

- `data/batch.py`

`Seq2SeqBatch` contains:

| Field | Shape | Meaning |
| --- | --- | --- |
| `src` | `(B, S)` | Source token ids, padded |
| `tgt_input` | `(B, T)` | Decoder input ids, right-shifted |
| `tgt_y` | `(B, T)` | Supervision target ids |
| `src_mask` | `(B, 1, S)` | Source padding mask |
| `tgt_mask` | `(B, T, T)` | Target padding mask + causal mask |
| `ntokens` | `int` | Number of non-pad tokens in `tgt_y` |

Mask semantics:

1. Mask dtype is `bool`.
2. `True` means the position is allowed to attend.
3. `False` means it is masked out.

`tgt_mask` is constructed as:

1. target padding mask `(B, 1, T)`
2. causal lower-triangular mask `(1, T, T)`
3. logical `and`

result:

```text
(B, T, T)
```

## 9. Real Example Shapes from This Workspace

Observed from a real batch built by:

```python
build_bpe_dataloader(
    src_path="data/wmt14_bpe_en_de/train.en",
    tgt_path="data/wmt14_bpe_en_de/train.de",
    batch_size=4,
    max_src_len=64,
    max_tgt_len=64,
)
```

One sampled batch produced:

- `src.shape = (4, 43)`
- `tgt_input.shape = (4, 38)`
- `tgt_y.shape = (4, 38)`
- `src_mask.shape = (4, 1, 43)`
- `tgt_mask.shape = (4, 38, 38)`
- `ntokens = 93`

These numbers are examples, not fixed constants. They vary batch to batch because sentence lengths vary.

## 10. Batch Construction Modes

`build_bpe_dataloader(...)` supports two batching modes.

### Mode A: fixed sentence count

Used when token budgets are disabled:

- `batch_size` must be a positive integer.

### Mode B: approximate token-budget batching

Used when both `src_token_budget` and `tgt_token_budget` are set:

1. Samples are collected into a local pool.
2. The pool is sorted by approximate sentence length.
3. Samples are packed until the source or target budget would be exceeded.

Training default in `train_transformer_base.py`:

- `src_token_budget = 4096`
- `tgt_token_budget = 4096`
- `batch_pool_size = 2048`

Validation default remains fixed-size batching.

## 11. Model Input/Output Contract

Code path:

- `nets/utils/encoder_decoder.py`
- `nets/build_transformer.py`
- `nets/utils/Generator.py`

Training/validation forward path:

```python
hidden_states = model(
    batch.src,
    batch.tgt_input,
    batch.src_mask,
    batch.tgt_mask,
)
logits = model.generator(hidden_states)
```

Shapes:

| Tensor | Shape |
| --- | --- |
| `batch.src` | `(B, S)` |
| `batch.tgt_input` | `(B, T)` |
| `batch.src_mask` | `(B, 1, S)` |
| `batch.tgt_mask` | `(B, T, T)` |
| `hidden_states` | `(B, T, d_model)` |
| `logits` | `(B, T, vocab_size)` |

Important convention:

`model.forward(...)` returns decoder hidden states, not logits. The projection to vocabulary space is a separate explicit step via `model.generator(...)`.

## 12. Loss Contract

Code path:

- `utils/label_smoothing.py`

`LabelSmoothingLoss.forward(logits, target)` expects:

1. `logits` shape `(B, T, V)`
2. `target` shape `(B, T)`

In this project, the correct pairing is:

```python
criterion(logits, batch.tgt_y)
```

The loss ignores `pad_idx` positions and returns:

- total loss
- NLL component
- smoothing component
- effective token count

## 13. Validation and Decoding Contract

### Validation

Code path:

- `train_utils/validate_one_epoch.py`

Validation uses the same teacher-forcing contract as training:

1. forward with `batch.tgt_input`
2. compare against `batch.tgt_y`

### Greedy decode for sample logging

`greedy_decode(...)`:

1. Encodes `src` once to `memory`
2. Starts target with `[<bos>]`
3. Autoregressively appends one token at a time
4. Stops at `<eos>` or `max_len`

Return shape:

- `ys: (B, T_pred)`

### BLEU evaluation

Code path:

- `evaluate_transformer_bleu.py`

This script uses:

1. BPE test files as input/reference
2. checkpoint averaging
3. beam search decoding
4. length penalty
5. final BPE detokenization by replacing `@@ `

It is separate from the training loop.

## 14. Non-Obvious Invariants

The following assumptions must remain true for the code to behave correctly:

1. Source and target text files must have exactly the same number of lines.
2. Every line pair must correspond to the same sentence pair.
3. `SharedVocab` ids for `<pad>`, `<bos>`, `<eos>`, `<unk>` must stay stable.
4. `tgt` passed into `Seq2SeqBatch.from_tensors(...)` must already contain `<bos>` and `<eos>`.
5. `src_mask` shape must stay compatible with attention broadcasting.
6. `tgt_mask` must remain causal and must also hide target padding.
7. `model.forward(...)` must keep returning hidden states, not softmax probabilities.

## 15. What to Check When Something Breaks

If training suddenly fails, check these in order:

1. English/German files still have identical line counts.
2. `vocab.json` still matches the BPE files you are loading.
3. `pad_id`, `bos_id`, `eos_id`, `unk_id` are unchanged.
4. `tgt_input` and `tgt_y` still come from the same shifted full target.
5. `src_mask` is `(B, 1, S)` and `tgt_mask` is `(B, T, T)`.
6. `criterion(...)` is still fed with `(logits, batch.tgt_y)`.
7. The configured `train_subset_ratio` is what you intended.
