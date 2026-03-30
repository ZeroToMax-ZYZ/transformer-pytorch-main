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


# 项目数据规范
## 一、适用范围
本文档定义了当前代码仓库遵循的**数据契约**：
1. WMT14原始数据集的导出方式与磁盘存储结构
2. 每个预处理阶段的数据构成
3. `build_bpe_dataloader(...)`函数的读取输入与返回输出
4. 模型、损失函数、解码逻辑要求的张量形状规范
5. 不可修改的固定规范与支持自定义的可配置项

核心目标：让开发者无需猜测隐藏默认规则，即可完整追溯端到端数据流。

## 二、目录级数据流
仓库当前划分了以下6个数据集预处理阶段：

| 阶段名称 | 存储目录 | 说明 |
| ---- | ---- | ---- |
| 原始导出数据 | `data/wmt14_raw_en_de/` | 直接从Hugging Face `wmt14/de-en`导出的纯文本 |
| 分词文本数据 | `data/wmt14_tok_en_de/` | 已完成基础分词的英德平行语料 |
| 清洗文本数据 | `data/wmt14_clean_en_de/` | 过滤脏数据后的标准训练文本 |
| BPE子词文本 | `data/wmt14_bpe_en_de/` | 最终用于训练/评测的子词切分文本 |
| BPE模型文件 | `data/wmt14_bpe_model/` | BPE编码规则与辅助词表文件 |
| 共享全局词表 | `data/wmt14_vocab/` | 模型最终使用的统一token-ID映射表 |

仓库内置可直接运行的校验脚本链路：
1. `script/dataset_part/01_download_dataset.py`
将`wmt14/de-en`数据集导出为对齐纯文本，存入`data/wmt14_raw_en_de/`目录
2. `script/dataset_part/02_check_download.py`
校验原始英语、德语文本行数完全一致
3. `script/dataset_part/03_check_tok.py`
校验分词后的平行文本仍严格逐行对齐
4. `script/dataset_part/04_check_clean.py`
校验清洗后的训练文本仍严格逐行对齐
5. `script/dataset_part/05_check_val_empty.py`
统计验证集/测试集分词文本的空行与token长度分布
6. `script/dataset_part/06_check_bpe_result.py`
校验BPE子词文本对齐性，并检查词表覆盖度
7. `script/dataset_part/07_build_shared_vocab.py`
基于BPE切分的`train.en`和`train.de`构建最终全局共享词表

### 重要限制说明
仓库已包含分词/清洗/BPE处理后的成品文件，以及配套校验脚本；**但未内置分词、文本清洗、BPE编码生成的原始执行脚本**。因此仅靠本仓库代码，无法完整从零复现全套预处理流程。

## 三、当前工作区文件统计
基于现有文件统计各划分集行数：

| 数据集划分 | 原始文本行数 | BPE子词行数 |
| ---- | ---- | ---- |
| 训练集 | 4,508,785 | 3,927,488 |
| 验证集 | 3,000 | 3,000 |
| 测试集 | 3,003 | 3,003 |

### 数据解读
1. 训练集在BPE最终预处理前已被精简过滤，行数大幅减少
2. 验证集、测试集行数与原始导出划分完全对齐，无删减

## 四、共享全局词表契约
代码路径：
- `data/shared_vocab.py`
- `script/dataset_part/07_build_shared_vocab.py`

模型采用**源语言-目标语言统一共享词表**，无独立词表拆分。

### 固定特殊占位符（不可修改ID）
全局词表强制预留以下固定ID，永久不变：

| 特殊Token | 固定ID |
| ---- | ---- |
| `<pad>`（填充符） | 0 |
| `<bos>`（句首起始符） | 1 |
| `<eos>`（句尾结束符） | 2 |
| `<unk>`（未知生僻符） | 3 |

当前工作区元数据（`data/wmt14_vocab/meta.json`）：
- 总词表大小 `vocab_size = 40236`

### 词表构建规则
`SharedVocab.build_from_files(...)`执行逻辑：
1. 仅读取**训练集BPE文本**，不读取验证/测试集
2. 统计英语`train.en`与德语`train.de`所有token全局词频
3. 优先插入4个固定特殊占位符
4. 普通词汇排序入库规则：优先按**词频降序**；词频相同时按**字典序升序**

### 核心约束结论
1. 验证集、测试集数据不会泄露到词表构建过程，杜绝数据泄露
2. 源语言嵌入层、目标语言嵌入层、输出分类层可共用同一ID空间与权重矩阵

## 五、数据加载器读取规范
代码路径：
- `data/wmt_14_bpe_dataset.py`

训练/评测数据加载器仅读取标准UTF-8纯文本文件，文件必须满足4项要求：
1. 每行严格对应1句完整句子
2. 英语、德语平行文件必须**逐行严格对齐**，无错位
3. 所有token已通过空格完成分隔
4. 保留BPE子词连接符（如`@@`），不提前删除

标准文件配对示例：
- 源文本：`data/wmt14_bpe_en_de/train.en`
- 目标文本：`data/wmt14_bpe_en_de/train.de`

### 单条样本拼接前原始格式
`ParallelBPEIterableDataset`迭代器输出原始数据：
```python
(src_tokens, tgt_tokens)
```
两个元素均为Python字符串列表`List[str]`，此时状态：
1. 未映射数字ID，仅原生字符串
2. 未添加`<pad>`填充符
3. 未添加`<bos>/<eos>`特殊起止符
4. 纯原始平行token字符串对

## 六、训练集子集比例裁剪
代码路径：
- `train_transformer_base.py`
- `resume_train_transformer.py`
- `data/wmt_14_bpe_dataset.py`

仓库新增训练集快速裁剪参数，支持命令行配置：
```bash
python train_transformer_base.py --train-subset-ratio 0.2
```

### 参数含义
1. 仅加载BPE训练集前20%数据，快速缩短调试周期
2. 验证集、测试集不受参数影响，加载全部数据
3. 参数持久化存入配置：`data.train_subset_ratio`
4. 断点续训时，自动复用检查点中保存的裁剪比例

### 实现细节
1. `resolve_num_samples_for_ratio(总样本数, 裁剪比例)`：比例转为整数有效样本数
2. 最终数值同时传入两个参数：`num_samples`、`sample_limit`
3. `sample_limit`在多进程分片前强制生效，避免分片错位

### 关键结果约束
该裁剪为**前缀顺序子集**，非全局随机采样子集

## 七、批处理拼接对齐契约
代码路径：
- `data/wmt14_bpe_dataset.py` 中的`BPEBatchCollator`
- `data/batch.py` 中的`Seq2SeqBatch`

每个小批次对齐拼接标准流程：
1. 源token通过`SharedVocab.encode(...)`映射数字ID
2. 目标token通过`SharedVocab.encode(...)`映射数字ID
3. 源文本末尾可选追加`<eos>`结束符
4. 目标文本强制固定格式：
```text
<bos> 原始目标token序列 <eos>
```
5. 变长序列统一填充`<pad>`，对齐至批次最大长度
6. 填充完成张量封装为标准`Seq2SeqBatch`类

### 源文本统一格式
`src`张量最终组成：
```text
源token序列 + [<eos>] + 后续<pad>填充符
```
⚠️ 源文本**无`<bos>`句首起始符**

### 目标文本统一格式
移位输入前完整目标序列：
```text
[<bos>] + 目标token序列 + [<eos>] + 后续<pad>填充符
```
再执行标准右移位拆分：
```python
tgt_input = tgt[:, :-1]  # 解码器输入
tgt_y = tgt[:, 1:]       # 监督训练标签
```

### 核心作用
1. `tgt_input`：解码器真实可见的输入序列
2. `tgt_y`：损失函数拟合的监督真值
3. 严格遵循训练/验证通用**教师强制训练**标准范式

## 八、Seq2SeqBatch张量标准契约
代码路径：
- `data/batch.py`

`Seq2SeqBatch`封装类所有字段、形状与含义固定：

| 字段名 | 张量形状 | 字段含义 |
| ---- | ---- | ---- |
| `src` | `(B, S)` | 源语言填充后数字ID张量 |
| `tgt_input` | `(B, T)` | 解码器右移位输入ID张量 |
| `tgt_y` | `(B, T)` | 监督训练真值ID张量 |
| `src_mask` | `(B, 1, S)` | 源语言填充位置掩码 |
| `tgt_mask` | `(B, T, T)` | 目标语言填充掩码+因果未来掩码 |
| `ntokens` | 整数`int` | `tgt_y`中非填充有效token总数 |

### 掩码通用语义
1. 掩码数据类型：布尔型`bool`
2. `True`：允许注意力权重访问该位置
3. `False`：屏蔽该位置，禁止注意力访问

### 目标掩码构建逻辑
1. 目标填充掩码：`(B, 1, T)`
2. 下三角因果未来掩码：`(1, T, T)`
3. 两者逻辑**与运算合并**
最终形状：`(B, T, T)`

## 九、工作区真实批次张量示例
调用以下函数生成真实批次数据：
```python
build_bpe_dataloader(
    src_path="data/wmt14_bpe_en_de/train.en",
    tgt_path="data/wmt14_bpe_en_de/train.de",
    batch_size=4,
    max_src_len=64,
    max_tgt_len=64,
)
```

随机采样1个批次真实形状：
- `src.shape = (4, 43)`
- `tgt_input.shape = (4, 38)`
- `tgt_y.shape = (4, 38)`
- `src_mask.shape = (4, 1, 43)`
- `tgt_mask.shape = (4, 38, 38)`
- `ntokens = 93`

⚠️ 仅为示例数值，非固定常量；批次间形状随句子长度动态变化

## 十、批次构建两种模式
`build_bpe_dataloader(...)`支持两种主流批处理策略

### 模式A：固定句子条数批次
禁用令牌配额时默认生效：
- 必须配置正整数`batch_size`，批次固定包含N条句子

### 模式B：近似令牌配额动态批次
同时配置`src_token_budget`与`tgt_token_budget`时生效：
1. 样本先存入本地缓存池
2. 缓存池按句子近似长度排序
3. 动态堆叠样本，直至即将超出源/目标任一令牌配额上限

### 训练/验证默认配置
训练集默认动态令牌配额：
- `src_token_budget = 4096`
- `tgt_token_budget = 4096`
- `batch_pool_size = 2048`

验证集默认固定句子条数批次

## 十一、模型输入输出标准契约
代码路径：
- `nets/utils/encoder_decoder.py`
- `nets/build_transformer.py`
- `nets/utils/Generator.py`

训练/验证前向传播固定调用链路：
```python
hidden_states = model(
    batch.src,
    batch.tgt_input,
    batch.src_mask,
    batch.tgt_mask,
)
logits = model.generator(hidden_states)
```

输入输出张量形状对应：

| 张量名 | 标准形状 |
| ---- | ---- |
| `batch.src` | `(B, S)` |
| `batch.tgt_input` | `(B, T)` |
| `batch.src_mask` | `(B, 1, S)` |
| `batch.tgt_mask` | `(B, T, T)` |
| `hidden_states` | `(B, T, d_model)` |
| `logits` | `(B, T, vocab_size)` |

### 重要固定规范
`model.forward(...)`仅返回解码器隐状态，**不直接输出分类对数概率**；词表空间投影必须单独调用`model.generator(...)`显式执行

## 十二、损失函数标准契约
代码路径：
- `utils/label_smoothing.py`

`LabelSmoothingLoss.forward(logits, target)`强制输入形状：
1. 模型输出对数概率`logits`：`(B, T, V)`
2. 监督真值标签`target`：`(B, T)`

本项目唯一正确调用配对：
```python
criterion(logits, batch.tgt_y)
```

损失函数自动忽略所有`<pad>`填充ID位置，最终返回5项指标：
- 总加权损失值
- 原始负对数似然（NLL）损失
- 标签平滑正则损失
- 批次有效token总数

## 十三、验证与解码标准契约
### 验证集推理
代码路径：
- `train_utils/validate_one_epoch.py`

完全复用训练阶段教师强制范式：
1. 输入`batch.tgt_input`前向传播
2. 与`batch.tgt_y`真值计算损失指标

### 贪心解码（日志采样查看）
`greedy_decode(...)`固定流程：
1. 源文本一次性编码生成记忆隐状态`memory`
2. 目标序列初始化为`[<bos>]`句首符
3. 自回归逐token贪心追加生成
4. 遇到`<eos>`或达到最大长度终止

返回张量形状：`ys: (B, T_pred)`

### BLEU分数评测
代码路径：
- `evaluate_transformer_bleu.py`

独立于训练主循环，专属评测脚本流程：
1. 输入/参考文本：BPE切分测试集文件
2. 权重优化：多检查点权重均值融合
3. 解码策略：束搜索生成+长度惩罚
4. 后处理：替换`@@ `还原原始文本，去除BPE子词标记

## 十四、不可修改隐含强制约束
以下规则必须严格遵守，破坏将直接导致代码异常：
1. 源/目标平行文本行数必须完全相等
2. 逐行文本必须严格一一对应平行句对
3. 4个特殊token的固定ID永久不可变更
4. 传入`Seq2SeqBatch.from_tensors(...)`的目标张量必须已含`<bos>`与`<eos>`
5. `src_mask`形状必须适配注意力广播机制
6. `tgt_mask`必须同时满足：因果未来屏蔽+填充位置屏蔽
7. 模型前向传播必须仅返回隐状态，禁止直接输出Softmax概率

## 十五、训练故障优先排查清单
训练突然报错/异常时，按优先级依次排查：
1. 英德平行文本行数是否仍完全一致
2. 加载词表`vocab.json`是否匹配当前BPE文本
3. 4个特殊token ID是否被意外修改
4. `tgt_input`与`tgt_y`是否仍为同一序列右移位拆分
5. 掩码形状合规：`src_mask=(B,1,S)`、`tgt_mask=(B,T,T)`
6. 损失函数是否正确传入`(logits, batch.tgt_y)`
7. 训练集裁剪比例`train_subset_ratio`是否配置错误