# Multi30k 数据预处理流程

本文档整理了 Multi30k En-De 数据集从下载到可以送入 Transformer 训练的完整预处理流程。

---

## 总览

```
① 下载原始数据          ② 学习 BPE 规则         ③ 应用 BPE 切分         ④ 构建词表           ⑤ 训练时加载
GitHub → 英德文本 → learn_bpe → tok 文本 → apply_bpe → BPE 文本 → build_from_files → vocab.json
      ↓                         ↓                       ↓                       ↓
01_prepare.py           01_prepare.py            01_prepare.py            01_prepare.py       DataLoader 训练时实时处理
(一键完成 ①~④)
```

| 数据集 | 下载来源 | 训练集规模 | 词表大小 |
|--------|---------|-----------|---------|
| Multi30k | GitHub (官方 tokenized 版本) | 29,000 对 | 9,712 |

一键完成全部离线预处理（①~④）：

```bash
python script/dataset_part/Multi30k/01_prepare.py
```

---

## 第一步：下载原始数据并导出为平行文本

### 对应代码

`script/dataset_part/Multi30k/01_prepare.py` → `export_official_tokenized_files()`

```
main()
 └── export_official_tokenized_files(raw_dir, force_download)
      └── download_file(url, out_path)          # 逐文件下载
           └── urllib.request.urlopen(url)       # 标准库下载
      └── check_parallel(out_en, out_de)         # 校验英德行数一致
```

### 数据来源

从 GitHub 下载官方已经分好词的版本：

```
URL: https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/tok/
文件映射:
  train.lc.norm.tok.en   →  data/multi30k_tok_en_de/train.en
  train.lc.norm.tok.de   →  data/multi30k_tok_en_de/train.de
  val.lc.norm.tok.en     →  data/multi30k_tok_en_de/valid.en
  val.lc.norm.tok.de     →  data/multi30k_tok_en_de/valid.de
  test_2016_flickr...en  →  data/multi30k_tok_en_de/test.en
  test_2016_flickr...de  →  data/multi30k_tok_en_de/test.de
```

### 输出格式

导出为严格对齐的双行文本文件，一行一句，英德对应：

```
data/multi30k_tok_en_de/
├── train.en        ← 29,000 行
├── train.de        ← 29,000 行
├── valid.en        ← 1,014 行
├── valid.de        ← 1,014 行
├── test.en         ← 1,000 行
└── test.de         ← 1,000 行
```

### 数据样例

train.en（已分词，空格分隔）：
```
two young , white males are outside near many bushes .
several men in hard hats are operating a giant pulley system .
a little girl climbing into a wooden playhouse .
```

train.de（已分词，空格分隔）：
```
zwei junge weiße männer sind im freien in der nähe vieler büsche .
mehrere männer mit schutzhelmen bedienen ein antriebsradsystem .
ein kleines mädchen klettert in ein spielhaus aus holz .
```

注意：此时的"分词"是基于空格的简单切分，还不是 BPE 子词。德语中 `antriebsradsystem` 是一整个词。

---

## 第二步：学习 BPE 合并规则

BPE（Byte Pair Encoding）的核心思想是：**从字符级别开始，反复合并出现频率最高的相邻 token 对**，直到达到预设的合并次数。

### 对应代码

`script/dataset_part/Multi30k/01_prepare.py` → `learn_joint_bpe_codes()`

```
main()
 └── learn_joint_bpe_codes(train_en_path, train_de_path, codes_path, num_merges, min_frequency)
      ├── shutil.copyfileobj(train_en, tmp)    # 拼接英文训练集
      ├── shutil.copyfileobj(train_de, tmp)    # 拼接德文训练集 → 合并为一个临时文件
      └── learn_bpe(                           # subword_nmt 库的核心函数
              infile=tmp,
              outfile=codes_path,
              num_symbols=10000,               # 最多 10000 次合并
              min_frequency=2,                 # 至少出现 2 次才合并
              num_workers=1,
          )
      └── tmp.unlink()                         # 删除临时文件
```

依赖库：`subword_nmt`（`pip install subword-nmt`）

```python
from subword_nmt.learn_bpe import learn_bpe
```

### 具体做法

1. 将 train.en 和 train.de **拼接**成一个临时文件（联合学习，共享子词表）
2. 调用 `learn_bpe` 从拼接语料中学习合并规则
3. 保存为 `codes.shared` 文件
4. 删除临时文件

### 输出文件

```
data/multi30k_bpe_model/codes.shared    ← 10,001 行（第 1 行是版本号注释）
```

### codes.shared 文件格式

```
#version: 0.2
i n              ← 第 1 次合并：'i' 和 'n' 经常相邻，合并成 'in'
e n</w>          ← 第 2 次合并：'e' 和 'n</w>'（词尾 n）合并
i n</w>          ← 第 3 次合并
e r</w>
e in
a n
c h
u n
e r
in g</w>
...
khaki farbener</w>
kell ner</w>
kehr en</w>      ← 第 10,000 次合并
```

每条规则表示"将这两个相邻 token 合并为一个"。`</w>` 表示词尾边界。

### 为什么英德联合学习？

- 英语和德语有大量共享的子词（数字、标点、拉丁词根）
- 联合学习可以让同一套 BPE 规则同时处理两种语言
- 最终只需要一个共享词表，不需要分别维护 src/tgt 两套词表

---

## 第三步：应用 BPE 对文本进行子词切分

用上一步学到的合并规则，对所有 split（train/valid/test）的英德文本进行 BPE 切分。

### 对应代码

`script/dataset_part/Multi30k/01_prepare.py` → `build_bpe_text()` → `apply_bpe_file()`

```
main()
 └── build_bpe_text(raw_dir, bpe_dir, codes_path, force_bpe)
      ├── BPE(codes=f_codes, merges=-1, separator="@@")   # 加载 BPE 规则
      └── apply_bpe_file(in_path, out_path, bpe)          # 对每个文件逐行处理
           └── bpe.process_line(line)                      # subword_nmt 的核心切分函数
```

关键参数：
- `separator="@@`：子词连接符，`pul@@ ley` 表示这两个子词原本属于同一个词
- `merges=-1`：使用 codes 文件中的全部合并规则

### 输出目录

```
data/multi30k_bpe_en_de/
├── train.en        ← BPE 处理后的英文（29,000 行）
├── train.de        ← BPE 处理后的德文（29,000 行）
├── valid.en        ← 1,014 行
├── valid.de        ← 1,014 行
├── test.en         ← 1,000 行
└── test.de         ← 1,000 行
```

### 数据样例对比

同一句话在 BPE 前后的变化：

```
BPE 前 (multi30k_tok_en_de/train.en):
several men in hard hats are operating a giant pulley system .

BPE 后 (multi30k_bpe_en_de/train.en):
several men in hard hats are operating a giant pul@@ ley system .
                                                    ↑         ↑
                                              "pulley" 被切成了 "pul@@ ley"
                                              @@ 表示这两个子词原本属于同一个词
```

```
BPE 前 (multi30k_tok_en_de/train.de):
mehrere männer mit schutzhelmen bedienen ein antriebsradsystem .

BPE 后 (multi30k_bpe_en_de/train.de):
mehrere männer mit schutzhelmen bedienen ein an@@ trie@@ b@@ s@@ rad@@ system .
                                                    ↑    ↑   ↑   ↑    ↑
                                              "antriebsradsystem" 被切成了 6 个子词
                                              长词、复合词会被切得更细
```

```
BPE 前 (multi30k_tok_en_de/train.de):
zwei männer stehen am herd und bereiten essen zu .

BPE 后 (multi30k_bpe_en_de/train.de):
zwei männer stehen am herd und bereiten essen zu .
                                              ↑
                                      短词保持不变，不被切分
```

### BPE 的效果

- 高频词保持完整（如 `the`, `a`, `und`, `ein`）
- 低频长词被拆成高频子词（如 `antriebsradsystem` → `an@@ trie@@ b@@ s@@ rad@@ system`）
- 词表大小可控（9,712 个子词 vs 如果不做 BPE 可能有几万个独立词）

---

## 第四步：构建共享词表

从 BPE 处理后的训练集（train.en + train.de）构建一个英德共享的词表。

### 对应代码

`script/dataset_part/Multi30k/01_prepare.py` → `build_shared_vocab_from_bpe()`

内部调用 `data/shared_vocab.py` → `SharedVocab.build_from_files()`

```
main()
 └── build_shared_vocab_from_bpe(bpe_dir, vocab_dir, force_vocab)
      └── SharedVocab.build_from_files(
              file_paths=["train.en", "train.de"],   # 只用训练集，防止信息泄露
              min_freq=1,
              special_tokens=SpecialTokens(),         # <pad>, <bos>, <eos>, <unk>
          )
           ├── Counter()                             # 统计所有 BPE token 频率
           │    └── for line in f: counter.update(line.strip().split())
           ├── 特殊 token 固定 id: pad=0, bos=1, eos=2, unk=3
           ├── 普通 token 排序: 词频降序 → 字典序
           └── 依次分配 id: 从 4 开始
      └── vocab.save(vocab_json, vocab_txt, meta_json)
```

### 输出文件

```
data/multi30k_vocab/
├── vocab.json      ← {"token": id} 的 JSON 映射
├── vocab.txt       ← 每行一个 token，行号即 id
└── meta.json       ← 词表元信息
```

### vocab.json 样例

```json
{
  "<pad>": 0,
  "<bos>": 1,
  "<eos>": 2,
  "<unk>": 3,
  ".": 4,
  "a": 5,
  "in": 6,
  "ein": 7,
  "einem": 8,
  ",": 9,
  "the": 10,
  "eine": 11,
  "und": 12,
  "mit": 13,
  "auf": 14
}
```

总词表大小：**9,712** 个 token（Multi30k 数据集较小，词表也较小）。

### meta.json 样例

```json
{
  "vocab_size": 9712,
  "pad_token": "<pad>",
  "bos_token": "<bos>",
  "eos_token": "<eos>",
  "unk_token": "<unk>",
  "pad_id": 0,
  "bos_id": 1,
  "eos_id": 2,
  "unk_id": 3
}
```

### prepare_meta.json（预处理元信息）

`01_prepare.py` 最后还会保存一份预处理元信息：

```json
{
  "dataset": "multi30k",
  "language_pair": "en-de",
  "source": "official_multi30k_task1_tokenized",
  "raw_dir": "data/multi30k_tok_en_de",
  "bpe_dir": "data/multi30k_bpe_en_de",
  "vocab_dir": "data/multi30k_vocab",
  "num_bpe_merges": 10000,
  "bpe_min_frequency": 2,
  "train_num_samples": 29000,
  "valid_num_samples": 1014,
  "test_num_samples": 1000,
  "vocab_size": 9712
}
```

### 为什么用共享词表？

- 英德共享同一套 token → id 映射
- 源端和目标端的 Embedding 可以**权重共享**（weight tying）
- Generator 的输出投影也可以和 Embedding 共享权重
- 这是原论文的做法

---

## 第五步：训练时实时处理（DataLoader）

前四步是离线预处理，产出的是 BPE 文本文件和词表文件。训练时，DataLoader 在每个 batch 实时完成以下转换。

### 对应代码

入口：`train_multi30k_base.py` → `build_bpe_dataloader()`

```
train_multi30k_base.py main()
 └── build_bpe_dataloader(...)                     # data/wmt_14_bpe_dataset.py
      ├── ParallelBPEIterableDataset(...)           # 流式读取 BPE 文本
      │    └── _buffer_shuffle_iterator()           # 流式 shuffle
      │         └── _line_iterator()                # 逐行读取 yield (List[str], List[str])
      ├── ApproxTokenBucketBatchDataset(...)        # [训练集] token-budget 动态分 batch
      │    └── _yield_batched_pool()                # 排序 + 逐条累加 + 切 batch
      └── BPEBatchCollator(...)                     # collate_fn: 编码 + padding + 构造 Seq2SeqBatch
           └── __call__(batch)
```

### 5.1 读取 BPE 文本

**代码位置**：`data/wmt_14_bpe_dataset.py` → `ParallelBPEIterableDataset._line_iterator()`

```python
src_tokens = src_line.strip().split()   # "several men in ..." → ["several", "men", "in", ...]
tgt_tokens = tgt_line.strip().split()
yield src_tokens, tgt_tokens
```

```
输入: "several men in hard hats are operating a giant pul@@ ley system ."
输出: ["several", "men", "in", "hard", "hats", "are", "operating", "a", "giant", "pul@@", "ley", "system", "."]
```

### 5.2 动态分 batch（仅训练集）

**代码位置**：`data/wmt_14_bpe_dataset.py` → `ApproxTokenBucketBatchDataset._yield_batched_pool()`

```python
# 1. pool 内按长度排序（减少 padding 浪费）
samples.sort(key=lambda x: max(len(x[0]), len(x[1])))

# 2. 逐条加入 batch，累加原始长度
src_len = len(src_tokens) + 1   # +1 是 <eos>
tgt_len = len(tgt_tokens) + 2   # +2 是 <bos> + <eos>

# 3. 如果超出预算（默认 src=4096, tgt=4096）或句子数达上限（128），切 batch
if batch_src_tokens + src_len > 4096 or batch_tgt_tokens + tgt_len > 4096:
    yield current_batch
```

验证集走固定 `batch_size=64` 路径，不经过 `ApproxTokenBucketBatchDataset`。

### 5.3 截断（max_src_len / max_tgt_len）

**代码位置**：`data/wmt_14_bpe_dataset.py` → `BPEBatchCollator.__call__()`

```python
if self.max_src_len is not None:
    src_tokens = src_tokens[:self.max_src_len]    # 最多保留 100 个 BPE token

if self.max_tgt_len is not None:
    tgt_tokens = tgt_tokens[:self.max_tgt_len]
```

### 5.4 词表编码（token → id）

**代码位置**：`data/shared_vocab.py` → `SharedVocab.encode()`

```python
src_ids = self.vocab.encode(src_tokens)
# 内部: [self.token2id(tok) for tok in src_tokens]
# token2id() 查 self.token_to_id 字典，找不到返回 self.unk_id(3)

# ["several", "men", "in", "hard", ...] → [245, 87, 6, 512, ...]
```

### 5.5 追加特殊 token

**代码位置**：`data/wmt_14_bpe_dataset.py` → `BPEBatchCollator.__call__()`

```python
# 源端：末尾追加 <eos>（id=2）
if self.add_src_eos:
    src_ids = src_ids + [self.vocab.eos_id]       # [245, 87, 6, ..., 2]

# 目标端：开头加 <bos>(1)，末尾加 <eos>(2)
tgt_ids = [self.vocab.bos_id] + tgt_ids + [self.vocab.eos_id]   # [1, 156, 4821, ..., 2]
```

### 5.6 Padding 对齐

**代码位置**：`data/batch.py` → `pad_sequences()`

```python
def pad_sequences(sequences, pad_idx):
    max_len = max(len(seq) for seq in sequences)
    out = torch.full((batch_size, max_len), pad_idx, dtype=torch.long)  # 全填 0
    for i, seq in enumerate(sequences):
        out[i, :len(seq)] = torch.tensor(seq)   # 左对齐写入
    return out
```

```
batch 内 3 条 src（编码后）:
  src[0] = [245, 87, 6, 512, 301, 2]           长度 6
  src[1] = [72, 533, 2]                         长度 3
  src[2] = [156, 4821, 87, 3012, 118, 55, 2]   长度 7

padding 后（右侧补 0）:
  src[0] = [245,  87,   6,  512,  301,   2,   0]
  src[1] = [ 72, 533,   2,    0,    0,   0,   0]
  src[2] = [156, 4821, 87, 3012, 118,  55,   2]

→ Tensor shape: (3, 7)
```

### 5.7 Shift Right（目标序列右移）

**代码位置**：`data/batch.py` → `shift_right()`

```python
tgt_input = tgt[:, :-1].contiguous()   # 丢掉最后一个 token
tgt_y     = tgt[:, 1:].contiguous()    # 丢掉第一个 token (<bos>)
```

```
tgt (含 <bos> 和 <eos>):
[<bos>, 156, 4821, 87, <eos>, <pad>, <pad>]
    │
    ├── tgt_input = tgt[:, :-1]   →  [<bos>, 156, 4821, 87, <eos>, <pad>]  送入 Decoder
    └── tgt_y     = tgt[:, 1:]    →  [156, 4821, 87, <eos>, <pad>, <pad>]  监督标签
```

### 5.8 构造 Mask

**代码位置**：`data/batch.py` → `make_src_mask()` + `make_tgt_mask()`

```python
# src_mask: 屏蔽 PAD
src_mask = (src != pad_idx).unsqueeze(1)                    # (B, S) → (B, 1, S)

# tgt_mask: 因果遮蔽 + 屏蔽 PAD
pad_mask    = (tgt_input != pad_idx).unsqueeze(1)           # (B, 1, T)
causal_mask = torch.tril(torch.ones(1, T, T, dtype=bool))   # (1, T, T) 下三角
tgt_mask    = pad_mask & causal_mask                         # (B, T, T) 广播
```

```
src_mask:  [T, T, T, T, T, T, F]   ← 最后一个位置是 PAD

tgt_mask (因果 + PAD):
  [[T, F, F, F, F, F],    ← 位置 0 (<bos>) 只能看自己
   [T, T, F, F, F, F],    ← 位置 1 能看 0,1
   [T, T, T, F, F, F],
   [T, T, T, T, F, F],
   [T, T, T, T, T, F],    ← 位置 4 (<eos>) 能看 0~4
   [F, F, F, F, F, F]]    ← PAD 位置全部屏蔽
```

### 5.9 统计有效 token 数

**代码位置**：`data/batch.py` → `Seq2SeqBatch.from_tensors()`

```python
ntokens = int((tgt_y != pad_idx).sum().item())
# 用于 loss 归一化：loss = sum(per_token_loss) / ntokens
```

### 最终输出：Seq2SeqBatch

**代码位置**：`data/batch.py` → `Seq2SeqBatch`

```python
batch.src        # (B, S)      源端 token id，已 padding
batch.tgt_input  # (B, T)      Decoder 输入，已右移
batch.tgt_y      # (B, T)      监督标签
batch.src_mask   # (B, 1, S)   源端 mask
batch.tgt_mask   # (B, T, T)   目标端 mask（因果 + PAD）
batch.ntokens    # 标量         有效 token 数（用于 loss 归一化）
```

---

## 完整流程示例

以一条训练样本为例，走完全流程：

```
┌──────────────────────────────────────────────────────────────────────────┐
│ ① 原始文本（下载自 GitHub）                                              │
│    EN: "Several men in hard hats are operating a giant pulley system ."  │
│    DE: "Mehrere Männer mit Schutzhelmen bedienen ein Antriebsradsystem ."│
│    代码: Multi30k/01_prepare.py → export_official_tokenized_files()      │
└──────────────────────────┬───────────────────────────────────────────────┘
                           ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ ② 空格分词 → multi30k_tok_en_de/                                        │
│    EN: "several men in hard hats are operating a giant pulley system ."  │
│    DE: "mehrere männer mit schutzhelmen bedienen ein antriebsradsystem ."│
│    （Multi30k 官方已做好小写 + 空格分词，下载即用）                        │
└──────────────────────────┬───────────────────────────────────────────────┘
                           ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ ③ BPE 子词切分 → multi30k_bpe_en_de/                                    │
│    代码: Multi30k/01_prepare.py → build_bpe_text() → apply_bpe_file()    │
│    核心: subword_nmt BPE.process_line()                                  │
│                                                                          │
│    EN: "several men in hard hats are operating a giant pul@@ ley system ."│
│    DE: "mehrere männer mit schutzhelmen bedienen ein an@@ trie@@ b@@     │
│          s@@ rad@@ system ."                                             │
└──────────────────────────┬───────────────────────────────────────────────┘
                           ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ ④ 构建共享词表 → multi30k_vocab/                                         │
│    代码: Multi30k/01_prepare.py → build_shared_vocab_from_bpe()          │
│    核心: data/shared_vocab.py → SharedVocab.build_from_files()           │
│                                                                          │
│    vocab.json: {"<pad>":0, "<bos>":1, "<eos>":2, "<unk>":3, ...}       │
│    总词表: 9,712 个 token                                                │
└──────────────────────────┬───────────────────────────────────────────────┘
                           ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ ⑤ 训练时 DataLoader 实时处理                                             │
│    入口: train_multi30k_base.py → build_bpe_dataloader()                 │
│    核心: data/wmt_14_bpe_dataset.py → BPEBatchCollator.__call__()        │
│         data/batch.py → Seq2SeqBatch.from_tensors()                      │
│                                                                          │
│    BPE 文本 → split() → encode() → 加 <bos>/<eos> → pad → shift right   │
│    → 构造 mask → Seq2SeqBatch                                            │
│                                                                          │
│    batch.src        (B, S)       送入 Encoder                            │
│    batch.tgt_input  (B, T)       送入 Decoder                            │
│    batch.tgt_y      (B, T)       计算 loss                               │
│    batch.src_mask   (B, 1, S)    Encoder/Cross-Attention mask            │
│    batch.tgt_mask   (B, T, T)    Decoder Self-Attention mask             │
└──────────────────────────┬───────────────────────────────────────────────┘
                           ▼
                      送入 Transformer 训练
```

---

## 脚本索引

| 脚本 | 作用 |
|------|------|
| `Multi30k/01_prepare.py` | Multi30k 一键预处理（①~④） |

### 关键函数

| 函数 | 作用 |
|------|------|
| `export_official_tokenized_files()` | 从 GitHub 下载 Multi30k 官方分词文件 |
| `learn_joint_bpe_codes()` | 英德联合学习 BPE 合并规则 |
| `build_bpe_text()` | 对所有 split 应用 BPE 切分 |
| `build_shared_vocab_from_bpe()` | 从 BPE 文本构建共享词表 |
| `save_prepare_meta()` | 保存预处理元信息 |

---

## 代码文件索引

| 文件 | 作用 | 关键函数/类 |
|------|------|------------|
| `script/dataset_part/Multi30k/01_prepare.py` | Multi30k 一键预处理（①~④） | `export_official_tokenized_files()`, `learn_joint_bpe_codes()`, `build_bpe_text()`, `build_shared_vocab_from_bpe()` |
| `data/shared_vocab.py` | 词表类 | `SharedVocab.build_from_files()`, `.encode()`, `.decode()`, `.token2id()`, `.save()`, `.load()` |
| `data/wmt_14_bpe_dataset.py` | DataLoader 构建 | `ParallelBPEIterableDataset`, `ApproxTokenBucketBatchDataset`, `BPEBatchCollator`, `build_bpe_dataloader()` |
| `data/batch.py` | Batch 封装 | `Seq2SeqBatch.from_tensors()`, `make_src_mask()`, `make_tgt_mask()`, `shift_right()`, `pad_sequences()` |
| `train_multi30k_base.py` | 训练入口 | `main()` 调用 `build_bpe_dataloader()` 构建 DataLoader |
