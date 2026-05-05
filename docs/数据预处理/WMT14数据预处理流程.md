# WMT14 数据预处理流程

本文档整理了 WMT14 En-De 数据集从下载到可以送入 Transformer 训练的完整预处理流程。

---

## 总览

```
① 下载原始数据      ② Moses 分词         ③ Moses 清洗         ④ 学习 BPE 规则      ⑤ 应用 BPE 切分      ⑥ 构建词表           ⑦ 训练时加载
HuggingFace → raw → normalize+tok → tok → clean-corpus-n → clean → learn-joint-bpe → codes → apply-bpe → BPE → build_from_files → vocab.json
  01_download         命令行               03_check_tok        04_check_clean        命令行                 命令行              07_build_vocab
  02_check_download                                                                                   06_check_bpe
```

| 数据集 | 下载来源 | 训练集规模 | 词表大小 |
|--------|---------|-----------|---------|
| WMT14 | HuggingFace `wmt14/de-en` | ~4,500,000 对（清洗后 ~3,900,000） | ~40,000 |

### 数据目录结构

```
data/
├── wmt14_raw_en_de/        ← ① 原始导出
│   ├── train.en / train.de
│   ├── valid.en / valid.de
│   └── test.en  / test.de
├── wmt14_tok_en_de/        ← ② Moses 分词后
│   ├── train.en / train.de
│   ├── valid.en / valid.de
│   └── test.en  / test.de
├── wmt14_clean_en_de/      ← ③ Moses 清洗后（仅 train）
│   ├── train.en
│   └── train.de
├── wmt14_bpe_model/        ← ④ BPE 规则文件
│   ├── codes.bpe
│   ├── vocab.en
│   └── vocab.de
├── wmt14_bpe_en_de/        ← ⑤ BPE 切分后（最终训练数据）
│   ├── train.en / train.de
│   ├── valid.en / valid.de
│   └── test.en  / test.de
└── wmt14_vocab/            ← ⑥ 共享词表
    ├── vocab.json
    ├── vocab.txt
    └── meta.json
```

---

## 第一步：下载原始数据并导出为平行文本

### 对应代码

`script/dataset_part/WMT14/01_download.py` → `export_all_wmt14()`

```
main()
 └── export_all_wmt14()
      └── load_dataset("wmt14", "de-en")        # HuggingFace datasets 库
      └── export_split_to_text()                 # 逐 split 导出
           └── sanitize_to_single_line()         # 强制单行化，去掉 \r \n
      └── check_parallel_files()                 # 校验行数一致
```

### 数据来源

```python
dataset = load_dataset("wmt14", "de-en")
# dataset["train"], dataset["validation"], dataset["test"]
```

### 输出格式

导出为严格对齐的双行文本文件，一行一句，英德对应：

```
data/wmt14_raw_en_de/
├── train.en        ← ~4,500,000 行
├── train.de        ← ~4,500,000 行
├── valid.en        ← ~3,000 行
├── valid.de        ← ~3,000 行
├── test.en         ← ~3,000 行
└── test.de         ← ~3,000 行
```

### 数据样例

train.en（原始文本）：
```
" Iron reinforces the determination of the international community ."
" The President spoke about the importance of climate change ."
```

train.de（原始文本）：
```
" Eisen stärkt die Entschlossenheit der internationalen Gemeinschaft ."
" Der Präsident sprach über die Bedeutung des Klimawandels ."
```

### 验证脚本

```bash
python script/dataset_part/WMT14/02_check_download.py
# 检查 train.en vs train.de 行数是否一致
# 检查 valid / test 同理
```

---

## 第二步：Moses 分词（Tokenization）

使用 Moses 工具包对原始文本进行标准化和分词处理。这一步是经典 NMT 预处理流程的标准做法（fairseq 也是这样做的）。

### 处理流程

对每个文件依次执行三个 Moses 脚本（管道串联）：

1. `normalize-punctuation.perl` — 标准化标点符号
2. `remove-non-printing-char.perl` — 移除不可打印字符
3. `tokenizer.perl -threads 8 -a -l $lang` — 空格分词

### 执行命令

**训练集（英/德）**：

```bash
# 英文
cmd /c "perl tools/mosesdecoder/scripts/tokenizer/normalize-punctuation.perl -l en < data/wmt14_raw_en_de/train.en | perl tools/mosesdecoder/scripts/tokenizer/remove-non-printing-char.perl | perl tools/mosesdecoder/scripts/tokenizer/tokenizer.perl -threads 8 -a -l en > data/wmt14_tok_en_de/train.en"

# 德文
cmd /c "perl tools/mosesdecoder/scripts/tokenizer/normalize-punctuation.perl -l de < data/wmt14_raw_en_de/train.de | perl tools/mosesdecoder/scripts/tokenizer/remove-non-printing-char.perl | perl tools/mosesdecoder/scripts/tokenizer/tokenizer.perl -threads 8 -a -l de > data/wmt14_tok_en_de/train.de"
```

**验证集（英/德）**：

```bash
cmd /c "perl tools/mosesdecoder/scripts/tokenizer/normalize-punctuation.perl -l en < data/wmt14_raw_en_de/valid.en | perl tools/mosesdecoder/scripts/tokenizer/remove-non-printing-char.perl | perl tools/mosesdecoder/scripts/tokenizer/tokenizer.perl -threads 8 -a -l en > data/wmt14_tok_en_de/valid.en"

cmd /c "perl tools/mosesdecoder/scripts/tokenizer/normalize-punctuation.perl -l de < data/wmt14_raw_en_de/valid.de | perl tools/mosesdecoder/scripts/tokenizer/remove-non-printing-char.perl | perl tools/mosesdecoder/scripts/tokenizer/tokenizer.perl -threads 8 -a -l de > data/wmt14_tok_en_de/valid.de"
```

**测试集（英/德）**：

```bash
cmd /c "perl tools/mosesdecoder/scripts/tokenizer/normalize-punctuation.perl -l en < data/wmt14_raw_en_de/test.en | perl tools/mosesdecoder/scripts/tokenizer/remove-non-printing-char.perl | perl tools/mosesdecoder/scripts/tokenizer/tokenizer.perl -threads 8 -a -l en > data/wmt14_tok_en_de/test.en"

cmd /c "perl tools/mosesdecoder/scripts/tokenizer/normalize-punctuation.perl -l de < data/wmt14_raw_en_de/test.de | perl tools/mosesdecoder/scripts/tokenizer/remove-non-printing-char.perl | perl tools/mosesdecoder/scripts/tokenizer/tokenizer.perl -threads 8 -a -l de > data/wmt14_tok_en_de/test.de"
```

### 输出目录

```
data/wmt14_tok_en_de/
├── train.en        ← ~4,500,000 行
├── train.de        ← ~4,500,000 行
├── valid.en        ← ~3,000 行
├── valid.de        ← ~3,000 行
├── test.en         ← ~3,000 行
└── test.de         ← ~3,000 行
```

### 数据样例对比

```
原始 (wmt14_raw_en_de/train.en):
" Iron reinforces the determination of the international community ."

分词后 (wmt14_tok_en_de/train.en):
& quot ; Iron reinforces the determination of the international community .
```

注意：`normalize-punctuation.perl` 会将引号等特殊字符标准化，`tokenizer.perl` 会在标点符号前后加空格。

### 验证脚本

```bash
python script/dataset_part/WMT14/03_check_tok.py
# 检查 data/wmt14_tok_en_de/ 下英德行数是否一致
```

---

## 第三步：Moses 清洗（Clean）

使用 Moses 的 `clean-corpus-n.perl` 对训练集进行清洗，去掉长度异常的句对。

### 为什么只清洗训练集？

- 训练集中可能存在长度异常的句对（过长或过短），会影响训练效果
- 验证集和测试集应保持原始状态，不做任何删减
- 这和 fairseq 脚本的做法一致：对 test 不做 `clean-corpus-n.perl`

### 执行命令

```bash
cmd /c "perl tools/mosesdecoder/scripts/training/clean-corpus-n.perl -ratio 1.5 data/wmt14_tok_en_de/train en de data/wmt14_clean_en_de/train 1 250"
```

### 参数说明

| 参数 | 值 | 含义 |
|------|-----|------|
| `-ratio` | `1.5` | 源/目标句子长度比超过 1.5 的句对会被删除 |
| 最小长度 | `1` | 保留至少 1 个 token 的句子 |
| 最大长度 | `250` | 保留最多 250 个 token 的句子 |

### 输出目录

```
data/wmt14_clean_en_de/
├── train.en        ← ~3,900,000 行（比原始少了约 60 万行）
└── train.de        ← ~3,900,000 行
```

### 清洗效果

- 原始训练集：~4,508,785 对
- 清洗后训练集：~3,927,488 对
- 删除了约 580,000 对长度异常的句对

### 验证脚本

```bash
python script/dataset_part/WMT14/04_check_clean.py
# 检查 data/wmt14_clean_en_de/ 下英德行数是否一致
```

---

## 第四步：检查验证集质量

检查验证集是否有空行，并统计句长分布：

```bash
python script/dataset_part/WMT14/05_check_val_empty.py
# 检查 valid/test 的空行数
# 统计最短/最长/平均句长
```

---

## 第五步：学习 BPE 合并规则

BPE（Byte Pair Encoding）的核心思想是：**从字符级别开始，反复合并出现频率最高的相邻 token 对**，直到达到预设的合并次数。

### 为什么用 subword-nmt 而不是 Moses 做 BPE？

- Moses 负责 tokenize/clean，subword-nmt 负责 BPE，这是经典分工
- fairseq 的公开脚本也是这样做的
- subword-nmt 的 `learn-joint-bpe-and-vocab` 可以同时学习 BPE 规则和双语词表

### 执行命令

```bash
subword-nmt learn-joint-bpe-and-vocab \
    --input data/wmt14_clean_en_de/train.en data/wmt14_clean_en_de/train.de \
    -s 37000 \
    -o data/wmt14_bpe_model/codes.bpe \
    --write-vocabulary data/wmt14_bpe_model/vocab.en data/wmt14_bpe_model/vocab.de
```

### 参数说明

| 参数 | 值 | 含义 |
|------|-----|------|
| `--input` | `train.en train.de` | 输入文件（英德联合学习） |
| `-s` | `37000` | BPE merge 次数（论文中 shared BPE 约 37k） |
| `-o` | `codes.bpe` | 输出的 BPE 规则文件 |
| `--write-vocabulary` | `vocab.en vocab.de` | 同时输出英德各自的子词词表 |

### 输出文件

```
data/wmt14_bpe_model/
├── codes.bpe       ← 37,000 条 BPE 合并规则
├── vocab.en        ← 英文子词词表（token → 频率）
└── vocab.de        ← 德文子词词表（token → 频率）
```

### codes.bpe 文件格式

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
```

每条规则表示"将这两个相邻 token 合并为一个"。`</w>` 表示词尾边界。

### 为什么英德联合学习？

- 英语和德语有大量共享的子词（数字、标点、拉丁词根）
- 联合学习可以让同一套 BPE 规则同时处理两种语言
- 最终只需要一个共享词表，不需要分别维护 src/tgt 两套词表
- 这是 Transformer 原论文的做法（shared source-target vocabulary）

---

## 第六步：应用 BPE 对文本进行子词切分

用上一步学到的合并规则，对所有 split 的英德文本进行 BPE 切分。

**关键区别**：
- **训练集**：从 `wmt14_clean_en_de/`（清洗后）读取
- **验证集/测试集**：从 `wmt14_tok_en_de/`（分词后，未清洗）读取

### 执行命令

**训练集（从清洗后的文本）**：

```bash
# 英文
cmd /c "subword-nmt apply-bpe -c data/wmt14_bpe_model/codes.bpe --vocabulary data/wmt14_bpe_model/vocab.en --vocabulary-threshold 50 < data/wmt14_clean_en_de/train.en > data/wmt14_bpe_en_de/train.en"

# 德文
cmd /c "subword-nmt apply-bpe -c data/wmt14_bpe_model/codes.bpe --vocabulary data/wmt14_bpe_model/vocab.de --vocabulary-threshold 50 < data/wmt14_clean_en_de/train.de > data/wmt14_bpe_en_de/train.de"
```

**验证集（从分词后的文本）**：

```bash
cmd /c "subword-nmt apply-bpe -c data/wmt14_bpe_model/codes.bpe --vocabulary data/wmt14_bpe_model/vocab.en --vocabulary-threshold 50 < data/wmt14_tok_en_de/valid.en > data/wmt14_bpe_en_de/valid.en"

cmd /c "subword-nmt apply-bpe -c data/wmt14_bpe_model/codes.bpe --vocabulary data/wmt14_bpe_model/vocab.de --vocabulary-threshold 50 < data/wmt14_tok_en_de/valid.de > data/wmt14_bpe_en_de/valid.de"
```

**测试集（从分词后的文本）**：

```bash
cmd /c "subword-nmt apply-bpe -c data/wmt14_bpe_model/codes.bpe --vocabulary data/wmt14_bpe_model/vocab.en --vocabulary-threshold 50 < data/wmt14_tok_en_de/test.en > data/wmt14_bpe_en_de/test.en"

cmd /c "subword-nmt apply-bpe -c data/wmt14_bpe_model/codes.bpe --vocabulary data/wmt14_bpe_model/vocab.de --vocabulary-threshold 50 < data/wmt14_tok_en_de/test.de > data/wmt14_bpe_en_de/test.de"
```

### 参数说明

| 参数 | 含义 |
|------|------|
| `-c codes.bpe` | BPE 规则文件 |
| `--vocabulary vocab.en/de` | 子词词表文件 |
| `--vocabulary-threshold 50` | 词频低于 50 的子词会被替换为 `<unk>` |

### 输出目录

```
data/wmt14_bpe_en_de/
├── train.en        ← BPE 处理后的英文（~3,900,000 行）
├── train.de        ← BPE 处理后的德文（~3,900,000 行）
├── valid.en        ← ~3,000 行
├── valid.de        ← ~3,000 行
├── test.en         ← ~3,000 行
└── test.de         ← ~3,000 行
```

### 数据样例对比

同一句话在 BPE 前后的变化：

```
BPE 前 (wmt14_tok_en_de/train.en):
Iron reinforces the determination of the international community .

BPE 后 (wmt14_bpe_en_de/train.en):
Iron re@@ in@@ force@@ s the d@@ eterm@@ in@@ ation of the inter@@ national community .
```

```
BPE 前 (wmt14_tok_en_de/train.de):
Eisen stärkt die Entschlossenheit der internationalen Gemeinschaft .

BPE 后 (wmt14_bpe_en_de/train.de):
Eisen stärkt die Ent@@ sch@@ lose@@ en@@ heit der inter@@ nationalen Ge@@ me@@ in@@ schaft .
```

### BPE 的效果

- 高频词保持完整（如 `the`, `a`, `und`, `ein`）
- 低频长词被拆成高频子词
- 词表大小可控（~40,000 个子词）

### 验证脚本

```bash
python script/dataset_part/WMT14/06_check_bpe.py
# 检查 BPE 后的 train/valid/test 英德行数是否一致
# 统计英德各自的 vocab 大小和合并后的 union vocab 大小
```

---

## 第七步：构建共享词表

从 BPE 处理后的训练集（train.en + train.de）构建一个英德共享的词表。

### 对应代码

`script/dataset_part/WMT14/07_build_vocab.py`

### 词表构建逻辑

```python
# 1. 扫描训练集，统计词频
counter = Counter()
for path in [train_en, train_de]:
    for line in f:
        tokens = line.strip().split()    # 按空格切分 BPE token
        counter.update(tokens)

# 2. 固定特殊 token（id 0~3）
token_to_id = {
    "<pad>": 0,
    "<bos>": 1,
    "<eos>": 2,
    "<unk>": 3,
}

# 3. 普通 token 按词频降序、字典序排序
normal_tokens = [(tok, freq) for tok, freq in counter.items() if freq >= 1]
normal_tokens.sort(key=lambda x: (-x[1], x[0]))

# 4. 依次分配 id
for tok, _ in normal_tokens:
    token_to_id[tok] = len(token_to_id)   # 从 4 开始
```

### 输出文件

```
data/wmt14_vocab/
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
  "the": 5,
  ",": 6,
  "of": 7,
  "to": 8,
  "and": 9,
  "in": 10
}
```

### 运行命令

```bash
python script/dataset_part/WMT14/07_build_vocab.py
```

---

## 第八步：验证 DataLoader

验证 DataLoader 能正常产出 Seq2SeqBatch：

```bash
python script/dataset_part/WMT14/08_test_dataloader.py
# 验证 vocab 加载
# 验证 batch 构造
# 打印 shape 信息
```

---

## 第九步：验证前向传播

验证真实 batch 能送入 Transformer 做前向传播：

```bash
python script/dataset_part/WMT14/09_test_forward.py
# 构建微型 Transformer
# 用真实 batch 做前向传播
# 验证输出 shape 正确
```

---

## 完整流程示例

以一条训练样本为例，走完全流程：

```
┌──────────────────────────────────────────────────────────────────────────┐
│ ① 原始文本（下载自 HuggingFace）                                         │
│    EN: " Iron reinforces the determination of the international community ."│
│    DE: " Eisen stärkt die Entschlossenheit der internationalen Gemeinschaft ."│
│    代码: WMT14/01_download.py → export_all_wmt14()                       │
└──────────────────────────┬───────────────────────────────────────────────┘
                           ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ ② Moses 分词 → wmt14_tok_en_de/                                         │
│    命令: normalize-punctuation → remove-non-printing → tokenizer.perl    │
│    EN: "Iron reinforces the determination of the international community ."│
│    DE: "Eisen stärkt die Entschlossenheit der internationalen Gemeinschaft ."│
└──────────────────────────┬───────────────────────────────────────────────┘
                           ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ ③ Moses 清洗 → wmt14_clean_en_de/（仅 train）                           │
│    命令: clean-corpus-n.perl -ratio 1.5 ... 1 250                        │
│    删除长度异常句对，训练集从 ~4,500,000 → ~3,900,000                     │
└──────────────────────────┬───────────────────────────────────────────────┘
                           ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ ④ 学习 BPE 规则 → wmt14_bpe_model/                                      │
│    命令: subword-nmt learn-joint-bpe-and-vocab -s 37000                  │
│    输出: codes.bpe (37,000 条合并规则) + vocab.en + vocab.de             │
└──────────────────────────┬───────────────────────────────────────────────┘
                           ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ ⑤ 应用 BPE 切分 → wmt14_bpe_en_de/                                      │
│    命令: subword-nmt apply-bpe -c codes.bpe --vocabulary-threshold 50    │
│    train 从 clean 读取，valid/test 从 tok 读取                           │
│    EN: "Iron re@@ in@@ force@@ s the d@@ eterm@@ in@@ ation of ..."     │
│    DE: "Eisen stärkt die Ent@@ sch@@ lose@@ en@@ heit der ..."          │
└──────────────────────────┬───────────────────────────────────────────────┘
                           ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ ⑥ 构建共享词表 → wmt14_vocab/                                            │
│    代码: WMT14/07_build_vocab.py                                         │
│    vocab.json: {"<pad>":0, "<bos>":1, "<eos>":2, "<unk>":3, ...}       │
│    总词表: ~40,000 个 token                                              │
└──────────────────────────┬───────────────────────────────────────────────┘
                           ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ ⑦ 训练时 DataLoader 实时处理                                             │
│    入口: train_multi30k_base.py → build_bpe_dataloader()                 │
│    核心: data/wmt_14_bpe_dataset.py → BPEBatchCollator.__call__()        │
│         data/batch.py → Seq2SeqBatch.from_tensors()                      │
│                                                                          │
│    BPE 文本 → split() → encode() → 加 <bos>/<eos> → pad → shift right   │
│    → 构造 mask → Seq2SeqBatch                                            │
└──────────────────────────┬───────────────────────────────────────────────┘
                           ▼
                      送入 Transformer 训练
```

---

## 脚本索引

| 脚本 | 作用 |
|------|------|
| `WMT14/01_download.py` | 从 HuggingFace 下载 WMT14 数据集并导出为文本文件 |
| `WMT14/02_check_download.py` | 检查下载后英德行数一致 |
| `WMT14/03_check_tok.py` | 检查分词后英德行数一致 |
| `WMT14/04_check_clean.py` | 检查清洗后英德行数一致 |
| `WMT14/05_check_val_empty.py` | 检查验证集无空行，统计句长分布 |
| `WMT14/06_check_bpe.py` | 检查 BPE 后行数一致 + 统计 vocab 大小 |
| `WMT14/07_build_vocab.py` | 从 BPE 文本构建共享词表 |
| `WMT14/08_test_dataloader.py` | 验证 DataLoader 能产出正确 shape 的 Seq2SeqBatch |
| `WMT14/09_test_forward.py` | 验证真实 batch 能送入 Transformer 前向传播 |

### 命令行操作（无脚本）

| 步骤 | 工具 | 命令 |
|------|------|------|
| Moses 分词 | `tools/mosesdecoder/` | `normalize-punctuation.perl` + `remove-non-printing-char.perl` + `tokenizer.perl` |
| Moses 清洗 | `tools/mosesdecoder/` | `clean-corpus-n.perl -ratio 1.5 ... 1 250` |
| BPE 学习 | `subword-nmt` | `learn-joint-bpe-and-vocab -s 37000` |
| BPE 应用 | `subword-nmt` | `apply-bpe -c codes.bpe --vocabulary-threshold 50` |
