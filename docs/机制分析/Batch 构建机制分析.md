# Batch 构建机制分析

## 1. 当前项目的 Batch 构建流程

整个流程分 4 层，从原始 BPE 文本到模型可消费的 `Seq2SeqBatch`：

```
BPE 文本文件 (.en / .de)
       │
       ▼
ParallelBPEIterableDataset    ← 逐行流式读取，yield (List[str], List[str])
       │
       ▼
ApproxTokenBucketBatchDataset ← 按 token 预算动态分 batch（可选）
       │
       ▼
BPEBatchCollator              ← 截断 → 编码 → 追加特殊 token → padding
       │
       ▼
Seq2SeqBatch                  ← shift right + 构造 mask
```

### 1.1 第一层：流式读取（ParallelBPEIterableDataset）

逐行同时打开 source 和 target BPE 文件，严格对齐：

```python
src_tokens = src_line.strip().split()   # List[str]，长度 s_i_raw
tgt_tokens = tgt_line.strip().split()   # List[str]，长度 t_i_raw
yield src_tokens, tgt_tokens
```

这一层**不产生 Tensor**，输出是 Python 的字符串列表。支持：
- 多 worker 按行号切片（`line_idx % total_shards != shard_id` 则跳过）
- 流式 buffer shuffle（固定大小缓冲区，近似全局打乱）
- `set_epoch(epoch)` 让每个 epoch 的 shuffle 种子不同

### 1.2 第二层：动态分 batch（ApproxTokenBucketBatchDataset）

这是训练集的核心。分 batch 的逻辑不是"固定 N 条一组"，而是**按 token 数量预算**动态决定：

```python
# 单条样本的代价
src_len = len(src_tokens) + 1   # +1 是追加的 <eos>
tgt_len = len(tokens) + 2       # +2 是 <bos> 和 <eos>
```

分 batch 规则（在 `_yield_batched_pool()` 中）：

1. 在一个 pool（默认 2048 条）内按 `max(src_len, tgt_len)` **排序**，减少 padding 浪费
2. 逐条加入当前 batch，累计 src/tgt token 数
3. 如果加入后**任一端超过预算**（默认 src=4096, tgt=4096），先产出当前 batch，开新 batch
4. 如果**句子数达到上限**（默认 128），也产出当前 batch

```
pool 内排序后逐条加入:
┌─────────────────────────────────────────────────────┐
│ sample 1: src=5,  tgt=7   → batch_src=5,  batch_tgt=7   │
│ sample 2: src=6,  tgt=8   → batch_src=11, batch_tgt=15  │
│ sample 3: src=8,  tgt=10  → batch_src=19, batch_tgt=25  │
│ ...                                                    │
│ sample N: src=12, tgt=15  → 如果 batch_src+12 > 4096   │
│                              → yield 当前 batch，开新    │
└─────────────────────────────────────────────────────┘
```

关键特点：
- **batch_size 是动态的**：每个 batch 的句子数取决于句子长度，不固定
- **预算约束的是"各样本原始长度之和"**，不是 padding 后的 `B × S`
- 验证集走另一条路：固定 `batch_size=64`

### 1.3 第三层：编码与 padding（BPEBatchCollator）

每个 batch 内的样本经过以下处理：

```
原始 BPE tokens:  ["A", "dog", "is", "running"]
                        │
                        ▼ 截断（max_src_len=100）
                   ["A", "dog", "is", "running"]
                        │
                        ▼ vocab.encode()：token → id
                   [156, 4821, 87, 3012]
                        │
                        ▼ 追加特殊 token
                   src: [156, 4821, 87, 3012, 2]      ← +<eos>
                   tgt: [1, 89, 203, 417, 2]           ← +<bos>...+<eos>
                        │
                        ▼ pad_sequences()：右侧补 0
                   src: [156, 4821, 87, 3012, 2, 0, 0]  → (B, S)
                   tgt: [1, 89, 203, 417, 2, 0, 0, 0]   → (B, T_full)
```

`pad_sequences()` 的实现：

```python
out = torch.full((batch_size, max_len), pad_idx, dtype=torch.long)  # 全填 PAD
for i, seq in enumerate(sequences):
    out[i, :len(seq)] = torch.tensor(seq)   # 左对齐写入
```

### 1.4 第四层：Shift Right + Mask 构造（Seq2SeqBatch）

`Seq2SeqBatch.from_tensors()` 完成训练前的最后准备：

```python
# 1. source mask: 屏蔽 PAD
src_mask = (src != 0).unsqueeze(1)          # (B, S) → (B, 1, S)

# 2. target 右移切分
tgt_input = tgt[:, :-1]                     # (B, T_full) → (B, T)  送入 Decoder
tgt_y     = tgt[:, 1:]                      # (B, T_full) → (B, T)  监督标签

# 3. target mask: 因果遮蔽 + PAD 遮蔽
pad_mask    = (tgt_input != 0).unsqueeze(1)  # (B, 1, T)
causal_mask = torch.tril(ones(1, T, T))      # (1, T, T)
tgt_mask    = pad_mask & causal_mask          # (B, T, T)

# 4. 有效 token 数
ntokens = (tgt_y != 0).sum()                 # 标量，用于 loss 归一化
```

最终 `Seq2SeqBatch` 包含：

| 字段 | shape | dtype | 用途 |
|------|-------|-------|------|
| `src` | `(B, S)` | int64 | Encoder 输入 |
| `tgt_input` | `(B, T)` | int64 | Decoder 输入 |
| `tgt_y` | `(B, T)` | int64 | 监督标签 |
| `src_mask` | `(B, 1, S)` | bool | Encoder/Cross-Attention mask |
| `tgt_mask` | `(B, T, T)` | bool | Decoder Self-Attention mask |
| `ntokens` | 标量 | int | loss 归一化用 |

---

## 2. 与传统 CV 任务的区别

### 2.1 输入形状：固定 vs 变长

| | CV | NLP（本项目） |
|---|---|---|
| 单样本 shape | `(C, H, W)` 固定 | `(seq_len,)` 变长 |
| batch shape | `(B, C, H, W)` 整齐 | `(B, S)` 需要 padding 对齐 |
| 对齐方式 | resize / center crop 到统一尺寸 | 右侧补 `<pad>` 到 batch 内最长 |

CV 的 `transforms.Resize(224)` 保证每张图都是 `224×224`，batch 天然整齐。NLP 的句子长度从 3 个 token 到 100 个 token 不等，**必须 padding 才能组成 batch tensor**。

### 2.2 Mask 机制：CV 不需要，NLP 必须

CV 的 batch 中每个样本的所有像素都参与计算，没有"无效区域"的概念（除了极少数检测任务的 padding）。

NLP 中 padding 出来的位置是**人为填充的 0**，不能参与：
- 注意力计算（不能让真实 token 去"关注"PAD）
- Loss 计算（不能让模型去"预测"PAD）

所以 NLP 需要两套 mask：

```
src_mask (B, 1, S):  屏蔽 Encoder 中的 PAD key
tgt_mask (B, T, T):  同时屏蔽 PAD key + 未来位置（因果遮蔽）
```

CV 任务中通常不需要 mask，因为所有像素都是真实的。

### 2.3 数据增强：像素级 vs 序列级

| | CV | NLP（本项目） |
|---|---|---|
| 增强方式 | 随机裁剪、翻转、颜色抖动、Mixup | 无（本项目未使用） |
| 作用对象 | 像素值 | — |
| Dropout 作用 | 无（或 DropPath） | Attention Dropout + Residual Dropout |

本项目没有使用数据增强（没有 BPE dropout、没有随机删除等），正则化完全靠 Dropout 和 Label Smoothing。

### 2.4 Batch 大小策略：固定 vs 动态

CV 通常用**固定 batch_size**（如 32、64、128），因为每张图的 shape 一致，显存消耗可预测。

本项目训练集用 **token-budget 动态 batch**：

```
固定 batch_size=64:
┌─────────────────────────────────────────────────┐
│ 64 条样本，最长 80 token → 每条都 pad 到 80       │
│ 总 token = 64 × 80 = 5120                        │
│ 短句浪费大量 padding                              │
└─────────────────────────────────────────────────┘

token-budget=4096:
┌─────────────────────────────────────────────────┐
│ 先排序，长短搭配                                  │
│ 50 条样本，最长 90 token → pad 到 90              │
│ 总 token ≈ 50 × 90 = 4500（更紧凑）               │
│ 或 80 条样本，最长 55 token → pad 到 55            │
│ 总 token ≈ 80 × 55 = 4400（更紧凑）               │
└─────────────────────────────────────────────────┘
```

动态 batch 的好处：
- **显存更稳定**：不管句子长短，每步的总 token 数大致相同
- **padding 更少**：排序后长短搭配，减少了 padding 浪费
- **训练更高效**：同样一个 epoch，有效 token 占比更高

### 2.5 数据集类型：MapDataset vs IterableDataset

| | CV | NLP（本项目） |
|---|---|---|
| 数据集类型 | `torch.utils.data.Dataset`（Map 式） | `IterableDataset`（流式） |
| 随机访问 | 支持（`dataset[idx]`） | 不支持（只能顺序读） |
| Shuffle | `sampler=RandomSampler` | 流式 buffer shuffle |
| 多 epoch 重排 | 每个 epoch 自动重排 | 需要 `set_epoch()` 改变种子 |

CV 的图片数据集通常可以随机访问（`ImageFolder` 直接按文件名索引），所以用 `RandomSampler` 打乱即可。

NLP 的语料文件通常很大（几十 GB），不适合全部加载到内存，所以用 `IterableDataset` 流式读取。流式读取天然不支持随机访问，只能用 buffer shuffle 近似打乱。

### 2.6 汇总对比表

| 维度 | CV（如 ImageNet 分类） | NLP（本项目 Transformer） |
|------|----------------------|--------------------------|
| 单样本 | `(3, 224, 224)` 固定 | `(seq_len,)` 变长 |
| 对齐方式 | Resize/Crop | 右侧 Pad |
| Mask | 不需要 | `src_mask` + `tgt_mask` |
| Batch 大小 | 固定 N 张 | 动态（按 token 预算） |
| 数据增强 | 像素级变换 | 无（靠 Dropout + LS） |
| 数据集类型 | MapDataset | IterableDataset |
| Shuffle | RandomSampler | Buffer Shuffle |
| Loss 归一化 | 除以 batch_size | 除以有效 token 数 |

---

## 3. 与原论文的对比

### 3.1 原论文的 Batch 描述

原论文（Attention Is All You Need, 2017）Section 5.1 描述：

> "Each training batch contained a set of sentence pairs containing approximately 25000 source tokens and 25000 target tokens."

论文用的是 **token batching**：按 token 数量而非句子数量来控制 batch 大小。

### 3.2 当前实现 vs 论文

| 论文描述 | 当前实现 | 是否一致 |
|---------|---------|---------|
| 按 token 数量分 batch（~25000 src + ~25000 tgt） | `ApproxTokenBucketBatchDataset` 按 token 预算分 batch | **思路一致，数值不同** |
| 多 GPU token batching | 单机 DataLoader + 多 worker | **架构不同** |
| 按长度排序减少 padding | pool 内按 `max(src, tgt)` 排序 | **一致** |
| BPE 分词（32000 合并规则） | BPE 分词，vocab_size 取自实际词表 | **一致** |
| Shared source-target vocabulary | `share_embeddings=True` | **一致** |
| `<bos>` / `<eos>` 特殊 token | `tgt_ids = [bos] + ids + [eos]`，src 追加 `<eos>` | **一致** |
| Warmup steps = 4000 | `warmup_steps = 2000`（可配置） | **数值不同，机制一致** |
| Adam (β1=0.9, β2=0.98, ε=1e-9) | 完全一致 | **一致** |
| Label Smoothing (ε=0.1) | `smoothing=0.1` | **一致** |

### 3.3 关键差异详解

#### 差异 1：Token Budget 数量级

论文用 **~25000 src + ~25000 tgt**（约 50000 token/batch）。这是在 8 块 P100 GPU 上、使用较大模型（base: d_model=512, big: d_model=1024）的配置。

当前项目默认 **4096 src + 4096 tgt**（约 8192 token/batch）。这是面向单卡/小规模训练的合理缩减。

```
论文:     ~25000 token/batch × 8 GPU = ~200000 token/step
当前:     ~4096  token/batch × 1 GPU = ~4096  token/step
```

token budget 影响的是**梯度估计的方差**：budget 越大，每个 step 看到的样本越多，梯度越稳定，但单 step 耗时也越长。

#### 差异 2：多 GPU Batching 策略

论文的 token batching 是跨 GPU 的：所有 GPU 上的样本拼起来才达到 25000 token。当前实现是单机 `DataLoader` 层面的 token budget，不涉及跨 GPU 拼接。

在 DDP 模式下，当前实现是每个 rank 各自分 batch，然后通过 `all_reduce` 同步梯度。这和论文的"全局 token budget"有本质区别。

#### 差异 3：Source 是否追加 EOS

论文没有明确提到 source 端追加 `<eos>`。当前实现中 `add_src_eos=True`（默认），会在 source 末尾追加一个 `<eos>` token。这是一个实现细节，对模型效果影响很小。

### 3.4 总结

当前项目的 batch 构建机制**在设计思路上与论文高度一致**：

- 按 token 数量而非句子数量控制 batch 大小（token batching）
- 按长度排序减少 padding 浪费
- Shared vocabulary + weight tying
- BPE 分词 + 特殊 token 处理

**数值和规模上有差异**，但这些差异是由训练硬件条件决定的工程折中，不影响模型架构和训练逻辑的正确性。token budget 从 25000 缩减到 4096 是合理的单卡适配，warmup steps 从 4000 缩减到 2000 是针对小数据集（Multi30k ~29000 条）的调参选择。
