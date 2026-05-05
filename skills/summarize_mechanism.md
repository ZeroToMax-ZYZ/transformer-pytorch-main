# Skill: 机制总结（summarize_mechanism）

当用户要求总结某个机制（如 batch 构建、数据加载、训练流程、解码策略等）时，按以下规范输出。

---

## 核心原则

**讲清楚"现在是怎么做的"，然后和"别人怎么做的"对比，最后说清楚"为什么这样做"。**

目标读者是"想理解当前项目的设计选择，以及这些选择和标准做法/原论文有什么异同"的人。

---

## 结构模板

### 1. 当前项目的实现流程

用分层/分步的方式，从输入到输出讲清楚整个机制。

#### 1.1 先给全局流程图

用 ASCII 图展示数据流经的主要组件：

```markdown
BPE 文本文件 (.en / .de)
       │
       ▼
ParallelBPEIterableDataset    ← 逐行流式读取
       │
       ▼
ApproxTokenBucketBatchDataset ← 动态分 batch
       │
       ▼
BPEBatchCollator              ← 编码 + padding
       │
       ▼
Seq2SeqBatch                  ← shift right + mask
```

**不要用文字描述替代流程图**。一张图胜过十段话。

#### 1.2 逐层/逐步详解

对流程图中的每个组件，用编号子节展开：

```markdown
### 1.1 第一层：流式读取（ParallelBPEIterableDataset）

逐行同时打开 source 和 target BPE 文件，严格对齐：

src_tokens = src_line.strip().split()   # List[str]，长度 s_i_raw
tgt_tokens = tgt_line.strip().split()   # List[str]，长度 t_i_raw
yield src_tokens, tgt_tokens

这一层**不产生 Tensor**，输出是 Python 的字符串列表。支持：
- 多 worker 按行号切片
- 流式 buffer shuffle
- set_epoch(epoch) 让每个 epoch 的 shuffle 种子不同
```

每个子节的结构：
1. **代码片段**（简短，指明调的是什么函数）
2. **具体数值示例**（如 `src_len = len(src_tokens) + 1`）
3. **关键特点**（用列表形式，每条一句话）

#### 1.3 用 ASCII 图展示内部逻辑

对于有循环/条件逻辑的组件，用 ASCII 图画出来：

```markdown
pool 内排序后逐条加入:
┌─────────────────────────────────────────────────────┐
│ sample 1: src=5,  tgt=7   → batch_src=5,  batch_tgt=7   │
│ sample 2: src=6,  tgt=8   → batch_src=11, batch_tgt=15  │
│ ...                                                    │
│ sample N: src=12, tgt=15  → 如果 batch_src+12 > 4096   │
│                              → yield 当前 batch，开新    │
└─────────────────────────────────────────────────────┘
```

#### 1.4 汇总表格

在流程讲解的最后，用一个表格汇总所有字段/输出：

```markdown
| 字段 | shape | dtype | 用途 |
|------|-------|-------|------|
| src | (B, S) | int64 | Encoder 输入 |
| tgt_input | (B, T) | int64 | Decoder 输入 |
| tgt_y | (B, T) | int64 | 监督标签 |
| src_mask | (B, 1, S) | bool | Encoder mask |
| tgt_mask | (B, T, T) | bool | Decoder mask |
| ntokens | 标量 | int | loss 归一化 |
```

---

### 2. 与相关概念/任务的对比

把当前机制放到更大的背景中，和读者可能熟悉的概念做对比。

#### 2.1 选择合适的对比对象

对比对象应该是读者**大概率熟悉**的、和当前机制**有相似性但又有区别**的东西。比如：

- Batch 构建 → 对比 CV 任务的 batch 构建
- 训练循环 → 对比 PyTorch 标准训练循环
- 解码策略 → 对比贪心搜索 vs 束搜索
- 数据加载 → 对比 MapDataset vs IterableDataset

#### 2.2 用对比表格

对比的核心工具是**表格**，而不是大段文字：

```markdown
| 维度 | CV（如 ImageNet 分类） | NLP（本项目 Transformer） |
|------|----------------------|--------------------------|
| 单样本 | (3, 224, 224) 固定 | (seq_len,) 变长 |
| 对齐方式 | Resize/Crop | 右侧 Pad |
| Mask | 不需要 | src_mask + tgt_mask |
| Batch 大小 | 固定 N 张 | 动态（按 token 预算） |
```

#### 2.3 对每个差异点展开说明

表格之后，对重要的差异点单独展开：

```markdown
### 2.4 Batch 大小策略：固定 vs 动态

CV 通常用**固定 batch_size**（如 32、64、128），因为每张图的 shape 一致，显存消耗可预测。

本项目训练集用 **token-budget 动态 batch**：

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
└─────────────────────────────────────────────────┘

动态 batch 的好处：
- **显存更稳定**：不管句子长短，每步的总 token 数大致相同
- **padding 更少**：排序后长短搭配，减少了 padding 浪费
- **训练更高效**：同样一个 epoch，有效 token 占比更高
```

**关键**：每个差异点都要解释"为什么这样做更好"或"为什么这样做是必要的"。

---

### 3. 与原论文的对比

如果当前机制是对某篇论文的实现，必须和原论文做对比。

#### 3.1 引用论文原文

直接引用论文中的相关描述，让读者知道"论文说了什么"：

```markdown
### 3.1 原论文的 Batch 描述

原论文（Attention Is All You Need, 2017）Section 5.1 描述：

> "Each training batch contained a set of sentence pairs containing
> approximately 25000 source tokens and 25000 target tokens."

论文用的是 **token batching**：按 token 数量而非句子数量来控制 batch 大小。
```

#### 3.2 逐点对比表格

用表格逐点对比论文描述和当前实现：

```markdown
### 3.2 当前实现 vs 论文

| 论文描述 | 当前实现 | 是否一致 |
|---------|---------|---------|
| 按 token 数量分 batch（~25000 src + ~25000 tgt） | ApproxTokenBucketBatchDataset 按 token 预算分 batch | **思路一致，数值不同** |
| 按长度排序减少 padding | pool 内按 max(src, tgt) 排序 | **一致** |
| BPE 分词（32000 合并规则） | BPE 分词，vocab_size 取自实际词表 | **一致** |
| Shared source-target vocabulary | share_embeddings=True | **一致** |
| Adam (β1=0.9, β2=0.98, ε=1e-9) | 完全一致 | **一致** |
```

对比结果分三种：
- **一致**：完全按照论文实现
- **思路一致，数值不同**：设计思路相同，但具体参数有差异
- **不同**：实现方式有本质区别

#### 3.3 对关键差异展开说明

对"思路一致，数值不同"和"不同"的项，单独展开说明**为什么有差异**：

```markdown
### 3.3 关键差异详解

#### 差异 1：Token Budget 数量级

论文用 ~25000 src + ~25000 tgt（约 50000 token/batch）。
这是在 8 块 P100 GPU 上、使用较大模型的配置。

当前项目默认 4096 src + 4096 tgt（约 8192 token/batch）。
这是面向单卡/小规模训练的合理缩减。

论文:     ~25000 token/batch × 8 GPU = ~200000 token/step
当前:     ~4096  token/batch × 1 GPU = ~4096  token/step

token budget 影响的是**梯度估计的方差**：budget 越大，梯度越稳定，
但单 step 耗时也越长。
```

每个差异都要解释：
1. **论文怎么做**（引用原文或公认做法）
2. **当前怎么做**（具体参数/代码）
3. **为什么不同**（硬件限制、数据集规模、工程折中等）
4. **影响是什么**（对模型效果/训练效率的影响）

#### 3.4 总结

用一段话总结"和论文的一致程度"，区分哪些是"核心一致"和"工程折中"：

```markdown
### 3.4 总结

当前项目的 batch 构建机制**在设计思路上与论文高度一致**：

- 按 token 数量而非句子数量控制 batch 大小（token batching）
- 按长度排序减少 padding 浪费
- Shared vocabulary + weight tying
- BPE 分词 + 特殊 token 处理

**数值和规模上有差异**，但这些差异是由训练硬件条件决定的工程折中，
不影响模型架构和训练逻辑的正确性。
```

---

### 4. 设计选择的理由（穿插在各节中）

每个设计选择都要解释"为什么这样做"，而不仅仅是"这样做"。常见的理由类型：

| 理由类型 | 示例 |
|---------|------|
| 硬件限制 | "token budget 从 25000 缩减到 4096 是单卡适配" |
| 数据集规模 | "warmup steps 从 4000 缩减到 2000 是针对小数据集的调参" |
| 训练效率 | "排序后长短搭配，减少 padding 浪费" |
| 数值稳定性 | "用 token 数而非 batch_size 归一化 loss，避免长句主导梯度" |
| 工程简洁性 | "用 IterableDataset 流式读取，避免加载几十 GB 语料到内存" |

---

## 写作禁忌

1. **不要只说"当前项目用了 X"**。必须说清楚 X 是什么、为什么用、和别的选择有什么区别。

2. **不要跳过对比**。如果这个机制在论文中有描述，必须对比。如果在其他领域有类似机制，也必须对比。

3. **不要用大段文字替代表格**。对比信息用表格呈现，文字只用来展开解释表格中的关键点。

4. **不要回避差异**。如果当前实现和论文不一致，必须明确指出，并解释为什么。不要让读者自己去发现差异。

5. **不要只给结论不给过程**。比如"token budget 从 25000 缩减到 4096"是结论，过程是"单卡显存有限，25000 会 OOM"。

6. **不要遗漏"为什么这样做更好"**。每个设计选择都要解释好处，而不仅仅是描述做法。

7. **不要用模糊的表述**。"大概差不多"、"基本一致"这种表述没有信息量。要说清楚"哪些一致、哪些不同、为什么不同"。

---

## 调试友好原则

1. **用可验证的数值**。token budget=4096、pool_size=2048 这些参数要写出来，读者可以对照检查自己的配置。

2. **展示边界情况**。比如"如果句子长度超过 max_src_len 会怎样"、"如果 batch 内只有一条样本会怎样"。

3. **给出排查方向**。比如：

   ```markdown
   如果你发现每个 batch 的句子数都是 1，说明 token budget 设置太大，
   或者句子太长。检查 max_src_len 和 src_token_budget 的关系。
   ```

---

## 与 skills/ 的关系

- `summary_dataflow.md`：关注**数据 shape 的全流程变化**，适合理解 tensor 怎么流动。
- `explain_code_detail.md`：关注**单个函数/变量的深度解析**，适合理解某个具体实现。
- `summarize_mechanism.md`（本文件）：关注**机制的设计选择和对比**，适合理解"为什么这样做"。

三者互补：dataflow 给你 shape 全景，detail 给你局部放大镜，mechanism 给你设计决策的背景。
