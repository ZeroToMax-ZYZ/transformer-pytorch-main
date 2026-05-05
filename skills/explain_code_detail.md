# Skill: 代码细节讲解（explain_code_detail）

当用户要求讲解某个函数、变量、模块的实现细节时，按以下规范输出。

---

## 核心原则

**先搞清楚"为什么存在"，再讨论"怎么实现"。**

目标读者是"正在读代码、想搞清楚某个部分到底在干什么"的人。文档应该让人**理解设计意图**，而不仅仅是知道 API 签名。

---

## 结构模板

### 1. 开头：这个东西要解决什么问题

用一两句话说清楚这个函数/变量存在的原因。不要上来就贴代码。

```markdown
## 1. target mask 要解决什么问题

Decoder self-attention 有两个约束：

1. 不能看未来 token：位置 `i` 只能看 `0..i`，不能看 `i+1..T-1`。
2. 不能关注 PAD token：padding 出来的列没有真实语义，不能作为 key/value 被其他位置关注。
```

**不要写成**："这个函数是用来生成 mask 的。" 这种废话。

### 2. 代码片段（简短）

只贴最关键的代码，让读者知道"实现长什么样"。不需要逐行注释。

```python
def make_tgt_mask(tgt_input: torch.Tensor, pad_idx: int) -> torch.Tensor:
    pad_mask = (tgt_input != pad_idx).unsqueeze(1)  # (B, 1, T)
    causal_mask = subsequent_mask(tgt_input.size(1), tgt_input.device)  # (1, T, T)
    tgt_mask = pad_mask & causal_mask
    return tgt_mask
```

### 3. shape 约定表

列出所有相关张量的 shape，用表格形式。明确每个维度的物理含义。

```markdown
| 张量 | shape | 说明 |
|---|---|---|
| `tgt_input` | `(B, T)` | Decoder 输入 token id |
| `pad_mask` | `(B, 1, T)` | 标记每个 key 位置是不是非 PAD |
| `causal_mask` | `(1, T, T)` | 下三角因果 mask |
| `tgt_mask` | `(B, T, T)` | 最终 Decoder self-attention mask |
```

**关键**：如果某个维度有特殊含义（比如 `tgt_mask` 的两个 `T` 分别代表 query 和 key），必须明确说明。

### 4. 分组件详解（每个组件一个子节）

对函数内部的每个逻辑步骤，单独一节讨论：

#### 4.1 先说"这个组件负责什么"

```markdown
## 3. causal_mask：只允许看当前和过去

`subsequent_mask(T)` 用 `torch.tril(...)` 生成下三角矩阵。
它只处理"未来不可见"，不关心哪些位置是 PAD。
```

#### 4.2 再用具体数值画出来

**必须用具体数值**，不要只写 `(1, T, T)`。

```markdown
当 `T = 6` 时：

causal_mask.shape = (1, 6, 6)

[
  [1, 0, 0, 0, 0, 0],
  [1, 1, 0, 0, 0, 0],
  [1, 1, 1, 0, 0, 0],
  [1, 1, 1, 1, 0, 0],
  [1, 1, 1, 1, 1, 0],
  [1, 1, 1, 1, 1, 1],
]
```

#### 4.3 解释读法

告诉读者怎么"读懂"这个张量：

```markdown
读法：

- 第 `0` 行表示位置 `0` 只能看位置 `0`。
- 第 `1` 行表示位置 `1` 可以看位置 `0, 1`。
- 第 `5` 行表示位置 `5` 可以看位置 `0..5`。
```

### 5. 多组件合并后的结果

当函数内部有多个 mask 做逻辑运算时，用具体数值展示合并过程：

```markdown
## 5. tgt_mask：pad_mask 与 causal_mask 的结果

pad_mask:
[
  [
    [1, 1, 1, 1, 0, 0]
  ]
]

causal_mask:
[
  [1, 0, 0, 0, 0, 0],
  [1, 1, 0, 0, 0, 0],
  ...
]

广播后按位与：

tgt_mask.shape = (1, 6, 6)
[
  [1, 0, 0, 0, 0, 0],
  [1, 1, 0, 0, 0, 0],
  [1, 1, 1, 0, 0, 0],
  [1, 1, 1, 1, 0, 0],
  [1, 1, 1, 1, 0, 0],
  [1, 1, 1, 1, 0, 0],
]
```

然后用表格逐行解释结果：

```markdown
| query 位置 | token | 允许关注的 key 位置 | 行内容 |
|---:|---|---|---|
| `0` | `<bos>` | `0` | `[1, 0, 0, 0, 0, 0]` |
| `1` | `y1` | `0, 1` | `[1, 1, 0, 0, 0, 0]` |
| `4` | `<pad>` | `0, 1, 2, 3` | `[1, 1, 1, 1, 0, 0]` |
```

### 6. 指出容易误解的地方（Gotchas）

**这是最重要的部分**。每个实现都有一些"看起来不对但实际上是对的"或者"看起来对但实际上有坑"的地方。必须主动指出。

```markdown
这里有一个容易误解的点：当前实现不会把 PAD query 行整行置为 0。

第 `4`、`5` 行对应的 query token 自己是 PAD，但它们仍然可以关注前面的真实 token。
这是因为当前 `pad_mask` 的 shape 是 `(B, 1, T)`，只屏蔽 key 维度上的 PAD 列，
不屏蔽 query 行。

这个设计在训练里通常没有问题，因为 loss 阶段会过滤 PAD 标签：
valid_mask = target_flat.ne(self.pad_idx)
```

### 7. 与上下游的关系

说明这个函数/变量在更大的数据流中处于什么位置：

```markdown
## 7. 和 target shift 的关系

`Seq2SeqBatch.from_tensors()` 先做 target shift，再基于 `tgt_input` 构造 mask：

tgt_input, tgt_y = shift_right(tgt)
tgt_mask = make_tgt_mask(tgt_input, pad_idx)
```

可以用"如果完整 target 是...那么..."的方式，用具体数值走一遍。

### 8. 多样本 batch 例子

给出一个 `B > 1` 的完整例子，展示不同样本的 mask 如何不同：

```markdown
## 8. 多样本 batch 例子

假设 B = 2, T = 6：

tgt_input =
[
  [<bos>, a1, a2, <eos>, <pad>, <pad>],
  [<bos>, b1, b2, b3,    b4,    <eos>],
]

pad_mask[0] = [[1, 1, 1, 1, 0, 0]]
pad_mask[1] = [[1, 1, 1, 1, 1, 1]]

tgt_mask[0] = ...  # 最后两列被 PAD mask 屏蔽
tgt_mask[1] = ...  # 没有 PAD，等于 causal mask
```

### 9. 如果想改成另一种实现（可选）

如果当前实现有替代方案，简单说明区别：

```markdown
## 9. 如果想屏蔽 PAD query 行

当前代码没有这么做。如果想让 PAD query 行也全为 0，可以构造 query mask：

query_pad_mask = (tgt_input != pad_idx).unsqueeze(2)  # (B, T, 1)
tgt_mask = key_pad_mask & query_pad_mask & causal_mask

但这不是当前项目的实现。当前项目只屏蔽 PAD key 列，PAD 位置的 loss 由 LabelSmoothingLoss 过滤。
```

### 10. 一句话总结

用一句话概括这个函数/变量的核心逻辑，最好用数学或伪代码表达：

```markdown
## 10. 一句话总结

tgt_mask[b, i, j] = (j <= i) and (tgt_input[b, j] != pad_idx)

也就是：
- j <= i 来自 causal_mask，保证不能看未来。
- tgt_input[b, j] != pad_idx 来自 pad_mask，保证不能关注 PAD key。
- 没有包含 tgt_input[b, i] != pad_idx，所以不会屏蔽 PAD query 行。
```

---

## 写作禁忌

1. **不要只说"这个函数生成 mask"**。必须说清楚 mask 要解决什么约束。

2. **不要只写 shape 不给具体数值**。`(B, T, T)` 是必要的，但必须同时给出"B=1, T=6, 最后两个是 PAD"的具体张量。

3. **不要跳过中间 shape**。比如 `pad_mask` 从 `(B, T)` 到 `(B, 1, T)` 的 `unsqueeze` 过程必须写出来。

4. **不要忽略广播机制**。`(B, 1, T)` 和 `(1, T, T)` 做 `&` 运算时如何广播，必须解释。

5. **不要回避"看起来不对但实际上是对的"的设计**。比如"PAD query 行没有被屏蔽"这种容易引起疑惑的点，必须主动解释为什么这样做是可以的。

6. **不要用"详见代码"替代解释**。如果某个细节值得讨论，就在这里讨论完，不要让读者自己去看代码。

7. **不要遗漏 dtype 信息**。mask 是 `bool` 还是 `float`，`True` 代表"允许"还是"禁止"，必须明确。

---

## 调试友好原则

讲解的目标之一是帮助读者**模拟 debug 过程**：

1. **用可验证的数值**。选 `B=1, T=6` 这样的小例子，读者可以手算验证。

2. **展示中间结果**。不要只给最终结果，把每一步的中间张量都画出来。

3. **标注"如果你看到的不是这样，说明哪里出问题了"**。比如：

   ```markdown
   如果你发现 tgt_mask[0] 的最后一行全是 1，说明 pad_mask 没有正确屏蔽 PAD 列。
   ```

4. **给出常见错误的排查方向**。比如：

   ```markdown
   常见错误：pad_idx 设置错误。如果 pad_idx=1 而不是 0，会导致 <bos> 位置被屏蔽。
   ```

---

## 与 skills/summary_dataflow.md 的区别

- `summary_dataflow.md`：关注**数据从输入到输出的全流程**，强调 shape 在每一步的变化。
- `explain_code_detail.md`：关注**单个函数/变量的深度解析**，强调设计意图、具体数值、易错点。

两者互补：dataflow 给你全景，detail 给你局部放大镜。
