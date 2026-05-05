# Target Mask 详细说明

本文只讨论当前代码里的 target mask。核心实现位于 `data/batch.py`：

```python
def subsequent_mask(size: int, device: torch.device) -> torch.Tensor:
    return torch.tril(
        torch.ones((1, size, size), dtype=torch.bool, device=device)
    )

def make_tgt_mask(tgt_input: torch.Tensor, pad_idx: int) -> torch.Tensor:
    pad_mask = (tgt_input != pad_idx).unsqueeze(1)  # (B, 1, T)
    causal_mask = subsequent_mask(tgt_input.size(1), tgt_input.device)  # (1, T, T)

    tgt_mask = pad_mask & causal_mask
    return tgt_mask
```

## 1. target mask 要解决什么问题

Decoder self-attention 有两个约束：

1. 不能看未来 token：位置 `i` 只能看 `0..i`，不能看 `i+1..T-1`。
2. 不能关注 PAD token：padding 出来的列没有真实语义，不能作为 key/value 被其他位置关注。

当前代码把这两个约束拆成两个 mask：

- `causal_mask`：负责“不能看未来”。
- `pad_mask`：负责“不能关注 PAD 列”。
- `tgt_mask = pad_mask & causal_mask`：两个条件都满足的位置才允许 attention。

在本项目中：

- mask 的 dtype 是 `torch.bool`。
- `True` 表示允许关注。
- `False` 表示禁止关注。

底层 attention 里会把禁止位置的 score 填成当前 dtype 的最小值：

```python
scores = scores.masked_fill(mask == 0, min_value)
```

## 2. shape 约定

假设：

- `tgt_input.shape = (B, T)`
- `B` 是 batch size。
- `T` 是 Decoder 输入长度。
- `pad_idx` 是 PAD id，当前项目默认是 `0`。

那么：

| 张量 | shape | 说明 |
|---|---|---|
| `tgt_input` | `(B, T)` | Decoder 输入 token id |
| `pad_mask` | `(B, 1, T)` | 标记每个 key 位置是不是非 PAD |
| `causal_mask` | `(1, T, T)` | 下三角因果 mask |
| `tgt_mask` | `(B, T, T)` | 最终 Decoder self-attention mask |

注意 `tgt_mask` 的两个 `T` 含义不同：

- 第二维是 query 位置，也就是“当前正在更新哪个位置”。
- 第三维是 key 位置，也就是“当前 query 可以看哪些历史位置”。

所以 `tgt_mask[b, i, j]` 表示：

```text
第 b 条样本中，query 位置 i 是否允许关注 key 位置 j
```

## 3. causal_mask：只允许看当前和过去

`subsequent_mask(T)` 用 `torch.tril(...)` 生成下三角矩阵。

当 `T = 6` 时：

```text
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

读法：

- 第 `0` 行表示位置 `0` 只能看位置 `0`。
- 第 `1` 行表示位置 `1` 可以看位置 `0, 1`。
- 第 `5` 行表示位置 `5` 可以看位置 `0..5`。

它只处理“未来不可见”，不关心哪些位置是 PAD。

## 4. pad_mask：只允许看非 PAD 列

当前代码：

```python
pad_mask = (tgt_input != pad_idx).unsqueeze(1)
```

如果 `B = 1`，`T = 6`，并且 `tgt_input` 最后两个位置是 PAD：

```text
tgt_input = [
  [<bos>, y1, y2, <eos>, <pad>, <pad>]
]
```

用 `1` 表示非 PAD，`0` 表示 PAD：

```text
tgt_input != pad_idx

[
  [1, 1, 1, 1, 0, 0]
]
```

`unsqueeze(1)` 后：

```text
pad_mask.shape = (1, 1, 6)

[
  [
    [1, 1, 1, 1, 0, 0]
  ]
]
```

这里的 `pad_mask` 是 key mask。它的作用是屏蔽 key/value 维度上的 PAD 列。

也就是说，任何 query 位置都不应该去关注第 `4`、`5` 列，因为这两列是 PAD。

## 5. tgt_mask：pad_mask 与 causal_mask 的结果

当前代码：

```python
tgt_mask = pad_mask & causal_mask
```

在 `B = 1`、`T = 6`、最后两个 token 是 PAD 的例子中：

```text
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
  [1, 1, 1, 0, 0, 0],
  [1, 1, 1, 1, 0, 0],
  [1, 1, 1, 1, 1, 0],
  [1, 1, 1, 1, 1, 1],
]
```

广播后按位与：

```text
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

逐行解释：

| query 位置 | token | 允许关注的 key 位置 | 行内容 |
|---:|---|---|---|
| `0` | `<bos>` | `0` | `[1, 0, 0, 0, 0, 0]` |
| `1` | `y1` | `0, 1` | `[1, 1, 0, 0, 0, 0]` |
| `2` | `y2` | `0, 1, 2` | `[1, 1, 1, 0, 0, 0]` |
| `3` | `<eos>` | `0, 1, 2, 3` | `[1, 1, 1, 1, 0, 0]` |
| `4` | `<pad>` | `0, 1, 2, 3` | `[1, 1, 1, 1, 0, 0]` |
| `5` | `<pad>` | `0, 1, 2, 3` | `[1, 1, 1, 1, 0, 0]` |

这里有一个容易误解的点：当前实现不会把 PAD query 行整行置为 0。

第 `4`、`5` 行对应的 query token 自己是 PAD，但它们仍然可以关注前面的真实 token。这是因为当前 `pad_mask` 的 shape 是 `(B, 1, T)`，只屏蔽 key 维度上的 PAD 列，不屏蔽 query 行。

这个设计在训练里通常没有问题，因为 loss 阶段会过滤 PAD 标签：

```python
valid_mask = target_flat.ne(self.pad_idx)
logits_valid = logits_flat[valid_mask]
target_valid = target_flat[valid_mask]
```

也就是说，PAD query 位置虽然会产生 hidden state 和 logits，但对应位置不会参与 loss。

## 6. mask 进入 MultiHeadedAttention 后的 shape

`MultiHeadedAttention.forward()` 里还有一步：

```python
if mask is not None:
    mask = mask.unsqueeze(1)
```

所以对于 target self-attention：

| 阶段 | shape |
|---|---|
| `tgt_mask` | `(B, T, T)` |
| MHA 内部 `mask.unsqueeze(1)` | `(B, 1, T, T)` |
| attention scores | `(B, h, T, T)` |
| mask 广播后 | `(B, h, T, T)` |

`h` 是 attention head 数。当前训练默认 `h = 8`。

所有 head 共用同一份 target mask。

## 7. 和 target shift 的关系

`Seq2SeqBatch.from_tensors()` 先做 target shift，再基于 `tgt_input` 构造 mask：

```python
tgt_input, tgt_y = shift_right(tgt)
tgt_mask = make_tgt_mask(tgt_input, pad_idx)
```

如果完整 target 是：

```text
tgt = [<bos>, y1, y2, <eos>, <pad>, <pad>, <pad>]
```

那么：

```text
tgt_input = [<bos>, y1, y2, <eos>, <pad>, <pad>]
tgt_y     = [y1,    y2, <eos>, <pad>, <pad>, <pad>]
```

`make_tgt_mask()` 看到的是 `tgt_input`，所以本例中 `T = 6`，最后两个是 PAD。

loss 看到的是 `tgt_y`，其中后三个是 PAD，所以这三个位置不会参与 loss。

## 8. 多样本 batch 例子

假设 `B = 2`，`T = 6`：

```text
tgt_input =
[
  [<bos>, a1, a2, <eos>, <pad>, <pad>],
  [<bos>, b1, b2, b3,    b4,    <eos>],
]
```

则：

```text
pad_mask.shape = (2, 1, 6)

pad_mask[0] =
[
  [1, 1, 1, 1, 0, 0]
]

pad_mask[1] =
[
  [1, 1, 1, 1, 1, 1]
]
```

`causal_mask` 对整个 batch 共享：

```text
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

最终：

```text
tgt_mask.shape = (2, 6, 6)

tgt_mask[0] =
[
  [1, 0, 0, 0, 0, 0],
  [1, 1, 0, 0, 0, 0],
  [1, 1, 1, 0, 0, 0],
  [1, 1, 1, 1, 0, 0],
  [1, 1, 1, 1, 0, 0],
  [1, 1, 1, 1, 0, 0],
]

tgt_mask[1] =
[
  [1, 0, 0, 0, 0, 0],
  [1, 1, 0, 0, 0, 0],
  [1, 1, 1, 0, 0, 0],
  [1, 1, 1, 1, 0, 0],
  [1, 1, 1, 1, 1, 0],
  [1, 1, 1, 1, 1, 1],
]
```

第 0 条样本的最后两列被 PAD mask 屏蔽；第 1 条样本没有 PAD，因此最终 mask 等于 causal mask。

## 9. 如果想屏蔽 PAD query 行

当前代码没有这么做。如果想让 PAD query 行也全为 0，可以构造 query mask：

```python
key_pad_mask = (tgt_input != pad_idx).unsqueeze(1)  # (B, 1, T)
query_pad_mask = (tgt_input != pad_idx).unsqueeze(2)  # (B, T, 1)
causal_mask = subsequent_mask(tgt_input.size(1), tgt_input.device)  # (1, T, T)
tgt_mask = key_pad_mask & query_pad_mask & causal_mask
```

仍然用 `T = 6`、最后两个是 PAD 的例子：

```text
query_pad_mask =
[
  [1],
  [1],
  [1],
  [1],
  [0],
  [0],
]
```

最终 mask 会变成：

```text
[
  [1, 0, 0, 0, 0, 0],
  [1, 1, 0, 0, 0, 0],
  [1, 1, 1, 0, 0, 0],
  [1, 1, 1, 1, 0, 0],
  [0, 0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0, 0],
]
```

但这不是当前项目的实现。当前项目只屏蔽 PAD key 列，PAD 位置的 loss 由 `LabelSmoothingLoss` 过滤。

## 10. 一句话总结

当前 `tgt_mask` 的逻辑是：

```text
tgt_mask[b, i, j] = (j <= i) and (tgt_input[b, j] != pad_idx)
```

也就是：

- `j <= i` 来自 `causal_mask`，保证不能看未来。
- `tgt_input[b, j] != pad_idx` 来自 `pad_mask`，保证不能关注 PAD key。
- 没有包含 `tgt_input[b, i] != pad_idx`，所以不会屏蔽 PAD query 行。

