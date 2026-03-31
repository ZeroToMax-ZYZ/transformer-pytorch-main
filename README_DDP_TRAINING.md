# DDP Training

这个项目现在支持两种训练方式：

- 单卡：直接 `python train_transformer_base.py`
- 多卡 DDP：使用 `torchrun`

当前实现特点：

- 单卡和多卡共用同一个训练入口 `train_transformer_base.py`
- 多卡时自动根据 `RANK / WORLD_SIZE / LOCAL_RANK` 初始化 DDP
- 训练集会按 DDP rank 切分，不同 GPU 不会重复训练同一批 batch
- 只有 rank 0 会写：
  - `experiments/.../config.json`
  - `experiments/.../metrics.csv`
  - TensorBoard 日志
  - checkpoint

## 1. 单卡训练

```bash
python train_transformer_base.py
```

如果你想限制 epoch 数或 token budget：

```bash
python train_transformer_base.py --num-epochs 20 --train-src-token-budget 2048 --train-tgt-token-budget 2048
```

## 2. 4 卡 3080 DDP 训练

推荐用 `torchrun`：

```bash
torchrun --standalone --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=29500 train_transformer_base.py
```

如果你的 3080 显存比较紧，先从更保守的每卡 token budget 开始：

```bash
torchrun --standalone --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=29500 train_transformer_base.py --train-src-token-budget 1536 --train-tgt-token-budget 1536
```

如果显存充足，再尝试默认的：

```bash
torchrun --standalone --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=29500 train_transformer_base.py --train-src-token-budget 2048 --train-tgt-token-budget 2048
```

## 3. 常用启动参数

- `--num-epochs`
  - 覆盖默认 epoch 数
- `--train-src-token-budget`
  - 训练集 source 端每卡 token budget
- `--train-tgt-token-budget`
  - 训练集 target 端每卡 token budget
- `--train-num-workers`
  - 每个 rank 的 DataLoader worker 数
- `--valid-num-workers`
  - 验证集 DataLoader worker 数
- `--output-dir`
  - 指定实验输出目录
- `--max-train-steps-per-epoch`
  - 限制每个 epoch 最多训练多少步，调试时有用

示例：

```bash
torchrun --standalone --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=29500 train_transformer_base.py \
  --num-epochs 30 \
  --train-src-token-budget 1536 \
  --train-tgt-token-budget 1536 \
  --train-num-workers 2 \
  --valid-num-workers 2
```

## 4. 输出位置

默认输出到：

```text
experiments/transformer_wmt14_en_de_base_时间戳/
```

其中包含：

- `config.json`
- `metrics.csv`
- `tb/`
- `checkpoints/`
- `plots/`

## 5. 4 卡 3080 的建议

建议先用这个命令起跑：

```bash
torchrun --standalone --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=29500 train_transformer_base.py --train-src-token-budget 1536 --train-tgt-token-budget 1536 --train-num-workers 2
```

如果稳定且没有 OOM，再提高到：

```bash
torchrun --standalone --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=29500 train_transformer_base.py --train-src-token-budget 2048 --train-tgt-token-budget 2048 --train-num-workers 2
```

## 6. 注意事项

- `train-src-token-budget` 和 `train-tgt-token-budget` 是“每张卡”的 budget，不是全局总 budget。
- 4 卡时，全局有效吞吐大约是单卡的 4 倍减去 DDP 通信损耗。
- 进度条只在 rank 0 显示。
- 验证、CSV、TensorBoard、checkpoint 只由 rank 0 写出。
