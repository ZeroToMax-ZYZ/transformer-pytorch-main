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


# Transformer 论文实现对齐说明
本文档对比了当前代码仓库与2017年Transformer经典论文《Attention Is All You Need》的实现差异。

本文并非笼统判定“一致”或“不同”，而是精准说明：代码实现**严格遵循论文**的细节、以及**基于工程考量刻意偏离论文**的设计。

## 一、与论文基本完全对齐的实现
### 1.1 模型规模与模块结构
代码路径：
- `train_transformer_base.py`
- `nets/build_transformer.py`
- `nets/utils/encoder_decoder.py`

当前默认配置：
- 编码器/解码器层数 $N = 6$
- 模型维度 $d_{model} = 512$
- 前馈网络维度 $d_{ff} = 2048$
- 注意力头数 $h = 8$
- 丢弃率 $dropout = 0.1$

该配置与论文中**基础版Transformer**参数完全一致。

### 1.2 缩放点积注意力
代码路径：
- `nets/utils/attention.py`

实现公式严格遵循论文：
```text
softmax(Q K^T / sqrt(d_k)) V
```

### 1.3 多头注意力结构
代码路径：
- `nets/utils/attention.py`

实现流程：
1. 对查询Q、键K、值V进行线性投影
2. 拆分多个注意力头
3. 每个头独立计算注意力权重
4. 拼接所有注意力头输出，再做最终线性投影

结构完全匹配论文定义。

### 1.4 逐位置前馈神经网络
代码路径：
- `nets/utils/PositionwiseFeedForward.py`

前馈网络结构严格对齐论文：
```text
线性层(模型维度→前馈维度) → ReLU激活 → 丢弃层 → 线性层(前馈维度→模型维度)
```

### 1.5 正弦位置编码
代码路径：
- `nets/utils/PositionalEncoding.py`

项目采用**固定正弦/余弦位置编码**，而非可学习位置嵌入，与论文默认方案一致。

### 1.6 嵌入向量缩放（乘以根号模型维度）
代码路径：
- `nets/utils/encoder_decoder.py`

嵌入向量输出添加位置编码前，会乘以 `sqrt(d_model)`，严格遵循论文细节。

### 1.7 共享词表与权重绑定
代码路径：
- `data/shared_vocab.py`
- `nets/build_transformer.py`

项目实现：
1. 源语言/目标语言共用一套词表
2. 源端嵌入层、目标端嵌入层、输出分类投影层权重绑定

与论文共享BPE词表、权重复用的设计一致。

### 1.8 训练损失与优化器配置
代码路径：
- `utils/label_smoothing.py`
- `utils/noam_scheduler.py`
- `train_transformer_base.py`

默认超参完全对齐论文：
- 标签平滑系数 `0.1`
- Adam优化器：$\beta_1=0.9$、$\beta_2=0.98$、极小值$\epsilon=1e-9$
- Noam学习率预热步数 `4000`

### 1.9 解码器掩码与自回归解码
代码路径：
- `data/batch.py`
- `train_utils/validate_one_epoch.py`
- `evaluate_transformer_bleu.py`

训练时解码器使用**因果掩码遮挡未来位置**，推理时采用自回归逐词解码，与论文机制一致。

## 二、刻意偏离论文的工程化设计
### 2.1 前置层归一化（Pre-LN）而非后置层归一化（Post-LN）
代码路径：
- `nets/utils/encoder_decoder.py`

当前实现（Pre-LN）：
```text
x + 子层计算(归一化(x))
```

论文原始公式（Post-LN）：
```text
归一化(x + 子层计算(x))
```

这是模型核心架构最大的差异点。

实际影响：
1. 整体Transformer模块层级结构不变
2. 归一化前置改变梯度传播特性，优化收敛行为与原版不同

### 2.2 训练循环基于轮次（Epoch），而非纯步数驱动
代码路径：
- `train_transformer_base.py`
- `train_utils/fit.py`

论文以更新步数统计训练进度，本仓库封装为轮次训练，并支持限制单轮最大步数：
- `max_train_steps_per_epoch`（单轮最大训练步数）
- `max_valid_steps_per_epoch`（单轮最大验证步数）

属于工程控制层优化，非模型算法层面改动。

### 2.3 默认训练上限适配调试，非论文标准训练配置
代码路径：
- `train_transformer_base.py`

默认调试友好配置：
- 训练总轮次 `num_epochs = 100`
- 单轮最大训练步数 `max_train_steps_per_epoch = 1000`
- 禁用验证文本采样 `valid_num_text_samples = 0`
- 禁用直方图日志 `histogram_interval = 0`

便于迭代调试，未复刻论文大规模长时训练方案。

### 2.4 令牌配额分批次仅近似复刻论文策略
代码路径：
- `data/wmt_14_bpe_dataset.py`
- `train_transformer_base.py`

对齐点：
1. 核心逻辑一致：按句子长度分组、以令牌总数限制批次大小

差异点：
1. 论文面向多卡分布式训练，批次按源/目标令牌数精准划分
2. 本仓库单机实现：迭代数据集+本地排序+令牌配额打包
3. 默认配额4096/4096，远小于论文超大批次规模

思路对齐，但非原版严格复刻。

### 2.5 数据预处理未完整复刻论文原生流程
代码路径：
- `script/dataset_part/01_download_dataset.py`
- `data/wmt14_raw_en_de/`
- `data/wmt14_tok_en_de/`
- `data/wmt14_clean_en_de/`
- `data/wmt14_bpe_en_de/`

项目遵循「原始文本→分词→清洗→BPE编码」标准流水线，思路贴合论文。

但存在三处不完整：
1. 原始语料取自Hugging Face `wmt14/de-en`，未复刻论文原生数据获取链路
2. 仓库未内置分词、文本清洗、BPE词表生成脚本
3. 本仓库最终共享词表大小40236，论文原版约37000

预处理仅参考论文思路，无法通过本仓库代码完整复现原版精准流程。

### 2.6 训练验证侧重损失/困惑度，非论文核心BLEU指标
代码路径：
- `train_utils/validate_one_epoch.py`
- `train_utils/fit.py`

训练过程监控指标：
- 验证损失、负对数似然损失、平滑损失
- 令牌准确率、困惑度

论文核心评估指标为翻译BLEU分数。本仓库虽支持BLEU计算，但不作为训练早停/模型优选的核心指标。

### 2.7 贪心采样日志仅用于调试，非论文评估方案
代码路径：
- `train_utils/validate_one_epoch.py`

验证阶段输出少量贪心解码文本样例，仅快速肉眼排查错误，不属于论文正式评测流程。

### 2.8 检查点平均与束搜索独立于训练主循环
代码路径：
- `evaluate_transformer_bleu.py`

论文最终翻译效果依赖推理优化：束搜索解码、多检查点权重平均。
本仓库支持以上功能，但仅集成在独立BLEU评测脚本中，未嵌入训练主流程。

### 2.9 混合精度训练：新增工程优化特性
代码路径：
- `train_transformer_base.py`
- `train_utils/train_one_epoch.py`

引入AMP自动混合精度、梯度缩放器，属于现代深度学习工程优化，2017原版论文未包含该设计。

### 2.10 训练集子集比例：本地调试专属参数
代码路径：
- `train_transformer_base.py`
- `resume_train_transformer.py`
- `data/wmt_14_bpe_dataset.py`

新增`train_subset_ratio`参数，快速缩小训练数据量、缩短调试迭代周期，完全独立于论文原生配置。

## 三、中性无关实现选择
以下改动不构成论文实质偏离，仅代码工程习惯差异：

### 3.1 输出分类层独立于模型前向传播
`EncoderDecoder.forward(...)`仅输出隐状态，词汇分类投影需单独调用`model.generator(...)`。
纯代码解耦设计，无算法模型改动。

### 3.2 注意力掩码采用数据类型极小值填充
代码路径：
- `nets/utils/attention.py`

掩码遮挡位置填充`torch.finfo(scores.dtype).min`（自适应极小值），而非固定常数。
适配混合精度训练数值稳定性，无建模逻辑改动。

### 3.3 Xavier初始化：通用实用默认配置
代码路径：
- `nets/build_transformer.py`

所有权重矩阵采用Xavier/Glort均匀初始化，行业通用标准实现，不改变模型本质。

## 四、核心总结
一句话概括本仓库：
**本代码仓库是扎实的基础版Transformer实现，架构、损失函数、优化器严格贴合原版论文；仅在归一化位置、工程化训练循环、预处理/评测流程完整性上刻意差异化设计。**

若需严格复刻原版论文实验效果，优先补齐4个核心缺口：
1. 前置层归一化（Pre-LN）改回论文后置层归一化（Post-LN）
2. 复刻原版步数调度与超大令牌配额分批次策略
3. 仓库内置完整分词/清洗/BPE词表生成可复现脚本
4. 将BLEU、束搜索、检查点平均嵌入主实验流程，而非独立评测脚本

