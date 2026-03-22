from __future__ import annotations

"""
功能：
1. 逐行读取 BPE 后的平行语料。
2. 使用 SharedVocab 将 token 编码为 id。
3. 在 collate_fn 中完成：
   - source EOS 追加（可选）
   - target BOS/EOS 追加
   - padding
   - target shifted right
   - src_mask / tgt_mask 构造
4. 返回 Seq2SeqBatch，供 Transformer 直接使用。

本版本面向正式训练准备，支持：
1. IterableDataset + 多 worker 分片
2. 流式 buffer shuffle
3. set_epoch(epoch)
4. DataLoader 的 persistent_workers / prefetch_factor

和论文的关系：
1. `ApproxTokenBucketBatchDataset` 保留了“按长度近似分组、按 token 数控制 batch”的思路。
2. 当前实现是单机 DataLoader 版本，不是论文中的多 GPU token batching 复刻。
"""

import random
from itertools import zip_longest
from typing import Callable, Iterator, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from data.batch import Seq2SeqBatch, pad_sequences
from data.shared_vocab import SharedVocab


def resolve_num_samples_for_ratio(total_num_samples: int, subset_ratio: float) -> int:
    """
    根据总样本数和训练子集比例，计算本轮实际使用的样本数。

    约定：
    1. `subset_ratio=1.0` 表示使用全部训练集。
    2. 当比例小于 1 时，返回向下取整后的样本数，但至少保留 1 条。
    """
    if total_num_samples <= 0:
        raise ValueError("total_num_samples must be positive.")
    if not (0.0 < subset_ratio <= 1.0):
        raise ValueError("subset_ratio must satisfy 0 < subset_ratio <= 1.")

    if subset_ratio == 1.0:
        return total_num_samples
    return max(1, int(total_num_samples * subset_ratio))


class ParallelBPEIterableDataset(IterableDataset):
    """
    逐行流式读取平行 BPE 文本。

    每个样本输出：
        (src_tokens, tgt_tokens)

    参数：
        src_path: BPE 源语言文件路径
        tgt_path: BPE 目标语言文件路径
        skip_empty: 是否跳过空样本
        shuffle_buffer_size: 流式 shuffle 缓冲区大小
        seed: 随机种子基值
        num_samples: 可选，总样本数，用于 __len__
        sample_limit: 可选，只读取前多少条平行样本
    """

    def __init__(
        self,
        src_path: str,
        tgt_path: str,
        skip_empty: bool = False,
        shuffle_buffer_size: int = 0,
        seed: int = 42,
        num_samples: Optional[int] = None,
        sample_limit: Optional[int] = None,
    ) -> None:
        super().__init__()
        if sample_limit is not None and sample_limit <= 0:
            raise ValueError("sample_limit must be positive when provided.")

        self.src_path = src_path
        self.tgt_path = tgt_path
        self.skip_empty = skip_empty
        self.shuffle_buffer_size = shuffle_buffer_size
        self.seed = seed
        self.epoch = 0
        self.num_samples = num_samples
        self.sample_limit = sample_limit

    def set_epoch(self, epoch: int) -> None:
        """
        设置当前 epoch 编号。

        作用：
            配合 buffer shuffle，让每个 epoch 的样本顺序不同。
        """
        self.epoch = epoch

    def __len__(self) -> int:
        if self.num_samples is None:
            raise TypeError("当前 IterableDataset 未提供 num_samples，无法可靠返回长度。")
        return self.num_samples

    def _line_iterator(self) -> Iterator[Tuple[List[str], List[str]]]:
        """
        最底层逐行读取器，不做 shuffle，只负责：
        1. 打开双语文件
        2. 保证逐行严格对齐
        3. 在多 worker 下按行号切片
        """
        worker_info = get_worker_info()
        worker_id = 0 if worker_info is None else worker_info.id
        num_workers = 1 if worker_info is None else worker_info.num_workers

        with open(self.src_path, "r", encoding="utf-8") as f_src, \
             open(self.tgt_path, "r", encoding="utf-8") as f_tgt:

            for line_idx, pair in enumerate(zip_longest(f_src, f_tgt, fillvalue=None)):
                if self.sample_limit is not None and line_idx >= self.sample_limit:
                    break

                src_line, tgt_line = pair

                if src_line is None or tgt_line is None:
                    raise RuntimeError("检测到源文件和目标文件在迭代时长度不一致。")

                # 多 worker 下按行切分
                if (line_idx % num_workers) != worker_id:
                    continue

                src_tokens = src_line.strip().split()
                tgt_tokens = tgt_line.strip().split()

                if self.skip_empty and (len(src_tokens) == 0 or len(tgt_tokens) == 0):
                    continue

                yield src_tokens, tgt_tokens

    def _buffer_shuffle_iterator(self) -> Iterator[Tuple[List[str], List[str]]]:
        """
        基于固定大小缓冲区的流式 shuffle。

        注意：
            这不是严格全局打乱，而是近似 shuffle。
            对大规模流式训练是一个非常实用的折中方案。
        """
        worker_info = get_worker_info()
        worker_id = 0 if worker_info is None else worker_info.id

        rng = random.Random(self.seed + self.epoch * 100003 + worker_id)

        buffer: List[Tuple[List[str], List[str]]] = []

        for sample in self._line_iterator():
            if len(buffer) < self.shuffle_buffer_size:
                buffer.append(sample)
            else:
                idx = rng.randrange(len(buffer))
                yield buffer[idx]
                buffer[idx] = sample

        rng.shuffle(buffer)
        for sample in buffer:
            yield sample

    def __iter__(self) -> Iterator[Tuple[List[str], List[str]]]:
        if self.shuffle_buffer_size is None or self.shuffle_buffer_size <= 1:
            yield from self._line_iterator()
        else:
            yield from self._buffer_shuffle_iterator()


class ApproxTokenBucketBatchDataset(IterableDataset):
    """
    近似 token-budget batching。

    设计意图：
    1. 在一个有限池内按长度排序，减少 padding 浪费。
    2. 同时约束源端和目标端 token 数，避免 batch 波动过大。
    3. 保留流式读取能力，不要求把整个数据集一次性载入内存。
    """

    def __init__(
        self,
        dataset: IterableDataset,
        src_token_budget: int,
        tgt_token_budget: int,
        add_src_eos: bool = True,
        max_sentences_per_batch: Optional[int] = None,
        pool_size: int = 2048,
    ) -> None:
        super().__init__()
        if src_token_budget <= 0 or tgt_token_budget <= 0:
            raise ValueError("src_token_budget and tgt_token_budget must be positive.")
        if pool_size <= 0:
            raise ValueError("pool_size must be positive.")

        self.dataset = dataset
        self.src_token_budget = src_token_budget
        self.tgt_token_budget = tgt_token_budget
        self.add_src_eos = add_src_eos
        self.max_sentences_per_batch = max_sentences_per_batch
        self.pool_size = pool_size

    def set_epoch(self, epoch: int) -> None:
        if hasattr(self.dataset, "set_epoch"):
            self.dataset.set_epoch(epoch)

    def _sample_cost(self, sample: Tuple[List[str], List[str]]) -> Tuple[int, int]:
        src_tokens, tgt_tokens = sample
        src_len = len(src_tokens) + (1 if self.add_src_eos else 0)
        tgt_len = len(tgt_tokens) + 2  # bos + eos
        return src_len, tgt_len

    def _yield_batched_pool(
        self,
        samples: List[Tuple[List[str], List[str]]],
    ) -> Iterator[List[Tuple[List[str], List[str]]]]:
        samples.sort(key=lambda x: max(len(x[0]), len(x[1])))

        batch: List[Tuple[List[str], List[str]]] = []
        batch_src_tokens = 0
        batch_tgt_tokens = 0

        for sample in samples:
            src_len, tgt_len = self._sample_cost(sample)

            budget_exceeded = (
                len(batch) > 0
                and (
                    batch_src_tokens + src_len > self.src_token_budget
                    or batch_tgt_tokens + tgt_len > self.tgt_token_budget
                )
            )
            sentence_limit_exceeded = (
                self.max_sentences_per_batch is not None
                and len(batch) >= self.max_sentences_per_batch
            )

            if budget_exceeded or sentence_limit_exceeded:
                yield batch
                batch = []
                batch_src_tokens = 0
                batch_tgt_tokens = 0

            batch.append(sample)
            batch_src_tokens += src_len
            batch_tgt_tokens += tgt_len

        if batch:
            yield batch

    def __iter__(self) -> Iterator[List[Tuple[List[str], List[str]]]]:
        pool: List[Tuple[List[str], List[str]]] = []
        for sample in self.dataset:
            pool.append(sample)
            if len(pool) >= self.pool_size:
                yield from self._yield_batched_pool(pool)
                pool = []

        if pool:
            yield from self._yield_batched_pool(pool)


class BPEBatchCollator:
    """
    将一批 BPE token 序列整理成模型可直接消费的 `Seq2SeqBatch`。
    """

    def __init__(
        self,
        vocab: SharedVocab,
        max_src_len: Optional[int] = None,
        max_tgt_len: Optional[int] = None,
        add_src_eos: bool = True,
    ) -> None:
        self.vocab = vocab
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len
        self.add_src_eos = add_src_eos

    def __call__(self, batch: Sequence[Tuple[List[str], List[str]]]) -> Seq2SeqBatch:
        src_id_list: List[List[int]] = []
        tgt_id_list: List[List[int]] = []

        for src_tokens, tgt_tokens in batch:
            if self.max_src_len is not None:
                src_tokens = src_tokens[:self.max_src_len]

            if self.max_tgt_len is not None:
                tgt_tokens = tgt_tokens[:self.max_tgt_len]

            src_ids = self.vocab.encode(src_tokens)
            tgt_ids = self.vocab.encode(tgt_tokens)

            if self.add_src_eos:
                src_ids = src_ids + [self.vocab.eos_id]

            tgt_ids = [self.vocab.bos_id] + tgt_ids + [self.vocab.eos_id]

            src_id_list.append(src_ids)
            tgt_id_list.append(tgt_ids)

        src_tensor = pad_sequences(src_id_list, pad_idx=self.vocab.pad_id)
        tgt_tensor = pad_sequences(tgt_id_list, pad_idx=self.vocab.pad_id)

        return Seq2SeqBatch.from_tensors(
            src=src_tensor,
            tgt=tgt_tensor,
            pad_idx=self.vocab.pad_id,
        )


def build_bpe_collate_fn(
    vocab: SharedVocab,
    max_src_len: Optional[int] = None,
    max_tgt_len: Optional[int] = None,
    add_src_eos: bool = True,
) -> Callable[[Sequence[Tuple[List[str], List[str]]]], Seq2SeqBatch]:
    """
    构造 DataLoader 的 collate_fn。
    """
    return BPEBatchCollator(
        vocab=vocab,
        max_src_len=max_src_len,
        max_tgt_len=max_tgt_len,
        add_src_eos=add_src_eos,
    )


def build_bpe_dataloader(
    src_path: str,
    tgt_path: str,
    vocab: SharedVocab,
    batch_size: Optional[int],
    num_workers: int = 0,
    pin_memory: bool = True,
    max_src_len: Optional[int] = None,
    max_tgt_len: Optional[int] = None,
    add_src_eos: bool = True,
    skip_empty: bool = False,
    shuffle_buffer_size: int = 0,
    seed: int = 42,
    num_samples: Optional[int] = None,
    sample_limit: Optional[int] = None,
    persistent_workers: bool = False,
    prefetch_factor: Optional[int] = None,
    src_token_budget: Optional[int] = None,
    tgt_token_budget: Optional[int] = None,
    max_sentences_per_batch: Optional[int] = None,
    batch_pool_size: int = 2048,
) -> DataLoader:
    """
    构建平行 BPE 文本 DataLoader。

    训练推荐：
        - train:
            shuffle_buffer_size = 10000 ~ 50000
            num_workers = 2 / 4 / 8（视机器而定）
            persistent_workers = True
        - valid/test:
            shuffle_buffer_size = 0
            num_workers = 0 或 2
            persistent_workers = False
    """
    dataset = ParallelBPEIterableDataset(
        src_path=src_path,
        tgt_path=tgt_path,
        skip_empty=skip_empty,
        shuffle_buffer_size=shuffle_buffer_size,
        seed=seed,
        num_samples=num_samples,
        sample_limit=sample_limit,
    )

    collate_fn = build_bpe_collate_fn(
        vocab=vocab,
        max_src_len=max_src_len,
        max_tgt_len=max_tgt_len,
        add_src_eos=add_src_eos,
    )

    use_token_budget_batching = (
        src_token_budget is not None
        or tgt_token_budget is not None
    )

    if use_token_budget_batching:
        if src_token_budget is None or tgt_token_budget is None:
            raise ValueError("src_token_budget and tgt_token_budget must be set together.")

        dataset = ApproxTokenBucketBatchDataset(
            dataset=dataset,
            src_token_budget=src_token_budget,
            tgt_token_budget=tgt_token_budget,
            add_src_eos=add_src_eos,
            max_sentences_per_batch=max_sentences_per_batch,
            pool_size=batch_pool_size,
        )

        dataloader_kwargs = dict(
            dataset=dataset,
            batch_size=None,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_fn,
            persistent_workers=(persistent_workers and num_workers > 0),
        )
    else:
        if batch_size is None or batch_size <= 0:
            raise ValueError("batch_size must be positive when token-budget batching is disabled.")

        dataloader_kwargs = dict(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_fn,
            persistent_workers=(persistent_workers and num_workers > 0),
        )

    if num_workers > 0 and prefetch_factor is not None:
        dataloader_kwargs["prefetch_factor"] = prefetch_factor

    loader = DataLoader(**dataloader_kwargs)
    return loader
