from __future__ import annotations

import random
from typing import Iterator, List, Optional, Sequence, Tuple

from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from data.shared_vocab import SharedVocab, SpecialTokens
from data.wmt_14_bpe_dataset import build_bpe_collate_fn


DEFAULT_COPY_ALPHABET = "abcdefghijklmnopqrstuvwxyz"


def _normalize_alphabet(alphabet: Sequence[str] | str) -> List[str]:
    if isinstance(alphabet, str):
        tokens = list(alphabet)
    else:
        tokens = list(alphabet)

    normalized: List[str] = []
    seen = set()
    for token in tokens:
        if not token:
            raise ValueError("alphabet tokens must be non-empty.")
        if any(char.isspace() for char in token):
            raise ValueError("alphabet tokens must not contain whitespace.")
        if token in seen:
            continue
        seen.add(token)
        normalized.append(token)

    if not normalized:
        raise ValueError("alphabet must contain at least one token.")

    return normalized


def build_char_copy_vocab(alphabet: Sequence[str] | str = DEFAULT_COPY_ALPHABET) -> SharedVocab:
    special_tokens = SpecialTokens()
    token_to_id = {}

    for token in special_tokens.as_list():
        token_to_id[token] = len(token_to_id)

    for token in _normalize_alphabet(alphabet):
        if token in token_to_id:
            raise ValueError(f"alphabet token conflicts with special token: {token}")
        token_to_id[token] = len(token_to_id)

    return SharedVocab(token_to_id=token_to_id, special_tokens=special_tokens)


class CharacterCopyIterableDataset(IterableDataset):
    """
    Yield fixed random character sequences where target is an exact copy of source.
    """

    def __init__(
        self,
        num_samples: int,
        alphabet: Sequence[str] | str = DEFAULT_COPY_ALPHABET,
        min_seq_len: int = 4,
        max_seq_len: int = 16,
        seed: int = 42,
        shuffle: bool = True,
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        super().__init__()
        if num_samples <= 0:
            raise ValueError("num_samples must be positive.")
        if min_seq_len <= 0:
            raise ValueError("min_seq_len must be positive.")
        if max_seq_len < min_seq_len:
            raise ValueError("max_seq_len must be >= min_seq_len.")
        if world_size <= 0:
            raise ValueError("world_size must be positive.")
        if not (0 <= rank < world_size):
            raise ValueError("rank must satisfy 0 <= rank < world_size.")

        self.num_samples = num_samples
        self.alphabet = _normalize_alphabet(alphabet)
        self.min_seq_len = min_seq_len
        self.max_seq_len = max_seq_len
        self.seed = seed
        self.shuffle = shuffle
        self.rank = rank
        self.world_size = world_size
        self.epoch = 0

    def __len__(self) -> int:
        if self.world_size > 1:
            return (self.num_samples + self.world_size - 1) // self.world_size
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _build_epoch_indices(self) -> List[int]:
        indices = list(range(self.num_samples))
        if self.shuffle:
            rng = random.Random(self.seed + self.epoch * 100_003)
            rng.shuffle(indices)
        return indices

    def _make_sequence(self, sample_index: int) -> List[str]:
        rng = random.Random(self.seed + sample_index * 1_000_003 + 17)
        seq_len = rng.randint(self.min_seq_len, self.max_seq_len)
        return [self.alphabet[rng.randrange(len(self.alphabet))] for _ in range(seq_len)]

    def __iter__(self) -> Iterator[Tuple[List[str], List[str]]]:
        worker_info = get_worker_info()
        worker_id = 0 if worker_info is None else worker_info.id
        num_workers = 1 if worker_info is None else worker_info.num_workers

        total_shards = self.world_size * num_workers
        shard_id = self.rank * num_workers + worker_id

        for stream_index, sample_index in enumerate(self._build_epoch_indices()):
            if (stream_index % total_shards) != shard_id:
                continue

            tokens = self._make_sequence(sample_index)
            yield tokens, list(tokens)


def build_char_copy_dataloader(
    num_samples: int,
    vocab: SharedVocab,
    batch_size: int,
    alphabet: Sequence[str] | str = DEFAULT_COPY_ALPHABET,
    min_seq_len: int = 4,
    max_seq_len: int = 16,
    shuffle: bool = True,
    seed: int = 42,
    num_workers: int = 0,
    pin_memory: bool = True,
    persistent_workers: bool = False,
    prefetch_factor: Optional[int] = None,
    rank: int = 0,
    world_size: int = 1,
) -> DataLoader:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    dataset = CharacterCopyIterableDataset(
        num_samples=num_samples,
        alphabet=alphabet,
        min_seq_len=min_seq_len,
        max_seq_len=max_seq_len,
        seed=seed,
        shuffle=shuffle,
        rank=rank,
        world_size=world_size,
    )

    dataloader_kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "collate_fn": build_bpe_collate_fn(
            vocab=vocab,
            max_src_len=max_seq_len,
            max_tgt_len=max_seq_len,
            add_src_eos=True,
        ),
        "persistent_workers": (persistent_workers and num_workers > 0),
    }

    if num_workers > 0 and prefetch_factor is not None:
        dataloader_kwargs["prefetch_factor"] = prefetch_factor

    return DataLoader(**dataloader_kwargs)
