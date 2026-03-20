from __future__ import annotations

"""
功能：
1. 从 BPE 文本训练集构建 shared vocabulary。
2. 提供 token -> id / id -> token 的双向映射。
3. 保存和加载 vocab 文件。
4. 为后续 DataLoader、target shifted right、mask 构造提供统一词表接口。

设计原则：
1. 只用 train 构建词表，不让 valid/test 泄露统计信息。
2. 源语言和目标语言共享一套 token-id 体系。
3. 固定特殊符号顺序，确保 pad/bos/eos/unk 的 id 稳定。
"""

import json
from collections import Counter
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence


@dataclass
class SpecialTokens:
    pad: str = "<pad>"
    bos: str = "<bos>"
    eos: str = "<eos>"
    unk: str = "<unk>"

    def as_list(self) -> List[str]:
        return [self.pad, self.bos, self.eos, self.unk]


class SharedVocab:
    """
    shared source-target vocabulary

    入口：
        - 通过 build_from_files 从 train.en/train.de 构建
        - 或通过 load 从已保存文件加载

    出口：
        - token_to_id / id_to_token
        - encode / decode
        - 各特殊 token 的 id
    """

    def __init__(
        self,
        token_to_id: Dict[str, int],
        special_tokens: SpecialTokens,
    ) -> None:
        self.token_to_id = token_to_id
        self.id_to_token = {idx: tok for tok, idx in token_to_id.items()}
        self.special_tokens = special_tokens

        self.pad_id = self.token_to_id[self.special_tokens.pad]
        self.bos_id = self.token_to_id[self.special_tokens.bos]
        self.eos_id = self.token_to_id[self.special_tokens.eos]
        self.unk_id = self.token_to_id[self.special_tokens.unk]

    def __len__(self) -> int:
        return len(self.token_to_id)

    def token2id(self, token: str) -> int:
        return self.token_to_id.get(token, self.unk_id)

    def id2token(self, idx: int) -> str:
        return self.id_to_token[idx]

    def encode(self, tokens: Sequence[str]) -> List[int]:
        """
        将 token 序列编码成 id 序列。
        """
        return [self.token2id(tok) for tok in tokens]

    def decode(self, ids: Sequence[int]) -> List[str]:
        """
        将 id 序列解码成 token 序列。
        """
        return [self.id2token(idx) for idx in ids]

    @classmethod
    def build_from_files(
        cls,
        file_paths: Sequence[str],
        min_freq: int = 1,
        special_tokens: Optional[SpecialTokens] = None,
    ) -> "SharedVocab":
        """
        从多个训练文本文件中构建共享词表。

        参数：
            file_paths: 参与构建词表的文件路径列表，通常是 train.en 和 train.de
            min_freq: 最小词频阈值
            special_tokens: 特殊 token 配置

        返回：
            SharedVocab 实例
        """
        if special_tokens is None:
            special_tokens = SpecialTokens()

        counter: Counter = Counter()

        for path in file_paths:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    tokens = line.strip().split()
                    counter.update(tokens)

        # 先放特殊 token，确保 id 稳定
        token_to_id: Dict[str, int] = {}
        for tok in special_tokens.as_list():
            token_to_id[tok] = len(token_to_id)

        # 再放普通 token
        # 排序策略：
        # 1. 先按词频降序
        # 2. 词频相同按字典序
        normal_tokens = [
            (tok, freq)
            for tok, freq in counter.items()
            if freq >= min_freq and tok not in token_to_id
        ]
        normal_tokens.sort(key=lambda x: (-x[1], x[0]))

        for tok, _ in normal_tokens:
            token_to_id[tok] = len(token_to_id)

        return cls(token_to_id=token_to_id, special_tokens=special_tokens)

    def save(self, vocab_json_path: str, vocab_txt_path: str, meta_json_path: str) -> None:
        """
        保存词表到磁盘。
        """
        with open(vocab_json_path, "w", encoding="utf-8") as f:
            json.dump(self.token_to_id, f, ensure_ascii=False, indent=2)

        with open(vocab_txt_path, "w", encoding="utf-8") as f:
            for idx in range(len(self)):
                f.write(self.id_to_token[idx] + "\n")

        meta = {
            "vocab_size": len(self),
            "pad_token": self.special_tokens.pad,
            "bos_token": self.special_tokens.bos,
            "eos_token": self.special_tokens.eos,
            "unk_token": self.special_tokens.unk,
            "pad_id": self.pad_id,
            "bos_id": self.bos_id,
            "eos_id": self.eos_id,
            "unk_id": self.unk_id,
        }

        with open(meta_json_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(
        cls,
        vocab_json_path: str,
        special_tokens: Optional[SpecialTokens] = None,
    ) -> "SharedVocab":
        """
        从 vocab.json 加载词表。
        """
        if special_tokens is None:
            special_tokens = SpecialTokens()

        with open(vocab_json_path, "r", encoding="utf-8") as f:
            token_to_id = json.load(f)

        return cls(token_to_id=token_to_id, special_tokens=special_tokens)