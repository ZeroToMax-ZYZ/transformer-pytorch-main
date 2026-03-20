def count_lines(path: str) -> int:
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)

pairs = [
    ("data/wmt14_bpe_en_de/train.en", "data/wmt14_bpe_en_de/train.de"),
    ("data/wmt14_bpe_en_de/valid.en", "data/wmt14_bpe_en_de/valid.de"),
    ("data/wmt14_bpe_en_de/test.en", "data/wmt14_bpe_en_de/test.de"),
]

for en_path, de_path in pairs:
    en_n = count_lines(en_path)
    de_n = count_lines(de_path)
    print(en_path, en_n)
    print(de_path, de_n)
    assert en_n == de_n, f"行数不一致: {en_path} vs {de_path}"


def collect_vocab(path: str) -> set:
    vocab = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            vocab.update(line.strip().split())
    return vocab

v_en = collect_vocab("data/wmt14_bpe_en_de/train.en")
v_de = collect_vocab("data/wmt14_bpe_en_de/train.de")
v_union = v_en | v_de

print(f"train.en vocab size: {len(v_en)}")
print(f"train.de vocab size: {len(v_de)}")
print(f"shared union vocab size: {len(v_union)}")