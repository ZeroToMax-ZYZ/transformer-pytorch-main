def count_empty_lines(path: str) -> int:
    cnt = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                cnt += 1
    return cnt

files = [
    "data/wmt14_tok_en_de/valid.en",
    "data/wmt14_tok_en_de/valid.de",
    "data/wmt14_tok_en_de/test.en",
    "data/wmt14_tok_en_de/test.de",
]



for path in files:
    n = count_empty_lines(path)
    print(path, "空行数 =", n)



def line_token_stats(path: str):
    lengths = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            lengths.append(len(line.strip().split()))
    print(path)
    print("  行数:", len(lengths))
    print("  最短:", min(lengths))
    print("  最长:", max(lengths))
    print("  平均:", sum(lengths) / len(lengths))

files = [
    "data/wmt14_tok_en_de/valid.en",
    "data/wmt14_tok_en_de/valid.de",
    "data/wmt14_tok_en_de/test.en",
    "data/wmt14_tok_en_de/test.de",
]

for path in files:
    line_token_stats(path)