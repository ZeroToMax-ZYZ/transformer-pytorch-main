import torch
import torch.nn as nn


class Generator(nn.Module):
    """
    Generator / 词表投影头

    功能：
        将 Decoder 最后一层输出的隐状态映射到词表维度。

    说明：
        这里默认使用 bias=False，原因是后续要做 weight tying，
        即让 generator.proj.weight 与 embedding.weight 共享同一组参数。
        这里返回的是 logits，不在模块内做 softmax；概率归一化交给损失函数或解码逻辑。
    """

    def __init__(self, d_model: int, vocab_size: int, bias: bool = False):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入：
            x: (B, T, d_model)

        输出：
            logits: (B, T, vocab_size)
        """
        return self.proj(x)
