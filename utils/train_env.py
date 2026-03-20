from __future__ import annotations

"""
功能：
1. 设置随机种子。
2. 获取训练设备。
3. 统计模型参数量。
4. 生成实验时间戳。
"""

import os
import random
from datetime import datetime
from typing import Dict

import numpy as np
import torch


def seed_everything(seed: int = 42, deterministic: bool = False) -> None:
    """
    设置全局随机种子。

    参数：
        seed: 随机种子
        deterministic: 是否开启更强的确定性模式
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def get_device() -> torch.device:
    """
    获取当前推荐训练设备。
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def count_trainable_parameters(model: torch.nn.Module) -> int:
    """
    统计模型可训练参数量。
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_timestamp_str() -> str:
    """
    获取实验时间戳字符串。
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")