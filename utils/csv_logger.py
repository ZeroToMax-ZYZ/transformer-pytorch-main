from __future__ import annotations

"""
功能：
1. 以追加方式写入训练/验证核心指标到 CSV。
2. 自动处理表头写入。
3. 保持字段稳定，便于后续画图和分析。
"""

import csv
import os
from typing import Dict, Iterable, List, Optional


class CSVMetricLogger:
    """
    简单可靠的 CSV 指标记录器。

    入口：
        csv_path: 输出 csv 路径
        fieldnames: 列名列表，建议训练前固定好

    出口：
        append_row(row_dict): 追加一行
    """

    def __init__(
        self,
        csv_path: str,
        fieldnames: List[str],
    ) -> None:
        self.csv_path = csv_path
        self.fieldnames = fieldnames

        os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)

        if not os.path.exists(self.csv_path):
            with open(self.csv_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()

    def append_row(self, row: Dict) -> None:
        """
        追加一行数据。
        """
        safe_row = {}
        for key in self.fieldnames:
            safe_row[key] = row.get(key, "")

        with open(self.csv_path, "a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(safe_row)