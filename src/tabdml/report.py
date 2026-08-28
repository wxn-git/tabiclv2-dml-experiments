from __future__ import annotations

from pathlib import Path

import pandas as pd


def _markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    rows = [
        "| " + " | ".join("" if pd.isna(value) else str(value) for value in row) + " |"
        for row in frame.itertuples(index=False, name=None)
    ]
    return "\n".join([header, separator, *rows])


def write_chinese_report(summary: pd.DataFrame, output: str | Path) -> Path:
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# TabICLv2-PLR-DML 仿真实验报告",
        "",
        "## 解释边界",
        "",
        "- 第一阶段每个配置仅运行 20 次，作用是筛选，不用于形成最终覆盖率结论。",
        "- 第二阶段配置由第一阶段结果数据驱动选择，因此属于重点验证。",
        "- TabICLv2 的预训练成本不计入下游运行时间；时间结果仅代表当前硬件上的使用成本。",
        "- 只有当差异相对于 Monte Carlo 不确定性稳定时，才应表述为“改进”。",
        "",
        "## 当前结果",
        "",
    ]
    if summary.empty:
        lines.append("尚无成功实验记录。")
    else:
        visible = [
            column
            for column in (
                "stage",
                "scenario",
                "n",
                "p",
                "learner",
                "tabicl_estimators",
                "bias",
                "rmse",
                "coverage",
                "mean_runtime_seconds",
                "success_count",
            )
            if column in summary.columns
        ]
        lines.append(_markdown_table(summary[visible]))
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path
