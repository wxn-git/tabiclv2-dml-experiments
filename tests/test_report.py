from pathlib import Path

import pandas as pd

from tabdml.report import write_chinese_report


def test_report_discloses_screening_selection_and_pretraining_cost(tmp_path):
    summary = pd.DataFrame(
        [
            {
                "stage": "stage1",
                "scenario": "linear",
                "n": 500,
                "p": 10,
                "learner": "lasso",
                "tabicl_estimators": 0,
                "rmse": 0.1,
                "bias": 0.01,
                "coverage": 0.95,
                "mean_runtime_seconds": 2.0,
                "success_count": 20,
            }
        ]
    )
    output = tmp_path / "report.md"
    write_chinese_report(summary, output)
    text = Path(output).read_text(encoding="utf-8")
    assert "20 次" in text
    assert "数据驱动" in text
    assert "预训练成本" in text

