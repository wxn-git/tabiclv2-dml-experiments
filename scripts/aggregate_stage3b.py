from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from tabdml.stage3b_aggregate import aggregate_dml_records, markdown_table


def _read_json(root: Path) -> list[dict]:
    return [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(root.glob("*.json"))
    ]


def main() -> int:
    output = Path("results/stage3b_analysis")
    output.mkdir(parents=True, exist_ok=True)
    batch_a_records = _read_json(Path("results/stage3b_batch_a_raw"))
    confirmation_records = _read_json(Path("results/stage3b_confirmation_raw"))
    screening_records = _read_json(Path("results/stage3b_screening_raw"))

    batch_a = aggregate_dml_records(batch_a_records, theta0=1.0)
    confirmation = aggregate_dml_records(confirmation_records, theta0=1.0)
    screening_frame = pd.DataFrame(screening_records)
    successful_screening = screening_frame[screening_frame["status"].eq("success")]
    screening = (
        successful_screening.groupby(
            ["candidate", "candidate_group", "learner_kind", "training_target", "config_hash"],
            dropna=False,
        )
        .agg(
            replications=("replication", "count"),
            mean_validation_d_mse=("validation_d_mse", "mean"),
            mean_validation_m0_mse=("validation_m0_mse", "mean"),
            mean_runtime_seconds=("runtime_seconds", "mean"),
        )
        .reset_index()
        .sort_values(["candidate_group", "mean_validation_d_mse"])
    )
    batch_a.to_csv(output / "batch_a_summary.csv", index=False)
    screening.to_csv(output / "screening_summary.csv", index=False)
    confirmation.to_csv(output / "confirmation_summary.csv", index=False)

    key_columns = [
        "learner_l",
        "learner_m",
        "bias",
        "rmse",
        "coverage",
        "mean_l_mse",
        "mean_m_mse",
        "mean_lm_error_cross",
        "mean_theta_proxy",
        "mean_proxy_error",
    ]
    report = [
        "# Stage 3B Tree机制诊断与处理模型筛选结果",
        "",
        "## Batch A：现有Stage 3A误差分解",
        "",
        markdown_table(batch_a, key_columns),
        "",
        "## Batch B：处理模型筛选",
        "",
        markdown_table(
            screening,
            [
                "candidate",
                "candidate_group",
                "training_target",
                "replications",
                "mean_validation_d_mse",
                "mean_validation_m0_mse",
                "mean_runtime_seconds",
            ],
        ),
        "",
        "## Batch C：独立确认",
        "",
        markdown_table(confirmation, key_columns),
        "",
        "说明：Batch C仍属于50次筛选后确认；论文最终覆盖率表需使用预先锁定配置和新的200至500次重复。",
    ]
    (output / "analysis_report_zh.md").write_text("\n".join(report), encoding="utf-8")
    print(f"Wrote Stage 3B analysis to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
