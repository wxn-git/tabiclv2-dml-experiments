from __future__ import annotations

import math

import numpy as np
import pandas as pd


GROUP_COLUMNS = [
    "stage",
    "scenario",
    "n",
    "p",
    "learner_l",
    "learner_m",
    "learner_l_config_hash",
    "learner_m_config_hash",
]


def _markdown_value(value) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.4f}"
    return str(value).replace("|", "\\|")


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    selected = frame[columns]
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    rows = [
        "| " + " | ".join(_markdown_value(value) for value in row) + " |"
        for row in selected.itertuples(index=False, name=None)
    ]
    return "\n".join([header, separator, *rows])


def aggregate_dml_records(records: list[dict], theta0: float) -> pd.DataFrame:
    frame = pd.DataFrame(records)
    if frame.empty:
        return pd.DataFrame()
    successful = frame[frame["status"].eq("success")].copy()
    if successful.empty:
        return pd.DataFrame()
    successful["squared_error"] = (successful["theta"] - theta0) ** 2
    successful["covered"] = (
        (successful["ci_lower"] <= theta0)
        & (theta0 <= successful["ci_upper"])
    )

    rows = []
    for key, group in successful.groupby(GROUP_COLUMNS, dropna=False):
        theta = group["theta"].to_numpy(dtype=float)
        coverage = group["covered"].to_numpy(dtype=float)
        count = len(group)
        row = dict(zip(GROUP_COLUMNS, key))
        row.update(
            success_count=count,
            bias=float(theta.mean() - theta0),
            rmse=float(math.sqrt(group["squared_error"].mean())),
            empirical_sd=float(theta.std(ddof=1)) if count > 1 else float("nan"),
            mean_standard_error=float(group["standard_error"].mean()),
            coverage=float(coverage.mean()),
            mean_interval_length=float((group["ci_upper"] - group["ci_lower"]).mean()),
            mean_l_mse=float(group["l_mse"].mean()),
            mean_m_mse=float(group["m_mse"].mean()),
            mean_lm_error_cross=float(group["lm_error_cross"].mean()),
            mean_theta_proxy=float(group["theta_proxy"].mean()),
            mean_proxy_error=float(group["proxy_error"].mean()),
            mean_runtime_seconds=float(group["runtime_seconds"].mean()),
            bias_mcse=(
                float(theta.std(ddof=1) / math.sqrt(count))
                if count > 1
                else float("nan")
            ),
            coverage_mcse=float(math.sqrt(coverage.mean() * (1 - coverage.mean()) / count)),
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(GROUP_COLUMNS).reset_index(drop=True)
