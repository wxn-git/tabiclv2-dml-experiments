from __future__ import annotations

import numpy as np
import pandas as pd


GROUP_COLUMNS = ["stage", "scenario", "n", "p", "learner", "tabicl_estimators"]


def summarize(records, theta0: float = 1.0) -> pd.DataFrame:
    frame = pd.DataFrame(records)
    if frame.empty:
        return pd.DataFrame()
    rows = []
    for keys, group in frame.groupby(GROUP_COLUMNS, dropna=False):
        success = group[group["status"] == "success"].copy()
        row = dict(zip(GROUP_COLUMNS, keys))
        row["success_count"] = len(success)
        row["failure_count"] = int((group["status"] == "failed").sum())
        row["oom_count"] = int((group["status"] == "oom").sum())
        if success.empty:
            rows.append(row)
            continue
        errors = success["theta"].astype(float) - theta0
        row.update(
            bias=float(errors.mean()),
            rmse=float(np.sqrt(np.mean(errors**2))),
            empirical_sd=float(success["theta"].astype(float).std(ddof=1))
            if len(success) > 1
            else np.nan,
            mean_standard_error=float(success["standard_error"].astype(float).mean()),
            coverage=float(
                ((success["ci_lower"] <= theta0) & (success["ci_upper"] >= theta0)).mean()
            ),
            mean_interval_length=float(
                (success["ci_upper"].astype(float) - success["ci_lower"].astype(float)).mean()
            ),
            mean_runtime_seconds=float(success["runtime_seconds"].astype(float).mean()),
            mean_l_mse=float(success["l_mse"].astype(float).mean()),
            mean_m_mse=float(success["m_mse"].astype(float).mean()),
            bias_mcse=float(success["theta"].astype(float).std(ddof=1) / np.sqrt(len(success)))
            if len(success) > 1
            else np.nan,
        )
        rows.append(row)
    return pd.DataFrame(rows)

