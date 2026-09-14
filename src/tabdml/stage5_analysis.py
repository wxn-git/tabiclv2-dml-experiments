"""Statistical summaries and plotting data for Stage 5."""
from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


GROUP = ["scenario", "n", "p", "method"]
REQUIRED = {
    *GROUP, "replication", "status", "theta0", "theta_hat", "standard_error",
    "ci_lower", "ci_upper", "covered", "squared_error", "l_mse", "m_mse",
    "runtime_seconds",
}


def _frame(records: pd.DataFrame | Iterable[Mapping[str, Any]]) -> pd.DataFrame:
    frame = records.copy() if isinstance(records, pd.DataFrame) else pd.DataFrame(records)
    missing = REQUIRED.difference(frame.columns)
    if missing:
        raise ValueError(f"Stage 5 records are missing columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("Stage 5 records are empty")
    if frame.duplicated([*GROUP, "replication"]).any():
        raise ValueError("Stage 5 records contain duplicate replication keys")
    if not frame["status"].isin(["success", "failed", "oom", "fallback"]).all():
        raise ValueError("Stage 5 records contain an unknown status")
    return frame


def bootstrap_mse_interval(
    squared_errors: Iterable[float], *, seed: int, resamples: int = 10_000,
) -> tuple[float, float]:
    values = np.asarray(list(squared_errors), dtype=float)
    if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("squared errors must be a nonempty finite nonnegative vector")
    if type(seed) is not int or type(resamples) is not int or resamples < 1:
        raise ValueError("seed and resamples must be native integers")
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), size=(resamples, len(values)))].mean(axis=1)
    lower, upper = np.percentile(means, [2.5, 97.5])
    return float(lower), float(upper)


def _seed(parts: tuple[Any, ...]) -> int:
    digest = hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def summarize_stage5(
    records: pd.DataFrame | Iterable[Mapping[str, Any]], *, bootstrap_resamples: int = 10_000,
) -> pd.DataFrame:
    frame = _frame(records)
    # All methods must use exactly the same replication IDs within a DGP/cell.
    for _, cell in frame.groupby(["scenario", "n", "p"], sort=False):
        sets = cell.groupby("method")["replication"].apply(lambda x: tuple(sorted(x.tolist())))
        if len(set(sets)) != 1:
            raise ValueError("Stage 5 methods do not have paired replication sets")
    rows = []
    for key, group in frame.groupby(GROUP, sort=False):
        counts = group["status"].value_counts()
        success = group.loc[group["status"].eq("success")].copy()
        if success.empty:
            raise ValueError(f"Stage 5 group has no successful records: {key}")
        numeric = [
            "theta0", "theta_hat", "standard_error", "ci_lower", "ci_upper",
            "squared_error", "l_mse", "m_mse", "runtime_seconds",
        ]
        for column in numeric:
            success[column] = pd.to_numeric(success[column], errors="raise")
        if not np.isfinite(success[numeric].to_numpy(dtype=float)).all():
            raise ValueError(f"Stage 5 group contains non-finite values: {key}")
        if success["theta0"].nunique() != 1:
            raise ValueError("Stage 5 theta0 varies within a group")
        theta0 = float(success["theta0"].iloc[0])
        errors = success["theta_hat"].to_numpy(float) - theta0
        squared = success["squared_error"].to_numpy(float)
        if not np.allclose(squared, errors**2):
            raise ValueError("Stage 5 squared_error is inconsistent with theta_hat")
        ci_low, ci_high = bootstrap_mse_interval(
            squared, seed=_seed(tuple(key)), resamples=bootstrap_resamples,
        )
        mse = float(squared.mean())
        rows.append({
            **dict(zip(GROUP, key)), "replications": int(len(group)),
            "success_count": int(len(success)),
            "failure_count": int(counts.get("failed", 0)),
            "oom_count": int(counts.get("oom", 0)),
            "fallback_count": int(counts.get("fallback", 0)),
            "bias": float(errors.mean()), "treatment_mse": mse,
            "rmse": float(np.sqrt(mse)),
            "empirical_sd": float(success["theta_hat"].std(ddof=1)) if len(success) > 1 else 0.0,
            "mean_se": float(success["standard_error"].mean()),
            "coverage": float(success["covered"].astype(bool).mean()),
            "interval_width": float((success["ci_upper"] - success["ci_lower"]).mean()),
            "l_mse": float(success["l_mse"].mean()),
            "m_mse": float(success["m_mse"].mean()),
            "mean_runtime": float(success["runtime_seconds"].mean()),
            "mse_ci_lower": ci_low, "mse_ci_upper": ci_high,
        })
    return pd.DataFrame(rows).sort_values(GROUP).reset_index(drop=True)


def build_stage5_plot_data(summary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {*GROUP, "treatment_mse", "mse_ci_lower", "mse_ci_upper"}
    if missing := required.difference(summary.columns):
        raise ValueError(f"Stage 5 summary is missing plotting columns: {sorted(missing)}")
    fixed_n = summary.loc[summary["n"].eq(1000) & summary["p"].isin([10, 50, 100])].copy()
    fixed_p = summary.loc[summary["p"].eq(50) & summary["n"].isin([500, 1000, 2000])].copy()
    if fixed_n.empty or fixed_p.empty:
        raise ValueError("Stage 5 summary does not contain both prescribed sweeps")
    return fixed_n.reset_index(drop=True), fixed_p.reset_index(drop=True)


def estimate_formal_runtime(
    preflight_records: pd.DataFrame | Iterable[Mapping[str, Any]], *,
    formal_replications: int = 100, cpu_workers: int = 5,
    ensemble_workers: int = 2, overhead_fraction: float = 0.05,
) -> dict[str, Any]:
    frame = _frame(preflight_records)
    success = frame.loc[frame["status"].eq("success")].copy()
    for value, name in ((formal_replications, "formal_replications"), (cpu_workers, "cpu_workers"), (ensemble_workers, "ensemble_workers")):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive native integer")
    reps = success["replication"].nunique()
    if reps < 1:
        raise ValueError("preflight has no successful replications")
    factor = formal_replications / reps
    runtime = pd.to_numeric(success["runtime_seconds"], errors="raise")
    if not np.isfinite(runtime).all() or (runtime < 0).any():
        raise ValueError("preflight runtimes must be finite and nonnegative")
    method_runtime = success.assign(runtime_seconds=runtime).groupby("method")["runtime_seconds"].sum()
    gpu = factor * float(method_runtime.reindex(["tabiclv2_1", "tabiclv2_8"], fill_value=0).sum())
    cpu = factor * float(method_runtime.reindex(["xgboost_tuned", "extra_trees", "lasso"], fill_value=0).sum()) / cpu_workers
    has_ensemble = "ensemble" in method_runtime.index
    ensemble = factor * float(method_runtime.get("ensemble", 0.0)) / ensemble_workers
    core = max(gpu, cpu) + ensemble
    projected = core * (1.0 + overhead_fraction)
    result = {
        "gpu_lane_seconds": gpu, "standard_cpu_lane_seconds": cpu,
        "composition_analysis_overhead_seconds": projected - core,
        "projected_elapsed_seconds": projected,
        "preflight_replications": int(reps), "formal_replications": formal_replications,
        "cpu_workers": cpu_workers,
        "hardware_dependent": True,
        "note": "Projection is hardware- and implementation-dependent.",
    }
    if has_ensemble:
        result.update({
            "ensemble_lane_seconds": ensemble,
            "ensemble_workers": ensemble_workers,
        })
    return result


def load_stage5_records(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.is_dir():
        values = [json.loads(item.read_text(encoding="utf-8")) for item in sorted(path.glob("*.json"))]
        return pd.DataFrame(values)
    if path.suffix.lower() == ".jsonl":
        return pd.read_json(path, lines=True)
    value = pd.read_json(path)
    return value if isinstance(value, pd.DataFrame) else value.to_frame().T
