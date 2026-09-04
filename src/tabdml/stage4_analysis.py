from __future__ import annotations

from collections.abc import Mapping, Sequence
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "tabdml-matplotlib")
)

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from .stage4_experiment import (
    iter_stage4_pairs,
    validate_frozen_tuning,
    validate_stage4_record,
    validate_stage4_selection,
)
from .stage4_selection import select_confirmation_cells


PRIMARY_METHODS = ("tabiclv2_1", "xgboost_tuned")
OFFICIAL_METHODS = (
    "tabiclv2_1",
    "tabiclv2_8",
    "xgboost",
    "xgboost_tuned",
    "extra_trees",
    "oracle",
)
CELL_COLUMNS = ["panel", "scenario", "n", "p"]
AGGREGATE_COLUMNS = [
    *CELL_COLUMNS,
    "method",
    "learner_l",
    "learner_m",
    "replications",
    "bias",
    "rmse",
    "empirical_se",
    "mean_reported_se",
    "coverage",
    "coverage_ci_lower",
    "coverage_ci_upper",
    "nominal_coverage_in_exact_interval",
    "mean_interval_width",
    "mean_l_mse",
    "mean_m_mse",
    "mean_nuisance_error_product",
    "mean_lm_error_cross",
    "mean_residual_d_variance",
    "mean_bias_numerator_proxy",
    "mean_theta_proxy",
    "mean_proxy_error",
    "mean_runtime_seconds",
    "mean_l_fit_seconds",
    "mean_m_fit_seconds",
    "mean_total_fit_seconds",
    "mean_peak_gpu_mb",
    "gpu_observation_count",
    "fallback_count",
]
PRIMARY_COMPARISON_COLUMNS = [
    *CELL_COLUMNS,
    "paired_count",
    "inference_status",
    "tab_rmse",
    "xgb_rmse",
    "rmse_improvement_pct",
    "mean_squared_error_difference",
    "difference_ci_lower",
    "difference_ci_upper",
    "paired_p_value",
    "holm_p_value",
    "tab_abs_error_win_rate",
    "tab_bias",
    "xgb_bias",
    "tab_coverage",
    "xgb_coverage",
    "tab_coverage_ci_lower",
    "tab_coverage_ci_upper",
    "xgb_coverage_ci_lower",
    "xgb_coverage_ci_upper",
    "coverage_difference",
    "symmetric_success",
    "superior",
    "failed_conditions",
]
RANKING_COLUMNS = [
    *CELL_COLUMNS,
    "mean_paired_squared_error_difference",
    "selection_rule",
]
COVERAGE_COLUMNS = [
    *CELL_COLUMNS,
    "method",
    "learner_l",
    "learner_m",
    "replications",
    "coverage",
    "coverage_ci_lower",
    "coverage_ci_upper",
    "nominal_coverage_in_exact_interval",
    "mean_interval_width",
]
NUISANCE_COLUMNS = [
    *CELL_COLUMNS,
    "method",
    "learner_l",
    "learner_m",
    "replications",
    "mean_l_mse",
    "mean_m_mse",
    "mean_nuisance_error_product",
    "mean_lm_error_cross",
    "mean_residual_d_variance",
    "mean_bias_numerator_proxy",
    "mean_theta_proxy",
    "mean_proxy_error",
    "mean_runtime_seconds",
    "mean_l_fit_seconds",
    "mean_m_fit_seconds",
    "mean_total_fit_seconds",
    "mean_peak_gpu_mb",
    "gpu_observation_count",
    "fallback_count",
]


def _finite_number(value: Any, name: str) -> float:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, float, np.integer, np.floating))
        or not np.isfinite(value)
    ):
        raise ValueError(f"{name} must be a finite numeric value")
    return float(value)


def _native_integer(value: Any, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be a native integer >= {minimum}")
    return value


def validate_stage4_alpha(alpha: Any) -> float:
    """Validate the predeclared confidence level for confirmatory analysis."""
    level = _finite_number(alpha, "alpha")
    if level != 0.05:
        raise ValueError("Stage 4 confirmatory alpha must be exactly 0.05")
    return level


def validate_stage4_theta0(theta0: Any) -> float:
    """Validate the fixed estimand used by every Stage 4 analysis interface."""
    try:
        truth = _finite_number(theta0, "theta0")
    except ValueError as error:
        raise ValueError(
            "fixed Stage 4 design requires theta0 exactly 1.0"
        ) from error
    if truth != 1.0:
        raise ValueError("fixed Stage 4 design requires theta0 exactly 1.0")
    return truth


def _validate_fixed_stage4_design(config: Mapping[str, Any]) -> None:
    if not isinstance(config, Mapping):
        raise ValueError("fixed Stage 4 design requires a config mapping")

    validate_stage4_theta0(config.get("theta0"))

    folds = config.get("folds")
    if isinstance(folds, bool) or not isinstance(folds, int) or folds != 5:
        raise ValueError("fixed Stage 4 design requires exactly 5 folds")

    for section, expected in (
        ("tuning", 10),
        ("screening", 20),
        ("confirmation", 100),
    ):
        phase = config.get(section)
        actual = phase.get("replications") if isinstance(phase, Mapping) else None
        if (
            isinstance(actual, bool)
            or not isinstance(actual, int)
            or actual != expected
        ):
            raise ValueError(
                "fixed Stage 4 design requires "
                f"{section} replications exactly {expected}"
            )


def _checked_mean(values: Any, name: str) -> float:
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            result = float(np.mean(np.asarray(values, dtype=float)))
    except FloatingPointError as error:
        raise ValueError(f"{name} calculation overflowed") from error
    if not np.isfinite(result):
        raise ValueError(f"{name} calculation produced a nonfinite value")
    return result


def _checked_std(values: Any, name: str) -> float:
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            result = float(np.std(np.asarray(values, dtype=float), ddof=1))
    except FloatingPointError as error:
        raise ValueError(f"{name} calculation overflowed") from error
    if not np.isfinite(result):
        raise ValueError(f"{name} calculation produced a nonfinite value")
    return result


def _checked_squares(values: Any, name: str) -> np.ndarray:
    try:
        with np.errstate(over="raise", invalid="raise"):
            result = np.square(np.asarray(values, dtype=float))
    except FloatingPointError as error:
        raise ValueError(f"{name} calculation overflowed") from error
    if not np.isfinite(result).all():
        raise ValueError(f"{name} calculation produced nonfinite values")
    return result


def holm_adjust(p_values: Sequence[float]) -> np.ndarray:
    if isinstance(p_values, (str, bytes)) or not isinstance(p_values, Sequence):
        raise ValueError("p_values must be a sequence")
    values = np.asarray(p_values, dtype=object)
    checked = np.asarray(
        [_finite_number(value, "p_values") for value in values], dtype=float
    )
    if np.any((checked < 0.0) | (checked > 1.0)):
        raise ValueError("p_values must be between 0 and 1")
    if not len(checked):
        return np.asarray([], dtype=float)
    order = np.argsort(checked, kind="stable")
    sorted_values = checked[order]
    scaled = sorted_values * np.arange(len(checked), 0, -1)
    adjusted_sorted = np.minimum(1.0, np.maximum.accumulate(scaled))
    adjusted = np.empty_like(adjusted_sorted)
    adjusted[order] = adjusted_sorted
    return adjusted


def exact_coverage_interval(
    covered: int,
    total: int,
    alpha: float = 0.05,
) -> tuple[float, float]:
    successes = _native_integer(covered, "covered")
    trials = _native_integer(total, "total", minimum=1)
    level = _finite_number(alpha, "alpha")
    if not 0.0 < level < 1.0:
        raise ValueError("alpha must be strictly between 0 and 1")
    if successes > trials:
        raise ValueError("covered must not exceed total")
    lower = (
        0.0
        if successes == 0
        else float(stats.beta.ppf(level / 2.0, successes, trials - successes + 1))
    )
    upper = (
        1.0
        if successes == trials
        else float(
            stats.beta.ppf(
                1.0 - level / 2.0,
                successes + 1,
                trials - successes,
            )
        )
    )
    if not np.isfinite((lower, upper)).all():
        raise ValueError("coverage interval calculation produced nonfinite bounds")
    return lower, upper


def apply_superiority_rule(comparison: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(comparison, Mapping):
        raise ValueError("comparison must be a mapping")
    improvement_value = comparison.get("rmse_improvement_pct")
    improvement_undefined = improvement_value is None or (
        isinstance(improvement_value, (float, np.floating))
        and np.isnan(improvement_value)
    )
    if improvement_undefined:
        xgb_rmse = _finite_number(comparison.get("xgb_rmse"), "xgb_rmse")
        tab_rmse = _finite_number(comparison.get("tab_rmse"), "tab_rmse")
        if xgb_rmse != 0.0 or tab_rmse < 0.0:
            raise ValueError(
                "undefined rmse_improvement_pct requires zero xgb_rmse"
            )
        improvement = None
    else:
        improvement = _finite_number(
            improvement_value, "rmse_improvement_pct"
        )
    numeric = {
        field: _finite_number(comparison.get(field), field)
        for field in (
            "holm_p_value",
            "tab_coverage",
            "xgb_coverage",
            "coverage_difference",
        )
    }
    for field in ("holm_p_value", "tab_coverage", "xgb_coverage"):
        if not 0.0 <= numeric[field] <= 1.0:
            raise ValueError(f"{field} must be between 0 and 1")
    if not -1.0 <= numeric["coverage_difference"] <= 1.0:
        raise ValueError("coverage_difference must be between -1 and 1")
    symmetric = comparison.get("symmetric_success")
    if not isinstance(symmetric, (bool, np.bool_)):
        raise ValueError("symmetric_success must be boolean")
    conditions = (
        (
            (
                "rmse_improvement_undefined"
                if improvement_undefined
                else "rmse_improvement_below_10pct"
            ),
            improvement is not None and improvement >= 10.0,
        ),
        ("holm_p_value_not_below_0.05", numeric["holm_p_value"] < 0.05),
        (
            "tab_coverage_more_than_0.05_below_xgb",
            numeric["coverage_difference"] >= -0.05,
        ),
        ("tab_coverage_below_0.90", numeric["tab_coverage"] >= 0.90),
        ("asymmetric_or_incomplete_results", bool(symmetric)),
    )
    failed = [name for name, passed in conditions if not passed]
    result = dict(comparison)
    result["superior"] = not failed
    result["failed_conditions"] = ";".join(failed)
    return result


def _validate_primary_record(record: Any, index: int) -> dict[str, Any]:
    if not isinstance(record, Mapping):
        raise ValueError(f"primary record {index} must be a mapping")
    if record.get("status") != "success":
        raise ValueError(f"primary record {index} status must be success")
    panel = record.get("panel")
    scenario = record.get("scenario")
    method_l = record.get("learner_l")
    method_m = record.get("learner_m")
    if not isinstance(panel, str) or not panel:
        raise ValueError(f"primary record {index} panel must be a string")
    if not isinstance(scenario, str) or not scenario:
        raise ValueError(f"primary record {index} scenario must be a string")
    if method_l != method_m or method_l not in PRIMARY_METHODS:
        raise ValueError(f"primary record {index} is not a declared primary method")
    result = {
        "panel": panel,
        "scenario": scenario,
        "n": _native_integer(record.get("n"), "n", minimum=1),
        "p": _native_integer(record.get("p"), "p", minimum=1),
        "replication": _native_integer(record.get("replication"), "replication"),
        "method": method_l,
        "theta": _finite_number(record.get("theta"), "theta"),
        "standard_error": _finite_number(
            record.get("standard_error"), "standard_error"
        ),
        "ci_lower": _finite_number(record.get("ci_lower"), "ci_lower"),
        "ci_upper": _finite_number(record.get("ci_upper"), "ci_upper"),
    }
    if result["standard_error"] < 0:
        raise ValueError("standard_error must be nonnegative")
    if result["ci_lower"] > result["theta"] or result["theta"] > result["ci_upper"]:
        raise ValueError("primary record confidence interval is invalid")
    return result


def _paired_test(delta: np.ndarray, alpha: float) -> tuple[float | None, float | None, float | None]:
    if len(delta) < 1:
        raise ValueError("paired comparisons require at least one replication")
    mean = _checked_mean(delta, "paired mean difference")
    if len(delta) == 1:
        return None, None, None
    scale = max(abs(mean), float(np.max(np.abs(delta))))
    numerically_constant = (
        float(np.ptp(delta)) <= 32.0 * np.finfo(float).eps * scale
    )
    if numerically_constant:
        p_value = 1.0 if mean == 0.0 else 0.0
        return p_value, mean, mean
    standard_error = float(stats.sem(delta, ddof=1))
    critical = float(stats.t.ppf(1.0 - alpha / 2.0, len(delta) - 1))
    test = stats.ttest_1samp(delta, 0.0, nan_policy="raise")
    p_value = float(test.pvalue)
    bounds = (mean - critical * standard_error, mean + critical * standard_error)
    if not np.isfinite((p_value, *bounds)).all():
        raise ValueError("paired inference produced nonfinite results")
    return p_value, float(bounds[0]), float(bounds[1])


def paired_primary_comparisons(
    records: Sequence[Mapping[str, Any]],
    theta0: float = 1.0,
    alpha: float = 0.05,
) -> pd.DataFrame:
    if isinstance(records, (str, bytes)) or not isinstance(records, Sequence):
        raise ValueError("records must be a sequence")
    truth = validate_stage4_theta0(theta0)
    level = validate_stage4_alpha(alpha)
    checked = [
        _validate_primary_record(record, index)
        for index, record in enumerate(records)
    ]
    if not checked:
        raise ValueError("primary records must not be empty")
    identities: dict[tuple[Any, ...], dict[str, Any]] = {}
    for record in checked:
        key = tuple(record[column] for column in CELL_COLUMNS) + (
            record["replication"],
            record["method"],
        )
        if key in identities:
            raise ValueError("duplicate primary cell/method/replication record")
        identities[key] = record

    cells = sorted(
        {key[:4] for key in identities},
        key=lambda key: (str(key[0]), str(key[1]), int(key[2]), int(key[3])),
    )
    rows = []
    for cell in cells:
        by_method = {
            method: {
                key[4]: record
                for key, record in identities.items()
                if key[:4] == cell and key[5] == method
            }
            for method in PRIMARY_METHODS
        }
        tab_reps = set(by_method["tabiclv2_1"])
        xgb_reps = set(by_method["xgboost_tuned"])
        if tab_reps != xgb_reps or not tab_reps:
            raise ValueError("primary methods must have identical paired replications")
        replications = sorted(tab_reps)
        tab = [by_method["tabiclv2_1"][replication] for replication in replications]
        xgb = [by_method["xgboost_tuned"][replication] for replication in replications]
        tab_errors = np.asarray([record["theta"] - truth for record in tab])
        xgb_errors = np.asarray([record["theta"] - truth for record in xgb])
        tab_squared_errors = _checked_squares(
            tab_errors, "Tab squared errors"
        )
        xgb_squared_errors = _checked_squares(
            xgb_errors, "XGB squared errors"
        )
        with np.errstate(over="raise", invalid="raise"):
            delta = tab_squared_errors - xgb_squared_errors
        if not np.isfinite(delta).all():
            raise ValueError(
                "paired squared-error differences produced nonfinite values"
            )
        tab_rmse = float(
            np.sqrt(_checked_mean(tab_squared_errors, "Tab mean squared error"))
        )
        xgb_rmse = float(
            np.sqrt(_checked_mean(xgb_squared_errors, "XGB mean squared error"))
        )
        improvement = (
            None
            if xgb_rmse == 0.0
            else float(100.0 * (xgb_rmse - tab_rmse) / xgb_rmse)
        )
        p_value, ci_lower, ci_upper = _paired_test(delta, level)
        tab_covered = sum(
            record["ci_lower"] <= truth <= record["ci_upper"] for record in tab
        )
        xgb_covered = sum(
            record["ci_lower"] <= truth <= record["ci_upper"] for record in xgb
        )
        count = len(replications)
        tab_coverage = tab_covered / count
        xgb_coverage = xgb_covered / count
        tab_coverage_ci = exact_coverage_interval(tab_covered, count, level)
        xgb_coverage_ci = exact_coverage_interval(xgb_covered, count, level)
        rows.append(
            {
                **dict(zip(CELL_COLUMNS, cell)),
                "paired_count": count,
                "tab_rmse": tab_rmse,
                "xgb_rmse": xgb_rmse,
                "rmse_improvement_pct": improvement,
                "mean_squared_error_difference": _checked_mean(
                    delta, "paired mean squared-error difference"
                ),
                "difference_ci_lower": ci_lower,
                "difference_ci_upper": ci_upper,
                "paired_p_value": p_value,
                "tab_abs_error_win_rate": float(
                    np.mean(np.abs(tab_errors) < np.abs(xgb_errors))
                ),
                "tab_bias": _checked_mean(tab_errors, "Tab bias"),
                "xgb_bias": _checked_mean(xgb_errors, "XGB bias"),
                "tab_coverage": tab_coverage,
                "xgb_coverage": xgb_coverage,
                "tab_coverage_ci_lower": tab_coverage_ci[0],
                "tab_coverage_ci_upper": tab_coverage_ci[1],
                "xgb_coverage_ci_lower": xgb_coverage_ci[0],
                "xgb_coverage_ci_upper": xgb_coverage_ci[1],
                "coverage_difference": tab_coverage - xgb_coverage,
                "symmetric_success": True,
            }
        )
    inference_available = all(row["paired_count"] >= 2 for row in rows)
    adjusted = (
        holm_adjust([row["paired_p_value"] for row in rows])
        if inference_available else [None] * len(rows)
    )
    completed = []
    for row, holm_p_value in zip(rows, adjusted, strict=True):
        if inference_available:
            row["inference_status"] = "available"
            row["holm_p_value"] = float(holm_p_value)
            completed.append(apply_superiority_rule(row))
        else:
            # Never apply Holm to a subset of an incomplete inference family.
            row.update(
                inference_status="implementation_smoke",
                paired_p_value=None, holm_p_value=None,
                difference_ci_lower=None, difference_ci_upper=None,
                superior=False, failed_conditions="inference_unavailable",
            )
            completed.append(row)
    result = pd.DataFrame(completed, columns=PRIMARY_COMPARISON_COLUMNS)
    _validate_comparison_frame(result)
    return result


def _missing(value: Any) -> bool:
    return value is None or (
        isinstance(value, (float, np.floating)) and np.isnan(value)
    )


def _validate_comparison_frame(frame: pd.DataFrame) -> None:
    if list(frame.columns) != PRIMARY_COMPARISON_COLUMNS or frame.empty:
        raise ValueError("primary comparison schema is invalid")
    nonnegative = {
        "paired_count",
        "tab_rmse",
        "xgb_rmse",
        "paired_p_value",
        "holm_p_value",
        "tab_abs_error_win_rate",
        "tab_coverage",
        "xgb_coverage",
        "tab_coverage_ci_lower",
        "tab_coverage_ci_upper",
        "xgb_coverage_ci_lower",
        "xgb_coverage_ci_upper",
    }
    numeric = set(PRIMARY_COMPARISON_COLUMNS).difference(
        {*CELL_COLUMNS, "inference_status", "symmetric_success", "superior", "failed_conditions"}
    )
    for index, row in frame.iterrows():
        unavailable = row["inference_status"] == "implementation_smoke"
        if row["inference_status"] not in {"available", "implementation_smoke"}:
            raise ValueError("invalid inference_status")
        inference_fields = {"paired_p_value", "holm_p_value", "difference_ci_lower", "difference_ci_upper"}
        if unavailable and (
            not all(_missing(row[field]) for field in inference_fields)
            or bool(row["superior"])
            or row["failed_conditions"] != "inference_unavailable"
            or not frame["paired_count"].eq(1).any()
        ):
            raise ValueError("unavailable inference must not contain inferential claims")
        for field in numeric:
            value = row[field]
            if unavailable and field in inference_fields:
                continue
            if field == "rmse_improvement_pct" and _missing(value):
                if _finite_number(row["xgb_rmse"], "xgb_rmse") != 0.0:
                    raise ValueError(
                        "undefined rmse improvement requires zero XGB RMSE"
                    )
                continue
            checked = _finite_number(value, f"comparison {index} {field}")
            if field in nonnegative and checked < 0.0:
                raise ValueError(f"comparison {field} must be nonnegative")
        for field in (
            "paired_p_value",
            "holm_p_value",
            "tab_abs_error_win_rate",
            "tab_coverage",
            "xgb_coverage",
            "tab_coverage_ci_lower",
            "tab_coverage_ci_upper",
            "xgb_coverage_ci_lower",
            "xgb_coverage_ci_upper",
        ):
            if unavailable and field in inference_fields:
                continue
            if not 0.0 <= float(row[field]) <= 1.0:
                raise ValueError(f"comparison {field} must be between 0 and 1")
        if not -1.0 <= float(row["coverage_difference"]) <= 1.0:
            raise ValueError("comparison coverage difference is out of range")
        if not isinstance(row["symmetric_success"], (bool, np.bool_)):
            raise ValueError("comparison symmetric_success must be boolean")
        if not isinstance(row["superior"], (bool, np.bool_)):
            raise ValueError("comparison superior must be boolean")
        if not isinstance(row["failed_conditions"], str):
            raise ValueError("comparison failed_conditions must be a string")
        mean_difference = float(row["mean_squared_error_difference"])
        if not unavailable and not (
            float(row["difference_ci_lower"])
            <= mean_difference
            <= float(row["difference_ci_upper"])
        ):
            raise ValueError("comparison difference interval is invalid")
        for method in ("tab", "xgb"):
            coverage = float(row[f"{method}_coverage"])
            if not (
                float(row[f"{method}_coverage_ci_lower"])
                <= coverage
                <= float(row[f"{method}_coverage_ci_upper"])
            ):
                raise ValueError(f"comparison {method} coverage interval is invalid")


def _method_label(learner_l: str, learner_m: str) -> str:
    return learner_l if learner_l == learner_m else f"{learner_l}/{learner_m}"


def _analysis_record(record: Any, index: int) -> dict[str, Any]:
    if not isinstance(record, Mapping):
        raise ValueError(f"analysis record {index} must be a mapping")
    if record.get("status") != "success":
        raise ValueError(f"analysis record {index} status must be success")
    panel = record.get("panel")
    scenario = record.get("scenario")
    learner_l = record.get("learner_l")
    learner_m = record.get("learner_m")
    if not all(
        isinstance(value, str) and value
        for value in (panel, scenario, learner_l, learner_m)
    ):
        raise ValueError("analysis record labels must be nonempty strings")
    result = {
        "panel": panel,
        "scenario": scenario,
        "n": _native_integer(record.get("n"), "n", minimum=1),
        "p": _native_integer(record.get("p"), "p", minimum=1),
        "replication": _native_integer(record.get("replication"), "replication"),
        "learner_l": learner_l,
        "learner_m": learner_m,
        "method": _method_label(learner_l, learner_m),
    }
    for field in (
        "theta",
        "standard_error",
        "ci_lower",
        "ci_upper",
        "l_mse",
        "m_mse",
        "nuisance_error_product",
        "lm_error_cross",
        "residual_d_variance",
        "bias_numerator_proxy",
        "theta_proxy",
        "proxy_error",
        "runtime_seconds",
    ):
        result[field] = _finite_number(record.get(field), field)
    for field in (
        "standard_error",
        "l_mse",
        "m_mse",
        "nuisance_error_product",
        "residual_d_variance",
        "runtime_seconds",
    ):
        if result[field] < 0:
            raise ValueError(f"{field} must be nonnegative")
    if result["ci_lower"] > result["theta"] or result["theta"] > result["ci_upper"]:
        raise ValueError("analysis record confidence interval is invalid")
    for target in ("l", "m"):
        values = record.get(f"{target}_fold_seconds")
        if not isinstance(values, list) or not values:
            raise ValueError(f"{target}_fold_seconds must be a nonempty list")
        checked = [_finite_number(value, f"{target}_fold_seconds") for value in values]
        if any(value < 0 for value in checked):
            raise ValueError(f"{target}_fold_seconds must be nonnegative")
        result[f"{target}_fit_seconds"] = float(sum(checked))
    peak = record.get("peak_gpu_mb")
    if peak is not None:
        peak = _finite_number(peak, "peak_gpu_mb")
        if peak < 0:
            raise ValueError("peak_gpu_mb must be nonnegative")
    result["peak_gpu_mb"] = peak
    fallback = record.get("fallback_reason")
    if fallback is not None and (not isinstance(fallback, str) or not fallback):
        raise ValueError("fallback_reason must be null or a nonempty string")
    result["fallback_reason"] = fallback
    return result


def aggregate_stage4(
    records: Sequence[Mapping[str, Any]],
    theta0: float = 1.0,
    alpha: float = 0.05,
) -> pd.DataFrame:
    if isinstance(records, (str, bytes)) or not isinstance(records, Sequence):
        raise ValueError("records must be a sequence")
    truth = validate_stage4_theta0(theta0)
    level = validate_stage4_alpha(alpha)
    checked = [_analysis_record(record, index) for index, record in enumerate(records)]
    if not checked:
        raise ValueError("analysis records must not be empty")
    identities = set()
    for record in checked:
        identity = tuple(record[column] for column in CELL_COLUMNS) + (
            record["learner_l"],
            record["learner_m"],
            record["replication"],
        )
        if identity in identities:
            raise ValueError("duplicate analysis cell/method/replication record")
        identities.add(identity)

    frame = pd.DataFrame(checked)
    rows = []
    group_columns = [*CELL_COLUMNS, "method", "learner_l", "learner_m"]
    for keys, group in frame.groupby(group_columns, sort=True, dropna=False):
        estimates = group["theta"].to_numpy(dtype=float)
        errors = estimates - truth
        covered = (
            (group["ci_lower"].to_numpy(dtype=float) <= truth)
            & (group["ci_upper"].to_numpy(dtype=float) >= truth)
        )
        count = len(group)
        coverage_count = int(np.sum(covered))
        coverage_interval = exact_coverage_interval(
            coverage_count, count, alpha=level
        )
        peak_values = group["peak_gpu_mb"].dropna().to_numpy(dtype=float)
        empirical_se = (
            _checked_std(estimates, "empirical SE") if count > 1 else None
        )
        squared_errors = _checked_squares(errors, "aggregate squared errors")
        row = dict(zip(group_columns, keys))
        row.update(
            {
                "replications": count,
                "bias": _checked_mean(errors, "aggregate bias"),
                "rmse": float(
                    np.sqrt(
                        _checked_mean(
                            squared_errors, "aggregate mean squared error"
                        )
                    )
                ),
                "empirical_se": empirical_se,
                "mean_reported_se": _checked_mean(
                    group["standard_error"], "mean reported SE"
                ),
                "coverage": coverage_count / count,
                "coverage_ci_lower": coverage_interval[0],
                "coverage_ci_upper": coverage_interval[1],
                "nominal_coverage_in_exact_interval": (
                    coverage_interval[0] <= 0.95 <= coverage_interval[1]
                ),
                "mean_interval_width": _checked_mean(
                    group["ci_upper"] - group["ci_lower"],
                    "mean interval width",
                ),
                "mean_l_mse": _checked_mean(group["l_mse"], "mean l MSE"),
                "mean_m_mse": _checked_mean(group["m_mse"], "mean m MSE"),
                "mean_nuisance_error_product": _checked_mean(
                    group["nuisance_error_product"],
                    "mean nuisance error product",
                ),
                "mean_lm_error_cross": _checked_mean(
                    group["lm_error_cross"], "mean lm error cross"
                ),
                "mean_residual_d_variance": _checked_mean(
                    group["residual_d_variance"],
                    "mean residual d variance",
                ),
                "mean_bias_numerator_proxy": _checked_mean(
                    group["bias_numerator_proxy"],
                    "mean bias numerator proxy",
                ),
                "mean_theta_proxy": _checked_mean(
                    group["theta_proxy"], "mean theta proxy"
                ),
                "mean_proxy_error": _checked_mean(
                    group["proxy_error"], "mean proxy error"
                ),
                "mean_runtime_seconds": _checked_mean(
                    group["runtime_seconds"], "mean runtime"
                ),
                "mean_l_fit_seconds": _checked_mean(
                    group["l_fit_seconds"], "mean l fit time"
                ),
                "mean_m_fit_seconds": _checked_mean(
                    group["m_fit_seconds"], "mean m fit time"
                ),
                "mean_total_fit_seconds": _checked_mean(
                    group["l_fit_seconds"] + group["m_fit_seconds"],
                    "mean total fit time",
                ),
                "mean_peak_gpu_mb": (
                    _checked_mean(peak_values, "mean peak GPU memory")
                    if len(peak_values)
                    else None
                ),
                "gpu_observation_count": len(peak_values),
                "fallback_count": int(group["fallback_reason"].notna().sum()),
            }
        )
        rows.append(row)
    result = pd.DataFrame(rows, columns=AGGREGATE_COLUMNS)
    _validate_aggregate_frame(result)
    return result


def _validate_aggregate_frame(frame: pd.DataFrame) -> None:
    if list(frame.columns) != AGGREGATE_COLUMNS or frame.empty:
        raise ValueError("aggregate schema is invalid")
    labels = {*CELL_COLUMNS, "method", "learner_l", "learner_m"}
    booleans = {"nominal_coverage_in_exact_interval"}
    signed = {
        "bias",
        "mean_lm_error_cross",
        "mean_bias_numerator_proxy",
        "mean_theta_proxy",
        "mean_proxy_error",
    }
    numeric = set(AGGREGATE_COLUMNS).difference(labels | booleans)
    for index, row in frame.iterrows():
        replications = _finite_number(
            row["replications"], f"aggregate {index} replications"
        )
        gpu_count = _finite_number(
            row["gpu_observation_count"],
            f"aggregate {index} gpu_observation_count",
        )
        for field in numeric:
            value = row[field]
            if field == "empirical_se" and _missing(value):
                if replications != 1.0:
                    raise ValueError(
                        "empirical_se may be missing only for one replication"
                    )
                continue
            if field == "mean_peak_gpu_mb" and _missing(value):
                if gpu_count != 0.0:
                    raise ValueError(
                        "mean_peak_gpu_mb missing with GPU observations"
                    )
                continue
            checked = _finite_number(value, f"aggregate {index} {field}")
            if field not in signed and checked < 0.0:
                raise ValueError(f"aggregate {field} must be nonnegative")
        for field in (
            "coverage",
            "coverage_ci_lower",
            "coverage_ci_upper",
        ):
            if not 0.0 <= float(row[field]) <= 1.0:
                raise ValueError(f"aggregate {field} must be between 0 and 1")
        if not isinstance(
            row["nominal_coverage_in_exact_interval"], (bool, np.bool_)
        ):
            raise ValueError(
                "nominal_coverage_in_exact_interval must be boolean"
            )
        for field in ("replications", "gpu_observation_count", "fallback_count"):
            value = float(row[field])
            if not value.is_integer():
                raise ValueError(f"aggregate {field} must be an integer")
        if replications < 1.0:
            raise ValueError("aggregate replications must be positive")
        if gpu_count > replications or float(row["fallback_count"]) > replications:
            raise ValueError("aggregate diagnostic count exceeds replications")
        coverage = float(row["coverage"])
        lower = float(row["coverage_ci_lower"])
        upper = float(row["coverage_ci_upper"])
        if not lower <= coverage <= upper:
            raise ValueError("aggregate coverage interval is invalid")
        nominal = lower <= 0.95 <= upper
        if bool(row["nominal_coverage_in_exact_interval"]) != nominal:
            raise ValueError("aggregate nominal coverage flag is inconsistent")


def _validate_ranking_frame(frame: pd.DataFrame) -> None:
    if list(frame.columns) != RANKING_COLUMNS or len(frame) != 24:
        raise ValueError("screening ranking schema is invalid")
    if frame[CELL_COLUMNS].duplicated().any():
        raise ValueError("screening ranking contains duplicate cells")
    for index, row in frame.iterrows():
        if not all(isinstance(row[field], str) and row[field] for field in CELL_COLUMNS[:2]):
            raise ValueError("screening ranking labels must be nonempty strings")
        for field in ("n", "p"):
            value = _finite_number(row[field], f"ranking {index} {field}")
            if value < 1.0 or not value.is_integer():
                raise ValueError(f"screening ranking {field} must be positive integer")
        _finite_number(
            row["mean_paired_squared_error_difference"],
            f"ranking {index} score",
        )
        if not isinstance(row["selection_rule"], str) or not row["selection_rule"]:
            raise ValueError("screening ranking selection_rule must be nonempty")


def _validate_analysis_outputs(analysis: Mapping[str, pd.DataFrame]) -> None:
    required = {
        "screening_summary",
        "screening_cell_ranking",
        "confirmation_summary",
        "primary_paired_comparisons",
        "coverage_diagnostics",
        "nuisance_diagnostics",
    }
    if set(analysis) != required:
        raise ValueError("analysis output tables are incomplete")
    _validate_aggregate_frame(analysis["screening_summary"])
    _validate_aggregate_frame(analysis["confirmation_summary"])
    _validate_comparison_frame(analysis["primary_paired_comparisons"])
    _validate_ranking_frame(analysis["screening_cell_ranking"])
    expected_lengths = {
        "screening_summary": 240,
        "screening_cell_ranking": 24,
        "confirmation_summary": 60,
        "primary_paired_comparisons": 6,
        "coverage_diagnostics": 60,
        "nuisance_diagnostics": 60,
    }
    for name, length in expected_lengths.items():
        if len(analysis[name]) != length:
            raise ValueError(f"analysis table {name} has an invalid row count")
    confirmation = analysis["confirmation_summary"]
    expected_diagnostics = {
        "coverage_diagnostics": confirmation[COVERAGE_COLUMNS],
        "nuisance_diagnostics": confirmation[NUISANCE_COLUMNS],
    }
    for name, expected in expected_diagnostics.items():
        frame = analysis[name]
        required_columns = (
            COVERAGE_COLUMNS if name == "coverage_diagnostics" else NUISANCE_COLUMNS
        )
        if list(frame.columns) != required_columns:
            raise ValueError(f"analysis table {name} schema is invalid")
        if not frame.equals(expected):
            raise ValueError(f"analysis table {name} does not match confirmation summary")
    for name, frame in analysis.items():
        if not isinstance(frame, pd.DataFrame) or frame.empty:
            raise ValueError(f"analysis table {name} must be a nonempty frame")
        for column in frame.columns:
            for value in frame[column]:
                if _missing(value):
                    if column not in {
                        "empirical_se",
                        "mean_peak_gpu_mb",
                        "rmse_improvement_pct",
                        "paired_p_value",
                        "holm_p_value",
                        "difference_ci_lower",
                        "difference_ci_upper",
                    }:
                        raise ValueError(
                            f"analysis table {name}/{column} has missing values"
                        )
                elif isinstance(value, (bool, np.bool_)):
                    continue
                elif isinstance(value, (int, float, np.integer, np.floating)):
                    _finite_number(value, f"analysis table {name}/{column}")


def _validate_record_universe(
    records: Sequence[Mapping[str, Any]],
    expected_pairs: Sequence[Any],
    phase: str,
) -> list[Mapping[str, Any]]:
    if isinstance(records, (str, bytes)) or not isinstance(records, Sequence):
        raise ValueError(f"{phase} records must be a sequence")
    expected = {pair.key: pair for pair in expected_pairs}
    if len(expected) != len(expected_pairs):
        raise ValueError(f"expected {phase} task keys are not unique")
    validated: dict[str, Mapping[str, Any]] = {}
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise ValueError(f"{phase} record {index} must be a mapping")
        task_key = record.get("task_key")
        if not isinstance(task_key, str) or task_key not in expected:
            raise ValueError(f"foreign {phase} task_key: {task_key}")
        if task_key in validated:
            raise ValueError(f"duplicate {phase} task_key: {task_key}")
        if "fallback_reason" not in record:
            raise ValueError(
                f"{phase} record {task_key} is missing fallback_reason provenance"
            )
        fallback = record["fallback_reason"]
        if fallback is not None and (
            not isinstance(fallback, str) or not fallback
        ):
            raise ValueError(
                f"{phase} record {task_key} has invalid fallback_reason"
            )
        validate_stage4_record(record, expected[task_key])
        validated[task_key] = record
    missing = set(expected).difference(validated)
    if missing:
        raise ValueError(
            f"incomplete {phase} task universe: missing {len(missing)} records"
        )
    return [validated[pair.key] for pair in expected_pairs]


def _selection_semantic_view(selection: Mapping[str, Any]) -> dict[str, Any]:
    def ordered_rows(name: str) -> list[dict[str, Any]]:
        return sorted(
            (dict(row) for row in selection[name]),
            key=lambda row: (
                row["panel"],
                row["scenario"],
                row["n"],
                row["p"],
            ),
        )

    return {
        key: selection[key]
        for key in selection
        if key not in {"screening_ranking", "cells"}
    } | {
        "screening_ranking": ordered_rows("screening_ranking"),
        "cells": ordered_rows("cells"),
    }


def build_stage4_analysis(
    screening_records: Sequence[Mapping[str, Any]],
    confirmation_records: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    frozen_tuning: Mapping[str, Any],
    selected_confirmation: Mapping[str, Any],
    execution_profile: str = "full",
    alpha: float = 0.05,
) -> dict[str, pd.DataFrame]:
    level = validate_stage4_alpha(alpha)
    if execution_profile not in {"full", "fast"}:
        raise ValueError("execution_profile must be 'full' or 'fast'")
    _validate_fixed_stage4_design(config)
    validate_frozen_tuning(config, frozen_tuning, execution_profile)
    selected_cells = validate_stage4_selection(
        config, selected_confirmation, execution_profile
    )
    screening_replications = (
        1
        if execution_profile == "fast"
        else int(config["screening"]["replications"])
    )
    confirmation_replications = (
        1
        if execution_profile == "fast"
        else int(config["confirmation"]["replications"])
    )
    screening_pairs = tuple(
        iter_stage4_pairs(
            config,
            "screening",
            frozen_tuning,
            replications=screening_replications,
            fast=execution_profile == "fast",
        )
    )
    expected_screening_count = 24 * screening_replications * 10
    expected_confirmation_count = 6 * confirmation_replications * 10
    if len(screening_pairs) != expected_screening_count:
        raise ValueError("strict config did not produce the exact screening universe")
    screening = _validate_record_universe(
        screening_records, screening_pairs, "screening"
    )
    recomputed_selection = select_confirmation_cells(
        screening,
        config,
        frozen_tuning,
        expected_replications=screening_replications,
        execution_profile=execution_profile,
    )
    if _selection_semantic_view(
        selected_confirmation
    ) != _selection_semantic_view(recomputed_selection):
        raise ValueError(
            "selected confirmation selection does not match supplied screening"
        )
    confirmation_pairs = tuple(
        iter_stage4_pairs(
            config,
            "confirmation",
            frozen_tuning,
            selected_confirmation=selected_confirmation,
            replications=confirmation_replications,
            fast=execution_profile == "fast",
        )
    )
    if len(confirmation_pairs) != expected_confirmation_count:
        raise ValueError(
            "strict config did not produce the exact confirmation universe"
        )
    confirmation = _validate_record_universe(
        confirmation_records, confirmation_pairs, "confirmation"
    )
    truth = validate_stage4_theta0(config.get("theta0"))
    screening_summary = aggregate_stage4(screening, truth, level)
    confirmation_summary = aggregate_stage4(confirmation, truth, level)
    primary_records = [
        record
        for pair, record in zip(confirmation_pairs, confirmation, strict=True)
        if pair.learner_l == pair.learner_m
        and pair.learner_l in PRIMARY_METHODS
    ]
    comparisons = paired_primary_comparisons(primary_records, truth, level)
    expected_cells = {
        (cell.panel, cell.scenario, cell.n, cell.p) for cell in selected_cells
    }
    actual_cells = {
        tuple(row[column] for column in CELL_COLUMNS)
        for row in comparisons.to_dict("records")
    }
    if len(comparisons) != 6 or actual_cells != expected_cells:
        raise ValueError("primary comparisons must cover exactly six frozen cells")
    screening_ranking = pd.DataFrame(
        selected_confirmation["screening_ranking"], columns=RANKING_COLUMNS
    ).sort_values(CELL_COLUMNS, kind="stable", ignore_index=True)
    analysis = {
        "screening_summary": screening_summary,
        "screening_cell_ranking": screening_ranking,
        "confirmation_summary": confirmation_summary,
        "primary_paired_comparisons": comparisons,
        "coverage_diagnostics": confirmation_summary[COVERAGE_COLUMNS].copy(),
        "nuisance_diagnostics": confirmation_summary[NUISANCE_COLUMNS].copy(),
    }
    _validate_analysis_outputs(analysis)
    return analysis


def _markdown_comparison_table(comparisons: pd.DataFrame) -> list[str]:
    lines = [
        "| panel | structure | n | p | Tab RMSE | XGB RMSE | 改善率 | 均值平方误差差 | "
        "差值 95% CI | 配对 p | Holm p | Tab 绝对误差胜率 | Tab coverage | "
        "XGB coverage | 优越 | 未满足条件 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in comparisons.to_dict("records"):
        display = dict(row)
        display["superior"] = "是" if row["superior"] else "否"
        unavailable = row["inference_status"] == "implementation_smoke"
        display["difference_interval"] = (
            "不可用" if unavailable else
            f"[{row['difference_ci_lower']:.6g}, {row['difference_ci_upper']:.6g}]"
        )
        display["paired_p"] = "不可用" if unavailable else f"{row['paired_p_value']:.6g}"
        display["holm_p"] = "不可用" if unavailable else f"{row['holm_p_value']:.6g}"
        if unavailable:
            display["superior"] = "未判定"
        improvement = row["rmse_improvement_pct"]
        if _missing(improvement):
            if row["xgb_rmse"] == 0.0 and row["tab_rmse"] == 0.0:
                display["rmse_improvement"] = (
                    "未定义（两者 RMSE 均为 0，结果为平局）"
                )
            elif row["xgb_rmse"] == 0.0 and row["tab_rmse"] > 0.0:
                display["rmse_improvement"] = (
                    "未定义（XGB RMSE 为 0，Tab 明确更差）"
                )
            else:
                raise ValueError("invalid undefined RMSE improvement")
        else:
            display["rmse_improvement"] = f"{float(improvement):.2f}%"
        lines.append(
            "| {panel} | {scenario} | {n} | {p} | {tab_rmse:.6f} | "
            "{xgb_rmse:.6f} | {rmse_improvement} | "
            "{mean_squared_error_difference:.6g} | "
            "{difference_interval} | {paired_p} | {holm_p} | "
            "{tab_abs_error_win_rate:.3f} | {tab_coverage:.3f} | "
            "{xgb_coverage:.3f} | {superior} | {failed_conditions} |".format(
                **display
            )
        )
    return lines


def _nuisance_only_cells(
    comparisons: pd.DataFrame, confirmation_summary: pd.DataFrame
) -> list[tuple[Any, ...]]:
    indexed = {
        tuple(row[column] for column in CELL_COLUMNS) + (row["method"],): row
        for row in confirmation_summary.to_dict("records")
    }
    cells = []
    for comparison in comparisons.to_dict("records"):
        cell = tuple(comparison[column] for column in CELL_COLUMNS)
        tab = indexed[cell + ("tabiclv2_1",)]
        xgb = indexed[cell + ("xgboost_tuned",)]
        nuisance_better = (
            tab["mean_l_mse"] < xgb["mean_l_mse"]
            and tab["mean_m_mse"] < xgb["mean_m_mse"]
        )
        dml_significant = (
            comparison["mean_squared_error_difference"] < 0.0
            and comparison["holm_p_value"] < 0.05
        )
        if nuisance_better and not dml_significant:
            cells.append(cell)
    return cells


def write_stage4_report(
    analysis: Mapping[str, pd.DataFrame], output_path: str | Path
) -> Path:
    comparisons = analysis["primary_paired_comparisons"]
    confirmation = analysis["confirmation_summary"]
    if len(comparisons) != 6:
        raise ValueError("Chinese report requires exactly six primary comparisons")
    if comparisons["inference_status"].eq("implementation_smoke").any():
        lines = [
            "# Stage 4 单次快速流程测试（非正式实验结果）", "",
            "仅验证流程，不作统计推断。配对 p 值、Holm p 值和差值置信区间不可用。",
            "这是 1 次快速实现测试，不是设计中的 5 次预检，也不替代 100 次正式确认。",
            "下列估计和覆盖率仅用于检查输出；不据此判断优越性、显著性或适用边界。", "",
            *_markdown_comparison_table(comparisons), "",
        ]
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lines), encoding="utf-8")
        return path
    qualifying = comparisons[comparisons["superior"].eq(True)]
    count = len(qualifying)
    standard_structures = set(
        qualifying.loc[qualifying["panel"].eq("standard"), "scenario"]
    )
    if count == 0:
        panel_claim = (
            "没有配置满足预设优越性规则；完整负面结果与平滑 DGP 结果共同界定适用边界。"
        )
    elif count == 1:
        panel_claim = "仅一个配置满足条件，因此只作配置特异性报告，不推广为一般树状结论。"
    elif len(standard_structures) >= 2:
        panel_claim = "优势跨越多个标准树结构。"
    elif count >= 2 and qualifying["panel"].eq("small_n_high_p").all():
        panel_claim = "优势主要集中于小样本高维树状环境。"
    else:
        panel_claim = "优势分布不足以支持预设的面板层推广，只按配置报告。"
    nuisance_only = _nuisance_only_cells(comparisons, confirmation)
    lines = [
        "# Stage 4 树状 DGP 确认性分析报告",
        "",
        "## 预先固定的优越性规则",
        "",
        "只有同一冻结配置同时满足以下五项条件，才声明 TabICLv2-1 优于 tuned-XGBoost：",
        "",
        "1. RMSE 改善率至少为 10%（`>= 10%`）；",
        "2. 六项主要比较的 Holm 校正 p 值严格小于 0.05；",
        "3. TabICLv2 coverage 不得比 tuned-XGBoost 低超过 0.05；",
        "4. TabICLv2 coverage 至少为 0.90；",
        "5. 两种方法按相同 replication 完整配对，且十个预定方法组合没有失败、OOM、缺失或静默 fallback。",
        "",
        "规则在查看确认结果前固定，本报告不作事后修改。",
        "",
        "## 六项主要配对比较",
        "",
        *_markdown_comparison_table(comparisons),
        "",
        f"符合全部五项优越性条件的配置数：{count}。",
        "",
        "## 面板层结论",
        "",
        panel_claim,
        "",
        "## Nuisance 诊断解释",
        "",
    ]
    if nuisance_only:
        lines.append(
            f"有 {len(nuisance_only)} 个配置改善了 nuisance prediction，"
            "但未转化为显著的处理效应估计优势。"
        )
    else:
        lines.append(
            "没有配置符合“nuisance prediction 改善但未转化为显著处理效应估计优势”的预定描述。"
        )
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


_METHOD_COLORS = {
    "tabiclv2_1": "#0072B2",
    "tabiclv2_8": "#56B4E9",
    "xgboost": "#E69F00",
    "xgboost_tuned": "#D55E00",
    "extra_trees": "#009E73",
    "oracle": "#CC79A7",
}


def _same_method_summary(summary: pd.DataFrame) -> pd.DataFrame:
    frame = summary[
        summary["learner_l"].eq(summary["learner_m"])
        & summary["method"].isin(OFFICIAL_METHODS)
    ].copy()
    if frame.empty:
        raise ValueError("no same-method confirmation summaries to plot")
    return frame


def _cell_labels(frame: pd.DataFrame) -> tuple[tuple[Any, ...], ...]:
    return tuple(
        sorted(
            {
                tuple(row[column] for column in CELL_COLUMNS)
                for row in frame.to_dict("records")
            },
            key=lambda cell: (str(cell[0]), str(cell[1]), int(cell[2]), int(cell[3])),
        )
    )


def _grouped_bars(
    axis: Any,
    frame: pd.DataFrame,
    cells: Sequence[tuple[Any, ...]],
    value_column: str,
    ylabel: str,
    with_coverage_interval: bool = False,
) -> None:
    x = np.arange(len(cells), dtype=float)
    width = 0.13
    by_identity = {
        tuple(row[column] for column in CELL_COLUMNS) + (row["method"],): row
        for row in frame.to_dict("records")
    }
    for method_index, method in enumerate(OFFICIAL_METHODS):
        values = [by_identity[cell + (method,)][value_column] for cell in cells]
        positions = x + (method_index - 2.5) * width
        yerr = None
        if with_coverage_interval:
            lower = [
                value - by_identity[cell + (method,)]["coverage_ci_lower"]
                for cell, value in zip(cells, values, strict=True)
            ]
            upper = [
                by_identity[cell + (method,)]["coverage_ci_upper"] - value
                for cell, value in zip(cells, values, strict=True)
            ]
            yerr = np.asarray([lower, upper])
        axis.bar(
            positions,
            values,
            width,
            label=method,
            color=_METHOD_COLORS[method],
            yerr=yerr,
            capsize=2 if yerr is not None else 0,
        )
    axis.set_xticks(x)
    axis.set_xticklabels(
        [f"{cell[1]}\nn={cell[2]}, p={cell[3]}" for cell in cells],
        rotation=15,
        ha="right",
    )
    axis.set_ylabel(ylabel)
    axis.grid(axis="y", alpha=0.25)


def write_stage4_figures(
    confirmation_summary: pd.DataFrame, output_dir: str | Path
) -> tuple[Path, Path, Path]:
    frame = _same_method_summary(confirmation_summary)
    panels = tuple(dict.fromkeys(frame["panel"].tolist()))
    if set(panels) != {"standard", "small_n_high_p"}:
        raise ValueError("figures require both prescribed panels")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    figure_paths = (
        output / "dml_rmse_by_panel.png",
        output / "nuisance_mse_by_panel.png",
        output / "coverage_by_panel.png",
    )
    figure, axes = plt.subplots(1, 2, figsize=(15, 5.5), sharey=True)
    for axis, panel in zip(axes, panels, strict=True):
        panel_frame = frame[frame["panel"].eq(panel)]
        cells = tuple(cell for cell in _cell_labels(panel_frame) if cell[0] == panel)
        _grouped_bars(axis, panel_frame, cells, "rmse", "DML RMSE")
        axis.set_title(f"Panel: {panel} | Metric: DML RMSE")
    axes[0].legend(title="Method", fontsize=8)
    figure.tight_layout()
    figure.savefig(figure_paths[0], dpi=180, bbox_inches="tight")
    plt.close(figure)

    figure, axes = plt.subplots(2, 2, figsize=(15, 10), sharex="row")
    for row_index, panel in enumerate(panels):
        panel_frame = frame[frame["panel"].eq(panel)]
        cells = tuple(cell for cell in _cell_labels(panel_frame) if cell[0] == panel)
        for column_index, (metric, label) in enumerate(
            (("mean_l_mse", "l nuisance MSE"), ("mean_m_mse", "m nuisance MSE"))
        ):
            axis = axes[row_index, column_index]
            _grouped_bars(axis, panel_frame, cells, metric, label)
            axis.set_title(f"Panel: {panel} | Metric: {label}")
    axes[0, 0].legend(title="Method", fontsize=8)
    figure.tight_layout()
    figure.savefig(figure_paths[1], dpi=180, bbox_inches="tight")
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(15, 5.5), sharey=True)
    for axis, panel in zip(axes, panels, strict=True):
        panel_frame = frame[frame["panel"].eq(panel)]
        cells = tuple(cell for cell in _cell_labels(panel_frame) if cell[0] == panel)
        _grouped_bars(
            axis,
            panel_frame,
            cells,
            "coverage",
            "Empirical 95% coverage",
            with_coverage_interval=True,
        )
        axis.axhline(0.95, color="#333333", linestyle="--", linewidth=1.2)
        axis.set_ylim(0.0, 1.05)
        axis.set_title(f"Panel: {panel} | Metric: Coverage")
    axes[0].legend(title="Method", fontsize=8)
    figure.tight_layout()
    figure.savefig(figure_paths[2], dpi=180, bbox_inches="tight")
    plt.close(figure)
    return figure_paths


def _write_analysis_bundle(
    analysis: Mapping[str, pd.DataFrame], output: Path
) -> list[Path]:
    _validate_analysis_outputs(analysis)
    outputs = []
    for name in (
        "screening_summary",
        "screening_cell_ranking",
        "confirmation_summary",
        "primary_paired_comparisons",
        "coverage_diagnostics",
        "nuisance_diagnostics",
    ):
        path = output / f"{name}.csv"
        analysis[name].to_csv(
            path,
            index=False,
            float_format="%.12g",
            lineterminator="\n",
            na_rep="NA",
        )
        outputs.append(path)
    outputs.append(write_stage4_report(analysis, output / "analysis_report_zh.md"))
    outputs.extend(
        write_stage4_figures(
            analysis["confirmation_summary"], output / "figures"
        )
    )
    if any(not path.is_file() or path.stat().st_size == 0 for path in outputs):
        raise ValueError("analysis bundle validation failed")
    if len(pd.read_csv(output / "primary_paired_comparisons.csv")) != 6:
        raise ValueError("analysis bundle must contain six primary comparisons")
    return outputs


def write_stage4_analysis(
    screening_records: Sequence[Mapping[str, Any]],
    confirmation_records: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    frozen_tuning: Mapping[str, Any],
    selected_confirmation: Mapping[str, Any],
    output_dir: str | Path,
    execution_profile: str = "full",
    alpha: float = 0.05,
) -> tuple[Path, ...]:
    analysis = build_stage4_analysis(
        screening_records,
        confirmation_records,
        config,
        frozen_tuning,
        selected_confirmation,
        execution_profile=execution_profile,
        alpha=alpha,
    )
    _validate_analysis_outputs(analysis)
    output = Path(output_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.stage-", dir=output.parent)
    )
    backup = staging.with_name(staging.name + ".backup")
    try:
        if output.exists():
            if not output.is_dir():
                raise ValueError("analysis output path must be a directory")
        staged_paths = _write_analysis_bundle(analysis, staging)
        relative_paths = [path.relative_to(staging) for path in staged_paths]
        if output.exists():
            os.replace(output, backup)
            try:
                os.replace(staging, output)
            except Exception:
                os.replace(backup, output)
                raise
            shutil.rmtree(backup)
        else:
            os.replace(staging, output)
        return tuple(output / path for path in relative_paths)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
        if backup.exists() and output.exists():
            shutil.rmtree(backup)
