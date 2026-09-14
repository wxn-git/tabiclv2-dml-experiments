from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd
import pytest

from scripts import analyze_stage4
import tabdml.stage4_analysis as stage4_analysis
from tabdml.config import derive_seed
from tabdml.stage3b_screen import _params_hash
from tabdml.stage4_analysis import (
    AGGREGATE_COLUMNS,
    build_stage4_analysis,
    apply_superiority_rule,
    exact_coverage_interval,
    holm_adjust,
    paired_primary_comparisons,
    write_stage4_analysis,
)
from tabdml.stage4_config import iter_tree_cells, load_stage4_config
from tabdml.stage4_experiment import iter_stage4_pairs
from tabdml.stage4_selection import select_confirmation_cells
from tabdml.stage4_tuning import tuning_run_fingerprint


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "stage4_tree_benchmark.yaml"


@pytest.fixture
def config():
    return load_stage4_config(CONFIG_PATH)


def _frozen_tuning(config, execution_profile="fast"):
    expected_replications = (
        1
        if execution_profile == "fast"
        else config["tuning"]["replications"]
    )
    candidate = config["tuning"]["xgboost_candidates"][0]
    nominal = dict(candidate["params"])
    effective = dict(nominal)
    if execution_profile == "fast":
        effective["n_estimators"] = 20
    cells = {}
    for cell in iter_tree_cells(config):
        cells[cell.key] = {
            target: {
                "candidate": candidate["name"],
                "learner_kind": "xgboost",
                "execution_profile": execution_profile,
                "nominal_params": nominal,
                "nominal_config_hash": _params_hash(nominal),
                "params": effective,
                "config_hash": _params_hash(effective),
                "replications": expected_replications,
                "mean_validation_observed_mse": 1.0,
                "mean_validation_truth_mse_diagnostic": 1.0,
                "selection_metric": (
                    "mean_validation_y_mse"
                    if target == "l"
                    else "mean_validation_d_mse"
                ),
            }
            for target in ("l", "m")
        }
    return {
        "tuning_stage": config["tuning"]["stage"],
        "tuning_seed_namespace": config["tuning"]["seed_namespace"],
        "tuning_run_fingerprint": tuning_run_fingerprint(
            config,
            expected_replications,
            execution_profile,
        ),
        "theta0": config["theta0"],
        "execution_profile": execution_profile,
        "selection_metric_l": "mean_validation_y_mse",
        "selection_metric_m": "mean_validation_d_mse",
        "expected_replications": expected_replications,
        "cells": cells,
    }


def _stage4_record(pair, theta):
    return {
        "task_key": pair.key,
        "stage": pair.stage,
        "seed_namespace": pair.effective_seed_namespace,
        "panel": pair.panel,
        "scenario": pair.scenario,
        "n": pair.n,
        "p": pair.p,
        "replication": pair.replication,
        "learner_l": pair.learner_l,
        "learner_m": pair.learner_m,
        "learner_l_config_hash": pair.learner_l_config_hash,
        "learner_m_config_hash": pair.learner_m_config_hash,
        "folds_count": pair.folds_count,
        "theta0": pair.theta0,
        "execution_profile": pair.execution_profile,
        "data_seed": derive_seed(
            pair.effective_seed_namespace,
            pair.scenario,
            pair.n,
            pair.p,
            pair.replication,
            "data",
        ),
        "fold_seed": derive_seed(
            pair.effective_seed_namespace,
            pair.scenario,
            pair.n,
            pair.p,
            pair.replication,
            "folds",
        ),
        "status": "success",
        "theta": theta,
        "standard_error": 0.04,
        "ci_lower": theta - 0.08,
        "ci_upper": theta + 0.08,
        "l_mse": 0.3,
        "m_mse": 0.2,
        "nuisance_error_product": 0.06,
        "lm_error_cross": 0.01,
        "residual_d_variance": 1.0,
        "bias_numerator_proxy": 0.01,
        "theta_proxy": theta,
        "proxy_error": theta - pair.theta0,
        "runtime_seconds": 0.5,
        "l_fold_seconds": [0.01] * pair.folds_count,
        "m_fold_seconds": [0.02] * pair.folds_count,
        "peak_gpu_mb": 128.0 if pair.learner_l.startswith("tabiclv2") else None,
        "fallback_reason": None,
    }


def _phase_records(
    config,
    frozen,
    phase,
    selected=None,
    execution_profile="fast",
):
    replications = (
        1
        if execution_profile == "fast"
        else config[phase]["replications"]
    )
    records = []
    for pair in iter_stage4_pairs(
        config,
        phase,
        frozen,
        selected_confirmation=selected,
        replications=replications,
        fast=execution_profile == "fast",
    ):
        error = 0.01 + 0.001 * pair.replication
        if pair.learner_l == pair.learner_m == "tabiclv2_1":
            error *= 1.2
        elif pair.learner_l == pair.learner_m == "xgboost_tuned":
            error *= 0.8
        records.append(_stage4_record(pair, config["theta0"] + error))
    return records


@pytest.fixture
def fast_inputs(config):
    frozen = _frozen_tuning(config)
    screening = _phase_records(config, frozen, "screening")
    selected = select_confirmation_cells(
        screening,
        config,
        frozen,
        execution_profile="fast",
    )
    confirmation = _phase_records(
        config,
        frozen,
        "confirmation",
        selected=selected,
    )
    return frozen, screening, selected, confirmation


def _comparison(**overrides):
    values = {
        "rmse_improvement_pct": 12.0,
        "holm_p_value": 0.01,
        "tab_coverage": 0.94,
        "xgb_coverage": 0.95,
        "coverage_difference": -0.01,
        "symmetric_success": True,
    }
    values.update(overrides)
    return values


def _primary_row(method, replication, theta):
    return {
        "status": "success",
        "panel": "standard",
        "scenario": "tree_stumps",
        "n": 1000,
        "p": 10,
        "replication": replication,
        "learner_l": method,
        "learner_m": method,
        "theta": theta,
        "standard_error": 0.03,
        "ci_lower": theta - 0.06,
        "ci_upper": theta + 0.06,
    }


def shuffled_primary_records():
    tab = [
        _primary_row("tabiclv2_1", replication, 1.0 + 0.005 * (-1) ** replication)
        for replication in range(100)
    ]
    xgb = [
        _primary_row(
            "xgboost_tuned", replication, 1.0 + 0.02 * (-1) ** replication
        )
        for replication in range(100)
    ]
    return list(reversed(xgb)) + tab


def test_holm_adjust_is_monotone_in_sorted_order():
    adjusted = holm_adjust([0.01, 0.04, 0.03])
    np.testing.assert_allclose(adjusted, [0.03, 0.06, 0.06])


def test_holm_adjust_preserves_ties_and_rejects_nonfinite_values():
    np.testing.assert_allclose(holm_adjust([0.02, 0.02, 0.5]), [0.06, 0.06, 0.5])
    with pytest.raises(ValueError, match="finite"):
        holm_adjust([0.01, np.nan])


def test_exact_coverage_interval_contains_observed_fraction():
    lower, upper = exact_coverage_interval(94, 100, alpha=0.05)
    assert lower < 0.94 < upper
    assert 0.87 < lower < 0.90
    assert 0.97 < upper < 1.0


def test_exact_coverage_interval_handles_boundaries_and_invalid_alpha():
    assert exact_coverage_interval(0, 100)[0] == 0.0
    assert exact_coverage_interval(100, 100)[1] == 1.0
    with pytest.raises(ValueError, match="alpha"):
        exact_coverage_interval(94, 100, alpha=1.0)


def test_superiority_requires_all_five_conditions():
    passing = _comparison()
    assert apply_superiority_rule(passing)["superior"] is True
    assert apply_superiority_rule({**passing, "tab_coverage": 0.89})[
        "superior"
    ] is False
    assert apply_superiority_rule({**passing, "rmse_improvement_pct": 9.9})[
        "superior"
    ] is False
    assert apply_superiority_rule({**passing, "holm_p_value": 0.05})[
        "superior"
    ] is False
    assert apply_superiority_rule({**passing, "coverage_difference": -0.051})[
        "superior"
    ] is False
    assert apply_superiority_rule({**passing, "symmetric_success": False})[
        "superior"
    ] is False


def test_superiority_thresholds_are_exact_and_ranges_are_validated():
    boundary = _comparison(
        rmse_improvement_pct=10.0,
        tab_coverage=0.90,
        xgb_coverage=0.95,
        coverage_difference=-0.05,
    )
    assert apply_superiority_rule(boundary)["superior"] is True
    with pytest.raises(ValueError, match="between 0 and 1"):
        apply_superiority_rule({**boundary, "tab_coverage": 1.01})


def test_paired_comparison_joins_on_replication_not_row_order():
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        result = paired_primary_comparisons(shuffled_primary_records(), theta0=1.0)
    assert result.iloc[0]["paired_count"] == 100
    assert result.iloc[0]["mean_squared_error_difference"] < 0


def test_paired_comparison_marks_both_zero_rmse_as_undefined_tie():
    records = [
        _primary_row(method, replication, 1.0)
        for method in ("tabiclv2_1", "xgboost_tuned")
        for replication in range(3)
    ]
    result = paired_primary_comparisons(records, theta0=1.0).iloc[0]
    assert result["paired_p_value"] == 1.0
    assert result["difference_ci_lower"] == 0.0
    assert result["difference_ci_upper"] == 0.0
    assert pd.isna(result["rmse_improvement_pct"])
    assert result["superior"] == False  # noqa: E712
    assert "rmse_improvement_undefined" in result["failed_conditions"]


def test_paired_comparison_marks_xgb_zero_tab_positive_as_undefined_loss():
    records = [
        _primary_row("tabiclv2_1", replication, 1.1)
        for replication in range(3)
    ] + [
        _primary_row("xgboost_tuned", replication, 1.0)
        for replication in range(3)
    ]
    result = paired_primary_comparisons(records, theta0=1.0).iloc[0]
    assert pd.isna(result["rmse_improvement_pct"])
    assert result["mean_squared_error_difference"] > 0
    assert result["superior"] == False  # noqa: E712
    assert "rmse_improvement_undefined" in result["failed_conditions"]


def test_paired_comparison_rejects_incomplete_pairs():
    records = shuffled_primary_records()
    records.pop()
    with pytest.raises(ValueError, match="paired replications"):
        paired_primary_comparisons(records, theta0=1.0)


def test_analysis_validates_exact_task5_universes_and_six_holm_tests(
    config, fast_inputs
):
    frozen, screening, selected, confirmation = fast_inputs
    analysis = build_stage4_analysis(
        list(reversed(screening)),
        list(reversed(confirmation)),
        config,
        frozen,
        selected,
        execution_profile="fast",
    )

    assert len(screening) == 24 * 1 * 10
    assert len(confirmation) == 6 * 1 * 10
    assert list(analysis["screening_summary"].columns) == AGGREGATE_COLUMNS
    assert len(analysis["screening_summary"]) == 24 * 10
    assert len(analysis["confirmation_summary"]) == 6 * 10
    assert len(analysis["primary_paired_comparisons"]) == 6
    assert analysis["primary_paired_comparisons"]["paired_count"].eq(1).all()
    comparisons = analysis["primary_paired_comparisons"]
    assert comparisons["inference_status"].eq("implementation_smoke").all()
    for field in ("paired_p_value", "holm_p_value", "difference_ci_lower", "difference_ci_upper"):
        assert comparisons[field].isna().all()
    assert not comparisons["superior"].any()
    assert set(analysis) == {
        "screening_summary",
        "screening_cell_ranking",
        "confirmation_summary",
        "primary_paired_comparisons",
        "coverage_diagnostics",
        "nuisance_diagnostics",
    }
    required_diagnostics = {
        "bias",
        "rmse",
        "empirical_se",
        "mean_reported_se",
        "coverage",
        "mean_interval_width",
        "mean_l_mse",
        "mean_m_mse",
        "mean_lm_error_cross",
        "mean_residual_d_variance",
        "mean_bias_numerator_proxy",
        "mean_runtime_seconds",
        "mean_total_fit_seconds",
        "mean_peak_gpu_mb",
    }
    assert required_diagnostics <= set(analysis["confirmation_summary"].columns)


def test_analysis_rejects_selection_stale_against_supplied_screening(
    config, fast_inputs
):
    frozen, screening, selected, confirmation = fast_inputs
    changed = deepcopy(screening)
    chosen = selected["cells"][0]
    alternative = next(
        cell
        for cell in iter_tree_cells(config)
        if cell.panel == chosen["panel"]
        and cell.scenario == chosen["scenario"]
        and (cell.n, cell.p) != (chosen["n"], chosen["p"])
    )
    tab = next(
        record
        for record in changed
        if record["panel"] == alternative.panel
        and record["scenario"] == alternative.scenario
        and record["n"] == alternative.n
        and record["p"] == alternative.p
        and record["learner_l"] == record["learner_m"] == "tabiclv2_1"
    )
    tab.update(
        theta=1.0,
        theta_proxy=1.0,
        proxy_error=0.0,
        ci_lower=0.92,
        ci_upper=1.08,
    )

    with pytest.raises(ValueError, match="selection.*screening"):
        build_stage4_analysis(
            changed,
            confirmation,
            config,
            frozen,
            selected,
            execution_profile="fast",
        )


def test_analysis_reconstructs_full_20_and_100_replication_universes(config):
    frozen = _frozen_tuning(config, execution_profile="full")
    screening = _phase_records(
        config,
        frozen,
        "screening",
        execution_profile="full",
    )
    selected = select_confirmation_cells(
        screening,
        config,
        frozen,
        execution_profile="full",
    )
    confirmation = _phase_records(
        config,
        frozen,
        "confirmation",
        selected=selected,
        execution_profile="full",
    )
    analysis = build_stage4_analysis(
        screening,
        confirmation,
        config,
        frozen,
        selected,
        execution_profile="full",
    )

    assert len(screening) == 24 * 20 * 10
    assert len(confirmation) == 6 * 100 * 10
    assert analysis["screening_summary"]["replications"].eq(20).all()
    assert analysis["confirmation_summary"]["replications"].eq(100).all()
    assert analysis["primary_paired_comparisons"]["paired_count"].eq(100).all()


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        (None, "theta0", 0.9),
        (None, "theta0", True),
        (None, "theta0", np.nan),
        (None, "folds", 4),
        ("tuning", "replications", 9),
        ("screening", "replications", 19),
        ("confirmation", "replications", 99),
    ],
)
def test_analysis_boundary_requires_fixed_stage4_design_constants(
    section, field, value, config, fast_inputs
):
    frozen, screening, selected, confirmation = fast_inputs
    altered = deepcopy(config)
    target = altered if section is None else altered[section]
    target[field] = value
    with pytest.raises(ValueError, match="fixed Stage 4 design"):
        build_stage4_analysis(
            screening,
            confirmation,
            altered,
            frozen,
            selected,
            execution_profile="fast",
        )


def test_confirmatory_analysis_rejects_non_005_alpha(config, fast_inputs):
    frozen, screening, selected, confirmation = fast_inputs
    with pytest.raises(ValueError, match="alpha.*0.05"):
        build_stage4_analysis(
            screening,
            confirmation,
            config,
            frozen,
            selected,
            execution_profile="fast",
            alpha=0.10,
        )
    with pytest.raises(ValueError, match="alpha.*0.05"):
        stage4_analysis.aggregate_stage4(
            confirmation[:1], theta0=1.0, alpha=0.10
        )
    primary = [
        record
        for record in confirmation
        if record["learner_l"] == record["learner_m"]
        and record["learner_l"] in {"tabiclv2_1", "xgboost_tuned"}
    ]
    with pytest.raises(ValueError, match="alpha.*0.05"):
        paired_primary_comparisons(primary, theta0=1.0, alpha=0.10)
    low, high = exact_coverage_interval(9, 10, alpha=0.10)
    assert 0.0 < low < 0.9 < high < 1.0


def test_statistical_analysis_interfaces_require_theta0_exactly_one(fast_inputs):
    _, _, _, confirmation = fast_inputs
    with pytest.raises(ValueError, match="theta0.*1.0"):
        stage4_analysis.aggregate_stage4(confirmation[:1], theta0=0.9)
    primary = [
        record
        for record in confirmation
        if record["learner_l"] == record["learner_m"]
        and record["learner_l"] in {"tabiclv2_1", "xgboost_tuned"}
    ]
    with pytest.raises(ValueError, match="theta0.*1.0"):
        paired_primary_comparisons(primary, theta0=0.9)


def test_analysis_cli_rejects_non_005_alpha_before_reading_inputs(
    monkeypatch,
):
    monkeypatch.setattr(
        sys,
        "argv",
        ["analyze_stage4.py", "--alpha", "0.10"],
    )
    with pytest.raises(ValueError, match="alpha.*0.05"):
        analyze_stage4.main()


@pytest.mark.parametrize(
    "mutation",
    [
        lambda rows: rows.pop(),
        lambda rows: rows.append(deepcopy(rows[0])),
        lambda rows: rows[0].update(status="failed"),
        lambda rows: rows[0].update(task_key="foreign-task"),
    ],
)
def test_analysis_never_filters_incomplete_duplicate_failed_or_foreign_records(
    config, fast_inputs, mutation
):
    frozen, screening, selected, confirmation = fast_inputs
    invalid = deepcopy(confirmation)
    mutation(invalid)
    with pytest.raises(ValueError):
        build_stage4_analysis(
            screening,
            invalid,
            config,
            frozen,
            selected,
            execution_profile="fast",
        )


def test_analysis_never_filters_invalid_screening_records(config, fast_inputs):
    frozen, screening, selected, confirmation = fast_inputs
    invalid = deepcopy(screening)
    invalid[0]["theta"] = np.nan
    with pytest.raises(ValueError, match="finite"):
        build_stage4_analysis(
            invalid,
            confirmation,
            config,
            frozen,
            selected,
            execution_profile="fast",
        )


@pytest.mark.parametrize(
    "field",
    [
        "standard_error",
        "l_mse",
        "m_mse",
        "nuisance_error_product",
        "residual_d_variance",
        "runtime_seconds",
    ],
)
def test_analysis_rejects_negative_nonnegative_domain_values(
    field, config, fast_inputs
):
    frozen, screening, selected, confirmation = fast_inputs
    invalid = deepcopy(confirmation)
    invalid[0][field] = -0.01
    with pytest.raises(ValueError, match="nonnegative"):
        build_stage4_analysis(
            screening,
            invalid,
            config,
            frozen,
            selected,
            execution_profile="fast",
        )


def test_analysis_keeps_signed_cross_error_and_bias_proxy(config, fast_inputs):
    frozen, screening, selected, confirmation = fast_inputs
    signed = deepcopy(confirmation)
    signed[0]["lm_error_cross"] = -0.25
    signed[0]["bias_numerator_proxy"] = -0.5
    analysis = build_stage4_analysis(
        screening,
        signed,
        config,
        frozen,
        selected,
        execution_profile="fast",
    )
    assert (analysis["confirmation_summary"]["mean_lm_error_cross"] < 0).any()
    assert (
        analysis["confirmation_summary"]["mean_bias_numerator_proxy"] < 0
    ).any()


@pytest.mark.parametrize("field", ["theta", "standard_error", "lm_error_cross"])
def test_aggregate_rejects_overflowed_derived_values(field, fast_inputs):
    _, _, _, confirmation = fast_inputs
    first = confirmation[0]
    records = deepcopy(
        [
            record
            for record in confirmation
            if record["panel"] == first["panel"]
            and record["scenario"] == first["scenario"]
            and record["n"] == first["n"]
            and record["p"] == first["p"]
            and record["learner_l"] == first["learner_l"]
            and record["learner_m"] == first["learner_m"]
        ]
    )
    if len(records) == 1:
        second = deepcopy(records[0])
        second["replication"] = 1
        records.append(second)
    for record in records:
        record[field] = 1e308
        if field == "theta":
            record["ci_lower"] = 1e308
            record["ci_upper"] = 1e308
    with pytest.raises(ValueError, match="overflow|nonfinite"):
        stage4_analysis.aggregate_stage4(records, theta0=1.0)


def test_overflow_fails_before_creating_or_replacing_output(
    tmp_path, config, fast_inputs
):
    frozen, screening, selected, confirmation = fast_inputs
    invalid = deepcopy(confirmation)
    invalid[0].update(theta=1e308, ci_lower=1e308, ci_upper=1e308)
    output = tmp_path / "nested" / "analysis"
    with pytest.raises(ValueError, match="overflow|nonfinite"):
        write_stage4_analysis(
            screening,
            invalid,
            config,
            frozen,
            selected,
            output,
            execution_profile="fast",
        )
    assert not output.exists()


@pytest.mark.parametrize(
    ("table", "field", "value"),
    [
        ("confirmation_summary", "mean_l_mse", -1.0),
        ("primary_paired_comparisons", "holm_p_value", np.inf),
        ("coverage_diagnostics", "coverage", 1.1),
        ("nuisance_diagnostics", "mean_runtime_seconds", -1.0),
    ],
)
def test_writer_revalidates_corrupted_derived_frames_before_output(
    monkeypatch, tmp_path, config, fast_inputs, table, field, value
):
    frozen, screening, selected, confirmation = fast_inputs
    analysis = build_stage4_analysis(
        screening,
        confirmation,
        config,
        frozen,
        selected,
        execution_profile="fast",
    )
    analysis = {name: frame.copy() for name, frame in analysis.items()}
    analysis[table].loc[analysis[table].index[0], field] = value
    monkeypatch.setattr(
        stage4_analysis,
        "build_stage4_analysis",
        lambda *args, **kwargs: analysis,
    )
    output = tmp_path / "analysis"
    with pytest.raises(ValueError, match="finite|nonnegative|range|match|unavailable inference"):
        write_stage4_analysis(
            screening,
            confirmation,
            config,
            frozen,
            selected,
            output,
            execution_profile="fast",
        )
    assert not output.exists()


def test_writer_rejects_invalid_derived_intervals_and_schema(
    monkeypatch, tmp_path, config, fast_inputs
):
    frozen, screening, selected, confirmation = fast_inputs
    original = build_stage4_analysis(
        screening,
        confirmation,
        config,
        frozen,
        selected,
        execution_profile="fast",
    )
    for mutation in ("comparison_interval", "extra_column"):
        analysis = {name: frame.copy() for name, frame in original.items()}
        if mutation == "comparison_interval":
            frame = analysis["primary_paired_comparisons"]
            index = frame.index[0]
            frame.loc[index, "difference_ci_lower"] = (
                frame.loc[index, "mean_squared_error_difference"] + 1.0
            )
        else:
            analysis["coverage_diagnostics"]["unexpected"] = 0.0
        monkeypatch.setattr(
            stage4_analysis,
            "build_stage4_analysis",
            lambda *args, _analysis=analysis, **kwargs: _analysis,
        )
        output = tmp_path / mutation
        with pytest.raises(ValueError, match="interval|schema|unavailable inference"):
            write_stage4_analysis(
                screening,
                confirmation,
                config,
                frozen,
                selected,
                output,
                execution_profile="fast",
            )
        assert not output.exists()


def test_report_distinguishes_undefined_rmse_tie_and_clear_loss(
    tmp_path, config, fast_inputs
):
    frozen, screening, selected, confirmation = fast_inputs
    analysis = build_stage4_analysis(
        screening,
        confirmation,
        config,
        frozen,
        selected,
        execution_profile="fast",
    )
    comparisons = analysis["primary_paired_comparisons"].copy()
    index = comparisons.index[0]
    comparisons.loc[index, ["tab_rmse", "xgb_rmse"]] = [0.0, 0.0]
    comparisons.loc[index, "rmse_improvement_pct"] = np.nan
    analysis["primary_paired_comparisons"] = comparisons
    tie_path = tmp_path / "tie.md"
    stage4_analysis.write_stage4_report(analysis, tie_path)
    assert "未定义（两者 RMSE 均为 0，结果为平局）" in tie_path.read_text(
        encoding="utf-8"
    )

    comparisons.loc[index, "tab_rmse"] = 0.1
    loss_path = tmp_path / "loss.md"
    stage4_analysis.write_stage4_report(analysis, loss_path)
    assert "未定义（XGB RMSE 为 0，Tab 明确更差）" in loss_path.read_text(
        encoding="utf-8"
    )


@pytest.mark.parametrize(
    ("tab_theta", "wording"),
    [
        (1.0, "未定义（两者 RMSE 均为 0，结果为平局）"),
        (1.1, "未定义（XGB RMSE 为 0，Tab 明确更差）"),
    ],
)
def test_zero_xgb_rmse_improvement_is_written_as_na_in_csv(
    tmp_path, config, fast_inputs, tab_theta, wording
):
    frozen, screening, selected, confirmation = fast_inputs
    tied = deepcopy(confirmation)
    cell = selected["cells"][0]
    for record in tied:
        if (
            record["panel"] == cell["panel"]
            and record["scenario"] == cell["scenario"]
            and record["n"] == cell["n"]
            and record["p"] == cell["p"]
            and record["learner_l"] == record["learner_m"]
            and record["learner_l"] in {"tabiclv2_1", "xgboost_tuned"}
        ):
            theta = (
                tab_theta if record["learner_l"] == "tabiclv2_1" else 1.0
            )
            record.update(
                theta=theta,
                theta_proxy=theta,
                proxy_error=theta - 1.0,
                ci_lower=0.92,
                ci_upper=max(1.08, theta),
            )
    output = tmp_path / "analysis"
    write_stage4_analysis(
        screening,
        tied,
        config,
        frozen,
        selected,
        output,
        execution_profile="fast",
    )
    csv = (output / "primary_paired_comparisons.csv").read_text(
        encoding="utf-8"
    )
    report = (output / "analysis_report_zh.md").read_text(encoding="utf-8")
    assert "NA" in csv
    assert wording in report


def test_analysis_rejects_missing_and_reports_explicit_fallback_provenance(
    config, fast_inputs
):
    frozen, screening, selected, confirmation = fast_inputs
    missing = deepcopy(confirmation)
    missing[0].pop("fallback_reason")
    with pytest.raises(ValueError, match="fallback_reason"):
        build_stage4_analysis(
            screening,
            missing,
            config,
            frozen,
            selected,
            execution_profile="fast",
        )

    explicit = deepcopy(confirmation)
    primary = next(
        record
        for record in explicit
        if record["learner_l"] == record["learner_m"] == "tabiclv2_1"
    )
    primary["fallback_reason"] = "recorded fallback"
    analysis = build_stage4_analysis(
        screening,
        explicit,
        config,
        frozen,
        selected,
        execution_profile="fast",
    )
    summary = analysis["confirmation_summary"]
    row = summary[
        summary["panel"].eq(primary["panel"])
        & summary["scenario"].eq(primary["scenario"])
        & summary["n"].eq(primary["n"])
        & summary["p"].eq(primary["p"])
        & summary["method"].eq("tabiclv2_1")
    ].iloc[0]
    assert row["fallback_count"] == 1
    comparison = analysis["primary_paired_comparisons"]
    comparison = comparison[
        comparison["panel"].eq(primary["panel"])
        & comparison["scenario"].eq(primary["scenario"])
        & comparison["n"].eq(primary["n"])
        & comparison["p"].eq(primary["p"])
    ].iloc[0]
    assert comparison["symmetric_success"] == True  # noqa: E712


def test_write_analysis_creates_deterministic_tables_report_and_figures(
    tmp_path, config, fast_inputs
):
    frozen, screening, selected, confirmation = fast_inputs
    output = tmp_path / "analysis"
    output.mkdir()
    (output / "stale.txt").write_text("old bundle", encoding="utf-8")
    written = write_stage4_analysis(
        screening,
        confirmation,
        config,
        frozen,
        selected,
        output,
        execution_profile="fast",
    )

    expected = {
        "screening_summary.csv",
        "screening_cell_ranking.csv",
        "confirmation_summary.csv",
        "primary_paired_comparisons.csv",
        "coverage_diagnostics.csv",
        "nuisance_diagnostics.csv",
        "analysis_report_zh.md",
        "figures/dml_rmse_by_panel.png",
        "figures/nuisance_mse_by_panel.png",
        "figures/coverage_by_panel.png",
    }
    relative = {
        str(path.relative_to(output)).replace("\\", "/") for path in written
    }
    assert relative == expected
    actual = {
        str(path.relative_to(output)).replace("\\", "/")
        for path in output.rglob("*")
        if path.is_file()
    }
    assert actual == expected
    comparison = pd.read_csv(output / "primary_paired_comparisons.csv")
    assert len(comparison) == 6
    report = (output / "analysis_report_zh.md").read_text(encoding="utf-8")
    assert "仅验证流程，不作统计推断" in report
    assert "5 次预检" in report and "100 次正式确认" in report
    assert "共同界定适用边界" not in report
    assert "符合全部五项优越性条件的配置数" not in report
    assert comparison["inference_status"].eq("implementation_smoke").all()
    assert comparison[["paired_p_value", "holm_p_value", "difference_ci_lower", "difference_ci_upper"]].isna().all().all()
    assert report.count("| standard |") == 3
    assert report.count("| small_n_high_p |") == 3
    for figure in expected:
        if figure.endswith(".png"):
            assert (output / figure).stat().st_size > 1_000


def test_report_keeps_one_panel_b_qualifier_configuration_specific(
    tmp_path, config, fast_inputs
):
    frozen = _frozen_tuning(config, execution_profile="full")
    screening = _phase_records(config, frozen, "screening", execution_profile="full")
    selected = select_confirmation_cells(screening, config, frozen, execution_profile="full")
    confirmation = _phase_records(config, frozen, "confirmation", selected=selected, execution_profile="full")
    analysis = build_stage4_analysis(
        screening,
        confirmation,
        config,
        frozen,
        selected,
        execution_profile="full",
    )
    comparisons = analysis["primary_paired_comparisons"].copy()
    comparisons["superior"] = False
    comparisons.loc[
        comparisons["panel"].eq("small_n_high_p").idxmax(), "superior"
    ] = True
    analysis["primary_paired_comparisons"] = comparisons
    report_path = tmp_path / "one-panel-b.md"
    stage4_analysis.write_stage4_report(analysis, report_path)
    report = report_path.read_text(encoding="utf-8")

    assert "仅一个配置满足条件，因此只作配置特异性报告" in report
    assert "优势主要集中于小样本高维树状环境" not in report


def test_failed_validation_creates_no_analysis_output(
    tmp_path, config, fast_inputs
):
    frozen, screening, selected, confirmation = fast_inputs
    output = tmp_path / "analysis"
    with pytest.raises(ValueError):
        write_stage4_analysis(
            screening,
            confirmation[:-1],
            config,
            frozen,
            selected,
            output,
            execution_profile="fast",
        )
    assert not output.exists()


def test_failed_bundle_generation_preserves_previous_complete_output(
    monkeypatch, tmp_path, config, fast_inputs
):
    frozen, screening, selected, confirmation = fast_inputs
    output = tmp_path / "analysis"
    output.mkdir()
    sentinel = output / "previous.txt"
    sentinel.write_text("complete previous analysis", encoding="utf-8")
    monkeypatch.setattr(
        stage4_analysis,
        "write_stage4_figures",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("plot failed")),
    )
    with pytest.raises(RuntimeError, match="plot failed"):
        write_stage4_analysis(
            screening,
            confirmation,
            config,
            frozen,
            selected,
            output,
            execution_profile="fast",
        )
    assert sentinel.read_text(encoding="utf-8") == "complete previous analysis"
    assert not (output / "screening_summary.csv").exists()


def test_analysis_cli_resolves_paths_from_repository_root(
    monkeypatch, tmp_path, config, fast_inputs
):
    frozen, screening, selected, confirmation = fast_inputs
    project = tmp_path / "project"
    (project / "configs").mkdir(parents=True)
    (project / "artifacts").mkdir()
    (project / "raw" / "screening").mkdir(parents=True)
    (project / "raw" / "confirmation").mkdir(parents=True)
    (project / "configs" / "stage4.yaml").write_text(
        CONFIG_PATH.read_text(encoding="utf-8"), encoding="utf-8"
    )
    (project / "artifacts" / "tuning.json").write_text(
        json.dumps(frozen), encoding="utf-8"
    )
    (project / "artifacts" / "selection.json").write_text(
        json.dumps(selected), encoding="utf-8"
    )
    for index, record in enumerate(screening):
        (project / "raw" / "screening" / f"{index}.json").write_text(
            json.dumps(record), encoding="utf-8"
        )
    for index, record in enumerate(confirmation):
        (project / "raw" / "confirmation" / f"{index}.json").write_text(
            json.dumps(record), encoding="utf-8"
        )
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.setattr(
        analyze_stage4,
        "__file__",
        str(project / "scripts" / "analyze_stage4.py"),
    )
    monkeypatch.chdir(elsewhere)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "analyze_stage4.py",
            "--config",
            "configs/stage4.yaml",
            "--screening-root",
            "raw/screening",
            "--confirmation-root",
            "raw/confirmation",
            "--tuned-models",
            "artifacts/tuning.json",
            "--selected-cells",
            "artifacts/selection.json",
            "--output-dir",
            "results/analysis",
            "--fast",
        ],
    )

    assert analyze_stage4.main() == 0
    assert (project / "results" / "analysis" / "analysis_report_zh.md").exists()
    assert not (elsewhere / "results").exists()
