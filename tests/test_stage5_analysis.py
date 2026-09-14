from __future__ import annotations

import numpy as np
import pandas as pd

from tabdml.stage5_analysis import (
    bootstrap_mse_interval,
    build_stage5_plot_data,
    estimate_formal_runtime,
    summarize_stage5,
)


SCENARIOS = ["linear", "smooth", "tree", "tree_stumps", "tree_hierarchical", "tree_forest_sum"]
METHODS = ["tabiclv2_1", "tabiclv2_8", "xgboost_tuned", "extra_trees", "lasso", "ensemble"]
CELLS = [(1000, 10), (1000, 50), (1000, 100), (500, 50), (2000, 50)]


def records(replications=5):
    rows = []
    for scenario in SCENARIOS:
        for n, p in CELLS:
            for method_index, method in enumerate(METHODS):
                for replication in range(replications):
                    error = 0.01 * (method_index + 1) + 0.001 * replication
                    rows.append({
                        "scenario": scenario, "n": n, "p": p, "method": method,
                        "replication": replication, "status": "success", "theta0": 1.0,
                        "theta_hat": 1.0 + error, "standard_error": 0.1,
                        "ci_lower": 0.8, "ci_upper": 1.2, "covered": True,
                        "squared_error": error**2, "l_mse": 0.2, "m_mse": 0.3,
                        "runtime_seconds": 1.0 + method_index,
                    })
    return pd.DataFrame(rows)


def test_summary_reports_required_causal_and_nuisance_metrics():
    summary = summarize_stage5(records())
    required = {
        "bias", "rmse", "treatment_mse", "empirical_sd", "mean_se",
        "coverage", "interval_width", "l_mse", "m_mse", "mean_runtime",
        "mse_ci_lower", "mse_ci_upper", "success_count", "failure_count",
    }
    assert required.issubset(summary.columns)
    assert len(summary) == 6 * 5 * 6


def test_bootstrap_is_deterministic_and_uses_replication_values():
    values = np.array([1.0, 4.0, 9.0, 16.0])
    first = bootstrap_mse_interval(values, seed=17, resamples=500)
    second = bootstrap_mse_interval(values, seed=17, resamples=500)
    assert first == second
    assert first[0] <= values.mean() <= first[1]


def test_center_cell_is_identical_in_both_plot_datasets():
    summary = summarize_stage5(records())
    fixed_n, fixed_p = build_stage5_plot_data(summary)
    left = fixed_n.query("n == 1000 and p == 50").sort_values(["scenario", "method"])
    right = fixed_p.query("n == 1000 and p == 50").sort_values(["scenario", "method"])
    pd.testing.assert_frame_equal(left.reset_index(drop=True), right.reset_index(drop=True))
    assert sorted(fixed_n.p.unique()) == [10, 50, 100]
    assert sorted(fixed_p.n.unique()) == [500, 1000, 2000]


def test_runtime_projection_separates_gpu_cpu_and_ensemble_lanes():
    projection = estimate_formal_runtime(records(), formal_replications=100, cpu_workers=5, ensemble_workers=2)
    assert projection["gpu_lane_seconds"] > 0
    assert projection["standard_cpu_lane_seconds"] > 0
    assert projection["ensemble_lane_seconds"] > 0
    assert projection["projected_elapsed_seconds"] >= projection["ensemble_lane_seconds"]
    assert projection["hardware_dependent"] is True

