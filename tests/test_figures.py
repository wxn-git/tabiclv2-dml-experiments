import pandas as pd
import pytest

from tabdml.figures import (
    make_accuracy_cost_figure,
    make_treatment_effect_p_trend_figure,
    prepare_treatment_effect_p_trend_data,
)


def test_accuracy_cost_figure_renders_without_gui(tmp_path):
    summary = pd.DataFrame(
        [
            {
                "learner": "lasso",
                "scenario": "linear",
                "rmse": 0.1,
                "mean_runtime_seconds": 1.0,
            },
            {
                "learner": "tabiclv2",
                "scenario": "linear",
                "rmse": 0.08,
                "mean_runtime_seconds": 5.0,
            },
        ]
    )
    output = make_accuracy_cost_figure(summary, tmp_path)
    assert output.exists()
    assert output.stat().st_size > 0


def test_prepare_treatment_effect_p_trend_data_selects_fixed_n_panels():
    stage2 = pd.DataFrame(
        [
            {"scenario": "linear", "n": 2000, "p": 50, "learner": learner, "replications": 100, "rmse": rmse}
            for learner, rmse in [("tabiclv2_1", 0.10), ("tabiclv2_8", 0.11), ("xgboost", 0.12)]
        ]
        + [
            {"scenario": scenario, "n": n, "p": p, "learner": learner, "replications": 100, "rmse": rmse}
            for scenario, n, dimensions in [("smooth", 1000, [50, 100]), ("tree", 5000, [10, 50])]
            for p in dimensions
            for learner, rmse in [("tabiclv2_1", 0.10), ("tabiclv2_8", 0.11), ("xgboost", 0.12)]
        ]
        + [
            {"scenario": "smooth", "n": 500, "p": 50, "learner": "tabiclv2_1", "replications": 100, "rmse": 9.9},
            {"scenario": "smooth", "n": 1000, "p": 50, "learner": "lasso", "replications": 100, "rmse": 9.9},
        ]
    )
    stage4 = pd.DataFrame(
        [
            {
                "panel": panel,
                "scenario": scenario,
                "n": n,
                "p": p,
                "method": method,
                "replications": 20,
                "rmse": rmse,
            }
            for panel, n in [("standard", 1000), ("small_n_high_p", 500)]
            for scenario in ["tree_stumps", "tree_hierarchical", "tree_forest_sum"]
            for p in [10, 50]
            for method, rmse in [
                ("tabiclv2_1", 0.10),
                ("tabiclv2_8", 0.11),
                ("xgboost", 0.12),
                ("xgboost_tuned", 0.09),
                ("extra_trees", 9.9),
            ]
        ]
    )

    result = prepare_treatment_effect_p_trend_data(stage2, stage4)

    assert len(result) == 39
    assert list(result["panel_key"].drop_duplicates()) == [
        "stage2_linear",
        "stage2_smooth",
        "stage2_tree",
        "stage4_tree_stumps",
        "stage4_tree_hierarchical",
        "stage4_tree_forest_sum",
    ]
    assert set(result["method_label"]) == {
        "TabICLv2-1",
        "TabICLv2-8",
        "XGBoost",
        "Tuned XGBoost",
    }
    assert result.groupby("panel_key")["n"].nunique().eq(1).all()
    assert result.loc[result["stage"] == "Stage 2", "replications"].eq(100).all()
    assert result.loc[result["stage"] == "Stage 4 screening", "replications"].eq(20).all()
    assert result.loc[result["method_label"] == "Tuned XGBoost", "stage"].eq("Stage 4 screening").all()
    assert result["treatment_effect_mse"].equals(result["rmse"].pow(2))
    assert result["treatment_effect_mse"].max() == pytest.approx(0.12**2)


def test_treatment_effect_p_trend_figure_exports_png_pdf_and_csv(tmp_path):
    panel_specs = [
        ("stage2_linear", "Linear", "Stage 2", 2000, 100),
        ("stage2_smooth", "Smooth", "Stage 2", 1000, 100),
        ("stage2_tree", "Original tree", "Stage 2", 5000, 100),
        ("stage4_tree_stumps", "Tree stumps", "Stage 4 screening", 1000, 20),
        ("stage4_tree_hierarchical", "Tree hierarchical", "Stage 4 screening", 1000, 20),
        ("stage4_tree_forest_sum", "Tree forest-sum", "Stage 4 screening", 1000, 20),
    ]
    plot_data = pd.DataFrame(
        [
            {
                "stage": stage,
                "panel_key": panel_key,
                "panel_title": title,
                "n": n,
                "p": p,
                "replications": replications,
                "method": "tabiclv2_1",
                "method_label": "TabICLv2-1",
                "rmse": 0.1,
                "treatment_effect_mse": 0.01,
            }
            for panel_key, title, stage, n, replications in panel_specs
            for p in [10, 50]
        ]
    )

    outputs = make_treatment_effect_p_trend_figure(plot_data, tmp_path)

    assert set(outputs) == {"png", "pdf", "csv"}
    assert all(path.exists() and path.stat().st_size > 0 for path in outputs.values())
    exported = pd.read_csv(outputs["csv"])
    assert len(exported) == len(plot_data)
    assert (exported["treatment_effect_mse"] > 0).all()
