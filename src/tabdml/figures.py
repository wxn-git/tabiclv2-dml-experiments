from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


_P_TREND_METHOD_LABELS = {
    "tabiclv2_1": "TabICLv2-1",
    "tabiclv2_8": "TabICLv2-8",
    "xgboost": "XGBoost",
    "xgboost_tuned": "Tuned XGBoost",
}

_P_TREND_PANELS = (
    {
        "panel_key": "stage2_linear",
        "stage": "Stage 2",
        "scenario": "linear",
        "n": 2000,
        "title": "Linear",
    },
    {
        "panel_key": "stage2_smooth",
        "stage": "Stage 2",
        "scenario": "smooth",
        "n": 1000,
        "title": "Smooth",
    },
    {
        "panel_key": "stage2_tree",
        "stage": "Stage 2",
        "scenario": "tree",
        "n": 5000,
        "title": "Original tree",
    },
    {
        "panel_key": "stage4_tree_stumps",
        "stage": "Stage 4 screening",
        "scenario": "tree_stumps",
        "n": 1000,
        "title": "Tree stumps",
    },
    {
        "panel_key": "stage4_tree_hierarchical",
        "stage": "Stage 4 screening",
        "scenario": "tree_hierarchical",
        "n": 1000,
        "title": "Tree hierarchical",
    },
    {
        "panel_key": "stage4_tree_forest_sum",
        "stage": "Stage 4 screening",
        "scenario": "tree_forest_sum",
        "n": 1000,
        "title": "Tree forest-sum",
    },
)


def make_accuracy_cost_figure(summary: pd.DataFrame, output_dir: str | Path) -> Path:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    successful = summary.dropna(subset=["rmse", "mean_runtime_seconds"])
    if successful.empty:
        raise ValueError("No successful summary rows to plot.")
    plt.figure(figsize=(8, 5))
    sns.scatterplot(
        successful,
        x="mean_runtime_seconds",
        y="rmse",
        hue="learner",
        style="scenario",
    )
    plt.xscale("log")
    plt.tight_layout()
    path = output / "accuracy_cost_pareto.png"
    plt.savefig(path, dpi=180)
    plt.close()
    return path


def prepare_treatment_effect_p_trend_data(
    stage2: pd.DataFrame,
    stage4: pd.DataFrame,
) -> pd.DataFrame:
    """Select the fixed-sample-size cells used by the exploratory p-trend plot."""

    selected: list[pd.DataFrame] = []
    for spec in _P_TREND_PANELS:
        source = stage2 if spec["stage"] == "Stage 2" else stage4
        method_column = "learner" if spec["stage"] == "Stage 2" else "method"
        required = {"scenario", "n", "p", method_column, "rmse"}
        missing = required.difference(source.columns)
        if missing:
            raise ValueError(f"Missing required columns for {spec['panel_key']}: {sorted(missing)}")

        frame = source.copy()
        frame["n"] = pd.to_numeric(frame["n"], errors="raise").astype(int)
        frame["p"] = pd.to_numeric(frame["p"], errors="raise").astype(int)
        frame["rmse"] = pd.to_numeric(frame["rmse"], errors="raise")
        mask = (
            frame["scenario"].eq(spec["scenario"])
            & frame["n"].eq(spec["n"])
            & frame[method_column].isin(_P_TREND_METHOD_LABELS)
        )
        if spec["stage"] != "Stage 2":
            if "panel" not in frame.columns:
                raise ValueError("Missing required Stage 4 column: panel")
            mask &= frame["panel"].eq("standard")
        else:
            mask &= ~frame[method_column].eq("xgboost_tuned")

        panel = frame.loc[mask, ["n", "p", method_column, "rmse"]].copy()
        replication_column = "replications" if "replications" in frame.columns else "success_count"
        if replication_column not in frame.columns:
            raise ValueError(f"Missing replication count for {spec['panel_key']}")
        panel["replications"] = pd.to_numeric(
            frame.loc[mask, replication_column], errors="raise"
        ).astype(int)
        panel = panel.rename(columns={method_column: "method"})
        panel["method_label"] = panel["method"].map(_P_TREND_METHOD_LABELS)
        panel["panel_key"] = spec["panel_key"]
        panel["panel_title"] = spec["title"]
        panel["stage"] = spec["stage"]
        selected.append(panel)

    result = pd.concat(selected, ignore_index=True)
    duplicate_key = ["panel_key", "method", "p"]
    if result.duplicated(duplicate_key).any():
        duplicates = result.loc[result.duplicated(duplicate_key, keep=False), duplicate_key]
        raise ValueError(f"Duplicate plotting rows:\n{duplicates.to_string(index=False)}")
    if not result.groupby("panel_key")["n"].nunique().eq(1).all():
        raise ValueError("Every panel must contain exactly one sample size.")
    if (result["rmse"] <= 0).any():
        raise ValueError("RMSE values must be positive for a logarithmic MSE axis.")

    panel_order = [spec["panel_key"] for spec in _P_TREND_PANELS]
    method_order = list(_P_TREND_METHOD_LABELS)
    result["treatment_effect_mse"] = result["rmse"].pow(2)
    result["_panel_order"] = result["panel_key"].map({key: i for i, key in enumerate(panel_order)})
    result["_method_order"] = result["method"].map({key: i for i, key in enumerate(method_order)})
    result = result.sort_values(["_panel_order", "p", "_method_order"]).drop(
        columns=["_panel_order", "_method_order"]
    )
    return result[
        [
            "stage",
            "panel_key",
            "panel_title",
            "n",
            "p",
            "replications",
            "method",
            "method_label",
            "rmse",
            "treatment_effect_mse",
        ]
    ].reset_index(drop=True)
