from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd
import seaborn as sns

from .stage5_analysis import build_stage5_plot_data


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


def make_treatment_effect_p_trend_figure(
    plot_data: pd.DataFrame,
    output_dir: str | Path,
) -> dict[str, Path]:
    """Render the six-panel exploratory feature-dimension figure."""

    required = {
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
    }
    missing = required.difference(plot_data.columns)
    if missing:
        raise ValueError(f"Missing plotting columns: {sorted(missing)}")
    if plot_data.empty:
        raise ValueError("No rows are available for the p-trend figure.")
    if (pd.to_numeric(plot_data["treatment_effect_mse"], errors="raise") <= 0).any():
        raise ValueError("Treatment-effect MSE must be positive for a logarithmic axis.")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    basename = "treatment_effect_mse_by_p_exploratory"
    paths = {
        "png": output / f"{basename}.png",
        "pdf": output / f"{basename}.pdf",
        "csv": output / f"{basename}_data.csv",
    }
    plot_data.to_csv(paths["csv"], index=False)

    styles = {
        "tabiclv2_1": {"color": "#D55E00", "marker": "o", "linestyle": "-"},
        "tabiclv2_8": {"color": "#CC79A7", "marker": "^", "linestyle": "--"},
        "xgboost": {"color": "#0072B2", "marker": "s", "linestyle": "-"},
        "xgboost_tuned": {"color": "#009E73", "marker": "D", "linestyle": ":"},
    }
    panel_order = [spec["panel_key"] for spec in _P_TREND_PANELS]
    method_order = list(_P_TREND_METHOD_LABELS)

    sns.set_theme(style="whitegrid", context="paper")
    fig, axes = plt.subplots(2, 3, figsize=(13.2, 7.4), sharey=True)
    legend_handles: dict[str, object] = {}
    for panel_index, (axis, panel_key) in enumerate(zip(axes.flat, panel_order, strict=True)):
        panel = plot_data.loc[plot_data["panel_key"].eq(panel_key)].copy()
        if panel.empty:
            raise ValueError(f"Missing panel data: {panel_key}")
        if panel["n"].nunique() != 1 or panel["replications"].nunique() != 1:
            raise ValueError(f"Panel {panel_key} mixes sample sizes or replication counts.")

        for method in method_order:
            series = panel.loc[panel["method"].eq(method)].sort_values("p")
            if series.empty:
                continue
            style = styles[method]
            (line,) = axis.plot(
                series["p"],
                series["treatment_effect_mse"],
                label=series["method_label"].iloc[0],
                color=style["color"],
                marker=style["marker"],
                linestyle=style["linestyle"] if len(series) > 1 else "None",
                linewidth=1.8,
                markersize=6,
                markeredgecolor="white",
                markeredgewidth=0.6,
            )
            legend_handles.setdefault(method, line)

        title = panel["panel_title"].iloc[0]
        stage = panel["stage"].iloc[0]
        n = int(panel["n"].iloc[0])
        replications = int(panel["replications"].iloc[0])
        axis.set_title(f"({chr(97 + panel_index)}) {title}\n{stage}, n={n}, R={replications}", pad=8)
        axis.set_xlabel("Feature dimension p")
        axis.set_xticks(sorted(panel["p"].unique()))
        axis.set_yscale("log")
        axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.4g}"))
        axis.grid(True, which="major", color="#D9D9D9", linewidth=0.7)
        axis.grid(True, which="minor", color="#EEEEEE", linewidth=0.45)
        for spine in axis.spines.values():
            spine.set_color("#333333")
            spine.set_linewidth(0.8)

    axes[0, 0].set_ylabel("Treatment-effect MSE (log scale)")
    axes[1, 0].set_ylabel("Treatment-effect MSE (log scale)")
    ordered_handles = [legend_handles[key] for key in method_order if key in legend_handles]
    ordered_labels = [_P_TREND_METHOD_LABELS[key] for key in method_order if key in legend_handles]
    fig.legend(
        ordered_handles,
        ordered_labels,
        loc="lower center",
        ncol=len(ordered_handles),
        frameon=False,
        bbox_to_anchor=(0.5, 0.02),
    )
    fig.suptitle("Treatment-effect MSE across feature dimensions", fontsize=15, y=0.99)
    fig.subplots_adjust(left=0.08, right=0.99, top=0.88, bottom=0.14, wspace=0.22, hspace=0.42)
    fig.savefig(paths["png"], dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(paths["pdf"], bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return paths


_STAGE5_SCENARIOS = (
    ("linear", "Linear"), ("smooth", "Smooth"), ("tree", "Original tree"),
    ("tree_stumps", "Tree stumps"),
    ("tree_hierarchical", "Hierarchical tree"),
    ("tree_forest_sum", "Forest-sum tree"),
)
_STAGE5_LABELS = {
    "tabiclv2_1": "TabICLv2-1", "tabiclv2_8": "TabICLv2-8",
    "xgboost_tuned": "Tuned XGBoost", "extra_trees": "Extra Trees",
    "lasso": "Lasso", "ensemble": "Ensemble",
}
_STAGE5_STYLES = {
    "tabiclv2_1": ("#D55E00", "o", "-"),
    "tabiclv2_8": ("#CC79A7", "^", "--"),
    "xgboost_tuned": ("#0072B2", "s", "-"),
    "extra_trees": ("#009E73", "D", "-."),
    "lasso": ("#E69F00", "v", ":"),
    "ensemble": ("#6A3D9A", "P", "-"),
}


def _stage5_panel_figure(
    data: pd.DataFrame, *, x: str, metric: str, lower: str | None,
    upper: str | None, title: str, ylabel: str, output: Path, basename: str,
    log_scale: bool = False,
) -> dict[str, Path]:
    paths = {"png": output / f"{basename}.png", "pdf": output / f"{basename}.pdf"}
    sns.set_theme(style="whitegrid", context="paper")
    fig, axes = plt.subplots(2, 3, figsize=(13.4, 7.5), sharey=False)
    handles = {}
    for index, (axis, (scenario, panel_title)) in enumerate(zip(axes.flat, _STAGE5_SCENARIOS, strict=True)):
        panel = data.loc[data["scenario"].eq(scenario)]
        if panel.empty:
            raise ValueError(f"Missing Stage 5 panel: {scenario}")
        for method in _STAGE5_LABELS:
            series = panel.loc[panel["method"].eq(method)].sort_values(x)
            if series.empty:
                raise ValueError(f"Missing Stage 5 method {method} in {scenario}")
            color, marker, linestyle = _STAGE5_STYLES[method]
            xvalues = series[x].to_numpy(dtype=float)
            yvalues = series[metric].to_numpy(dtype=float)
            if not np.isfinite(yvalues).all():
                raise ValueError("Stage 5 plotting values must be finite")
            (line,) = axis.plot(xvalues, yvalues, color=color, marker=marker,
                                linestyle=linestyle, linewidth=1.6, markersize=5,
                                label=_STAGE5_LABELS[method])
            handles.setdefault(method, line)
            if lower and upper:
                lo = series[lower].to_numpy(dtype=float)
                hi = series[upper].to_numpy(dtype=float)
                axis.fill_between(xvalues, lo, hi, color=color, alpha=0.12, linewidth=0)
        axis.set_title(f"({chr(97 + index)}) {panel_title}")
        axis.set_xlabel("Feature dimension p" if x == "p" else "Sample size n")
        axis.set_xticks(sorted(panel[x].unique()))
        if log_scale:
            axis.set_yscale("log")
            axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.4g}"))
        axis.grid(True, which="both", linewidth=0.5, alpha=0.55)
    axes[0, 0].set_ylabel(ylabel)
    axes[1, 0].set_ylabel(ylabel)
    fig.legend([handles[m] for m in _STAGE5_LABELS], list(_STAGE5_LABELS.values()),
               loc="lower center", ncol=6, frameon=False, bbox_to_anchor=(0.5, 0.01))
    fig.suptitle(title, fontsize=15, y=0.99)
    fig.subplots_adjust(left=0.075, right=0.99, top=0.91, bottom=0.13, wspace=0.26, hspace=0.34)
    fig.savefig(paths["png"], dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(paths["pdf"], bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return paths


def make_stage5_sensitivity_figures(summary: pd.DataFrame, output_dir: str | Path) -> dict[str, dict[str, Path]]:
    """Export two primary MSE figures and four two-sweep supplements."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    fixed_n, fixed_p = build_stage5_plot_data(summary)
    if (fixed_n["treatment_mse"] <= 0).any() or (fixed_p["treatment_mse"] <= 0).any():
        raise ValueError("Treatment-effect MSE must be positive for a logarithmic axis")
    fixed_n_path = output / "treatment_effect_mse_fixed_n_data.csv"
    fixed_p_path = output / "treatment_effect_mse_fixed_p_data.csv"
    fixed_n.to_csv(fixed_n_path, index=False)
    fixed_p.to_csv(fixed_p_path, index=False)
    outputs = {
        "fixed_n": _stage5_panel_figure(
            fixed_n, x="p", metric="treatment_mse", lower="mse_ci_lower", upper="mse_ci_upper",
            title="Treatment-effect MSE at fixed sample size (n=1000)",
            ylabel="Treatment-effect MSE (log scale)", output=output,
            basename="treatment_effect_mse_fixed_n", log_scale=True,
        ),
        "fixed_p": _stage5_panel_figure(
            fixed_p, x="n", metric="treatment_mse", lower="mse_ci_lower", upper="mse_ci_upper",
            title="Treatment-effect MSE at fixed feature dimension (p=50)",
            ylabel="Treatment-effect MSE (log scale)", output=output,
            basename="treatment_effect_mse_fixed_p", log_scale=True,
        ),
    }
    outputs["fixed_n"]["csv"] = fixed_n_path
    outputs["fixed_p"]["csv"] = fixed_p_path

    metric_labels = {
        "bias": "Bias", "coverage": "95% CI coverage",
        "l_mse": "Outcome nuisance MSE", "m_mse": "Treatment nuisance MSE",
    }
    # Each supplement is a 2x6 figure: top row fixed n, bottom row fixed p.
    for metric, label in metric_labels.items():
        basename = f"stage5_{metric}_supplement"
        paths = {"png": output / f"{basename}.png", "pdf": output / f"{basename}.pdf"}
        fig, axes = plt.subplots(2, 6, figsize=(19, 6.5), sharey="row")
        handles = {}
        for row, (sweep, x, xlabel) in enumerate((("fixed_n", "p", "p"), ("fixed_p", "n", "n"))):
            source = fixed_n if row == 0 else fixed_p
            for col, (scenario, panel_title) in enumerate(_STAGE5_SCENARIOS):
                axis = axes[row, col]
                panel = source.loc[source["scenario"].eq(scenario)]
                for method in _STAGE5_LABELS:
                    series = panel.loc[panel["method"].eq(method)].sort_values(x)
                    color, marker, linestyle = _STAGE5_STYLES[method]
                    (line,) = axis.plot(series[x], series[metric], color=color, marker=marker,
                                       linestyle=linestyle, linewidth=1.3, markersize=4)
                    handles.setdefault(method, line)
                axis.set_title(panel_title if row == 0 else "")
                axis.set_xlabel(xlabel)
                axis.set_xticks(sorted(panel[x].unique()))
                if metric in {"l_mse", "m_mse"} and (panel[metric] > 0).all():
                    axis.set_yscale("log")
                    axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.4g}"))
                if col == 0:
                    axis.set_ylabel(f"{label}\n{'fixed n=1000' if row == 0 else 'fixed p=50'}")
        fig.legend([handles[m] for m in _STAGE5_LABELS], list(_STAGE5_LABELS.values()),
                   loc="lower center", ncol=6, frameon=False)
        fig.suptitle(f"Stage 5 {label}", fontsize=14)
        fig.subplots_adjust(left=0.055, right=0.995, top=0.88, bottom=0.16, wspace=0.25, hspace=0.35)
        fig.savefig(paths["png"], dpi=300, bbox_inches="tight", facecolor="white")
        fig.savefig(paths["pdf"], bbox_inches="tight", facecolor="white")
        plt.close(fig)
        outputs[metric] = paths
    return outputs
