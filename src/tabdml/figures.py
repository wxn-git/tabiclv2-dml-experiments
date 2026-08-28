from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


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

