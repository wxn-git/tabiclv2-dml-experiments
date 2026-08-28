from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from .config import ExperimentConfig, TaskSpec


def enumerate_stage1_tasks(config: ExperimentConfig) -> Iterable[TaskSpec]:
    for scenario in config.scenarios:
        for n in config.sample_sizes:
            for p in config.dimensions:
                for replication in range(config.replications):
                    for learner in config.learners:
                        estimators = config.tabicl_estimators if learner == "tabiclv2" else 0
                        yield TaskSpec(
                            config.stage, scenario, n, p, replication, learner, estimators
                        )


def select_stage2(
    summary: pd.DataFrame,
    baseline: tuple[str, int, int] = ("linear", 2000, 50),
) -> list[dict]:
    traditional = summary[~summary["learner"].str.startswith("tabiclv2")]
    best = (
        traditional.groupby(["scenario", "n", "p"], as_index=False)["rmse"]
        .min()
        .rename(columns={"rmse": "traditional_rmse"})
    )
    tab = (
        summary[summary["learner"].str.startswith("tabiclv2")]
        .groupby(["scenario", "n", "p"], as_index=False)["rmse"]
        .min()
        .rename(columns={"rmse": "tabicl_rmse"})
    )
    ranked = best.merge(tab, on=["scenario", "n", "p"], how="inner")
    ranked["difference"] = ranked["tabicl_rmse"] - ranked["traditional_rmse"]

    selected: list[tuple[str, int, int]] = []
    baseline_key = (str(baseline[0]), int(baseline[1]), int(baseline[2]))
    for frame in (ranked.nsmallest(len(ranked), "difference"), ranked.nlargest(len(ranked), "difference")):
        target = 3 if len(selected) < 3 else 6
        for row in frame.itertuples(index=False):
            key = (str(row.scenario), int(row.n), int(row.p))
            if key != baseline_key and key not in selected:
                selected.append(key)
            if len(selected) == target:
                break
    selected = selected[:6]
    if baseline_key not in selected:
        selected.append(baseline_key)
    for row in ranked.itertuples(index=False):
        key = (str(row.scenario), int(row.n), int(row.p))
        if key not in selected:
            selected.append(key)
        if len(selected) == 7:
            break
    return [{"scenario": s, "n": n, "p": p} for s, n, p in selected[:7]]

