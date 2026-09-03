from __future__ import annotations

import json
import os
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.model_selection import train_test_split

from .config import derive_seed
from .dgp import simulate_plr
from .learners import make_configured_tree_learner
from .runner import classify_failure
from .sharding import belongs_to_shard, validate_shard
from .stage3b_screen import _params_hash
from .stage4_config import TreeBenchmarkCell, iter_tree_cells
from .storage import ResultStore


@dataclass(frozen=True)
class Stage4TuningTask:
    stage: str
    seed_namespace: str
    panel: str
    scenario: str
    n: int
    p: int
    replication: int
    target: str
    candidate: str
    params: dict[str, Any]
    validation_fraction: float

    def __post_init__(self) -> None:
        if self.target not in {"l", "m"}:
            raise ValueError("target must be 'l' or 'm'")
        if not 0 < self.validation_fraction < 1:
            raise ValueError("validation_fraction must lie between zero and one")

    @property
    def config_hash(self) -> str:
        return _params_hash(self.params)

    @property
    def key(self) -> str:
        return (
            f"{self.stage}__{self.panel}__{self.scenario}__n{self.n}__p{self.p}"
            f"__r{self.replication:03d}__target-{self.target}__{self.candidate}"
            f"__h{self.config_hash}"
        )


def iter_tuning_tasks(
    config: Mapping[str, Any],
    replications: int,
    num_shards: int = 1,
    shard_index: int = 0,
):
    validate_shard(num_shards, shard_index)
    if replications < 1:
        raise ValueError("replications must be at least 1")
    tuning = config["tuning"]
    for cell in iter_tree_cells(config):
        for target in tuning["targets"]:
            for candidate in tuning["xgboost_candidates"]:
                for replication in range(replications):
                    task = Stage4TuningTask(
                        stage=str(tuning["stage"]),
                        seed_namespace=str(tuning["seed_namespace"]),
                        panel=cell.panel,
                        scenario=cell.scenario,
                        n=cell.n,
                        p=cell.p,
                        replication=replication,
                        target=str(target),
                        candidate=str(candidate["name"]),
                        params=dict(candidate["params"]),
                        validation_fraction=float(tuning["validation_fraction"]),
                    )
                    if belongs_to_shard(task.key, num_shards, shard_index):
                        yield task


def run_tuning_task(
    task: Stage4TuningTask,
    theta0: float = 1.0,
    output_root: str | Path = "results/stage4_tree_tuning_raw",
    retry_failed: bool = False,
    fast: bool = False,
) -> dict[str, Any]:
    store = ResultStore(output_root)
    result_path = Path(output_root) / f"{task.key}.json"
    if result_path.exists():
        with result_path.open("r", encoding="utf-8") as handle:
            previous_status = json.load(handle).get("status")
        if previous_status == "success" or not retry_failed:
            return {"task_key": task.key, "status": "skipped"}

    started = time.perf_counter()
    data_seed = derive_seed(
        task.seed_namespace,
        task.panel,
        task.scenario,
        task.n,
        task.p,
        task.replication,
        "data",
    )
    split_seed = derive_seed(
        task.seed_namespace,
        task.panel,
        task.scenario,
        task.n,
        task.p,
        task.replication,
        "tuning_split",
    )
    learner_seed = derive_seed(task.key, "learner")
    metric_name = "validation_y_mse" if task.target == "l" else "validation_d_mse"
    base = {
        "task_key": task.key,
        "stage": task.stage,
        "seed_namespace": task.seed_namespace,
        "panel": task.panel,
        "scenario": task.scenario,
        "n": task.n,
        "p": task.p,
        "replication": task.replication,
        "target": task.target,
        "candidate": task.candidate,
        "learner_kind": "xgboost",
        "params": task.params,
        "config_hash": task.config_hash,
        "validation_fraction": task.validation_fraction,
        "data_seed": data_seed,
        "split_seed": split_seed,
        "learner_seed": learner_seed,
        "selection_metric": metric_name,
    }

    try:
        data = simulate_plr(task.scenario, task.n, task.p, data_seed, theta0)
        train, validation = train_test_split(
            np.arange(task.n),
            test_size=task.validation_fraction,
            random_state=split_seed,
            shuffle=True,
        )
        response = data.y if task.target == "l" else data.d
        truth = data.l0 if task.target == "l" else data.m0
        model = make_configured_tree_learner(
            "xgboost", task.params, learner_seed, fast=fast
        )
        model.fit(data.X[train], response[train])
        prediction = np.asarray(model.predict(data.X[validation]), dtype=float)
        if not np.isfinite(prediction).all():
            raise ValueError("Tuning predictions must be finite")
        record = {
            **base,
            "status": "success",
            "validation_observed_mse": float(
                np.mean((prediction - response[validation]) ** 2)
            ),
            "validation_truth_mse_diagnostic": float(
                np.mean((prediction - truth[validation]) ** 2)
            ),
            "runtime_seconds": time.perf_counter() - started,
        }
    except Exception as error:
        record = {
            **base,
            "status": classify_failure(error),
            "error_type": type(error).__name__,
            "error_message": str(error)[:1000],
            "traceback": traceback.format_exc(limit=8),
            "runtime_seconds": time.perf_counter() - started,
        }
    store.write(record)
    return record


def _cell_key(record: Mapping[str, Any]) -> str:
    return TreeBenchmarkCell(
        str(record["panel"]),
        str(record["scenario"]),
        int(record["n"]),
        int(record["p"]),
    ).key


def select_tuned_xgboost(
    records: Sequence[Mapping[str, Any]],
    expected_replications: int,
    expected_candidates: Sequence[str] | None = None,
    expected_cells: Sequence[TreeBenchmarkCell] | None = None,
) -> dict[str, Any]:
    if expected_replications < 1:
        raise ValueError("expected_replications must be at least 1")
    eligible = [record for record in records if record.get("status") == "success"]
    if not eligible:
        raise ValueError("No successful Stage 4 tuning records")

    grouped: dict[tuple[str, str, str], list[Mapping[str, Any]]] = {}
    for record in eligible:
        key = (_cell_key(record), str(record["target"]), str(record["candidate"]))
        grouped.setdefault(key, []).append(record)

    cell_keys = (
        [cell.key for cell in expected_cells]
        if expected_cells is not None
        else sorted({_cell_key(record) for record in records})
    )
    candidate_names = (
        [str(candidate) for candidate in expected_candidates]
        if expected_candidates is not None
        else sorted({str(record["candidate"]) for record in records})
    )
    for cell_key in cell_keys:
        for target in ("l", "m"):
            for candidate in candidate_names:
                if (cell_key, target, candidate) not in grouped:
                    raise ValueError(
                        f"Incomplete tuning records for {cell_key}/{target}/{candidate}"
                    )

    expected_set = set(range(expected_replications))
    ranked: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for (cell_key, target, candidate), values in grouped.items():
        replications = [int(value["replication"]) for value in values]
        if (
            len(replications) != expected_replications
            or set(replications) != expected_set
        ):
            raise ValueError(
                f"Incomplete tuning records for {cell_key}/{target}/{candidate}"
            )
        configurations = {
            (str(value["config_hash"]), _params_hash(dict(value["params"])))
            for value in values
        }
        if len(configurations) != 1:
            raise ValueError(
                f"Inconsistent tuning configuration for {cell_key}/{target}/{candidate}"
            )
        summary = {
            "candidate": candidate,
            "learner_kind": "xgboost",
            "params": dict(values[0]["params"]),
            "config_hash": str(values[0]["config_hash"]),
            "replications": expected_replications,
            "mean_validation_observed_mse": float(
                np.mean([value["validation_observed_mse"] for value in values])
            ),
            "mean_validation_truth_mse_diagnostic": float(
                np.mean(
                    [value["validation_truth_mse_diagnostic"] for value in values]
                )
            ),
            "selection_metric": (
                "mean_validation_y_mse" if target == "l" else "mean_validation_d_mse"
            ),
        }
        ranked.setdefault((cell_key, target), []).append(summary)

    selected_cells: dict[str, dict[str, Any]] = {}
    for cell_key in cell_keys:
        selected_cells[cell_key] = {}
        for target in ("l", "m"):
            choices = ranked.get((cell_key, target), [])
            if not choices:
                raise ValueError(f"Incomplete tuning records for {cell_key}/{target}")
            selected_cells[cell_key][target] = min(
                choices,
                key=lambda value: (
                    value["mean_validation_observed_mse"],
                    value["candidate"],
                ),
            )

    return {
        "selection_metric_l": "mean_validation_y_mse",
        "selection_metric_m": "mean_validation_d_mse",
        "expected_replications": expected_replications,
        "cells": selected_cells,
    }


def write_tuned_xgboost(
    records: Sequence[Mapping[str, Any]],
    output_path: str | Path,
    expected_replications: int,
    expected_candidates: Sequence[str] | None = None,
    expected_cells: Sequence[TreeBenchmarkCell] | None = None,
) -> dict[str, Any]:
    selected = select_tuned_xgboost(
        records,
        expected_replications,
        expected_candidates=expected_candidates,
        expected_cells=expected_cells,
    )
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(selected, handle, ensure_ascii=False, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return selected
