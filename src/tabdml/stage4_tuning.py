from __future__ import annotations

import hashlib
import json
import os
import time
import traceback
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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


_EXECUTION_PROFILES = frozenset({"full", "fast"})


def _exact_finite_theta0(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("theta0 must be an exact finite number")
    try:
        theta0 = float(value)
    except OverflowError as error:
        raise ValueError("theta0 must be an exact finite number") from error
    if not np.isfinite(theta0):
        raise ValueError("theta0 must be an exact finite number")
    return theta0


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
    theta0: float
    execution_profile: str = "full"

    def __post_init__(self) -> None:
        object.__setattr__(self, "theta0", _exact_finite_theta0(self.theta0))
        if self.target not in {"l", "m"}:
            raise ValueError("target must be 'l' or 'm'")
        if not 0 < self.validation_fraction < 1:
            raise ValueError("validation_fraction must lie between zero and one")
        if self.execution_profile not in _EXECUTION_PROFILES:
            raise ValueError("execution_profile must be 'full' or 'fast'")

    @property
    def effective_params(self) -> dict[str, Any]:
        configured = dict(self.params)
        if self.execution_profile == "fast":
            configured["n_estimators"] = min(
                int(configured.get("n_estimators", 20)), 20
            )
        return configured

    @property
    def nominal_config_hash(self) -> str:
        return _params_hash(self.params)

    @property
    def config_hash(self) -> str:
        return _params_hash(self.effective_params)

    @property
    def key(self) -> str:
        return (
            f"{self.stage}__{self.panel}__{self.scenario}__n{self.n}__p{self.p}"
            f"__r{self.replication:03d}__target-{self.target}__{self.candidate}"
            f"__theta0-{self.theta0.hex()}"
            f"__profile-{self.execution_profile}__h{self.config_hash}"
        )


def derive_tuning_seeds(task: Stage4TuningTask) -> dict[str, int]:
    return {
        "data_seed": derive_seed(
            task.seed_namespace,
            task.panel,
            task.scenario,
            task.n,
            task.p,
            task.replication,
            "data",
        ),
        "split_seed": derive_seed(
            task.seed_namespace,
            task.panel,
            task.scenario,
            task.n,
            task.p,
            task.replication,
            "tuning_split",
        ),
        "learner_seed": derive_seed(task.key, "learner"),
    }


def iter_tuning_tasks(
    config: Mapping[str, Any],
    replications: int,
    num_shards: int = 1,
    shard_index: int = 0,
    fast: bool = False,
):
    validate_shard(num_shards, shard_index)
    if replications < 1:
        raise ValueError("replications must be at least 1")
    theta0 = _exact_finite_theta0(config["theta0"])
    tuning = config["tuning"]
    raw_targets = tuning["targets"]
    if isinstance(raw_targets, (str, bytes)) or not isinstance(
        raw_targets, Sequence
    ):
        raise ValueError("tuning requires the exact ordered targets ('l', 'm')")
    targets = tuple(raw_targets)
    if targets != ("l", "m"):
        raise ValueError("tuning requires the exact ordered targets ('l', 'm')")
    for cell in iter_tree_cells(config):
        for target in targets:
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
                        theta0=theta0,
                        execution_profile="fast" if fast else "full",
                    )
                    if belongs_to_shard(task.key, num_shards, shard_index):
                        yield task


def run_tuning_task(
    task: Stage4TuningTask,
    output_root: str | Path = "results/stage4_tree_tuning_raw",
    retry_failed: bool = False,
    fast: bool | None = None,
) -> dict[str, Any]:
    task_fast = task.execution_profile == "fast"
    if fast is not None and fast != task_fast:
        raise ValueError("fast must match the task execution profile")
    store = ResultStore(output_root)
    result_path = Path(output_root) / f"{task.key}.json"
    if result_path.exists():
        with result_path.open("r", encoding="utf-8") as handle:
            previous = json.load(handle)
        previous_status = previous.get("status")
        if previous_status == "success":
            _validate_record_metadata(previous, task)
            return {"task_key": task.key, "status": "skipped"}
        if not retry_failed:
            return {"task_key": task.key, "status": "skipped"}

    started = time.perf_counter()
    seeds = derive_tuning_seeds(task)
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
        "theta0": task.theta0,
        "learner_kind": "xgboost",
        "execution_profile": task.execution_profile,
        "nominal_params": task.params,
        "nominal_config_hash": task.nominal_config_hash,
        "params": task.effective_params,
        "config_hash": task.config_hash,
        "validation_fraction": task.validation_fraction,
        **seeds,
        "selection_metric": metric_name,
    }

    try:
        data = simulate_plr(
            task.scenario, task.n, task.p, seeds["data_seed"], task.theta0
        )
        train, validation = train_test_split(
            np.arange(task.n),
            test_size=task.validation_fraction,
            random_state=seeds["split_seed"],
            shuffle=True,
        )
        response = data.y if task.target == "l" else data.d
        truth = data.l0 if task.target == "l" else data.m0
        model = make_configured_tree_learner(
            "xgboost",
            task.effective_params,
            seeds["learner_seed"],
            fast=task_fast,
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


def _validate_record_metadata(
    record: Mapping[str, Any], task: Stage4TuningTask
) -> None:
    try:
        record_theta0 = _exact_finite_theta0(record.get("theta0"))
    except ValueError as error:
        raise ValueError(
            f"Invalid tuning record {task.key}: theta0 mismatch"
        ) from error
    if record_theta0 != task.theta0:
        raise ValueError(f"Invalid tuning record {task.key}: theta0 mismatch")
    expected_fields = {
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
        "theta0": task.theta0,
        "learner_kind": "xgboost",
        "execution_profile": task.execution_profile,
        "validation_fraction": task.validation_fraction,
        "nominal_config_hash": task.nominal_config_hash,
        "config_hash": task.config_hash,
        **derive_tuning_seeds(task),
    }
    for field, expected in expected_fields.items():
        if record.get(field) != expected:
            raise ValueError(f"Invalid tuning record {task.key}: {field} mismatch")
    if record.get("nominal_params") != task.params:
        raise ValueError(f"Invalid tuning record {task.key}: nominal_params mismatch")
    if record.get("params") != task.effective_params:
        raise ValueError(f"Invalid tuning record {task.key}: params mismatch")
    params = record.get("params")
    if (
        not isinstance(params, Mapping)
        or _params_hash(dict(params)) != task.config_hash
    ):
        raise ValueError(f"Invalid tuning record {task.key}: config_hash mismatch")


def _validate_other_profile_record(
    record: Mapping[str, Any],
    templates: Mapping[tuple[Any, ...], Stage4TuningTask],
    execution_profile: str,
) -> None:
    if execution_profile not in _EXECUTION_PROFILES:
        raise ValueError("Invalid tuning record: unknown execution_profile")
    identity = (
        record.get("panel"),
        record.get("scenario"),
        record.get("n"),
        record.get("p"),
        record.get("target"),
        record.get("candidate"),
    )
    template = templates.get(identity)
    replication = record.get("replication")
    if (
        template is None
        or isinstance(replication, bool)
        or not isinstance(replication, int)
        or replication < 0
    ):
        raise ValueError("Invalid tuning record from other execution profile")
    other_task = Stage4TuningTask(
        stage=template.stage,
        seed_namespace=template.seed_namespace,
        panel=template.panel,
        scenario=template.scenario,
        n=template.n,
        p=template.p,
        replication=replication,
        target=template.target,
        candidate=template.candidate,
        params=template.params,
        validation_fraction=template.validation_fraction,
        theta0=template.theta0,
        execution_profile=execution_profile,
    )
    _validate_record_metadata(record, other_task)


def _canonical_tuning_run_manifest(
    expected_tasks: Sequence[Stage4TuningTask],
    expected_replications: int,
) -> dict[str, Any]:
    if (
        isinstance(expected_replications, bool)
        or not isinstance(expected_replications, int)
        or expected_replications < 1
    ):
        raise ValueError("expected_replications must be at least 1")
    tasks = tuple(expected_tasks)
    if not tasks:
        raise ValueError("expected_tasks must not be empty")

    task_keys = [task.key for task in tasks]
    if len(task_keys) != len(set(task_keys)):
        raise ValueError("Expected tuning task keys must be unique")

    profiles = {task.execution_profile for task in tasks}
    stages = {task.stage for task in tasks}
    seed_namespaces = {task.seed_namespace for task in tasks}
    validation_fractions = {task.validation_fraction for task in tasks}
    theta0_values = {task.theta0 for task in tasks}
    if any(
        len(values) != 1
        for values in (
            profiles,
            stages,
            seed_namespaces,
            validation_fractions,
            theta0_values,
        )
    ):
        raise ValueError(
            "Expected tuning tasks must use one exact stage, seed namespace, "
            "theta0, validation fraction, and execution profile"
        )

    cells = sorted(
        {(task.panel, task.scenario, task.n, task.p) for task in tasks},
        key=lambda cell: (cell[0], cell[1], cell[2], cell[3]),
    )
    targets = ("l", "m")
    candidate_specs: dict[str, dict[str, Any]] = {}
    for task in tasks:
        candidate_spec = {
            "candidate": task.candidate,
            "nominal_params": dict(task.params),
            "nominal_config_hash": task.nominal_config_hash,
            "effective_params": task.effective_params,
            "effective_config_hash": task.config_hash,
        }
        previous = candidate_specs.setdefault(task.candidate, candidate_spec)
        if previous != candidate_spec:
            raise ValueError(
                f"Expected tuning candidate {task.candidate} has mixed configurations"
            )
    candidate_names = sorted(candidate_specs)

    expected_identities = {
        (*cell, target, candidate, replication)
        for cell in cells
        for target in targets
        for candidate in candidate_names
        for replication in range(expected_replications)
    }
    actual_identities = {
        (
            task.panel,
            task.scenario,
            task.n,
            task.p,
            task.target,
            task.candidate,
            task.replication,
        )
        for task in tasks
    }
    if len(actual_identities) != len(tasks) or actual_identities != expected_identities:
        raise ValueError(
            "Expected tuning tasks must form the exact cell, target, candidate, "
            "and replication product"
        )

    task_manifest = []
    for task in sorted(tasks, key=lambda value: value.key):
        task_manifest.append(
            {
                "task_key": task.key,
                "stage": task.stage,
                "seed_namespace": task.seed_namespace,
                "execution_profile": task.execution_profile,
                "panel": task.panel,
                "scenario": task.scenario,
                "n": task.n,
                "p": task.p,
                "replication": task.replication,
                "target": task.target,
                "candidate": task.candidate,
                "theta0": task.theta0,
                "nominal_params": dict(task.params),
                "nominal_config_hash": task.nominal_config_hash,
                "effective_params": task.effective_params,
                "effective_config_hash": task.config_hash,
                "validation_fraction": task.validation_fraction,
                **derive_tuning_seeds(task),
            }
        )

    return {
        "schema": "stage4_tuning_run_v2",
        "stage": next(iter(stages)),
        "seed_namespace": next(iter(seed_namespaces)),
        "theta0": next(iter(theta0_values)),
        "execution_profile": next(iter(profiles)),
        "replications": expected_replications,
        "validation_fraction": next(iter(validation_fractions)),
        "cells": [
            {"panel": panel, "scenario": scenario, "n": n, "p": p}
            for panel, scenario, n, p in cells
        ],
        "targets": list(targets),
        "candidates": [candidate_specs[name] for name in candidate_names],
        "tasks": task_manifest,
    }


def tuning_task_universe_fingerprint(
    expected_tasks: Sequence[Stage4TuningTask],
    expected_replications: int,
) -> str:
    manifest = _canonical_tuning_run_manifest(
        expected_tasks,
        expected_replications,
    )
    return _tuning_run_manifest_fingerprint(manifest)


def _tuning_run_manifest_fingerprint(manifest: Mapping[str, Any]) -> str:
    payload = json.dumps(
        manifest,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def tuning_run_fingerprint(
    config: Mapping[str, Any],
    replications: int,
    execution_profile: str = "full",
) -> str:
    if execution_profile not in _EXECUTION_PROFILES:
        raise ValueError("execution_profile must be 'full' or 'fast'")
    expected_tasks = tuple(
        iter_tuning_tasks(
            config,
            replications,
            fast=execution_profile == "fast",
        )
    )
    return tuning_task_universe_fingerprint(expected_tasks, replications)


def select_tuned_xgboost(
    records: Sequence[Mapping[str, Any]],
    expected_replications: int,
    expected_tasks: Sequence[Stage4TuningTask] | None = None,
    expected_candidates: Sequence[str] | None = None,
    expected_cells: Sequence[TreeBenchmarkCell] | None = None,
) -> dict[str, Any]:
    if expected_replications < 1:
        raise ValueError("expected_replications must be at least 1")
    if expected_tasks is None:
        raise ValueError("expected_tasks is required for exact selection validation")
    expected_tasks = tuple(expected_tasks)
    if not expected_tasks:
        raise ValueError("expected_tasks must not be empty")
    if expected_candidates is not None or expected_cells is not None:
        raise ValueError("expected_tasks replaces expected_candidates and expected_cells")

    tuning_run_manifest = _canonical_tuning_run_manifest(
        expected_tasks,
        expected_replications,
    )
    tuning_run_identity = _tuning_run_manifest_fingerprint(tuning_run_manifest)

    expected_by_key = {task.key: task for task in expected_tasks}
    if len(expected_by_key) != len(expected_tasks):
        raise ValueError("Expected tuning task keys must be unique")
    profiles = {task.execution_profile for task in expected_tasks}
    stages = {task.stage for task in expected_tasks}
    seed_namespaces = {task.seed_namespace for task in expected_tasks}
    validation_fractions = {task.validation_fraction for task in expected_tasks}
    theta0_values = {task.theta0 for task in expected_tasks}
    expected_attributes = (
        profiles,
        stages,
        seed_namespaces,
        validation_fractions,
        theta0_values,
    )
    if any(len(values) != 1 for values in expected_attributes):
        raise ValueError(
            "Expected tuning tasks must use one exact stage, seed namespace, "
            "theta0, validation fraction, and execution profile"
        )
    expected_profile = next(iter(profiles))

    cell_keys = list(
        dict.fromkeys(
            TreeBenchmarkCell(task.panel, task.scenario, task.n, task.p).key
            for task in expected_tasks
        )
    )
    candidate_names = list(dict.fromkeys(task.candidate for task in expected_tasks))
    other_profile_templates = {
        (
            task.panel,
            task.scenario,
            task.n,
            task.p,
            task.target,
            task.candidate,
        ): task
        for task in expected_tasks
    }
    for cell_key in cell_keys:
        for target in ("l", "m"):
            for candidate in candidate_names:
                replications = {
                    task.replication
                    for task in expected_tasks
                    if TreeBenchmarkCell(
                        task.panel, task.scenario, task.n, task.p
                    ).key
                    == cell_key
                    and task.target == target
                    and task.candidate == candidate
                }
                if replications != set(range(expected_replications)):
                    raise ValueError(
                        f"Incomplete expected tuning universe for {cell_key}/{target}/{candidate}"
                    )

    selected_records: dict[str, Mapping[str, Any]] = {}
    for record in records:
        profile = record.get("execution_profile")
        if profile != expected_profile:
            _validate_other_profile_record(
                record,
                other_profile_templates,
                str(profile),
            )
            continue
        task_key = str(record.get("task_key"))
        task = expected_by_key.get(task_key)
        if task is None:
            raise ValueError(f"Invalid tuning record: unexpected task_key {task_key}")
        if task_key in selected_records:
            raise ValueError(f"Invalid tuning record: duplicate task_key {task_key}")
        if record.get("status") != "success":
            raise ValueError(f"Failed tuning record in expected profile: {task_key}")
        _validate_record_metadata(record, task)
        for metric in (
            "validation_observed_mse",
            "validation_truth_mse_diagnostic",
        ):
            value = record.get(metric)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not np.isfinite(value)
            ):
                raise ValueError(f"Invalid tuning record {task_key}: {metric}")
        selected_records[task_key] = record

    missing = set(expected_by_key).difference(selected_records)
    if missing:
        raise ValueError(
            f"Incomplete tuning records: missing {len(missing)} expected task keys"
        )

    grouped: dict[tuple[str, str, str], list[Mapping[str, Any]]] = {}
    for task_key, record in selected_records.items():
        task = expected_by_key[task_key]
        key = (
            TreeBenchmarkCell(task.panel, task.scenario, task.n, task.p).key,
            task.target,
            task.candidate,
        )
        grouped.setdefault(key, []).append(record)

    ranked: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for (cell_key, target, candidate), values in grouped.items():
        values = sorted(values, key=lambda value: int(value["replication"]))
        summary = {
            "candidate": candidate,
            "learner_kind": "xgboost",
            "execution_profile": expected_profile,
            "nominal_params": dict(values[0]["nominal_params"]),
            "nominal_config_hash": str(values[0]["nominal_config_hash"]),
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
        "tuning_stage": tuning_run_manifest["stage"],
        "tuning_seed_namespace": tuning_run_manifest["seed_namespace"],
        "tuning_run_fingerprint": tuning_run_identity,
        "theta0": tuning_run_manifest["theta0"],
        "execution_profile": expected_profile,
        "selection_metric_l": "mean_validation_y_mse",
        "selection_metric_m": "mean_validation_d_mse",
        "expected_replications": expected_replications,
        "cells": selected_cells,
    }


def write_tuned_xgboost(
    records: Sequence[Mapping[str, Any]],
    output_path: str | Path,
    expected_replications: int,
    expected_tasks: Sequence[Stage4TuningTask] | None = None,
    expected_candidates: Sequence[str] | None = None,
    expected_cells: Sequence[TreeBenchmarkCell] | None = None,
) -> dict[str, Any]:
    selected = select_tuned_xgboost(
        records,
        expected_replications,
        expected_tasks=expected_tasks,
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
