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
from .stage5_config import stage5_config_fingerprint
from .storage import ResultStore


_EXECUTION_PROFILES = frozenset({"full", "fast"})


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    converted = float(value)
    if not np.isfinite(converted):
        raise ValueError(f"{name} must be a finite number")
    return converted


def _type_strict_equal(actual: Any, expected: Any) -> bool:
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, Mapping):
        return set(actual) == set(expected) and all(
            _type_strict_equal(actual[key], expected[key]) for key in expected
        )
    if isinstance(expected, Sequence) and not isinstance(expected, (str, bytes)):
        return len(actual) == len(expected) and all(
            _type_strict_equal(left, right)
            for left, right in zip(actual, expected)
        )
    return actual == expected


@dataclass(frozen=True)
class Stage5TuningTask:
    stage: str
    seed_namespace: str
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
    config_fingerprint: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "theta0", _finite_number(self.theta0, "theta0"))
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
            f"{self.stage}__{self.scenario}__n{self.n}__p{self.p}"
            f"__r{self.replication:03d}__target-{self.target}__{self.candidate}"
            f"__theta0-{self.theta0.hex()}__profile-{self.execution_profile}"
            f"__h{self.config_hash}"
        )


def derive_stage5_tuning_seeds(task: Stage5TuningTask) -> dict[str, int]:
    shared = (
        task.seed_namespace,
        task.scenario,
        task.n,
        task.p,
        task.replication,
    )
    return {
        "data_seed": derive_seed(*shared, "data"),
        "split_seed": derive_seed(*shared, "tuning_split"),
        "learner_seed": derive_seed(task.key, "learner"),
    }


def _resolve_replications(
    config: Mapping[str, Any], replications: int | None, execution_profile: str
) -> int:
    if execution_profile not in _EXECUTION_PROFILES:
        raise ValueError("execution_profile must be 'full' or 'fast'")
    configured = config["tuning"]["replications"]
    if type(configured) is not int or configured < 1:
        raise ValueError("config tuning.replications must be a positive native integer")
    expected = 1 if execution_profile == "fast" else configured
    value = expected if replications is None else replications
    if type(value) is not int or value != expected:
        raise ValueError(
            f"{execution_profile} execution profile requires exactly "
            f"{expected} replication{'s' if expected != 1 else ''}"
        )
    return value


def iter_stage5_tuning_tasks(
    config: Mapping[str, Any],
    replications: int | None = None,
    execution_profile: str = "full",
    num_shards: int = 1,
    shard_index: int = 0,
):
    validate_shard(num_shards, shard_index)
    count = _resolve_replications(config, replications, execution_profile)
    tuning = config["tuning"]
    targets = tuple(tuning["targets"])
    if targets != ("l", "m"):
        raise ValueError("tuning requires the exact ordered targets ('l', 'm')")
    center = tuning["center"]
    fingerprint = stage5_config_fingerprint(config)
    for scenario in config["scenarios"]:
        for target in targets:
            for candidate in tuning["xgboost_candidates"]:
                for replication in range(count):
                    task = Stage5TuningTask(
                        stage=str(tuning["stage"]),
                        seed_namespace=str(tuning["seed_namespace"]),
                        scenario=str(scenario),
                        n=int(center["n"]),
                        p=int(center["p"]),
                        replication=replication,
                        target=target,
                        candidate=str(candidate["name"]),
                        params=dict(candidate["params"]),
                        validation_fraction=float(tuning["validation_fraction"]),
                        theta0=config["theta0"],
                        execution_profile=execution_profile,
                        config_fingerprint=fingerprint,
                    )
                    if belongs_to_shard(task.key, num_shards, shard_index):
                        yield task


def _record_base(task: Stage5TuningTask) -> dict[str, Any]:
    return {
        "task_key": task.key,
        "stage": task.stage,
        "seed_namespace": task.seed_namespace,
        "scenario": task.scenario,
        "n": task.n,
        "p": task.p,
        "replication": task.replication,
        "target": task.target,
        "candidate": task.candidate,
        "theta0": task.theta0,
        "learner_kind": "xgboost",
        "execution_profile": task.execution_profile,
        "config_fingerprint": task.config_fingerprint,
        "nominal_params": dict(task.params),
        "nominal_config_hash": task.nominal_config_hash,
        "params": task.effective_params,
        "config_hash": task.config_hash,
        "validation_fraction": task.validation_fraction,
        **derive_stage5_tuning_seeds(task),
        "selection_metric": (
            "validation_y_mse" if task.target == "l" else "validation_d_mse"
        ),
    }


def _validate_record_metadata(record: Mapping[str, Any], task: Stage5TuningTask) -> None:
    expected = _record_base(task)
    for field, value in expected.items():
        if not _type_strict_equal(record.get(field), value):
            raise ValueError(f"Invalid Stage 5 tuning record {task.key}: {field} mismatch")
    if _params_hash(dict(record["nominal_params"])) != task.nominal_config_hash:
        raise ValueError(f"Invalid Stage 5 tuning record {task.key}: nominal_config_hash mismatch")
    if _params_hash(dict(record["params"])) != task.config_hash:
        raise ValueError(f"Invalid Stage 5 tuning record {task.key}: config_hash mismatch")


def run_stage5_tuning_task(
    task: Stage5TuningTask,
    output_root: str | Path,
    retry_failed: bool = False,
) -> dict[str, Any]:
    output_root = Path(output_root)
    result_path = output_root / f"{task.key}.json"
    if result_path.exists():
        with result_path.open("r", encoding="utf-8") as handle:
            previous = json.load(handle)
        status = previous.get("status")
        if status == "success":
            _validate_record_metadata(previous, task)
            return {"task_key": task.key, "status": "skipped"}
        if status not in {"failed", "oom"}:
            raise ValueError(f"Invalid Stage 5 tuning record {task.key}: status")
        _validate_record_metadata(previous, task)
        if not retry_failed:
            return {"task_key": task.key, "status": "skipped"}

    store = ResultStore(output_root)
    started = time.perf_counter()
    base = _record_base(task)
    seeds = derive_stage5_tuning_seeds(task)
    try:
        data = simulate_plr(task.scenario, task.n, task.p, seeds["data_seed"], task.theta0)
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
            fast=task.execution_profile == "fast",
        )
        model.fit(data.X[train], response[train])
        prediction = np.asarray(model.predict(data.X[validation]), dtype=float)
        if not np.isfinite(prediction).all():
            raise ValueError("Tuning predictions must be finite")
        record = {
            **base,
            "status": "success",
            "validation_observed_mse": float(np.mean((prediction - response[validation]) ** 2)),
            "validation_truth_mse_diagnostic": float(np.mean((prediction - truth[validation]) ** 2)),
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


def _expected_tasks(
    config: Mapping[str, Any], execution_profile: str
) -> tuple[Stage5TuningTask, ...]:
    replications = _resolve_replications(config, None, execution_profile)
    return tuple(iter_stage5_tuning_tasks(config, replications, execution_profile))


def _run_manifest(
    config: Mapping[str, Any], expected_replications: int, execution_profile: str
) -> dict[str, Any]:
    tasks = tuple(
        iter_stage5_tuning_tasks(config, expected_replications, execution_profile)
    )
    return {
        "schema_version": "stage5_tuning_run_v1",
        "config_fingerprint": stage5_config_fingerprint(config),
        "tuning_stage": config["tuning"]["stage"],
        "tuning_seed_namespace": config["tuning"]["seed_namespace"],
        "theta0": config["theta0"],
        "execution_profile": execution_profile,
        "expected_replications": expected_replications,
        "task_keys": sorted(task.key for task in tasks),
    }


def stage5_tuning_run_fingerprint(
    config: Mapping[str, Any],
    expected_replications: int,
    execution_profile: str,
) -> str:
    _resolve_replications(config, expected_replications, execution_profile)
    payload = json.dumps(
        _run_manifest(config, expected_replications, execution_profile),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def select_stage5_tuning(
    records: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    execution_profile: str,
) -> dict[str, Any]:
    expected_tasks = _expected_tasks(config, execution_profile)
    expected_by_key = {task.key: task for task in expected_tasks}
    candidate_order = {
        candidate["name"]: index
        for index, candidate in enumerate(config["tuning"]["xgboost_candidates"])
    }
    selected: dict[str, Mapping[str, Any]] = {}
    for record in records:
        key = str(record.get("task_key"))
        if key in selected:
            raise ValueError(f"Invalid Stage 5 tuning record: duplicate task_key {key}")
        task = expected_by_key.get(key)
        if task is None:
            if record.get("execution_profile") != execution_profile:
                raise ValueError("Invalid Stage 5 tuning record: execution_profile mismatch")
            raise ValueError(f"Invalid Stage 5 tuning record: unexpected task_key {key}")
        if record.get("status") != "success":
            raise ValueError(f"Failed Stage 5 tuning record: {key}")
        _validate_record_metadata(record, task)
        for metric in ("validation_observed_mse", "validation_truth_mse_diagnostic", "runtime_seconds"):
            _finite_number(record.get(metric), metric)
        selected[key] = record
    missing = set(expected_by_key).difference(selected)
    if missing:
        raise ValueError(f"Incomplete Stage 5 tuning records: missing {len(missing)} task keys")

    grouped: dict[tuple[str, str, str], list[Mapping[str, Any]]] = {}
    for key, record in selected.items():
        task = expected_by_key[key]
        grouped.setdefault((task.scenario, task.target, task.candidate), []).append(record)

    rankings: dict[str, dict[str, list[dict[str, Any]]]] = {}
    winners: dict[str, dict[str, dict[str, Any]]] = {}
    replications = _resolve_replications(config, None, execution_profile)
    for scenario in config["scenarios"]:
        rankings[scenario] = {}
        winners[scenario] = {}
        for target in ("l", "m"):
            summaries = []
            for candidate in candidate_order:
                values = grouped[(scenario, target, candidate)]
                first = values[0]
                summaries.append(
                    {
                        "candidate": candidate,
                        "learner_kind": "xgboost",
                        "execution_profile": execution_profile,
                        "nominal_params": dict(first["nominal_params"]),
                        "nominal_config_hash": first["nominal_config_hash"],
                        "effective_params": dict(first["params"]),
                        "effective_config_hash": first["config_hash"],
                        "replications": replications,
                        "mean_validation_observed_mse": float(np.mean([value["validation_observed_mse"] for value in values])),
                        "mean_validation_truth_mse_diagnostic": float(np.mean([value["validation_truth_mse_diagnostic"] for value in values])),
                        "selection_metric": "mean_validation_y_mse" if target == "l" else "mean_validation_d_mse",
                    }
                )
            summaries.sort(key=lambda value: (value["mean_validation_observed_mse"], candidate_order[value["candidate"]]))
            rankings[scenario][target] = summaries
            winners[scenario][target] = dict(summaries[0])

    return {
        "schema_version": "stage5_tuning_v1",
        "config_fingerprint": stage5_config_fingerprint(config),
        "tuning_stage": config["tuning"]["stage"],
        "tuning_seed_namespace": config["tuning"]["seed_namespace"],
        "tuning_run_fingerprint": stage5_tuning_run_fingerprint(config, replications, execution_profile),
        "theta0": config["theta0"],
        "execution_profile": execution_profile,
        "expected_replications": replications,
        "selection_metric_names": {"l": "mean_validation_y_mse", "m": "mean_validation_d_mse"},
        "candidate_ranking": rankings,
        "scenarios": winners,
    }


def validate_frozen_stage5_tuning(
    frozen: Mapping[str, Any],
    config: Mapping[str, Any],
    execution_profile: str,
) -> Mapping[str, Any]:
    if not isinstance(frozen, Mapping):
        raise ValueError("Frozen Stage 5 tuning artifact must be a mapping")
    replications = _resolve_replications(config, None, execution_profile)
    expected = {
        "schema_version": "stage5_tuning_v1",
        "config_fingerprint": stage5_config_fingerprint(config),
        "tuning_stage": config["tuning"]["stage"],
        "tuning_seed_namespace": config["tuning"]["seed_namespace"],
        "tuning_run_fingerprint": stage5_tuning_run_fingerprint(
            config, replications, execution_profile
        ),
        "theta0": config["theta0"],
        "execution_profile": execution_profile,
        "expected_replications": replications,
        "selection_metric_names": {
            "l": "mean_validation_y_mse",
            "m": "mean_validation_d_mse",
        },
    }
    for field, value in expected.items():
        if not _type_strict_equal(frozen.get(field), value):
            raise ValueError(f"Frozen Stage 5 tuning {field} mismatch")

    candidates = {
        str(value["name"]): dict(value["params"])
        for value in config["tuning"]["xgboost_candidates"]
    }
    rankings = frozen.get("candidate_ranking")
    scenarios = frozen.get("scenarios")
    expected_scenarios = set(config["scenarios"])
    if not isinstance(rankings, Mapping) or set(rankings) != expected_scenarios:
        raise ValueError("Frozen Stage 5 tuning candidate_ranking scenarios mismatch")
    if not isinstance(scenarios, Mapping) or set(scenarios) != expected_scenarios:
        raise ValueError("Frozen Stage 5 tuning scenarios mismatch")
    for scenario in config["scenarios"]:
        if set(rankings[scenario]) != {"l", "m"} or set(scenarios[scenario]) != {"l", "m"}:
            raise ValueError(f"Frozen Stage 5 tuning targets mismatch for {scenario}")
        for target in ("l", "m"):
            values = rankings[scenario][target]
            if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
                raise ValueError(f"Frozen Stage 5 tuning ranking invalid for {scenario}/{target}")
            ranked_names = [value.get("candidate") for value in values]
            if len(ranked_names) != len(candidates) or set(ranked_names) != set(candidates):
                raise ValueError(f"Frozen Stage 5 tuning candidates mismatch for {scenario}/{target}")
            candidate_indices = {name: index for index, name in enumerate(candidates)}
            rank_keys = []
            for value in values:
                candidate = value.get("candidate")
                nominal = value.get("nominal_params")
                effective = value.get("effective_params")
                expected_effective = dict(candidates[candidate])
                if execution_profile == "fast":
                    expected_effective["n_estimators"] = min(
                        int(expected_effective.get("n_estimators", 20)), 20
                    )
                checks = {
                    "learner_kind": "xgboost",
                    "execution_profile": execution_profile,
                    "nominal_params": candidates[candidate],
                    "nominal_config_hash": _params_hash(candidates[candidate]),
                    "effective_params": expected_effective,
                    "effective_config_hash": _params_hash(expected_effective),
                    "replications": replications,
                    "selection_metric": "mean_validation_y_mse" if target == "l" else "mean_validation_d_mse",
                }
                for field, expected_value in checks.items():
                    if not _type_strict_equal(value.get(field), expected_value):
                        raise ValueError(f"Frozen Stage 5 tuning {field} mismatch for {scenario}/{target}")
                for metric in ("mean_validation_observed_mse", "mean_validation_truth_mse_diagnostic"):
                    _finite_number(value.get(metric), metric)
                loss = float(value["mean_validation_observed_mse"])
                rank_keys.append((loss, candidate_indices[candidate]))
            if rank_keys != sorted(rank_keys):
                raise ValueError(f"Frozen Stage 5 tuning ranking order mismatch for {scenario}/{target}")
            if not values or not _type_strict_equal(scenarios[scenario][target], values[0]):
                raise ValueError(f"Frozen Stage 5 tuning winner mismatch for {scenario}/{target}")
    return frozen


def load_stage5_tuning(
    path: str | Path,
    config: Mapping[str, Any],
    execution_profile: str,
) -> Mapping[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        frozen = json.load(handle)
    return validate_frozen_stage5_tuning(frozen, config, execution_profile)


def write_stage5_tuning(
    records: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    output_path: str | Path,
    execution_profile: str,
) -> dict[str, Any]:
    frozen = select_stage5_tuning(records, config, execution_profile)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(frozen, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return frozen
