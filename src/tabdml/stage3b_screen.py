from __future__ import annotations

import hashlib
import json
import os
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from .config import derive_seed
from .crossfit import _cuda_helpers
from .dgp import simulate_plr
from .learners import make_configured_tree_learner, make_learner
from .runner import classify_failure
from .sharding import replication_belongs_to_shard, validate_shard
from .storage import ResultStore


def _params_hash(params: dict) -> str:
    payload = json.dumps(params, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


@dataclass(frozen=True)
class ScreeningTaskSpec:
    stage: str
    seed_namespace: str
    scenario: str
    n: int
    p: int
    replication: int
    candidate: str
    candidate_group: str
    learner_kind: str
    params: dict
    training_target: str
    validation_fraction: float

    def __post_init__(self):
        if self.training_target not in {"d", "m0"}:
            raise ValueError("training_target must be 'd' or 'm0'.")
        if not 0 < self.validation_fraction < 1:
            raise ValueError("validation_fraction must lie between zero and one.")

    @property
    def config_hash(self) -> str:
        return _params_hash(
            {
                "candidate_group": self.candidate_group,
                "learner_kind": self.learner_kind,
                "params": self.params,
                "training_target": self.training_target,
                "validation_fraction": self.validation_fraction,
            }
        )

    @property
    def key(self) -> str:
        return (
            f"{self.stage}__{self.scenario}__n{self.n}__p{self.p}"
            f"__r{self.replication:03d}__{self.candidate}"
            f"__target-{self.training_target}__h{self.config_hash}"
        )


def iter_screening_tasks(
    config: dict,
    replications: int,
    candidate_groups: set[str] | None = None,
    candidate_names: set[str] | None = None,
    num_shards: int = 1,
    shard_index: int = 0,
):
    validate_shard(num_shards, shard_index)
    screening = config["screening"]
    candidates = tuple(screening["candidates"])
    names = [str(candidate["name"]) for candidate in candidates]
    if len(names) != len(set(names)):
        raise ValueError("Stage 3B screening candidate names must be unique.")
    known_groups = {str(candidate["group"]) for candidate in candidates}
    unknown_groups = set(candidate_groups or ()) - known_groups
    if unknown_groups:
        raise ValueError(f"Unknown screening groups: {sorted(unknown_groups)}")
    unknown_names = set(candidate_names or ()) - set(names)
    if unknown_names:
        raise ValueError(f"Unknown screening candidates: {sorted(unknown_names)}")

    selected = config["selected_configuration"]
    for replication in range(replications):
        if not replication_belongs_to_shard(replication, num_shards, shard_index):
            continue
        for candidate in candidates:
            if candidate_groups and str(candidate["group"]) not in candidate_groups:
                continue
            if candidate_names and str(candidate["name"]) not in candidate_names:
                continue
            yield ScreeningTaskSpec(
                stage=str(screening["stage"]),
                seed_namespace=str(screening["seed_namespace"]),
                scenario=str(selected["scenario"]),
                n=int(selected["n"]),
                p=int(selected["p"]),
                replication=replication,
                candidate=str(candidate["name"]),
                candidate_group=str(candidate["group"]),
                learner_kind=str(candidate["learner_kind"]),
                params=dict(candidate.get("params", {})),
                training_target=str(candidate.get("training_target", "d")),
                validation_fraction=float(screening["validation_fraction"]),
            )


def _make_screening_model(task: ScreeningTaskSpec, seed: int, fast: bool):
    if task.candidate_group in {"xgboost_tuned", "extra_trees"}:
        return make_configured_tree_learner(
            task.learner_kind,
            task.params,
            seed,
            fast=fast,
        )
    return make_learner(
        task.learner_kind,
        seed,
        (),
        tabicl_estimators=8 if task.learner_kind == "tabiclv2_8" else 1,
        fast=fast,
    )


def run_screening_task(
    task: ScreeningTaskSpec,
    output_root: str | Path = "results/stage3b_screening_raw",
    retry_failed: bool = False,
    fast: bool = False,
) -> dict:
    store = ResultStore(output_root)
    if store.exists(task) and not retry_failed:
        return {"task_key": task.key, "status": "skipped"}

    started = time.perf_counter()
    data_seed = derive_seed(
        task.seed_namespace,
        task.scenario,
        task.n,
        task.p,
        task.replication,
        "data",
    )
    split_seed = derive_seed(
        task.seed_namespace,
        task.scenario,
        task.n,
        task.p,
        task.replication,
        "screening_split",
    )
    learner_seed = derive_seed(task.key, "learner")
    base = {
        "task_key": task.key,
        "stage": task.stage,
        "seed_namespace": task.seed_namespace,
        "scenario": task.scenario,
        "n": task.n,
        "p": task.p,
        "replication": task.replication,
        "candidate": task.candidate,
        "candidate_group": task.candidate_group,
        "learner_kind": task.learner_kind,
        "params": task.params,
        "config_hash": task.config_hash,
        "training_target": task.training_target,
        "validation_fraction": task.validation_fraction,
        "data_seed": data_seed,
        "split_seed": split_seed,
        "learner_seed": learner_seed,
    }
    try:
        data = simulate_plr(task.scenario, task.n, task.p, data_seed, theta0=1.0)
        train, validation = train_test_split(
            np.arange(task.n),
            test_size=task.validation_fraction,
            random_state=split_seed,
            shuffle=True,
        )
        torch = (
            _cuda_helpers() if task.learner_kind.startswith("tabiclv2") else None
        )
        if torch is not None:
            torch.cuda.reset_peak_memory_stats()
        model = _make_screening_model(task, learner_seed, fast)
        target = data.d if task.training_target == "d" else data.m0
        model.fit(data.X[train], target[train])
        prediction = np.asarray(model.predict(data.X[validation]), dtype=float)
        if torch is not None:
            torch.cuda.synchronize()
        if not np.isfinite(prediction).all():
            raise ValueError("Screening predictions must be finite.")
        peak_gpu_mb = (
            None
            if torch is None
            else float(torch.cuda.max_memory_allocated() / 1024**2)
        )
        record = {
            **base,
            "status": "success",
            "validation_d_mse": float(
                np.mean((prediction - data.d[validation]) ** 2)
            ),
            "validation_m0_mse": float(
                np.mean((prediction - data.m0[validation]) ** 2)
            ),
            "peak_gpu_mb": peak_gpu_mb,
            "fallback_reason": getattr(model, "fallback_reason_", None),
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


def select_screening_winner(records: list[dict], candidate_group: str) -> dict:
    eligible = [
        record
        for record in records
        if record.get("status") == "success"
        and record.get("candidate_group") == candidate_group
        and record.get("training_target") == "d"
    ]
    if not eligible:
        raise ValueError(f"No eligible screening records for {candidate_group}.")
    grouped: dict[str, list[dict]] = {}
    for record in eligible:
        grouped.setdefault(str(record["candidate"]), []).append(record)
    ranked = []
    for candidate, values in grouped.items():
        default_kind = (
            "xgboost" if candidate_group == "xgboost_tuned" else candidate_group
        )
        ranked.append(
            {
                "candidate": candidate,
                "candidate_group": candidate_group,
                "learner_kind": values[0].get("learner_kind", default_kind),
                "params": values[0]["params"],
                "config_hash": values[0].get("config_hash"),
                "replications": len(values),
                "mean_validation_d_mse": float(
                    np.mean([value["validation_d_mse"] for value in values])
                ),
                "mean_validation_m0_mse_diagnostic": float(
                    np.mean([value["validation_m0_mse"] for value in values])
                ),
                "selection_metric": "mean_validation_d_mse",
            }
        )
    return min(ranked, key=lambda value: (value["mean_validation_d_mse"], value["candidate"]))


def write_screening_winners(
    records: list[dict],
    output_path: str | Path,
) -> dict:
    result = {
        "selection_metric": "mean_validation_d_mse",
        "xgboost_tuned": select_screening_winner(records, "xgboost_tuned"),
        "extra_trees": select_screening_winner(records, "extra_trees"),
    }
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return result
