from __future__ import annotations

import time
import traceback
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import TaskSpec, derive_seed
from .crossfit import crossfit_nuisance_pair, make_folds
from .dgp import simulate_plr
from .dml import estimate_plr_dml
from .runner import classify_failure
from .sharding import replication_belongs_to_shard, validate_shard
from .storage import ResultStore


@dataclass(frozen=True)
class Stage3TaskSpec:
    stage: str
    seed_namespace: str
    scenario: str
    n: int
    p: int
    replication: int
    learner_l: str
    learner_m: str
    tabicl_estimators: int

    @property
    def key(self) -> str:
        return (
            f"{self.stage}__{self.scenario}__n{self.n}__p{self.p}"
            f"__r{self.replication:03d}__l{self.learner_l}__m{self.learner_m}"
            f"__e{self.tabicl_estimators}"
        )


def _estimators_for_legacy_key(task: Stage3TaskSpec, learner_name: str) -> int:
    if learner_name == "tabiclv2_8":
        return 8
    if learner_name in {"tabiclv2", "tabiclv2_1"}:
        return task.tabicl_estimators
    return 0


def legacy_learner_seed(task: Stage3TaskSpec, learner_name: str) -> int:
    legacy = TaskSpec(
        task.seed_namespace,
        task.scenario,
        task.n,
        task.p,
        task.replication,
        learner_name,
        _estimators_for_legacy_key(task, learner_name),
    )
    return derive_seed(legacy.key, "learners")


def iter_stage3_tasks(
    config: dict,
    replications: int,
    pair_names: set[str] | None,
    num_shards: int,
    shard_index: int,
):
    validate_shard(num_shards, shard_index)
    pairs = tuple(config["learner_pairs"])
    names = [str(pair["name"]) for pair in pairs]
    if len(names) != len(set(names)):
        raise ValueError("Stage 3 learner pair names must be unique.")
    unknown = set(pair_names or ()) - set(names)
    if unknown:
        raise ValueError(f"Unknown Stage 3 pair names: {sorted(unknown)}")

    for selected in config["selected_configurations"]:
        for replication in range(replications):
            if not replication_belongs_to_shard(
                replication, num_shards, shard_index
            ):
                continue
            for pair in pairs:
                if pair_names and str(pair["name"]) not in pair_names:
                    continue
                yield Stage3TaskSpec(
                    stage=str(config["stage"]),
                    seed_namespace=str(config["seed_namespace"]),
                    scenario=str(selected["scenario"]),
                    n=int(selected["n"]),
                    p=int(selected["p"]),
                    replication=replication,
                    learner_l=str(pair["learner_l"]),
                    learner_m=str(pair["learner_m"]),
                    tabicl_estimators=int(config["tabicl_estimators"]),
                )


def run_stage3_task(
    task: Stage3TaskSpec,
    folds_count: int = 5,
    theta0: float = 1.0,
    output_root: str | Path = "results/stage3_tree_diagnosis_raw",
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
    fold_seed = derive_seed(
        task.seed_namespace,
        task.scenario,
        task.n,
        task.p,
        task.replication,
        "folds",
    )
    base = {
        "task_key": task.key,
        "stage": task.stage,
        "seed_namespace": task.seed_namespace,
        "scenario": task.scenario,
        "n": task.n,
        "p": task.p,
        "replication": task.replication,
        "learner_l": task.learner_l,
        "learner_m": task.learner_m,
        "tabicl_estimators": task.tabicl_estimators,
        "data_seed": data_seed,
        "fold_seed": fold_seed,
    }

    try:
        data = simulate_plr(task.scenario, task.n, task.p, data_seed, theta0)
        folds = make_folds(task.n, folds_count, fold_seed)
        crossfit = crossfit_nuisance_pair(
            data,
            task.learner_l,
            task.learner_m,
            folds,
            seed_l=legacy_learner_seed(task, task.learner_l),
            seed_m=legacy_learner_seed(task, task.learner_m),
            tabicl_estimators=task.tabicl_estimators,
            fast=fast,
        )
        estimate = estimate_plr_dml(data.y, data.d, crossfit.l_hat, crossfit.m_hat)
        l_mse = float(np.mean((crossfit.l_hat - data.l0) ** 2))
        m_mse = float(np.mean((crossfit.m_hat - data.m0) ** 2))
        record = {
            **base,
            "status": "success",
            "theta": estimate.theta,
            "standard_error": estimate.standard_error,
            "ci_lower": estimate.ci_lower,
            "ci_upper": estimate.ci_upper,
            "l_mse": l_mse,
            "m_mse": m_mse,
            "nuisance_error_product": float(np.sqrt(l_mse) * np.sqrt(m_mse)),
            "fold_seconds": list(crossfit.fold_seconds),
            "peak_gpu_mb": crossfit.peak_gpu_mb,
            "fallback_reason": crossfit.fallback_reason,
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
