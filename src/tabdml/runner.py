from __future__ import annotations

import time
import traceback
from pathlib import Path

import numpy as np

from .config import TaskSpec, derive_seed
from .crossfit import crossfit_nuisances, make_folds
from .dgp import simulate_plr
from .dml import estimate_plr_dml
from .storage import ResultStore


def classify_failure(error: BaseException) -> str:
    message = str(error).lower()
    if isinstance(error, MemoryError) or "out of memory" in message or "cuda oom" in message:
        return "oom"
    return "failed"


def run_task(
    task: TaskSpec,
    folds_count: int = 5,
    theta0: float = 1.0,
    output_root: str | Path = "results/raw",
    retry_failed: bool = False,
    fast: bool = False,
) -> dict:
    store = ResultStore(output_root)
    if store.exists(task) and not retry_failed:
        return {"task_key": task.key, "status": "skipped"}
    started = time.perf_counter()
    data_seed = derive_seed(task.stage, task.scenario, task.n, task.p, task.replication, "data")
    fold_seed = derive_seed(task.stage, task.scenario, task.n, task.p, task.replication, "folds")
    base = {
        "task_key": task.key,
        "stage": task.stage,
        "scenario": task.scenario,
        "n": task.n,
        "p": task.p,
        "replication": task.replication,
        "learner": task.learner,
        "tabicl_estimators": task.tabicl_estimators,
        "data_seed": data_seed,
        "fold_seed": fold_seed,
    }
    try:
        data = simulate_plr(task.scenario, task.n, task.p, data_seed, theta0)
        folds = make_folds(task.n, folds_count, fold_seed)
        crossfit = crossfit_nuisances(
            data,
            task.learner,
            folds,
            derive_seed(task.key, "learners"),
            task.tabicl_estimators,
            fast=fast,
        )
        estimate = estimate_plr_dml(data.y, data.d, crossfit.l_hat, crossfit.m_hat)
        record = {
            **base,
            "status": "success",
            "theta": estimate.theta,
            "standard_error": estimate.standard_error,
            "ci_lower": estimate.ci_lower,
            "ci_upper": estimate.ci_upper,
            "l_mse": float(np.mean((crossfit.l_hat - data.l0) ** 2)),
            "m_mse": float(np.mean((crossfit.m_hat - data.m0) ** 2)),
            "nuisance_error_product": float(
                np.sqrt(np.mean((crossfit.l_hat - data.l0) ** 2))
                * np.sqrt(np.mean((crossfit.m_hat - data.m0) ** 2))
            ),
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

