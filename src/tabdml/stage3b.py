from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import derive_seed
from .crossfit import crossfit_single_nuisance, make_folds
from .dgp import simulate_plr
from .diagnostics import compute_nuisance_diagnostics
from .dml import estimate_plr_dml
from .nuisance_cache import CachedNuisanceResult, NuisanceCache, NuisanceTaskSpec
from .stage3 import Stage3TaskSpec, legacy_learner_seed


@dataclass(frozen=True)
class Stage3BPairSpec:
    stage: str
    seed_namespace: str
    scenario: str
    n: int
    p: int
    replication: int
    learner_l: str
    learner_m: str
    folds_count: int
    theta0: float
    learner_l_config_hash: str = "default"
    learner_m_config_hash: str = "default"

    @property
    def key(self) -> str:
        return (
            f"{self.stage}__{self.scenario}__n{self.n}__p{self.p}"
            f"__r{self.replication:03d}__l{self.learner_l}__m{self.learner_m}"
            f"__hl{self.learner_l_config_hash}__hm{self.learner_m_config_hash}"
        )


def _tabicl_estimators(learner: str) -> int:
    if learner == "tabiclv2_8":
        return 8
    if learner.startswith("tabiclv2"):
        return 1
    return 0


def build_nuisance_spec(pair: Stage3BPairSpec, target: str) -> NuisanceTaskSpec:
    if target not in {"l", "m"}:
        raise ValueError("target must be 'l' or 'm'.")
    learner = pair.learner_l if target == "l" else pair.learner_m
    config_hash = (
        pair.learner_l_config_hash if target == "l" else pair.learner_m_config_hash
    )
    seed_task = Stage3TaskSpec(
        stage=pair.stage,
        seed_namespace=pair.seed_namespace,
        scenario=pair.scenario,
        n=pair.n,
        p=pair.p,
        replication=pair.replication,
        learner_l=learner,
        learner_m=learner,
        tabicl_estimators=1,
    )
    return NuisanceTaskSpec(
        seed_namespace=pair.seed_namespace,
        scenario=pair.scenario,
        n=pair.n,
        p=pair.p,
        replication=pair.replication,
        target=target,
        learner=learner,
        tabicl_estimators=_tabicl_estimators(learner),
        folds_count=pair.folds_count,
        learner_seed=legacy_learner_seed(seed_task, learner),
        learner_config_hash=config_hash,
    )


def _data_and_folds(task: NuisanceTaskSpec, theta0: float):
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
    data = simulate_plr(task.scenario, task.n, task.p, data_seed, theta0)
    return data, make_folds(task.n, task.folds_count, fold_seed)


def fit_cached_nuisance(
    task: NuisanceTaskSpec,
    cache_root: str | Path,
    theta0: float,
    fast: bool = False,
    learner_kind: str | None = None,
    learner_params: dict | None = None,
) -> CachedNuisanceResult:
    cache = NuisanceCache(cache_root)
    if cache.exists(task):
        return cache.read(task, expected_length=task.n)
    data, folds = _data_and_folds(task, theta0)
    result = crossfit_single_nuisance(
        data,
        task.target,
        task.learner,
        folds,
        seed=task.learner_seed,
        tabicl_estimators=task.tabicl_estimators,
        fast=fast,
        learner_kind=learner_kind,
        learner_params=learner_params,
    )
    cache.write(
        task,
        result.prediction,
        result.fold_seconds,
        result.peak_gpu_mb,
        result.fallback_reason,
    )
    return cache.read(task, expected_length=task.n)


def compose_dml_record(
    pair: Stage3BPairSpec,
    l_result: CachedNuisanceResult,
    m_result: CachedNuisanceResult,
) -> dict:
    data_seed = derive_seed(
        pair.seed_namespace,
        pair.scenario,
        pair.n,
        pair.p,
        pair.replication,
        "data",
    )
    fold_seed = derive_seed(
        pair.seed_namespace,
        pair.scenario,
        pair.n,
        pair.p,
        pair.replication,
        "folds",
    )
    data = simulate_plr(pair.scenario, pair.n, pair.p, data_seed, pair.theta0)
    estimate = estimate_plr_dml(
        data.y,
        data.d,
        l_result.prediction,
        m_result.prediction,
    )
    diagnostics = compute_nuisance_diagnostics(
        data,
        l_result.prediction,
        m_result.prediction,
        pair.theta0,
    )
    peak_values = [
        value
        for value in (l_result.peak_gpu_mb, m_result.peak_gpu_mb)
        if value is not None
    ]
    fallback_reasons = [
        value
        for value in (l_result.fallback_reason, m_result.fallback_reason)
        if value
    ]
    return {
        "task_key": pair.key,
        "stage": pair.stage,
        "seed_namespace": pair.seed_namespace,
        "scenario": pair.scenario,
        "n": pair.n,
        "p": pair.p,
        "replication": pair.replication,
        "learner_l": pair.learner_l,
        "learner_m": pair.learner_m,
        "learner_l_config_hash": pair.learner_l_config_hash,
        "learner_m_config_hash": pair.learner_m_config_hash,
        "folds_count": pair.folds_count,
        "theta0": pair.theta0,
        "data_seed": data_seed,
        "fold_seed": fold_seed,
        "status": "success",
        "theta": estimate.theta,
        "standard_error": estimate.standard_error,
        "ci_lower": estimate.ci_lower,
        "ci_upper": estimate.ci_upper,
        "l_mse": diagnostics.l_mse,
        "m_mse": diagnostics.m_mse,
        "nuisance_error_product": float(
            np.sqrt(diagnostics.l_mse) * np.sqrt(diagnostics.m_mse)
        ),
        "lm_error_cross": diagnostics.lm_error_cross,
        "residual_d_variance": diagnostics.residual_d_variance,
        "bias_numerator_proxy": diagnostics.bias_numerator_proxy,
        "theta_proxy": diagnostics.theta_proxy,
        "proxy_error": estimate.theta - diagnostics.theta_proxy,
        "l_fold_seconds": list(l_result.fold_seconds),
        "m_fold_seconds": list(m_result.fold_seconds),
        "peak_gpu_mb": max(peak_values) if peak_values else None,
        "fallback_reason": "; ".join(fallback_reasons) or None,
        "runtime_seconds": float(
            sum(l_result.fold_seconds) + sum(m_result.fold_seconds)
        ),
    }
