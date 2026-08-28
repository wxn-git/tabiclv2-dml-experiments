from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import NDArray
from sklearn.model_selection import KFold

from .config import derive_seed
from .dgp import SimulatedData
from .learners import make_configured_tree_learner, make_learner


@dataclass(frozen=True)
class CrossfitResult:
    l_hat: NDArray[np.float64]
    m_hat: NDArray[np.float64]
    fold_seconds: tuple[float, ...]
    peak_gpu_mb: float | None
    fallback_reason: str | None


@dataclass(frozen=True)
class SingleNuisanceResult:
    prediction: NDArray[np.float64]
    fold_seconds: tuple[float, ...]
    peak_gpu_mb: float | None
    fallback_reason: str | None


def make_folds(n: int, folds: int, seed: int):
    splitter = KFold(n_splits=folds, shuffle=True, random_state=seed)
    indices = np.arange(n)
    return tuple((train, test) for train, test in splitter.split(indices))


def _cuda_helpers():
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    return torch


def crossfit_nuisances(
    data: SimulatedData,
    learner_name: str,
    folds: Sequence[tuple[NDArray[np.int_], NDArray[np.int_]]],
    seed: int,
    tabicl_estimators: int,
    fast: bool = False,
) -> CrossfitResult:
    return crossfit_nuisance_pair(
        data,
        learner_name,
        learner_name,
        folds,
        seed_l=seed,
        seed_m=seed,
        tabicl_estimators=tabicl_estimators,
        fast=fast,
    )


def crossfit_nuisance_pair(
    data: SimulatedData,
    learner_l_name: str,
    learner_m_name: str,
    folds: Sequence[tuple[NDArray[np.int_], NDArray[np.int_]]],
    seed_l: int,
    seed_m: int,
    tabicl_estimators: int,
    fast: bool = False,
) -> CrossfitResult:
    l_hat = np.full(len(data.y), np.nan)
    m_hat = np.full(len(data.y), np.nan)
    fold_seconds: list[float] = []
    fallback_reasons: list[str] = []
    uses_tabicl = learner_l_name.startswith("tabiclv2") or learner_m_name.startswith(
        "tabiclv2"
    )
    torch = _cuda_helpers() if uses_tabicl else None
    if torch is not None:
        torch.cuda.reset_peak_memory_stats()

    for fold_index, (train, test) in enumerate(folds):
        started = time.perf_counter()
        l_model = None
        m_model = None
        if learner_l_name == "oracle":
            l_hat[test] = data.l0[test]
        else:
            l_model = make_learner(
                learner_l_name,
                derive_seed(seed_l, learner_l_name, fold_index),
                data.categorical_indices,
                tabicl_estimators,
                fast=fast,
            )
            l_model.fit(data.X[train], data.y[train])
            l_hat[test] = l_model.predict(data.X[test])
        if learner_m_name == "oracle":
            m_hat[test] = data.m0[test]
        else:
            m_fold_seed = derive_seed(seed_m, learner_m_name, fold_index)
            m_model = make_learner(
                learner_m_name,
                derive_seed(m_fold_seed, "m"),
                data.categorical_indices,
                tabicl_estimators,
                fast=fast,
            )
            m_model.fit(data.X[train], data.d[train])
            m_hat[test] = m_model.predict(data.X[test])
        for model in (l_model, m_model):
            if model is None:
                continue
            reason = getattr(model, "fallback_reason_", None)
            if reason:
                fallback_reasons.append(str(reason))
        if torch is not None:
            torch.cuda.synchronize()
        fold_seconds.append(time.perf_counter() - started)

    peak_gpu_mb = None
    if torch is not None:
        peak_gpu_mb = float(torch.cuda.max_memory_allocated() / 1024**2)
    return CrossfitResult(
        l_hat=l_hat,
        m_hat=m_hat,
        fold_seconds=tuple(fold_seconds),
        peak_gpu_mb=peak_gpu_mb,
        fallback_reason="; ".join(fallback_reasons) or None,
    )


def crossfit_single_nuisance(
    data: SimulatedData,
    target: str,
    learner_name: str,
    folds: Sequence[tuple[NDArray[np.int_], NDArray[np.int_]]],
    seed: int,
    tabicl_estimators: int,
    fast: bool = False,
    learner_kind: str | None = None,
    learner_params: dict | None = None,
) -> SingleNuisanceResult:
    if target not in {"l", "m"}:
        raise ValueError("target must be 'l' or 'm'.")

    prediction = np.full(len(data.y), np.nan)
    fold_seconds: list[float] = []
    fallback_reasons: list[str] = []
    torch = _cuda_helpers() if learner_name.startswith("tabiclv2") else None
    if torch is not None:
        torch.cuda.reset_peak_memory_stats()

    oracle_values = data.l0 if target == "l" else data.m0
    response = data.y if target == "l" else data.d
    for fold_index, (train, test) in enumerate(folds):
        started = time.perf_counter()
        if learner_name == "oracle":
            prediction[test] = oracle_values[test]
        else:
            model_seed = derive_seed(seed, learner_name, fold_index)
            if target == "m":
                model_seed = derive_seed(model_seed, "m")
            if learner_params is not None:
                if learner_kind is None:
                    raise ValueError("Configured nuisance learner requires learner_kind.")
                model = make_configured_tree_learner(
                    learner_kind,
                    learner_params,
                    model_seed,
                    fast=fast,
                )
            else:
                model = make_learner(
                    learner_name,
                    model_seed,
                    data.categorical_indices,
                    tabicl_estimators,
                    fast=fast,
                )
            model.fit(data.X[train], response[train])
            prediction[test] = model.predict(data.X[test])
            reason = getattr(model, "fallback_reason_", None)
            if reason:
                fallback_reasons.append(str(reason))
        if torch is not None:
            torch.cuda.synchronize()
        fold_seconds.append(time.perf_counter() - started)

    peak_gpu_mb = None
    if torch is not None:
        peak_gpu_mb = float(torch.cuda.max_memory_allocated() / 1024**2)
    return SingleNuisanceResult(
        prediction=prediction,
        fold_seconds=tuple(fold_seconds),
        peak_gpu_mb=peak_gpu_mb,
        fallback_reason="; ".join(fallback_reasons) or None,
    )
