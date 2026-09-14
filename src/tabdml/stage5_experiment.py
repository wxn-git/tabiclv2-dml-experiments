from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .config import derive_seed
from .crossfit import crossfit_single_nuisance, make_folds
from .dgp import simulate_plr
from .diagnostics import compute_nuisance_diagnostics
from .dml import estimate_plr_dml
from .nuisance_cache import NuisanceCache, NuisanceTaskSpec
from .stage3b_screen import _params_hash
from .stage5_config import (
    iter_sensitivity_cells,
    resolve_stage5_profile,
    stage5_config_fingerprint,
)
from .stage5_tuning import (
    stage5_tuning_run_fingerprint,
    validate_frozen_stage5_tuning,
)


_DEVICE_METHODS = {
    "gpu": ("tabiclv2_1", "tabiclv2_8"),
    "cpu": ("xgboost_tuned", "extra_trees", "lasso"),
    "ensemble": ("ensemble",),
}
_STAGE5_RECORD_FIELDS = frozenset(
    {
        "task_key", "stage", "profile", "seed_namespace", "scenario", "n", "p",
        "replication", "method", "learner_l", "learner_m", "config_fingerprint",
        "tuning_fingerprint", "learner_l_config_hash", "learner_m_config_hash",
        "folds_count", "theta0", "data_seed", "fold_seed", "status", "theta_hat",
        "theta", "standard_error", "ci_lower", "ci_upper", "covered",
        "squared_error", "l_mse", "m_mse", "nuisance_error_product",
        "lm_error_cross", "residual_d_variance", "bias_numerator_proxy",
        "theta_proxy", "proxy_error", "l_fold_seconds", "m_fold_seconds",
        "l_fit_time", "m_fit_time", "runtime_seconds", "peak_gpu_mb",
        "requested_device", "observed_device", "fallback_reason",
    }
)


@dataclass(frozen=True)
class Stage5PairSpec:
    profile: str
    stage: str
    seed_namespace: str
    scenario: str
    n: int
    p: int
    replication: int
    method: str
    folds_count: int
    theta0: float
    config_fingerprint: str
    tuning_fingerprint: str
    learner_l_config_hash: str
    learner_m_config_hash: str

    @property
    def data_seed(self) -> int:
        return derive_seed(
            self.seed_namespace,
            self.scenario,
            self.n,
            self.p,
            self.replication,
            "data",
        )

    @property
    def fold_seed(self) -> int:
        return derive_seed(
            self.seed_namespace,
            self.scenario,
            self.n,
            self.p,
            self.replication,
            "folds",
        )

    @property
    def key(self) -> str:
        identity = json.dumps(
            {
                "profile": self.profile,
                "stage": self.stage,
                "seed_namespace": self.seed_namespace,
                "scenario": self.scenario,
                "n": self.n,
                "p": self.p,
                "replication": self.replication,
                "method": self.method,
                "folds_count": self.folds_count,
                "theta0": self.theta0,
                "config_fingerprint": self.config_fingerprint,
                "tuning_fingerprint": self.tuning_fingerprint,
                "learner_l_config_hash": self.learner_l_config_hash,
                "learner_m_config_hash": self.learner_m_config_hash,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return f"{self.stage}__{self.method}__{hashlib.sha256(identity.encode()).hexdigest()}"


@dataclass(frozen=True)
class ResolvedStage5Method:
    learner: str
    learner_kind: str | None
    params: dict[str, Any] | None
    config_hash: str
    requested_device: str


@dataclass(frozen=True)
class Stage5NuisanceResult:
    prediction: NDArray[np.float64]
    fold_seconds: tuple[float, ...]
    peak_gpu_mb: float | None
    fallback_reason: str | None
    fit_time: float
    total_time: float
    requested_device: str
    observed_device: str


def methods_for_device(group: str) -> tuple[str, ...]:
    try:
        return _DEVICE_METHODS[group]
    except KeyError as error:
        raise ValueError("unknown Stage 5 device group") from error


def _profile_tuning_name(profile: str) -> str:
    return "fast" if profile == "smoke" else "full"


def _method_config_hashes(
    method: str,
    scenario: str,
    config: Mapping[str, Any],
    frozen: Mapping[str, Any],
) -> tuple[str, str]:
    if method == "xgboost_tuned":
        winners = frozen["scenarios"][scenario]
        return (
            str(winners["l"]["effective_config_hash"]),
            str(winners["m"]["effective_config_hash"]),
        )
    if method == "extra_trees":
        params = dict(config["extra_trees"]["params"])
        if frozen["execution_profile"] == "fast":
            params["n_estimators"] = min(int(params["n_estimators"]), 20)
        value = _params_hash(params)
        return value, value
    value = _params_hash({"learner": method})
    return value, value


def iter_stage5_pairs(
    config: Mapping[str, Any],
    frozen_tuning: Mapping[str, Any],
    profile: str,
):
    resolved_profile = resolve_stage5_profile(config, profile)
    tuning_profile = _profile_tuning_name(profile)
    validate_frozen_stage5_tuning(frozen_tuning, config, tuning_profile)
    config_fingerprint = stage5_config_fingerprint(config)
    tuning_fingerprint = stage5_tuning_run_fingerprint(
        config,
        1 if tuning_profile == "fast" else int(config["tuning"]["replications"]),
        tuning_profile,
    )
    for cell in iter_sensitivity_cells(config):
        for replication in range(resolved_profile.replications):
            for method in config["methods"]:
                l_hash, m_hash = _method_config_hashes(
                    str(method), cell.scenario, config, frozen_tuning
                )
                yield Stage5PairSpec(
                    profile=profile,
                    stage=resolved_profile.stage,
                    seed_namespace=resolved_profile.seed_namespace,
                    scenario=cell.scenario,
                    n=cell.n,
                    p=cell.p,
                    replication=replication,
                    method=str(method),
                    folds_count=int(config["folds"]),
                    theta0=float(config["theta0"]),
                    config_fingerprint=config_fingerprint,
                    tuning_fingerprint=tuning_fingerprint,
                    learner_l_config_hash=l_hash,
                    learner_m_config_hash=m_hash,
                )


def _validate_pair_provenance(
    pair: Stage5PairSpec,
    config: Mapping[str, Any],
    frozen: Mapping[str, Any],
) -> None:
    profile = resolve_stage5_profile(config, pair.profile)
    if (
        pair.stage != profile.stage
        or pair.seed_namespace != profile.seed_namespace
        or pair.config_fingerprint != stage5_config_fingerprint(config)
        or pair.scenario not in config["scenarios"]
        or pair.method not in config["methods"]
        or pair.folds_count != config["folds"]
        or pair.theta0 != config["theta0"]
        or type(pair.replication) is not int
        or not 0 <= pair.replication < profile.replications
    ):
        raise ValueError("Stage 5 pair provenance mismatch")
    if (pair.scenario, pair.n, pair.p) not in {
        (cell.scenario, cell.n, cell.p) for cell in iter_sensitivity_cells(config)
    }:
        raise ValueError("Stage 5 pair cell provenance mismatch")
    tuning_profile = _profile_tuning_name(pair.profile)
    validate_frozen_stage5_tuning(frozen, config, tuning_profile)
    expected_tuning = stage5_tuning_run_fingerprint(
        config,
        1 if tuning_profile == "fast" else int(config["tuning"]["replications"]),
        tuning_profile,
    )
    if pair.tuning_fingerprint != expected_tuning:
        raise ValueError("Stage 5 tuning fingerprint mismatch")
    if (
        pair.learner_l_config_hash,
        pair.learner_m_config_hash,
    ) != _method_config_hashes(pair.method, pair.scenario, config, frozen):
        raise ValueError("Stage 5 learner configuration provenance mismatch")


def resolve_stage5_method(
    pair: Stage5PairSpec,
    target: str,
    config: Mapping[str, Any],
    frozen_tuning: Mapping[str, Any],
) -> ResolvedStage5Method:
    if target not in {"l", "m"}:
        raise ValueError("target must be 'l' or 'm'")
    _validate_pair_provenance(pair, config, frozen_tuning)
    if pair.method == "xgboost_tuned":
        winner = frozen_tuning["scenarios"][pair.scenario][target]
        return ResolvedStage5Method(
            learner="xgboost_tuned",
            learner_kind="xgboost",
            params=dict(winner["effective_params"]),
            config_hash=str(winner["effective_config_hash"]),
            requested_device="cpu",
        )
    if pair.method == "extra_trees":
        params = dict(config["extra_trees"]["params"])
        if pair.profile == "smoke":
            params["n_estimators"] = min(int(params["n_estimators"]), 20)
        return ResolvedStage5Method(
            "extra_trees", "extra_trees", params, _params_hash(params), "cpu"
        )
    requested = "cuda" if pair.method.startswith("tabiclv2") else "cpu"
    return ResolvedStage5Method(
        pair.method, None, None, _params_hash({"learner": pair.method}), requested
    )


def _estimators(learner: str) -> int:
    return 8 if learner == "tabiclv2_8" else 1 if learner == "tabiclv2_1" else 0


def build_stage5_nuisance_spec(
    pair: Stage5PairSpec,
    target: str,
    resolved: ResolvedStage5Method,
) -> NuisanceTaskSpec:
    if target not in {"l", "m"}:
        raise ValueError("target must be 'l' or 'm'")
    expected_hash = pair.learner_l_config_hash if target == "l" else pair.learner_m_config_hash
    if resolved.config_hash != expected_hash:
        raise ValueError("Resolved learner configuration mismatch")
    provenance = json.dumps(
        {
            "learner_config_hash": resolved.config_hash,
            "config_fingerprint": pair.config_fingerprint,
            "tuning_fingerprint": pair.tuning_fingerprint,
            "profile": pair.profile,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    provenance_hash = (
        f"{resolved.config_hash}__stage5-"
        f"{hashlib.sha256(provenance.encode()).hexdigest()[:32]}"
    )
    return NuisanceTaskSpec(
        seed_namespace=pair.seed_namespace,
        scenario=pair.scenario,
        n=pair.n,
        p=pair.p,
        replication=pair.replication,
        target=target,
        learner=resolved.learner,
        tabicl_estimators=_estimators(resolved.learner),
        folds_count=pair.folds_count,
        learner_seed=derive_seed(
            pair.seed_namespace,
            pair.scenario,
            pair.n,
            pair.p,
            pair.replication,
            resolved.learner,
            target,
            "learner",
        ),
        learner_config_hash=provenance_hash,
    )


def stage5_nuisance_metadata_path(
    cache: NuisanceCache, task: NuisanceTaskSpec
) -> Path:
    digest = hashlib.sha256(task.key.encode("utf-8")).hexdigest()
    return cache.root / f"stage5-meta-{digest}.json"


def _validate_nuisance_result(
    task: NuisanceTaskSpec, result: Stage5NuisanceResult
) -> Stage5NuisanceResult:
    if not isinstance(result.prediction, np.ndarray):
        raise ValueError("Stage 5 nuisance predictions are invalid")
    prediction = np.asarray(result.prediction, dtype=float)
    if not isinstance(result.fold_seconds, tuple) or any(
        isinstance(value, bool) or not isinstance(value, (int, float))
        for value in result.fold_seconds
    ):
        raise ValueError("Stage 5 nuisance fold times are invalid")
    fold_seconds = np.asarray(result.fold_seconds, dtype=float)
    if prediction.ndim != 1 or len(prediction) != task.n or not np.isfinite(prediction).all():
        raise ValueError("Stage 5 nuisance predictions are invalid")
    if (
        fold_seconds.ndim != 1
        or len(fold_seconds) != task.folds_count
        or not np.isfinite(fold_seconds).all()
        or np.any(fold_seconds < 0)
    ):
        raise ValueError("Stage 5 nuisance fold times are invalid")
    for field in ("fit_time", "total_time"):
        value = getattr(result, field)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value < 0:
            raise ValueError(f"Stage 5 nuisance {field} is invalid")
    if result.peak_gpu_mb is not None and (
        isinstance(result.peak_gpu_mb, bool)
        or not np.isfinite(result.peak_gpu_mb)
        or result.peak_gpu_mb < 0
    ):
        raise ValueError("Stage 5 nuisance peak GPU memory is invalid")
    if result.requested_device not in {"cpu", "cuda"} or result.observed_device not in {"cpu", "cuda"}:
        raise ValueError("Stage 5 nuisance device metadata is invalid")
    if result.requested_device == "cpu" and result.observed_device != "cpu":
        raise ValueError("Stage 5 nuisance device metadata is invalid")
    if result.fallback_reason is not None and (
        not isinstance(result.fallback_reason, str) or not result.fallback_reason
    ):
        raise ValueError("Stage 5 nuisance fallback metadata is invalid")
    return result


def _read_stage5_cache(
    cache: NuisanceCache,
    task: NuisanceTaskSpec,
    resolved: ResolvedStage5Method,
    full_settings: bool,
) -> Stage5NuisanceResult:
    path = stage5_nuisance_metadata_path(cache, task)
    try:
        cached = cache.read(task, expected_length=task.n)
        metadata = json.loads(path.read_text(encoding="utf-8"))
    except Exception as error:
        raise ValueError(f"Invalid Stage 5 nuisance metadata: {path}") from error
    if not isinstance(metadata, Mapping):
        raise ValueError(f"Invalid Stage 5 nuisance metadata object: {path}")
    expected = {
        "schema_version": "stage5_nuisance_metadata_v1",
        "task": asdict(task),
        "resolved": {
            "learner": resolved.learner,
            "learner_kind": resolved.learner_kind,
            "params": resolved.params,
            "config_hash": resolved.config_hash,
        },
        "requested_device": resolved.requested_device,
        "full_settings": full_settings,
        "data_seed": derive_seed(
            task.seed_namespace, task.scenario, task.n, task.p,
            task.replication, "data",
        ),
        "fold_seed": derive_seed(
            task.seed_namespace, task.scenario, task.n, task.p,
            task.replication, "folds",
        ),
    }
    if set(metadata) != {*expected, "observed_device", "fit_time", "total_time"}:
        raise ValueError("Stage 5 nuisance metadata schema mismatch")
    for field, value in expected.items():
        actual = metadata.get(field)
        if actual != value or type(actual) is not type(value):
            raise ValueError(f"Stage 5 nuisance metadata {field} mismatch")
    result = Stage5NuisanceResult(
        prediction=cached.prediction,
        fold_seconds=cached.fold_seconds,
        peak_gpu_mb=cached.peak_gpu_mb,
        fallback_reason=cached.fallback_reason,
        fit_time=metadata.get("fit_time"),
        total_time=metadata.get("total_time"),
        requested_device=metadata["requested_device"],
        observed_device=metadata.get("observed_device"),
    )
    return _validate_nuisance_result(task, result)


def fit_stage5_nuisance(
    task: NuisanceTaskSpec,
    resolved: ResolvedStage5Method,
    cache_root: str | Path,
    theta0: float,
    full_settings: bool,
    retry_failed: bool = False,
) -> Stage5NuisanceResult:
    if task.learner != resolved.learner:
        raise ValueError("Stage 5 nuisance task and resolved learner mismatch")
    if not task.learner_config_hash.startswith(
        f"{resolved.config_hash}__stage5-"
    ):
        raise ValueError("Stage 5 nuisance learner configuration mismatch")
    if type(full_settings) is not bool:
        raise ValueError("full_settings must be boolean")
    started = time.perf_counter()
    cache = NuisanceCache(cache_root)
    if cache.path(task).exists():
        try:
            cached = _read_stage5_cache(cache, task, resolved, full_settings)
            if not (retry_failed and cached.fallback_reason):
                return cached
        except ValueError:
            if not retry_failed:
                raise
        cache.path(task).unlink(missing_ok=True)
        stage5_nuisance_metadata_path(cache, task).unlink(missing_ok=True)
    data_seed = derive_seed(
        task.seed_namespace, task.scenario, task.n, task.p, task.replication, "data"
    )
    fold_seed = derive_seed(
        task.seed_namespace, task.scenario, task.n, task.p, task.replication, "folds"
    )
    data = simulate_plr(task.scenario, task.n, task.p, data_seed, theta0)
    folds = make_folds(task.n, task.folds_count, fold_seed)
    result = crossfit_single_nuisance(
        data,
        task.target,
        task.learner,
        folds,
        seed=task.learner_seed,
        tabicl_estimators=task.tabicl_estimators,
        fast=not full_settings,
        learner_kind=resolved.learner_kind,
        learner_params=resolved.params,
    )
    fit_time = float(sum(result.fold_seconds))
    total_time = max(float(time.perf_counter() - started), fit_time)
    observed_device = (
        "cuda"
        if resolved.requested_device == "cuda" and result.peak_gpu_mb is not None
        else "cpu"
    )
    checked = _validate_nuisance_result(
        task,
        Stage5NuisanceResult(
            np.asarray(result.prediction, dtype=float),
            tuple(float(value) for value in result.fold_seconds),
            result.peak_gpu_mb,
            result.fallback_reason,
            fit_time,
            total_time,
            resolved.requested_device,
            observed_device,
        ),
    )
    cache.write(
        task,
        checked.prediction,
        checked.fold_seconds,
        checked.peak_gpu_mb,
        checked.fallback_reason,
    )
    metadata = {
        "schema_version": "stage5_nuisance_metadata_v1",
        "task": asdict(task),
        "resolved": {
            "learner": resolved.learner,
            "learner_kind": resolved.learner_kind,
            "params": resolved.params,
            "config_hash": resolved.config_hash,
        },
        "data_seed": data_seed,
        "fold_seed": fold_seed,
        "full_settings": full_settings,
        "requested_device": resolved.requested_device,
        "observed_device": observed_device,
        "fit_time": fit_time,
        "total_time": total_time,
    }
    path = stage5_nuisance_metadata_path(cache, task)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, sort_keys=True, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return _read_stage5_cache(cache, task, resolved, full_settings)


def read_stage5_nuisance(
    task: NuisanceTaskSpec,
    resolved: ResolvedStage5Method,
    cache_root: str | Path,
    full_settings: bool,
) -> Stage5NuisanceResult:
    """Read and validate one complete Stage 5 cache entry."""
    return _read_stage5_cache(
        NuisanceCache(cache_root), task, resolved, full_settings
    )


def compose_stage5_record(
    pair: Stage5PairSpec,
    l_result: Stage5NuisanceResult,
    m_result: Stage5NuisanceResult,
) -> dict[str, Any]:
    results = (l_result, m_result)
    for result in results:
        if len(result.prediction) != pair.n or not np.isfinite(result.prediction).all():
            raise ValueError("Stage 5 nuisance predictions must be finite")
        if len(result.fold_seconds) != pair.folds_count:
            raise ValueError("Stage 5 fold timings must be complete")
        times = (*result.fold_seconds, result.fit_time, result.total_time)
        if any(not np.isfinite(value) or value < 0 for value in times):
            raise ValueError("Stage 5 timings must be finite and nonnegative")
        if result.requested_device == "cuda" and result.observed_device != "cuda":
            raise ValueError("TabICLv2 must be observed on CUDA")
    data = simulate_plr(pair.scenario, pair.n, pair.p, pair.data_seed, pair.theta0)
    estimate = estimate_plr_dml(data.y, data.d, l_result.prediction, m_result.prediction)
    diagnostics = compute_nuisance_diagnostics(
        data, l_result.prediction, m_result.prediction, pair.theta0
    )
    fallback = "; ".join(
        value for value in (l_result.fallback_reason, m_result.fallback_reason) if value
    ) or None
    peak_values = [
        value for value in (l_result.peak_gpu_mb, m_result.peak_gpu_mb) if value is not None
    ]
    expected_device = "cuda" if pair.method.startswith("tabiclv2") else "cpu"
    if any(result.requested_device != expected_device for result in results):
        raise ValueError("Stage 5 requested device does not match method")
    if l_result.observed_device != m_result.observed_device:
        raise ValueError("Stage 5 nuisance observed devices do not match")
    requested = expected_device
    observed = l_result.observed_device
    record = {
        "task_key": pair.key,
        "stage": pair.stage,
        "profile": pair.profile,
        "seed_namespace": pair.seed_namespace,
        "scenario": pair.scenario,
        "n": pair.n,
        "p": pair.p,
        "replication": pair.replication,
        "method": pair.method,
        "learner_l": pair.method,
        "learner_m": pair.method,
        "config_fingerprint": pair.config_fingerprint,
        "tuning_fingerprint": pair.tuning_fingerprint,
        "learner_l_config_hash": pair.learner_l_config_hash,
        "learner_m_config_hash": pair.learner_m_config_hash,
        "folds_count": pair.folds_count,
        "theta0": pair.theta0,
        "data_seed": pair.data_seed,
        "fold_seed": pair.fold_seed,
        "status": "fallback" if fallback else "success",
        "theta_hat": estimate.theta,
        "theta": estimate.theta,
        "standard_error": estimate.standard_error,
        "ci_lower": estimate.ci_lower,
        "ci_upper": estimate.ci_upper,
        "covered": bool(estimate.ci_lower <= pair.theta0 <= estimate.ci_upper),
        "squared_error": float((estimate.theta - pair.theta0) ** 2),
        "l_mse": diagnostics.l_mse,
        "m_mse": diagnostics.m_mse,
        "nuisance_error_product": float(np.sqrt(diagnostics.l_mse * diagnostics.m_mse)),
        "lm_error_cross": diagnostics.lm_error_cross,
        "residual_d_variance": diagnostics.residual_d_variance,
        "bias_numerator_proxy": diagnostics.bias_numerator_proxy,
        "theta_proxy": diagnostics.theta_proxy,
        "proxy_error": float(estimate.theta - diagnostics.theta_proxy),
        "l_fold_seconds": list(l_result.fold_seconds),
        "m_fold_seconds": list(m_result.fold_seconds),
        "l_fit_time": l_result.fit_time,
        "m_fit_time": m_result.fit_time,
        "runtime_seconds": float(l_result.total_time + m_result.total_time),
        "peak_gpu_mb": max(peak_values) if peak_values else None,
        "requested_device": requested,
        "observed_device": observed,
        "fallback_reason": fallback,
    }
    numeric = [
        value
        for key, value in record.items()
        if key in {
            "theta_hat", "theta", "standard_error", "ci_lower", "ci_upper",
            "squared_error", "l_mse", "m_mse", "nuisance_error_product",
            "lm_error_cross", "residual_d_variance", "bias_numerator_proxy",
            "theta_proxy", "proxy_error", "l_fit_time", "m_fit_time", "runtime_seconds",
        }
    ]
    if not all(np.isfinite(value) for value in numeric) or record["ci_lower"] > record["ci_upper"]:
        raise ValueError("Stage 5 composed estimates must be finite and ordered")
    return dict(validate_stage5_record(record, pair))


def validate_stage5_record(
    record: Mapping[str, Any], pair: Stage5PairSpec
) -> Mapping[str, Any]:
    if set(record) != _STAGE5_RECORD_FIELDS:
        raise ValueError(f"Invalid Stage 5 record {pair.key}: schema mismatch")
    identity = {
        "task_key": pair.key,
        "stage": pair.stage,
        "profile": pair.profile,
        "seed_namespace": pair.seed_namespace,
        "scenario": pair.scenario,
        "n": pair.n,
        "p": pair.p,
        "replication": pair.replication,
        "method": pair.method,
        "learner_l": pair.method,
        "learner_m": pair.method,
        "config_fingerprint": pair.config_fingerprint,
        "tuning_fingerprint": pair.tuning_fingerprint,
        "learner_l_config_hash": pair.learner_l_config_hash,
        "learner_m_config_hash": pair.learner_m_config_hash,
        "folds_count": pair.folds_count,
        "theta0": pair.theta0,
        "data_seed": pair.data_seed,
        "fold_seed": pair.fold_seed,
    }
    for field, expected in identity.items():
        actual = record.get(field)
        if actual != expected or type(actual) is not type(expected):
            raise ValueError(f"Invalid Stage 5 record {pair.key}: {field} mismatch")
    status = record.get("status")
    if not isinstance(status, str) or status not in {"success", "fallback"}:
        raise ValueError(f"Invalid Stage 5 record {pair.key}: status")
    numeric_fields = (
        "theta_hat", "theta", "standard_error", "ci_lower", "ci_upper",
        "squared_error", "l_mse", "m_mse", "nuisance_error_product",
        "lm_error_cross", "residual_d_variance", "bias_numerator_proxy",
        "theta_proxy", "proxy_error", "l_fit_time", "m_fit_time",
        "runtime_seconds",
    )
    for field in numeric_fields:
        value = record.get(field)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
            raise ValueError(f"Invalid Stage 5 record {pair.key}: {field} must be finite")
    if record["theta_hat"] != record["theta"]:
        raise ValueError("Stage 5 theta aliases mismatch")
    if not record["ci_lower"] <= record["theta_hat"] <= record["ci_upper"]:
        raise ValueError("Stage 5 confidence interval is not ordered")
    if any(
        record[field] < 0
        for field in (
            "standard_error", "squared_error", "l_mse", "m_mse",
            "nuisance_error_product", "l_fit_time", "m_fit_time",
            "runtime_seconds",
        )
    ):
        raise ValueError("Stage 5 nonnegative metric is invalid")
    expected_covered = record["ci_lower"] <= pair.theta0 <= record["ci_upper"]
    if type(record.get("covered")) is not bool or record["covered"] != expected_covered:
        raise ValueError("Stage 5 coverage indicator is invalid")
    if record["squared_error"] != (record["theta_hat"] - pair.theta0) ** 2:
        raise ValueError("Stage 5 squared_error is invalid")
    expected_product = float(np.sqrt(record["l_mse"] * record["m_mse"]))
    if not np.isclose(record["nuisance_error_product"], expected_product):
        raise ValueError("Stage 5 nuisance error product is invalid")
    if not np.isclose(
        record["proxy_error"], record["theta_hat"] - record["theta_proxy"]
    ):
        raise ValueError("Stage 5 proxy_error is invalid")
    if record["runtime_seconds"] < record["l_fit_time"] + record["m_fit_time"]:
        raise ValueError("Stage 5 runtime is inconsistent with fit times")
    for target in ("l", "m"):
        values = record.get(f"{target}_fold_seconds")
        if not isinstance(values, list) or any(
            isinstance(value, bool) or not isinstance(value, (int, float))
            for value in values
        ):
            raise ValueError(f"Stage 5 {target} fold times are invalid")
        array = np.asarray(values, dtype=float)
        if len(array) != pair.folds_count or not np.isfinite(array).all() or np.any(array < 0):
            raise ValueError(f"Stage 5 {target} fold times are invalid")
    peak = record.get("peak_gpu_mb")
    if peak is not None and (
        isinstance(peak, bool)
        or not isinstance(peak, (int, float))
        or not np.isfinite(peak)
        or peak < 0
    ):
        raise ValueError("Stage 5 peak_gpu_mb is invalid")
    expected_device = "cuda" if pair.method.startswith("tabiclv2") else "cpu"
    if record.get("requested_device") != expected_device:
        raise ValueError("Stage 5 requested_device is invalid")
    if record.get("observed_device") not in {"cpu", "cuda"}:
        raise ValueError("Stage 5 observed_device is invalid")
    if expected_device == "cuda" and record["observed_device"] != "cuda":
        raise ValueError("TabICLv2 requires observed CUDA")
    if expected_device == "cpu" and record["observed_device"] != "cpu":
        raise ValueError("Stage 5 observed_device is invalid")
    fallback = record.get("fallback_reason")
    if status == "success" and fallback is not None:
        raise ValueError("Stage 5 success cannot contain fallback_reason")
    if status == "fallback" and (not isinstance(fallback, str) or not fallback):
        raise ValueError("Stage 5 fallback requires fallback_reason")
    return record


def validate_stage5_resume_record(
    record: Mapping[str, Any], pair: Stage5PairSpec
) -> str:
    """Validate identity and status before reusing a composed Stage 5 result."""
    status = record.get("status") if isinstance(record, Mapping) else None
    if status not in {"success", "fallback", "failed", "oom"}:
        raise ValueError(f"Invalid Stage 5 record {pair.key}: status")
    identity = {
        "task_key": pair.key,
        "stage": pair.stage,
        "profile": pair.profile,
        "seed_namespace": pair.seed_namespace,
        "scenario": pair.scenario,
        "n": pair.n,
        "p": pair.p,
        "replication": pair.replication,
        "method": pair.method,
        "learner_l": pair.method,
        "learner_m": pair.method,
        "config_fingerprint": pair.config_fingerprint,
        "tuning_fingerprint": pair.tuning_fingerprint,
        "learner_l_config_hash": pair.learner_l_config_hash,
        "learner_m_config_hash": pair.learner_m_config_hash,
        "folds_count": pair.folds_count,
        "theta0": pair.theta0,
        "data_seed": pair.data_seed,
        "fold_seed": pair.fold_seed,
    }
    for field, expected in identity.items():
        actual = record.get(field)
        if actual != expected or type(actual) is not type(expected):
            raise ValueError(f"Invalid Stage 5 record {pair.key}: {field} mismatch")
    if status in {"success", "fallback"}:
        validate_stage5_record(record, pair)
    return str(status)
