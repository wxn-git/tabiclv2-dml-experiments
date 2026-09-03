from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from .config import derive_seed
from .nuisance_cache import CachedNuisanceResult, NuisanceCache, NuisanceTaskSpec
from .sharding import belongs_to_shard, validate_shard
from .stage3b import (
    Stage3BPairSpec,
    build_nuisance_spec,
    compose_dml_record,
    fit_cached_nuisance,
)
from .stage3b_screen import _params_hash
from .stage4_config import TreeBenchmarkCell, iter_tree_cells


_EXECUTION_PROFILES = frozenset({"full", "fast"})
_STAGE4_METHODS = (
    "tabiclv2_1",
    "tabiclv2_8",
    "xgboost",
    "xgboost_tuned",
    "extra_trees",
    "oracle",
)
_ORACLE_DIAGNOSTICS = (
    ("oracle", "xgboost_tuned"),
    ("xgboost_tuned", "oracle"),
    ("oracle", "tabiclv2_1"),
    ("tabiclv2_1", "oracle"),
)


@dataclass(frozen=True)
class Stage4PairSpec:
    stage: str
    seed_namespace: str
    panel: str
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
    execution_profile: str = "full"

    def __post_init__(self) -> None:
        if self.execution_profile not in _EXECUTION_PROFILES:
            raise ValueError("execution_profile must be 'full' or 'fast'")

    @property
    def effective_seed_namespace(self) -> str:
        return f"{self.seed_namespace}__{self.panel}"

    @property
    def key(self) -> str:
        return (
            f"{self.stage}__{self.panel}__{self.scenario}__n{self.n}__p{self.p}"
            f"__r{self.replication:03d}__l{self.learner_l}__m{self.learner_m}"
            f"__hl{self.learner_l_config_hash}__hm{self.learner_m_config_hash}"
            f"__profile-{self.execution_profile}"
        )


@dataclass(frozen=True)
class ResolvedMethod:
    learner: str
    learner_kind: str | None
    params: dict[str, Any] | None
    config_hash: str


def _cell_key(pair: Stage4PairSpec) -> str:
    return f"{pair.panel}__{pair.scenario}__n{pair.n}__p{pair.p}"


def _effective_tree_params(
    params: Mapping[str, Any], execution_profile: str
) -> dict[str, Any]:
    effective = dict(params)
    if execution_profile == "fast":
        effective["n_estimators"] = min(
            int(effective.get("n_estimators", 20)), 20
        )
    return effective


def validate_frozen_tuning(
    config: Mapping[str, Any],
    frozen_tuning: Mapping[str, Any],
    execution_profile: str,
) -> Mapping[str, Any]:
    if execution_profile not in _EXECUTION_PROFILES:
        raise ValueError("execution_profile must be 'full' or 'fast'")
    if not isinstance(frozen_tuning, Mapping):
        raise ValueError("Frozen tuning selection must be a mapping")
    if frozen_tuning.get("execution_profile") != execution_profile:
        raise ValueError("Frozen tuning execution_profile mismatch")
    if frozen_tuning.get("selection_metric_l") != "mean_validation_y_mse":
        raise ValueError("Frozen tuning selection_metric_l mismatch")
    if frozen_tuning.get("selection_metric_m") != "mean_validation_d_mse":
        raise ValueError("Frozen tuning selection_metric_m mismatch")
    expected_replications = frozen_tuning.get("expected_replications")
    if (
        isinstance(expected_replications, bool)
        or not isinstance(expected_replications, int)
        or expected_replications < 1
    ):
        raise ValueError("Frozen tuning expected_replications is invalid")

    expected_cells = {cell.key for cell in iter_tree_cells(config)}
    cells = frozen_tuning.get("cells")
    if not isinstance(cells, Mapping) or set(cells) != expected_cells:
        raise ValueError("Frozen tuning cell keys do not match the Stage 4 config")
    candidates = {
        str(candidate["name"]): dict(candidate["params"])
        for candidate in config["tuning"]["xgboost_candidates"]
    }
    for cell_key in sorted(expected_cells):
        targets = cells[cell_key]
        if not isinstance(targets, Mapping) or set(targets) != {"l", "m"}:
            raise ValueError(f"Frozen tuning targets are invalid for {cell_key}")
        for target in ("l", "m"):
            winner = targets[target]
            location = f"{cell_key}/{target}"
            if not isinstance(winner, Mapping):
                raise ValueError(f"Frozen tuning winner is invalid for {location}")
            candidate_name = winner.get("candidate")
            if candidate_name not in candidates:
                raise ValueError(f"Frozen tuning candidate mismatch for {location}")
            if winner.get("learner_kind") != "xgboost":
                raise ValueError(f"Frozen tuning learner_kind mismatch for {location}")
            if winner.get("execution_profile") != execution_profile:
                raise ValueError(
                    f"Frozen tuning execution_profile mismatch for {location}"
                )
            nominal_params = winner.get("nominal_params")
            params = winner.get("params")
            if not isinstance(nominal_params, Mapping) or dict(
                nominal_params
            ) != candidates[candidate_name]:
                raise ValueError(
                    f"Frozen tuning nominal_params mismatch for {location}"
                )
            expected_params = _effective_tree_params(
                nominal_params, execution_profile
            )
            if not isinstance(params, Mapping) or dict(params) != expected_params:
                raise ValueError(f"Frozen tuning params mismatch for {location}")
            if winner.get("nominal_config_hash") != _params_hash(
                dict(nominal_params)
            ):
                raise ValueError(
                    f"Frozen tuning nominal_config_hash mismatch for {location}"
                )
            if winner.get("config_hash") != _params_hash(dict(params)):
                raise ValueError(f"Frozen tuning config_hash mismatch for {location}")
            if winner.get("replications") != expected_replications:
                raise ValueError(f"Frozen tuning replications mismatch for {location}")
            expected_metric = (
                "mean_validation_y_mse"
                if target == "l"
                else "mean_validation_d_mse"
            )
            if winner.get("selection_metric") != expected_metric:
                raise ValueError(
                    f"Frozen tuning selection_metric mismatch for {location}"
                )
            for metric in (
                "mean_validation_observed_mse",
                "mean_validation_truth_mse_diagnostic",
            ):
                value = winner.get(metric)
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not np.isfinite(value)
                ):
                    raise ValueError(
                        f"Frozen tuning {metric} is invalid for {location}"
                    )
    return frozen_tuning


def resolve_method(
    pair: Stage4PairSpec,
    target: str,
    frozen_tuning: Mapping[str, Any],
    extra_trees_params: Mapping[str, Any],
) -> ResolvedMethod:
    if target not in {"l", "m"}:
        raise ValueError("target must be 'l' or 'm'")
    learner = pair.learner_l if target == "l" else pair.learner_m
    frozen_profile = frozen_tuning.get("execution_profile")
    if frozen_profile is not None and frozen_profile != pair.execution_profile:
        raise ValueError("Frozen tuning execution_profile mismatch")
    if learner in {"tabiclv2_1", "tabiclv2_8", "xgboost", "oracle"}:
        return ResolvedMethod(learner, None, None, "default")
    if learner == "extra_trees":
        params = _effective_tree_params(
            extra_trees_params, pair.execution_profile
        )
        return ResolvedMethod(
            learner="extra_trees",
            learner_kind="extra_trees",
            params=params,
            config_hash=_params_hash(params),
        )
    if learner != "xgboost_tuned":
        raise ValueError(f"Unknown Stage 4 method: {learner}")

    cells = frozen_tuning.get("cells")
    if not isinstance(cells, Mapping) or _cell_key(pair) not in cells:
        raise ValueError(f"Missing frozen tuning cell: {_cell_key(pair)}")
    cell = cells[_cell_key(pair)]
    if not isinstance(cell, Mapping) or target not in cell:
        raise ValueError(f"Missing frozen tuning target: {_cell_key(pair)}/{target}")
    winner = cell[target]
    if not isinstance(winner, Mapping):
        raise ValueError(f"Invalid frozen tuning target: {_cell_key(pair)}/{target}")
    params = winner.get("params")
    config_hash = winner.get("config_hash")
    if winner.get("learner_kind") != "xgboost":
        raise ValueError("Frozen tuned method must use learner_kind xgboost")
    winner_profile = winner.get("execution_profile")
    if winner_profile is not None and winner_profile != pair.execution_profile:
        raise ValueError("Frozen tuned method execution_profile mismatch")
    if (
        not isinstance(params, Mapping)
        or not isinstance(config_hash, str)
        or not config_hash
    ):
        raise ValueError("Frozen tuned method requires params and config_hash")
    if winner_profile is not None and _params_hash(dict(params)) != config_hash:
        raise ValueError("Frozen tuned method config_hash mismatch")
    return ResolvedMethod(
        learner="xgboost_tuned",
        learner_kind="xgboost",
        params=dict(params),
        config_hash=config_hash,
    )


def _as_stage3b_pair(
    pair: Stage4PairSpec,
    learner_l_config_hash: str | None = None,
    learner_m_config_hash: str | None = None,
) -> Stage3BPairSpec:
    profile = pair.execution_profile
    l_hash = learner_l_config_hash or pair.learner_l_config_hash
    m_hash = learner_m_config_hash or pair.learner_m_config_hash
    return Stage3BPairSpec(
        stage=pair.stage,
        seed_namespace=pair.effective_seed_namespace,
        scenario=pair.scenario,
        n=pair.n,
        p=pair.p,
        replication=pair.replication,
        learner_l=pair.learner_l,
        learner_m=pair.learner_m,
        folds_count=pair.folds_count,
        theta0=pair.theta0,
        learner_l_config_hash=f"{l_hash}__profile-{profile}",
        learner_m_config_hash=f"{m_hash}__profile-{profile}",
    )


def build_stage4_nuisance_spec(
    pair: Stage4PairSpec,
    target: str,
    resolved: ResolvedMethod | None = None,
) -> NuisanceTaskSpec:
    resolved_hash = None if resolved is None else resolved.config_hash
    stage3b_pair = _as_stage3b_pair(
        pair,
        learner_l_config_hash=resolved_hash if target == "l" else None,
        learner_m_config_hash=resolved_hash if target == "m" else None,
    )
    return build_nuisance_spec(stage3b_pair, target)


def fit_stage4_nuisance(
    pair: Stage4PairSpec,
    target: str,
    frozen_tuning: Mapping[str, Any],
    extra_trees_params: Mapping[str, Any],
    cache_root: str | Path,
    fast: bool = False,
    retry_failed: bool = False,
) -> CachedNuisanceResult:
    profile = "fast" if fast else pair.execution_profile
    effective_pair = (
        pair
        if pair.execution_profile == profile
        else replace(pair, execution_profile=profile)
    )
    resolved = resolve_method(
        effective_pair,
        target,
        frozen_tuning,
        extra_trees_params,
    )
    task = build_stage4_nuisance_spec(effective_pair, target, resolved)
    cache = NuisanceCache(cache_root)
    if cache.exists(task):
        try:
            return cache.read(task, expected_length=task.n)
        except ValueError:
            if not retry_failed:
                raise
            cache.path(task).unlink()
    return fit_cached_nuisance(
        task,
        cache_root=cache_root,
        theta0=effective_pair.theta0,
        fast=profile == "fast",
        learner_kind=resolved.learner_kind,
        learner_params=resolved.params,
    )


def _selected_confirmation_cells(
    config: Mapping[str, Any],
    selected_confirmation: Mapping[str, Any] | None,
) -> tuple[TreeBenchmarkCell, ...]:
    if not isinstance(selected_confirmation, Mapping):
        raise ValueError("confirmation requires six selected confirmation cells")
    raw_cells = selected_confirmation.get("cells")
    if isinstance(raw_cells, (str, bytes)) or not isinstance(raw_cells, Sequence):
        raise ValueError("confirmation requires six selected confirmation cells")
    if len(raw_cells) != 6:
        raise ValueError("confirmation requires six selected confirmation cells")
    configured = {cell.key: cell for cell in iter_tree_cells(config)}
    selected: list[TreeBenchmarkCell] = []
    for raw in raw_cells:
        if not isinstance(raw, Mapping):
            raise ValueError("Invalid selected confirmation cell")
        try:
            cell = TreeBenchmarkCell(
                panel=str(raw["panel"]),
                scenario=str(raw["scenario"]),
                n=int(raw["n"]),
                p=int(raw["p"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("Invalid selected confirmation cell") from error
        if cell.key not in configured:
            raise ValueError(
                f"Selected confirmation cell is not configured: {cell.key}"
            )
        selected.append(configured[cell.key])
    if len({cell.key for cell in selected}) != 6:
        raise ValueError("Selected confirmation cells must be unique")
    expected_groups = {
        (panel, scenario)
        for panel in config["panels"]
        for scenario in config["structures"]
    }
    if {(cell.panel, cell.scenario) for cell in selected} != expected_groups:
        raise ValueError(
            "Selected confirmation cells must cover every panel and structure"
        )
    return tuple(selected)


def iter_stage4_pairs(
    config: Mapping[str, Any],
    phase: str,
    frozen_tuning: Mapping[str, Any],
    selected_confirmation: Mapping[str, Any] | None = None,
    replications: int | None = None,
    num_shards: int = 1,
    shard_index: int = 0,
    fast: bool = False,
):
    validate_shard(num_shards, shard_index)
    if phase not in {"screening", "confirmation"}:
        raise ValueError("phase must be 'screening' or 'confirmation'")
    profile = "fast" if fast else "full"
    validate_frozen_tuning(config, frozen_tuning, profile)
    phase_config = config[phase]
    methods = tuple(phase_config["methods"])
    if methods != _STAGE4_METHODS:
        raise ValueError("Stage 4 methods do not match the exact prescribed order")
    if replications is None:
        replications = int(
            phase_config["smoke_replications"]
            if phase == "confirmation" and fast
            else phase_config["replications"]
        )
    if isinstance(replications, bool) or replications < 1:
        raise ValueError("replications must be at least 1")
    cells = (
        iter_tree_cells(config)
        if phase == "screening"
        else _selected_confirmation_cells(config, selected_confirmation)
    )
    method_pairs = tuple((method, method) for method in methods) + _ORACLE_DIAGNOSTICS
    for cell in cells:
        for replication in range(replications):
            for learner_l, learner_m in method_pairs:
                pair = Stage4PairSpec(
                    stage=str(phase_config["stage"]),
                    seed_namespace=str(phase_config["seed_namespace"]),
                    panel=cell.panel,
                    scenario=cell.scenario,
                    n=cell.n,
                    p=cell.p,
                    replication=replication,
                    learner_l=learner_l,
                    learner_m=learner_m,
                    folds_count=int(config["folds"]),
                    theta0=float(config["theta0"]),
                    execution_profile=profile,
                )
                l_method = resolve_method(
                    pair, "l", frozen_tuning, config["extra_trees"]["params"]
                )
                m_method = resolve_method(
                    pair, "m", frozen_tuning, config["extra_trees"]["params"]
                )
                pair = replace(
                    pair,
                    learner_l_config_hash=l_method.config_hash,
                    learner_m_config_hash=m_method.config_hash,
                )
                if belongs_to_shard(pair.key, num_shards, shard_index):
                    yield pair


def validate_stage4_cached_result(
    pair: Stage4PairSpec,
    result: CachedNuisanceResult,
    target: str,
) -> None:
    prediction = np.asarray(result.prediction, dtype=float)
    if prediction.ndim != 1 or len(prediction) != pair.n:
        raise ValueError(f"Cached {target} predictions have invalid length")
    if not np.isfinite(prediction).all():
        raise ValueError(f"Cached {target} predictions must be finite")
    fold_seconds = np.asarray(result.fold_seconds, dtype=float)
    if (
        fold_seconds.ndim != 1
        or len(fold_seconds) != pair.folds_count
        or not np.isfinite(fold_seconds).all()
        or np.any(fold_seconds < 0)
    ):
        raise ValueError(f"Cached {target} fold times must be finite and complete")
    if result.peak_gpu_mb is not None and (
        not np.isfinite(result.peak_gpu_mb) or result.peak_gpu_mb < 0
    ):
        raise ValueError(f"Cached {target} peak GPU memory must be finite")


def compose_stage4_record(
    pair: Stage4PairSpec,
    l_result: CachedNuisanceResult,
    m_result: CachedNuisanceResult,
) -> dict[str, Any]:
    validate_stage4_cached_result(pair, l_result, "l")
    validate_stage4_cached_result(pair, m_result, "m")
    record = compose_dml_record(_as_stage3b_pair(pair), l_result, m_result)
    record.update(
        {
            "task_key": pair.key,
            "panel": pair.panel,
            "execution_profile": pair.execution_profile,
            "learner_l_config_hash": pair.learner_l_config_hash,
            "learner_m_config_hash": pair.learner_m_config_hash,
        }
    )
    numeric_fields = (
        "theta0",
        "theta",
        "standard_error",
        "ci_lower",
        "ci_upper",
        "l_mse",
        "m_mse",
        "nuisance_error_product",
        "lm_error_cross",
        "residual_d_variance",
        "bias_numerator_proxy",
        "theta_proxy",
        "proxy_error",
        "runtime_seconds",
    )
    if any(not np.isfinite(record[field]) for field in numeric_fields):
        raise ValueError("Stage 4 DML record values must be finite")
    if (
        record["ci_lower"] > record["theta"]
        or record["theta"] > record["ci_upper"]
    ):
        raise ValueError("Stage 4 DML confidence interval is invalid")
    return record


def validate_stage4_record(
    record: Mapping[str, Any], pair: Stage4PairSpec
) -> Mapping[str, Any]:
    expected = {
        "task_key": pair.key,
        "stage": pair.stage,
        "seed_namespace": pair.effective_seed_namespace,
        "panel": pair.panel,
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
        "execution_profile": pair.execution_profile,
        "data_seed": derive_seed(
            pair.effective_seed_namespace,
            pair.scenario,
            pair.n,
            pair.p,
            pair.replication,
            "data",
        ),
        "fold_seed": derive_seed(
            pair.effective_seed_namespace,
            pair.scenario,
            pair.n,
            pair.p,
            pair.replication,
            "folds",
        ),
        "status": "success",
    }
    for field, expected_value in expected.items():
        if record.get(field) != expected_value:
            raise ValueError(f"Invalid Stage 4 record {pair.key}: {field} mismatch")
    numeric_fields = (
        "theta",
        "standard_error",
        "ci_lower",
        "ci_upper",
        "l_mse",
        "m_mse",
        "nuisance_error_product",
        "lm_error_cross",
        "residual_d_variance",
        "bias_numerator_proxy",
        "theta_proxy",
        "proxy_error",
        "runtime_seconds",
    )
    for field in numeric_fields:
        value = record.get(field)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not np.isfinite(value)
        ):
            raise ValueError(
                f"Invalid Stage 4 record {pair.key}: {field} must be finite"
            )
    if (
        record["ci_lower"] > record["theta"]
        or record["theta"] > record["ci_upper"]
    ):
        raise ValueError(f"Invalid Stage 4 record {pair.key}: confidence interval")
    for target in ("l", "m"):
        values = record.get(f"{target}_fold_seconds")
        if (
            not isinstance(values, list)
            or len(values) != pair.folds_count
            or not np.isfinite(np.asarray(values, dtype=float)).all()
            or np.any(np.asarray(values, dtype=float) < 0)
        ):
            raise ValueError(
                f"Invalid Stage 4 record {pair.key}: {target}_fold_seconds"
            )
    peak = record.get("peak_gpu_mb")
    if peak is not None and (
        isinstance(peak, bool)
        or not isinstance(peak, (int, float))
        or not np.isfinite(peak)
        or peak < 0
    ):
        raise ValueError(f"Invalid Stage 4 record {pair.key}: peak_gpu_mb")
    return record
