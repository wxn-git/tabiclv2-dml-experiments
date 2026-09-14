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
from .stage4_config import (
    TreeBenchmarkCell,
    iter_tree_cells,
    resolve_stage4_replications,
)
from .stage4_tuning import tuning_run_fingerprint


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
STAGE4_SELECTION_RULE = "minimum_mean_tab_minus_xgb_squared_error"
_SELECTION_FIELDS = frozenset(
    {
        "execution_profile",
        "screening_stage",
        "screening_seed_namespace",
        "expected_screening_replications",
        "selection_rule",
        "config_fingerprint",
        "screening_ranking",
        "cells",
    }
)
_SELECTION_ROW_FIELDS = frozenset(
    {
        "panel",
        "scenario",
        "n",
        "p",
        "mean_paired_squared_error_difference",
        "selection_rule",
    }
)
_RESUMABLE_STATUSES = frozenset({"success", "failed", "oom"})


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


def stage4_configuration_fingerprint(config: Mapping[str, Any]) -> str:
    iter_tree_cells(config)
    return _params_hash(dict(config))


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
    required_replications = (
        1
        if execution_profile == "fast"
        else int(config["tuning"]["replications"])
    )
    if expected_replications != required_replications or isinstance(
        expected_replications, bool
    ):
        raise ValueError(
            "Frozen tuning expected_replications does not match the "
            f"{execution_profile} profile contract"
        )
    expected_run_provenance = {
        "tuning_stage": config["tuning"]["stage"],
        "tuning_seed_namespace": config["tuning"]["seed_namespace"],
        "tuning_run_fingerprint": tuning_run_fingerprint(
            config,
            required_replications,
            execution_profile,
        ),
    }
    for field, expected_value in expected_run_provenance.items():
        value = frozen_tuning.get(field)
        if not isinstance(value, str) or value != expected_value:
            raise ValueError(f"Frozen tuning {field} mismatch")
    theta0 = frozen_tuning.get("theta0")
    if (
        isinstance(theta0, bool)
        or not isinstance(theta0, (int, float))
        or not np.isfinite(theta0)
        or theta0 != config["theta0"]
    ):
        raise ValueError("Frozen tuning theta0 mismatch")

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


def _require_native_cell_integer(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"Selection {field} must be a native integer")
    return value


def _validate_selection_row(
    row: Any,
    configured: Mapping[str, TreeBenchmarkCell],
    location: str,
) -> tuple[TreeBenchmarkCell, float]:
    if not isinstance(row, Mapping) or set(row) != _SELECTION_ROW_FIELDS:
        raise ValueError(f"Invalid selection row schema at {location}")
    panel = row["panel"]
    scenario = row["scenario"]
    if not isinstance(panel, str) or not isinstance(scenario, str):
        raise ValueError(f"Invalid panel/scenario at {location}")
    n = _require_native_cell_integer(row["n"], "n")
    p = _require_native_cell_integer(row["p"], "p")
    cell_key = f"{panel}__{scenario}__n{n}__p{p}"
    if cell_key not in configured:
        raise ValueError(f"Selection cell is not configured: {cell_key}")
    if row["selection_rule"] != STAGE4_SELECTION_RULE:
        raise ValueError(f"Invalid selection_rule at {location}")
    score = row["mean_paired_squared_error_difference"]
    if (
        isinstance(score, bool)
        or not isinstance(score, (int, float))
        or not np.isfinite(score)
    ):
        raise ValueError(f"Selection score must be finite at {location}")
    return configured[cell_key], float(score)


def validate_stage4_selection(
    config: Mapping[str, Any],
    selected_confirmation: Mapping[str, Any],
    execution_profile: str,
) -> tuple[TreeBenchmarkCell, ...]:
    if execution_profile not in _EXECUTION_PROFILES:
        raise ValueError("execution_profile must be 'full' or 'fast'")
    if not isinstance(selected_confirmation, Mapping):
        raise ValueError("Selected confirmation artifact must be a mapping")
    if set(selected_confirmation) != _SELECTION_FIELDS:
        raise ValueError("Selected confirmation artifact schema is invalid")
    if selected_confirmation["execution_profile"] != execution_profile:
        raise ValueError("Selected confirmation execution_profile mismatch")
    screening = config["screening"]
    if selected_confirmation["screening_stage"] != screening["stage"]:
        raise ValueError("Selected confirmation screening_stage mismatch")
    if (
        selected_confirmation["screening_seed_namespace"]
        != screening["seed_namespace"]
    ):
        raise ValueError(
            "Selected confirmation screening_seed_namespace mismatch"
        )
    expected_replications = (
        1
        if execution_profile == "fast"
        else int(screening["replications"])
    )
    actual_replications = selected_confirmation[
        "expected_screening_replications"
    ]
    if (
        isinstance(actual_replications, bool)
        or actual_replications != expected_replications
    ):
        raise ValueError(
            "Selected confirmation screening replications do not match "
            f"the {execution_profile} profile contract"
        )
    if selected_confirmation["selection_rule"] != STAGE4_SELECTION_RULE:
        raise ValueError("Selected confirmation selection_rule mismatch")
    if (
        selected_confirmation["config_fingerprint"]
        != stage4_configuration_fingerprint(config)
    ):
        raise ValueError("Selected confirmation config_fingerprint mismatch")

    configured = {cell.key: cell for cell in iter_tree_cells(config)}
    ranking = selected_confirmation["screening_ranking"]
    if isinstance(ranking, (str, bytes)) or not isinstance(ranking, Sequence):
        raise ValueError("screening_ranking must contain all 24 cells")
    if len(ranking) != len(configured):
        raise ValueError("screening_ranking must contain all 24 cells")
    ranked: dict[str, tuple[Mapping[str, Any], TreeBenchmarkCell, float]] = {}
    for index, row in enumerate(ranking):
        cell, score = _validate_selection_row(
            row, configured, f"screening_ranking[{index}]"
        )
        if cell.key in ranked:
            raise ValueError(f"Duplicate screening_ranking cell: {cell.key}")
        ranked[cell.key] = (row, cell, score)
    if set(ranked) != set(configured):
        raise ValueError("screening_ranking must contain all 24 configured cells")

    raw_cells = selected_confirmation["cells"]
    if isinstance(raw_cells, (str, bytes)) or not isinstance(
        raw_cells, Sequence
    ) or len(raw_cells) != 6:
        raise ValueError("confirmation requires six selected confirmation cells")
    chosen: dict[tuple[str, str], tuple[Mapping[str, Any], TreeBenchmarkCell]] = {}
    selected_cells: list[TreeBenchmarkCell] = []
    for index, row in enumerate(raw_cells):
        cell, _ = _validate_selection_row(row, configured, f"cells[{index}]")
        ranked_row = ranked[cell.key][0]
        if dict(row) != dict(ranked_row):
            raise ValueError("Chosen cells must exactly match screening_ranking rows")
        group = (cell.panel, cell.scenario)
        if group in chosen:
            raise ValueError(
                "Chosen cells must contain one cell per panel and scenario"
            )
        chosen[group] = (row, cell)
        selected_cells.append(cell)

    grouped_ranking: dict[
        tuple[str, str], list[tuple[Mapping[str, Any], TreeBenchmarkCell, float]]
    ] = {}
    for value in ranked.values():
        cell = value[1]
        grouped_ranking.setdefault((cell.panel, cell.scenario), []).append(value)
    if set(chosen) != set(grouped_ranking):
        raise ValueError("Chosen cells must contain one cell per panel and scenario")
    for group, values in grouped_ranking.items():
        expected = min(values, key=lambda value: (value[2], value[1].n, value[1].p))
        if dict(chosen[group][0]) != dict(expected[0]):
            raise ValueError("Chosen cells must be the deterministic minima")
    return tuple(selected_cells)


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
    repairing = False
    if cache.exists(task):
        try:
            cached = cache.read(task, expected_length=task.n)
            validate_stage4_cached_result(effective_pair, cached, target)
            return cached
        except ValueError as error:
            if not retry_failed:
                raise ValueError(
                    f"Stage 4 nuisance cache integrity error for {task.key}: "
                    f"{error}"
                ) from error
            cache.path(task).unlink()
            repairing = True
    result = fit_cached_nuisance(
        task,
        cache_root=cache_root,
        theta0=effective_pair.theta0,
        fast=profile == "fast",
        learner_kind=resolved.learner_kind,
        learner_params=resolved.params,
    )
    try:
        validate_stage4_cached_result(effective_pair, result, target)
    except ValueError as error:
        cache.path(task).unlink(missing_ok=True)
        operation = "rebuilt nuisance" if repairing else "fitted nuisance"
        raise ValueError(
            f"Stage 4 {operation} failed integrity validation for "
            f"{task.key}: {error}"
        ) from error
    return result


def _selected_confirmation_cells(
    config: Mapping[str, Any],
    selected_confirmation: Mapping[str, Any] | None,
    execution_profile: str,
) -> tuple[TreeBenchmarkCell, ...]:
    if selected_confirmation is None:
        raise ValueError("confirmation requires six selected confirmation cells")
    return validate_stage4_selection(
        config, selected_confirmation, execution_profile
    )


def validate_stage4_preflight(
    config: Mapping[str, Any],
    phase: str,
    replications: int | None,
    *,
    fast: bool = False,
    preflight: bool = False,
) -> int | None:
    """Resolve the opt-in full-model preflight count without changing config."""
    if not preflight:
        return replications
    return resolve_stage4_replications(
        config, phase, replications, fast=fast, preflight=True
    )


def iter_stage4_pairs(
    config: Mapping[str, Any],
    phase: str,
    frozen_tuning: Mapping[str, Any],
    selected_confirmation: Mapping[str, Any] | None = None,
    replications: int | None = None,
    num_shards: int = 1,
    shard_index: int = 0,
    fast: bool = False,
    preflight: bool = False,
):
    validate_shard(num_shards, shard_index)
    replications = resolve_stage4_replications(
        config, phase, replications, fast=fast, preflight=preflight,
    )
    if phase not in {"screening", "confirmation"}:
        raise ValueError("phase must be 'screening' or 'confirmation'")
    profile = "fast" if fast else "full"
    validate_frozen_tuning(config, frozen_tuning, profile)
    phase_config = config[phase]
    methods = tuple(phase_config["methods"])
    if methods != _STAGE4_METHODS:
        raise ValueError("Stage 4 methods do not match the exact prescribed order")
    cells = (
        iter_tree_cells(config)
        if phase == "screening"
        else _selected_confirmation_cells(
            config, selected_confirmation, profile
        )
    )
    method_pairs = tuple((method, method) for method in methods) + _ORACLE_DIAGNOSTICS
    # Do not mutate config: selections/frozen models bind to the original design.
    # Stage separates DML keys; namespace separates nuisance keys and data/folds.
    suffix = "_preflight" if preflight else ""
    for cell in cells:
        for replication in range(replications):
            for learner_l, learner_m in method_pairs:
                pair = Stage4PairSpec(
                    stage=str(phase_config["stage"]) + suffix,
                    seed_namespace=str(phase_config["seed_namespace"]) + suffix,
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


def _expected_stage4_record_identity(pair: Stage4PairSpec) -> dict[str, Any]:
    return {
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
    }


def _validate_stage4_record_identity(
    record: Mapping[str, Any], pair: Stage4PairSpec
) -> None:
    expected = _expected_stage4_record_identity(pair)
    string_fields = frozenset(
        {
            "task_key",
            "stage",
            "seed_namespace",
            "panel",
            "scenario",
            "learner_l",
            "learner_m",
            "learner_l_config_hash",
            "learner_m_config_hash",
            "execution_profile",
        }
    )
    integer_fields = frozenset(
        {
            "n",
            "p",
            "replication",
            "folds_count",
            "data_seed",
            "fold_seed",
        }
    )
    for field, expected_value in expected.items():
        value = record.get(field)
        if field in string_fields:
            valid_type = isinstance(value, str)
        elif field in integer_fields:
            valid_type = isinstance(value, int) and not isinstance(value, bool)
        else:
            valid_type = (
                not isinstance(value, bool)
                and isinstance(value, (int, float))
                and np.isfinite(value)
            )
        if not valid_type or value != expected_value:
            raise ValueError(f"Invalid Stage 4 record {pair.key}: {field} mismatch")


def validate_stage4_resume_record(
    record: Mapping[str, Any], pair: Stage4PairSpec
) -> str:
    status = record.get("status")
    if status not in _RESUMABLE_STATUSES:
        raise ValueError(
            f"Invalid Stage 4 record {pair.key}: status is not resumable"
        )
    _validate_stage4_record_identity(record, pair)
    if status == "success":
        validate_stage4_record(record, pair)
    return str(status)


def validate_stage4_record(
    record: Mapping[str, Any], pair: Stage4PairSpec
) -> Mapping[str, Any]:
    _validate_stage4_record_identity(record, pair)
    if record.get("status") != "success":
        raise ValueError(f"Invalid Stage 4 record {pair.key}: status mismatch")
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
