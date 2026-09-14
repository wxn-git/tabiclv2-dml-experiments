"""Audited provenance migration from legacy six-method Stage 5 caches."""
from __future__ import annotations

import copy
import hashlib
import json
import os
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .config import derive_seed
from .nuisance_cache import NuisanceCache
from .stage5_config import stage5_config_fingerprint
from .stage5_experiment import (
    build_stage5_nuisance_spec, iter_stage5_pairs, read_stage5_nuisance,
    resolve_stage5_method, stage5_nuisance_metadata_path,
)
from .stage5_tuning import (
    load_stage5_tuning, stage5_tuning_run_fingerprint,
    validate_frozen_stage5_tuning,
)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(dict(value), handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _without_methods(config: Mapping[str, Any]) -> dict[str, Any]:
    value = copy.deepcopy(dict(config))
    value.pop("methods", None)
    return value


def rebind_stage5_tuning(
    source_config: Mapping[str, Any], destination_config: Mapping[str, Any],
    source_path: str | Path, destination_path: str | Path,
    execution_profile: str = "full",
) -> dict[str, Any]:
    if _without_methods(source_config) != _without_methods(destination_config):
        raise ValueError("Stage 5 configs differ in computation-affecting fields")
    source = dict(load_stage5_tuning(source_path, source_config, execution_profile))
    rebound = copy.deepcopy(source)
    replications = 1 if execution_profile == "fast" else destination_config["tuning"]["replications"]
    rebound["config_fingerprint"] = stage5_config_fingerprint(destination_config)
    rebound["tuning_run_fingerprint"] = stage5_tuning_run_fingerprint(
        destination_config, replications, execution_profile,
    )
    validate_frozen_stage5_tuning(rebound, destination_config, execution_profile)
    _atomic_json(Path(destination_path), rebound)
    return rebound


def _task_map(config, frozen, profile):
    result = {}
    for pair in iter_stage5_pairs(config, frozen, profile):
        for target in ("l", "m"):
            identity = (pair.scenario, pair.n, pair.p, pair.replication, pair.method, target)
            resolved = resolve_stage5_method(pair, target, config, frozen)
            task = build_stage5_nuisance_spec(pair, target, resolved)
            value = (pair, task, resolved)
            if identity in result and result[identity] != value:
                raise ValueError(f"Stage 5 migration identity collision: {identity}")
            result[identity] = value
    return result


def _prediction_hash(prediction) -> str:
    array = np.asarray(prediction, dtype="<f8").reshape(-1)
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def migrate_stage5_cache(
    source_config: Mapping[str, Any], destination_config: Mapping[str, Any],
    source_tuning_path: str | Path, destination_tuning_path: str | Path,
    source_cache_root: str | Path, destination_cache_root: str | Path,
    manifest_path: str | Path, *, profile: str = "preflight", resume: bool = False,
) -> dict[str, Any]:
    if _without_methods(source_config) != _without_methods(destination_config):
        raise ValueError("Stage 5 configs differ in computation-affecting fields")
    if tuple(destination_config["methods"]) != (
        "tabiclv2_1", "tabiclv2_8", "xgboost_tuned", "extra_trees", "lasso"
    ):
        raise ValueError("Destination must be the exact five-method protocol")
    source_tuning = load_stage5_tuning(source_tuning_path, source_config, "full")
    destination_tuning = rebind_stage5_tuning(
        source_config, destination_config, source_tuning_path,
        destination_tuning_path, "full",
    )
    old = _task_map(source_config, source_tuning, profile)
    new = _task_map(destination_config, destination_tuning, profile)
    if set(new) - set(old):
        raise ValueError("Destination tasks are not a subset of source tasks")
    expected = 30 * 5 * len(destination_config["methods"]) * 2
    if len(new) != expected:
        raise ValueError(f"Migration requires exactly {expected} nuisance tasks")
    source_cache = Path(source_cache_root)
    destination_cache = NuisanceCache(destination_cache_root)
    entries = []
    counts = Counter()
    for identity in sorted(new):
        old_pair, old_task, old_resolved = old[identity]
        new_pair, new_task, new_resolved = new[identity]
        if (old_pair.data_seed, old_pair.fold_seed, old_task.learner_seed) != (
            new_pair.data_seed, new_pair.fold_seed, new_task.learner_seed
        ):
            raise ValueError(f"Migration seed mismatch: {identity}")
        if (old_resolved.learner, old_resolved.learner_kind, old_resolved.params,
            old_resolved.requested_device) != (
            new_resolved.learner, new_resolved.learner_kind, new_resolved.params,
            new_resolved.requested_device,
        ):
            raise ValueError(f"Migration learner mismatch: {identity}")
        old_result = read_stage5_nuisance(
            old_task, old_resolved, source_cache, full_settings=True,
        )
        if old_result.fallback_reason:
            raise ValueError(f"Migration rejects fallback: {identity}")
        destination_npz = destination_cache.path(new_task)
        destination_meta = stage5_nuisance_metadata_path(destination_cache, new_task)
        if destination_npz.exists() or destination_meta.exists():
            if not resume or not (destination_npz.exists() and destination_meta.exists()):
                raise ValueError(f"Conflicting migration destination: {identity}")
            existing = read_stage5_nuisance(
                new_task, new_resolved, destination_cache.root, full_settings=True,
            )
            if _prediction_hash(existing.prediction) != _prediction_hash(old_result.prediction):
                raise ValueError(f"Migrated prediction mismatch: {identity}")
        else:
            destination_cache.write(
                new_task, old_result.prediction, old_result.fold_seconds,
                old_result.peak_gpu_mb, old_result.fallback_reason,
            )
            metadata = {
                "schema_version": "stage5_nuisance_metadata_v1",
                "task": asdict(new_task),
                "resolved": {
                    "learner": new_resolved.learner,
                    "learner_kind": new_resolved.learner_kind,
                    "params": new_resolved.params,
                    "config_hash": new_resolved.config_hash,
                },
                "data_seed": derive_seed(new_task.seed_namespace, new_task.scenario, new_task.n, new_task.p, new_task.replication, "data"),
                "fold_seed": derive_seed(new_task.seed_namespace, new_task.scenario, new_task.n, new_task.p, new_task.replication, "folds"),
                "full_settings": True,
                "requested_device": old_result.requested_device,
                "observed_device": old_result.observed_device,
                "fit_time": old_result.fit_time,
                "total_time": old_result.total_time,
            }
            _atomic_json(destination_meta, metadata)
        checked = read_stage5_nuisance(
            new_task, new_resolved, destination_cache.root, full_settings=True,
        )
        source_hash = _prediction_hash(old_result.prediction)
        destination_hash = _prediction_hash(checked.prediction)
        if source_hash != destination_hash:
            raise ValueError(f"Post-write prediction mismatch: {identity}")
        counts[identity[4]] += 1
        entries.append({"identity": list(identity), "source_task_key": old_task.key,
                        "destination_task_key": new_task.key,
                        "source_prediction_sha256": source_hash,
                        "destination_prediction_sha256": destination_hash})
    expected_npz = {destination_cache.path(task) for _, task, _ in new.values()}
    expected_meta = {stage5_nuisance_metadata_path(destination_cache, task) for _, task, _ in new.values()}
    if set(destination_cache.root.glob("*.npz")) != expected_npz:
        raise ValueError("Migrated NPZ universe is not exact")
    if set(destination_cache.root.glob("stage5-meta-*.json")) != expected_meta:
        raise ValueError("Migrated metadata universe is not exact")
    manifest = {
        "schema_version": "stage5_cache_migration_v1", "profile": profile,
        "source_config_fingerprint": stage5_config_fingerprint(source_config),
        "destination_config_fingerprint": stage5_config_fingerprint(destination_config),
        "source_tuning_fingerprint": source_tuning["tuning_run_fingerprint"],
        "destination_tuning_fingerprint": destination_tuning["tuning_run_fingerprint"],
        "expected_caches": expected, "migrated_caches": len(entries),
        "per_method": dict(sorted(counts.items())), "violations": 0,
        "source_cache_unchanged": True, "entries": entries,
    }
    _atomic_json(Path(manifest_path), manifest)
    return manifest

