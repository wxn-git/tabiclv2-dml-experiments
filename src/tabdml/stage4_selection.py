from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from .stage4_config import TreeBenchmarkCell, iter_tree_cells
from .stage4_experiment import (
    STAGE4_SELECTION_RULE,
    Stage4PairSpec,
    stage4_configuration_fingerprint,
    validate_stage4_record,
    validate_stage4_selection,
)
from .stage4_tuning import iter_tuning_tasks


_SELECTION_METHODS = ("tabiclv2_1", "xgboost_tuned")
_EXECUTION_PROFILES = frozenset({"full", "fast"})


def _finite_number(value: Any, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not np.isfinite(value)
    ):
        raise ValueError(f"{name} must be a finite numeric value")
    return float(value)


def paired_squared_error_advantage(
    theta_tab: float,
    theta_xgb: float,
    theta0: float,
) -> float:
    tab = _finite_number(theta_tab, "theta_tab")
    xgb = _finite_number(theta_xgb, "theta_xgb")
    truth = _finite_number(theta0, "theta0")
    result = (tab - truth) ** 2 - (xgb - truth) ** 2
    if not np.isfinite(result):
        raise ValueError("paired squared error difference must be a finite numeric value")
    return result


def _required_replications(
    config: Mapping[str, Any],
    execution_profile: str,
    expected_replications: int | None,
) -> int:
    if execution_profile not in _EXECUTION_PROFILES:
        raise ValueError("execution_profile must be 'full' or 'fast'")
    required = (
        1
        if execution_profile == "fast"
        else int(config["screening"]["replications"])
    )
    actual = required if expected_replications is None else expected_replications
    if (
        isinstance(actual, bool)
        or not isinstance(actual, int)
        or actual != required
    ):
        raise ValueError(
            "expected_replications does not match the "
            f"{execution_profile} profile contract ({required})"
        )
    return required


def _allowed_tuned_hashes(
    config: Mapping[str, Any], execution_profile: str
) -> dict[tuple[str, str, int, int, str], frozenset[str]]:
    hashes: dict[tuple[str, str, int, int, str], set[str]] = {}
    for task in iter_tuning_tasks(
        config,
        replications=1,
        fast=execution_profile == "fast",
    ):
        identity = (task.panel, task.scenario, task.n, task.p, task.target)
        hashes.setdefault(identity, set()).add(task.config_hash)
    return {identity: frozenset(values) for identity, values in hashes.items()}


def _record_cell(
    record: Mapping[str, Any], configured: Mapping[str, TreeBenchmarkCell]
) -> TreeBenchmarkCell:
    panel = record.get("panel")
    scenario = record.get("scenario")
    n = record.get("n")
    p = record.get("p")
    if (
        not isinstance(panel, str)
        or not isinstance(scenario, str)
        or isinstance(n, bool)
        or not isinstance(n, int)
        or isinstance(p, bool)
        or not isinstance(p, int)
    ):
        raise ValueError("Relevant screening record has an invalid configured cell")
    key = f"{panel}__{scenario}__n{n}__p{p}"
    if key not in configured:
        raise ValueError(f"Relevant screening record is not a configured cell: {key}")
    return configured[key]


def _record_pair(
    record: Mapping[str, Any],
    cell: TreeBenchmarkCell,
    config: Mapping[str, Any],
    execution_profile: str,
    method: str,
    replication: int,
    allowed_hashes: Mapping[tuple[str, str, int, int, str], frozenset[str]],
) -> Stage4PairSpec:
    l_hash = record.get("learner_l_config_hash")
    m_hash = record.get("learner_m_config_hash")
    if not isinstance(l_hash, str) or not l_hash:
        raise ValueError("Relevant screening record learner_l_config_hash mismatch")
    if not isinstance(m_hash, str) or not m_hash:
        raise ValueError("Relevant screening record learner_m_config_hash mismatch")
    if method == "tabiclv2_1":
        if l_hash != "default" or m_hash != "default":
            raise ValueError("TabICLv2 screening record config_hash mismatch")
    else:
        base = (cell.panel, cell.scenario, cell.n, cell.p)
        if l_hash not in allowed_hashes[(*base, "l")]:
            raise ValueError("Tuned-XGBoost learner_l_config_hash mismatch")
        if m_hash not in allowed_hashes[(*base, "m")]:
            raise ValueError("Tuned-XGBoost learner_m_config_hash mismatch")
    return Stage4PairSpec(
        stage=str(config["screening"]["stage"]),
        seed_namespace=str(config["screening"]["seed_namespace"]),
        panel=cell.panel,
        scenario=cell.scenario,
        n=cell.n,
        p=cell.p,
        replication=replication,
        learner_l=method,
        learner_m=method,
        folds_count=int(config["folds"]),
        theta0=float(config["theta0"]),
        learner_l_config_hash=l_hash,
        learner_m_config_hash=m_hash,
        execution_profile=execution_profile,
    )


def select_confirmation_cells(
    records: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    expected_replications: int | None = None,
    execution_profile: str = "full",
) -> dict[str, Any]:
    cells = iter_tree_cells(config)
    configured = {cell.key: cell for cell in cells}
    replications = _required_replications(
        config, execution_profile, expected_replications
    )
    if isinstance(records, (str, bytes)) or not isinstance(records, Sequence):
        raise ValueError("screening records must be a sequence")

    allowed_hashes = _allowed_tuned_hashes(config, execution_profile)
    selected_records: dict[tuple[str, int, str], Mapping[str, Any]] = {}
    selected_task_keys: set[str] = set()
    tuned_hashes_by_cell: dict[str, tuple[str, str]] = {}
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise ValueError(f"screening record {index} must be a mapping")
        learner_l = record.get("learner_l")
        learner_m = record.get("learner_m")
        touches_selection = (
            learner_l in _SELECTION_METHODS or learner_m in _SELECTION_METHODS
        )
        if not touches_selection:
            continue
        if learner_l != learner_m or learner_l not in _SELECTION_METHODS:
            raise ValueError("Relevant screening record has asymmetric methods")
        method = str(learner_l)
        if record.get("status") != "success":
            raise ValueError("Relevant screening record status must be success")

        cell = _record_cell(record, configured)
        replication = record.get("replication")
        if (
            isinstance(replication, bool)
            or not isinstance(replication, int)
            or replication not in range(replications)
        ):
            raise ValueError("Relevant screening record replication is unexpected")
        pair = _record_pair(
            record,
            cell,
            config,
            execution_profile,
            method,
            replication,
            allowed_hashes,
        )
        validate_stage4_record(record, pair)

        task_key = record.get("task_key")
        if task_key in selected_task_keys:
            raise ValueError(f"duplicate screening task_key: {task_key}")
        selected_task_keys.add(str(task_key))
        identity = (cell.key, replication, method)
        if identity in selected_records:
            raise ValueError(
                "duplicate screening cell/method/replication identity: "
                f"{cell.key}/{method}/{replication}"
            )
        selected_records[identity] = record

        if method == "xgboost_tuned":
            hashes = (
                str(record["learner_l_config_hash"]),
                str(record["learner_m_config_hash"]),
            )
            previous = tuned_hashes_by_cell.setdefault(cell.key, hashes)
            if previous != hashes:
                raise ValueError(
                    f"Tuned-XGBoost config_hash changed within cell {cell.key}"
                )

    expected_identities = {
        (cell.key, replication, method)
        for cell in cells
        for replication in range(replications)
        for method in _SELECTION_METHODS
    }
    if set(selected_records) != expected_identities:
        missing = expected_identities.difference(selected_records)
        extra = set(selected_records).difference(expected_identities)
        raise ValueError(
            "Screening records must contain complete paired replications for "
            f"all 24 configured cells (missing={len(missing)}, extra={len(extra)})"
        )

    theta0 = _finite_number(config["theta0"], "theta0")
    ranking = []
    for cell in cells:
        deltas = []
        for replication in range(replications):
            tab = selected_records[(cell.key, replication, "tabiclv2_1")]
            xgb = selected_records[(cell.key, replication, "xgboost_tuned")]
            deltas.append(
                paired_squared_error_advantage(
                    tab.get("theta"), xgb.get("theta"), theta0
                )
            )
        ranking.append(
            {
                "panel": cell.panel,
                "scenario": cell.scenario,
                "n": cell.n,
                "p": cell.p,
                "mean_paired_squared_error_difference": float(np.mean(deltas)),
                "selection_rule": STAGE4_SELECTION_RULE,
            }
        )

    groups = list(
        dict.fromkeys((row["panel"], row["scenario"]) for row in ranking)
    )
    chosen = [
        min(
            (
                row
                for row in ranking
                if (row["panel"], row["scenario"]) == group
            ),
            key=lambda row: (
                row["mean_paired_squared_error_difference"],
                row["n"],
                row["p"],
            ),
        )
        for group in groups
    ]
    artifact = {
        "execution_profile": execution_profile,
        "screening_stage": config["screening"]["stage"],
        "screening_seed_namespace": config["screening"]["seed_namespace"],
        "expected_screening_replications": replications,
        "selection_rule": STAGE4_SELECTION_RULE,
        "config_fingerprint": stage4_configuration_fingerprint(config),
        "screening_ranking": ranking,
        "cells": chosen,
    }
    validate_stage4_selection(config, artifact, execution_profile)
    return artifact


def write_confirmation_cells(
    records: Sequence[Mapping[str, Any]],
    output_path: str | Path,
    config: Mapping[str, Any],
    expected_replications: int | None = None,
    execution_profile: str = "full",
) -> dict[str, Any]:
    selected = select_confirmation_cells(
        records,
        config,
        expected_replications=expected_replications,
        execution_profile=execution_profile,
    )
    validate_stage4_selection(config, selected, execution_profile)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(selected, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return selected
