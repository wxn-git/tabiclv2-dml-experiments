from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from .stage4_config import iter_tree_cells
from .stage4_experiment import (
    STAGE4_SELECTION_RULE,
    iter_stage4_pairs,
    stage4_configuration_fingerprint,
    validate_frozen_tuning,
    validate_stage4_record,
    validate_stage4_selection,
)


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


def select_confirmation_cells(
    records: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    frozen_tuning: Mapping[str, Any],
    expected_replications: int | None = None,
    execution_profile: str = "full",
) -> dict[str, Any]:
    cells = iter_tree_cells(config)
    replications = _required_replications(
        config, execution_profile, expected_replications
    )
    if isinstance(records, (str, bytes)) or not isinstance(records, Sequence):
        raise ValueError("screening records must be a sequence")

    validate_frozen_tuning(config, frozen_tuning, execution_profile)
    expected_pairs = tuple(
        iter_stage4_pairs(
            config,
            "screening",
            frozen_tuning,
            replications=replications,
            fast=execution_profile == "fast",
        )
    )
    expected_by_key = {pair.key: pair for pair in expected_pairs}
    if len(expected_by_key) != len(expected_pairs):
        raise ValueError("Expected Stage 4 screening task keys must be unique")

    validated_records: dict[str, Mapping[str, Any]] = {}
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise ValueError(f"screening record {index} must be a mapping")
        task_key = record.get("task_key")
        if not isinstance(task_key, str) or task_key not in expected_by_key:
            raise ValueError(f"Invalid screening record: unexpected task_key {task_key}")
        if task_key in validated_records:
            raise ValueError(f"duplicate screening task_key: {task_key}")
        validate_stage4_record(record, expected_by_key[task_key])
        validated_records[task_key] = record

    if set(validated_records) != set(expected_by_key):
        missing = set(expected_by_key).difference(validated_records)
        raise ValueError(
            "Screening records must contain the complete screening task universe "
            f"for all 24 configured cells (missing={len(missing)})"
        )

    primary_records = {
        (
            f"{pair.panel}__{pair.scenario}__n{pair.n}__p{pair.p}",
            pair.replication,
            pair.learner_l,
        ): validated_records[pair.key]
        for pair in expected_pairs
        if pair.learner_l == pair.learner_m
        and pair.learner_l in _SELECTION_METHODS
    }

    theta0 = _finite_number(config["theta0"], "theta0")
    ranking = []
    for cell in cells:
        deltas = []
        for replication in range(replications):
            tab = primary_records[(cell.key, replication, "tabiclv2_1")]
            xgb = primary_records[(cell.key, replication, "xgboost_tuned")]
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
    frozen_tuning: Mapping[str, Any],
    expected_replications: int | None = None,
    execution_profile: str = "full",
) -> dict[str, Any]:
    selected = select_confirmation_cells(
        records,
        config,
        frozen_tuning,
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
