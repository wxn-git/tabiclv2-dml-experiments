from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import numpy as np

from .dgp import simulate_plr


_DECLARED_ROOTS = (
    ("tree_stumps", "m", 0),
    ("tree_stumps", "m", 1),
    ("tree_stumps", "m", 2),
    ("tree_stumps", "g", 0),
    ("tree_stumps", "g", 3),
    ("tree_stumps", "g", 4),
    ("tree_hierarchical", "m", 0),
    ("tree_hierarchical", "g", 0),
    ("tree_forest_sum", "m", 0),
    ("tree_forest_sum", "m", 3),
    ("tree_forest_sum", "g", 0),
    ("tree_forest_sum", "g", 3),
)

_AUDIT_FIELDS = (
    "scenario",
    "target",
    "root_variable",
    "threshold",
    "split_gain",
    "left_probability",
    "left_mean",
    "right_mean",
)


def split_gain(values, feature, threshold=0.0) -> float:
    values = np.asarray(values, dtype=float)
    left = np.asarray(feature) <= threshold
    right = ~left
    if not left.any() or not right.any():
        raise ValueError("A split must have observations on both sides.")
    parent = float(np.var(values))
    child = float(
        left.mean() * np.var(values[left])
        + right.mean() * np.var(values[right])
    )
    return parent - child


def audit_tree_structures(
    n: int = 200_000,
    seed: int = 20260903,
) -> list[dict]:
    rows = []
    simulated = {
        scenario: simulate_plr(scenario, n=n, p=10, seed=seed)
        for scenario in {row[0] for row in _DECLARED_ROOTS}
    }
    for scenario, target, root_variable in _DECLARED_ROOTS:
        data = simulated[scenario]
        values = data.m0 if target == "m" else data.g0
        feature = data.X[:, root_variable]
        left = feature <= 0.0
        rows.append(
            {
                "scenario": scenario,
                "target": target,
                "root_variable": root_variable,
                "threshold": 0.0,
                "split_gain": split_gain(values, feature),
                "left_probability": float(left.mean()),
                "left_mean": float(values[left].mean()),
                "right_mean": float(values[~left].mean()),
            }
        )
    return rows


def write_structure_audit(records, output_dir) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = list(records)

    json_path = output_dir / "structure_checks.json"
    json_temporary = json_path.with_suffix(".json.tmp")
    with json_temporary.open("w", encoding="utf-8") as handle:
        json.dump(records, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(json_temporary, json_path)

    csv_path = output_dir / "structure_checks.csv"
    csv_temporary = csv_path.with_suffix(".csv.tmp")
    with csv_temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_AUDIT_FIELDS)
        writer.writeheader()
        writer.writerows(records)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(csv_temporary, csv_path)
