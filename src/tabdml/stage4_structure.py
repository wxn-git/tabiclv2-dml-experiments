from __future__ import annotations

import csv
import itertools
import json
import math
import os
from pathlib import Path

import numpy as np

from .dgp import simulate_plr


AUDIT_SCHEMA = "stage4_structure_audit_v2"
AUDIT_MIN_N = 200_000
AUDIT_P = 10
GAIN_TOLERANCE = 1e-3
MSE_TOLERANCE = 1e-24

# These executable declarations are the code audit. Only axis-aligned stumps and
# depth-two trees can be represented; reconstruction against simulate_plr guards
# against declarations drifting away from the actual DGP implementation.
_FORMULA_DECLARATIONS = (
    {
        "scenario": "tree_stumps",
        "target": "m",
        "terms": (
            {"kind": "stump", "root_variable": 0, "threshold": 0.0, "coefficient": 0.9},
            {"kind": "stump", "root_variable": 1, "threshold": 0.0, "coefficient": -0.7},
            {"kind": "stump", "root_variable": 2, "threshold": 0.0, "coefficient": 0.5},
        ),
    },
    {
        "scenario": "tree_stumps",
        "target": "g",
        "terms": (
            {"kind": "stump", "root_variable": 0, "threshold": 0.0, "coefficient": 0.8},
            {"kind": "stump", "root_variable": 3, "threshold": 0.0, "coefficient": 0.6},
            {"kind": "stump", "root_variable": 4, "threshold": 0.0, "coefficient": -0.5},
        ),
    },
    {
        "scenario": "tree_hierarchical",
        "target": "m",
        "terms": (
            {
                "kind": "depth_two_tree",
                "root_variable": 0,
                "positive_child_variable": 1,
                "nonpositive_child_variable": 2,
                "threshold": 0.0,
                "root_positive_value": 0.8,
                "positive_child_increment": 0.6,
                "nonpositive_child_increment": -0.4,
            },
        ),
    },
    {
        "scenario": "tree_hierarchical",
        "target": "g",
        "terms": (
            {
                "kind": "depth_two_tree",
                "root_variable": 0,
                "positive_child_variable": 3,
                "nonpositive_child_variable": 4,
                "threshold": 0.0,
                "root_positive_value": 0.7,
                "positive_child_increment": 0.5,
                "nonpositive_child_increment": -0.4,
            },
        ),
    },
    {
        "scenario": "tree_forest_sum",
        "target": "m",
        "terms": (
            {
                "kind": "depth_two_tree",
                "root_variable": 0,
                "positive_child_variable": 1,
                "nonpositive_child_variable": 2,
                "threshold": 0.0,
                "root_positive_value": 0.55,
                "positive_child_increment": 0.40,
                "nonpositive_child_increment": -0.30,
            },
            {
                "kind": "depth_two_tree",
                "root_variable": 3,
                "positive_child_variable": 4,
                "nonpositive_child_variable": 5,
                "threshold": 0.0,
                "root_positive_value": 0.45,
                "positive_child_increment": -0.35,
                "nonpositive_child_increment": 0.30,
            },
        ),
    },
    {
        "scenario": "tree_forest_sum",
        "target": "g",
        "terms": (
            {
                "kind": "depth_two_tree",
                "root_variable": 0,
                "positive_child_variable": 6,
                "nonpositive_child_variable": 7,
                "threshold": 0.0,
                "root_positive_value": 0.50,
                "positive_child_increment": 0.35,
                "nonpositive_child_increment": -0.25,
            },
            {
                "kind": "depth_two_tree",
                "root_variable": 3,
                "positive_child_variable": 8,
                "nonpositive_child_variable": 9,
                "threshold": 0.0,
                "root_positive_value": 0.40,
                "positive_child_increment": -0.30,
                "nonpositive_child_increment": 0.25,
            },
        ),
    },
)

_DECLARED_ROOTS = tuple(
    (formula["scenario"], formula["target"], term["root_variable"])
    for formula in _FORMULA_DECLARATIONS
    for term in formula["terms"]
)

_AUDIT_FIELDS = (
    "record_type",
    "scenario",
    "target",
    "tree_index",
    "root_variable",
    "branch",
    "check",
    "n",
    "seed",
    "p",
    "gain_tolerance",
    "mse_tolerance",
    "threshold",
    "theoretical_probability",
    "monte_carlo_probability",
    "theoretical_leaf_value",
    "theoretical_left_probability",
    "monte_carlo_left_probability",
    "theoretical_right_probability",
    "monte_carlo_right_probability",
    "theoretical_left_mean",
    "monte_carlo_left_mean",
    "theoretical_right_mean",
    "monte_carlo_right_mean",
    "theoretical_split_gain",
    "monte_carlo_split_gain",
    "reconstruction_kind",
    "tree_count",
    "raw_reconstruction_mse",
    "violations",
    "passed",
)

_ROOT_FIELDS = {
    "scenario", "target", "root_variable", "threshold",
    "theoretical_left_probability", "monte_carlo_left_probability",
    "theoretical_right_probability", "monte_carlo_right_probability",
    "theoretical_left_mean", "monte_carlo_left_mean",
    "theoretical_right_mean", "monte_carlo_right_mean",
    "theoretical_split_gain", "monte_carlo_split_gain", "passed",
}
_LEAF_FIELDS = {
    "scenario", "target", "tree_index", "root_variable", "branch",
    "theoretical_probability", "monte_carlo_probability",
    "theoretical_leaf_value", "passed",
}
_RECONSTRUCTION_FIELDS = {
    "scenario", "target", "reconstruction_kind", "tree_count",
    "raw_reconstruction_mse", "passed",
}
_PROHIBITED_FIELDS = {"check", "violations", "passed"}


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


def _term_values(term, X):
    root_positive = X[:, term["root_variable"]] > term["threshold"]
    if term["kind"] == "stump":
        return term["coefficient"] * root_positive
    if term["kind"] == "depth_two_tree":
        positive_child = X[:, term["positive_child_variable"]] > term["threshold"]
        nonpositive_child = X[:, term["nonpositive_child_variable"]] > term["threshold"]
        return np.asarray(
            term["root_positive_value"] * root_positive
            + term["positive_child_increment"] * (root_positive & positive_child)
            + term["nonpositive_child_increment"] * (~root_positive & nonpositive_child),
            dtype=float,
        )
    raise ValueError(f"Unsupported structured truth term: {term['kind']}")


def _formula_values(formula, X):
    return sum((_term_values(term, X) for term in formula["terms"]), np.zeros(len(X)))


def _theoretical_design():
    return np.asarray(tuple(itertools.product((-1.0, 1.0), repeat=AUDIT_P)))


def _leaf_masks(term, X):
    root_positive = X[:, term["root_variable"]] > term["threshold"]
    if term["kind"] == "stump":
        return (
            ("root_nonpositive", ~root_positive),
            ("root_positive", root_positive),
        )
    positive_child = X[:, term["positive_child_variable"]] > term["threshold"]
    nonpositive_child = X[:, term["nonpositive_child_variable"]] > term["threshold"]
    return (
        ("root_nonpositive_child_nonpositive", ~root_positive & ~nonpositive_child),
        ("root_nonpositive_child_positive", ~root_positive & nonpositive_child),
        ("root_positive_child_nonpositive", root_positive & ~positive_child),
        ("root_positive_child_positive", root_positive & positive_child),
    )


def _root_check(formula, term, raw_values, theoretical_values, X, theoretical_X):
    root = term["root_variable"]
    threshold = term["threshold"]
    monte_left = X[:, root] <= threshold
    theory_left = theoretical_X[:, root] <= threshold
    theoretical_gain = split_gain(theoretical_values, theoretical_X[:, root], threshold)
    monte_carlo_gain = split_gain(raw_values, X[:, root], threshold)
    row = {
        "scenario": formula["scenario"],
        "target": formula["target"],
        "root_variable": root,
        "threshold": threshold,
        "theoretical_left_probability": float(theory_left.mean()),
        "monte_carlo_left_probability": float(monte_left.mean()),
        "theoretical_right_probability": float((~theory_left).mean()),
        "monte_carlo_right_probability": float((~monte_left).mean()),
        "theoretical_left_mean": float(theoretical_values[theory_left].mean()),
        "monte_carlo_left_mean": float(raw_values[monte_left].mean()),
        "theoretical_right_mean": float(theoretical_values[~theory_left].mean()),
        "monte_carlo_right_mean": float(raw_values[~monte_left].mean()),
        "theoretical_split_gain": theoretical_gain,
        "monte_carlo_split_gain": monte_carlo_gain,
    }
    row["passed"] = bool(
        theoretical_gain > GAIN_TOLERANCE
        and monte_carlo_gain > GAIN_TOLERANCE
        and 0 < row["monte_carlo_left_probability"] < 1
    )
    return row


def _leaf_checks(formula, term, tree_index, X, theoretical_X):
    monte_values = _term_values(term, X)
    theoretical_values = _term_values(term, theoretical_X)
    monte_masks = dict(_leaf_masks(term, X))
    theoretical_masks = dict(_leaf_masks(term, theoretical_X))
    rows = []
    for branch, theoretical_mask in theoretical_masks.items():
        monte_mask = monte_masks[branch]
        row = {
            "scenario": formula["scenario"],
            "target": formula["target"],
            "tree_index": tree_index,
            "root_variable": term["root_variable"],
            "branch": branch,
            "theoretical_probability": float(theoretical_mask.mean()),
            "monte_carlo_probability": float(monte_mask.mean()),
            "theoretical_leaf_value": float(theoretical_values[theoretical_mask][0]),
        }
        row["passed"] = bool(0 < row["monte_carlo_probability"] < 1)
        rows.append(row)
    return rows


def _reconstruction_check(formula, raw_values, observed_target):
    kinds = {term["kind"] for term in formula["terms"]}
    if kinds == {"stump"}:
        reconstruction_kind = "sum_of_stumps"
    elif len(formula["terms"]) == 1:
        reconstruction_kind = "depth_two_truth_tree"
    else:
        reconstruction_kind = "sum_of_two_depth_two_truth_trees"
    observed_raw = observed_target * raw_values.std() + raw_values.mean()
    raw_reconstruction_mse = float(np.mean((raw_values - observed_raw) ** 2))
    return {
        "scenario": formula["scenario"],
        "target": formula["target"],
        "reconstruction_kind": reconstruction_kind,
        "tree_count": len(formula["terms"]),
        "raw_reconstruction_mse": raw_reconstruction_mse,
        "passed": bool(raw_reconstruction_mse <= MSE_TOLERANCE),
    }


def _prohibited_form_checks(root_checks):
    terms = [term for formula in _FORMULA_DECLARATIONS for term in formula["terms"]]
    kinds = {term["kind"] for term in terms}
    root_cancellations = [
        f"{row['scenario']}:{row['target']}:X{row['root_variable']}"
        for row in root_checks
        if row["theoretical_left_mean"] == row["theoretical_right_mean"]
    ]
    checks = (
        ("product_threshold_absent", [kind for kind in kinds if kind == "product_threshold"]),
        ("sum_threshold_absent", [kind for kind in kinds if kind == "sum_threshold"]),
        ("pure_xor_absent", [kind for kind in kinds if kind == "pure_xor"]),
        ("exact_branch_mean_cancellation_absent", root_cancellations),
    )
    return [
        {"check": check, "violations": violations, "passed": not violations}
        for check, violations in checks
    ]


def audit_tree_structures(n: int = AUDIT_MIN_N, seed: int = 20260903) -> dict:
    if type(n) is not int or n < 10:
        raise ValueError("Structure audit requires integer n >= 10.")
    if type(seed) is not int:
        raise ValueError("Structure audit seed must be an integer.")
    theoretical_X = _theoretical_design()
    simulated = {
        scenario: simulate_plr(scenario, n=n, p=AUDIT_P, seed=seed)
        for scenario in dict.fromkeys(formula["scenario"] for formula in _FORMULA_DECLARATIONS)
    }
    root_checks = []
    leaf_checks = []
    reconstruction_checks = []
    for formula in _FORMULA_DECLARATIONS:
        data = simulated[formula["scenario"]]
        raw_values = _formula_values(formula, data.X)
        theoretical_values = _formula_values(formula, theoretical_X)
        for tree_index, term in enumerate(formula["terms"]):
            root_checks.append(
                _root_check(formula, term, raw_values, theoretical_values, data.X, theoretical_X)
            )
            leaf_checks.extend(_leaf_checks(formula, term, tree_index, data.X, theoretical_X))
        observed = data.m0 if formula["target"] == "m" else data.g0
        reconstruction_checks.append(_reconstruction_check(formula, raw_values, observed))

    audit = {
        "schema": AUDIT_SCHEMA,
        "parameters": {
            "n": n,
            "seed": seed,
            "p": AUDIT_P,
            "gain_tolerance": GAIN_TOLERANCE,
            "mse_tolerance": MSE_TOLERANCE,
        },
        "formula_declarations": json.loads(json.dumps(_FORMULA_DECLARATIONS)),
        "root_checks": root_checks,
        "leaf_checks": leaf_checks,
        "reconstruction_checks": reconstruction_checks,
        "prohibited_form_checks": _prohibited_form_checks(root_checks),
        "passed": False,
    }
    audit["passed"] = not _structure_audit_failures(audit, check_overall=False)
    return audit


def _number(value):
    return type(value) in (int, float) and math.isfinite(value)


def _canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _structure_audit_failures(audit, *, check_overall):
    failures = []
    expected_top = {
        "schema",
        "parameters",
        "formula_declarations",
        "root_checks",
        "leaf_checks",
        "reconstruction_checks",
        "prohibited_form_checks",
        "passed",
    }
    if not isinstance(audit, dict) or set(audit) != expected_top:
        return ["schema: invalid structure audit envelope"]
    if audit["schema"] != AUDIT_SCHEMA:
        failures.append("schema: unexpected structure audit version")
    parameters = audit["parameters"]
    expected_parameter_keys = {"n", "seed", "p", "gain_tolerance", "mse_tolerance"}
    if not isinstance(parameters, dict) or set(parameters) != expected_parameter_keys:
        failures.append("parameters: invalid audit parameters")
        return failures
    if type(parameters["n"]) is not int or parameters["n"] < AUDIT_MIN_N:
        failures.append(f"sample-size n: requires n >= {AUDIT_MIN_N}")
    if type(parameters["seed"]) is not int:
        failures.append("seed: requires an integer seed")
    if type(parameters["p"]) is not int or parameters["p"] != AUDIT_P:
        failures.append(f"p: requires p = {AUDIT_P}")
    if parameters["gain_tolerance"] != GAIN_TOLERANCE:
        failures.append("gain tolerance: publication tolerance cannot be changed")
    if parameters["mse_tolerance"] != MSE_TOLERANCE:
        failures.append("MSE tolerance: publication tolerance cannot be changed")
    if _canonical_json(audit["formula_declarations"]) != _canonical_json(_FORMULA_DECLARATIONS):
        failures.append("formula declarations: unexpected or prohibited formula code")

    roots = audit["root_checks"]
    if not isinstance(roots, list) or len(roots) != len(_DECLARED_ROOTS):
        failures.append("root checks: requires every declared root")
    else:
        identities = []
        for row in roots:
            try:
                if set(row) != _ROOT_FIELDS or type(row["root_variable"]) is not int:
                    raise ValueError
                identity = (row["scenario"], row["target"], row["root_variable"])
                identities.append(identity)
                numeric = (
                    row["threshold"],
                    row["theoretical_left_probability"],
                    row["monte_carlo_left_probability"],
                    row["theoretical_right_probability"],
                    row["monte_carlo_right_probability"],
                    row["theoretical_left_mean"],
                    row["monte_carlo_left_mean"],
                    row["theoretical_right_mean"],
                    row["monte_carlo_right_mean"],
                    row["theoretical_split_gain"],
                    row["monte_carlo_split_gain"],
                )
                if not all(_number(value) for value in numeric):
                    raise ValueError
                if (
                    row["threshold"] != 0
                    or row["theoretical_split_gain"] <= GAIN_TOLERANCE
                    or row["monte_carlo_split_gain"] <= GAIN_TOLERANCE
                    or not 0 < row["monte_carlo_left_probability"] < 1
                    or not 0 < row["monte_carlo_right_probability"] < 1
                    or row["passed"] is not True
                ):
                    failures.append(f"root gate: {identity}")
            except (KeyError, TypeError, ValueError):
                failures.append("root checks: invalid row")
        if tuple(identities) != _DECLARED_ROOTS:
            failures.append("root checks: unexpected or duplicate roots")

    leaves = audit["leaf_checks"]
    expected_leaf_count = sum(
        2 if term["kind"] == "stump" else 4
        for formula in _FORMULA_DECLARATIONS
        for term in formula["terms"]
    )
    if not isinstance(leaves, list) or len(leaves) != expected_leaf_count:
        failures.append("leaf checks: requires every truth-tree leaf")
    else:
        groups = {}
        for row in leaves:
            try:
                if (
                    set(row) != _LEAF_FIELDS
                    or type(row["tree_index"]) is not int
                    or type(row["root_variable"]) is not int
                ):
                    raise ValueError
                key = (row["scenario"], row["target"], row["tree_index"])
                groups.setdefault(key, []).append(row)
                if (
                    not _number(row["theoretical_probability"])
                    or not _number(row["monte_carlo_probability"])
                    or not _number(row["theoretical_leaf_value"])
                    or not 0 < row["theoretical_probability"] <= 1
                    or not 0 < row["monte_carlo_probability"] <= 1
                    or row["passed"] is not True
                ):
                    failures.append(f"leaf gate: {key}")
            except (KeyError, TypeError):
                failures.append("leaf checks: invalid row")
        for key, rows in groups.items():
            if not math.isclose(sum(row["theoretical_probability"] for row in rows), 1.0) or not math.isclose(
                sum(row["monte_carlo_probability"] for row in rows), 1.0
            ):
                failures.append(f"leaf probabilities: {key}")

    reconstructions = audit["reconstruction_checks"]
    expected_reconstructions = {
        (formula["scenario"], formula["target"]) for formula in _FORMULA_DECLARATIONS
    }
    if not isinstance(reconstructions, list) or len(reconstructions) != len(expected_reconstructions):
        failures.append("reconstruction checks: requires every target")
    else:
        identities = set()
        for row in reconstructions:
            try:
                if set(row) != _RECONSTRUCTION_FIELDS or type(row["tree_count"]) is not int:
                    raise ValueError
                identity = (row["scenario"], row["target"])
                identities.add(identity)
                if (
                    not _number(row["raw_reconstruction_mse"])
                    or row["raw_reconstruction_mse"] > MSE_TOLERANCE
                    or row["passed"] is not True
                ):
                    failures.append(f"reconstruction gate: {identity}")
            except (KeyError, TypeError):
                failures.append("reconstruction checks: invalid row")
        if identities != expected_reconstructions:
            failures.append("reconstruction checks: unexpected or duplicate targets")

    prohibited = audit["prohibited_form_checks"]
    expected_prohibited = {
        "product_threshold_absent",
        "sum_threshold_absent",
        "pure_xor_absent",
        "exact_branch_mean_cancellation_absent",
    }
    if not isinstance(prohibited, list) or len(prohibited) != len(expected_prohibited):
        failures.append("prohibited-form checks: incomplete")
    else:
        names = set()
        for row in prohibited:
            try:
                if set(row) != _PROHIBITED_FIELDS:
                    raise ValueError
                names.add(row["check"])
                if row["violations"] != [] or row["passed"] is not True:
                    failures.append(f"prohibited-form gate: {row['check']}")
            except (KeyError, TypeError):
                failures.append("prohibited-form checks: invalid row")
        if names != expected_prohibited:
            failures.append("prohibited-form checks: unexpected or duplicate checks")

    if check_overall and audit["passed"] is not (not failures):
        failures.append("overall gate: passed flag disagrees with component gates")
    return failures


def structure_audit_failures(audit) -> list[str]:
    """Return all independently evaluated publication-gate failures."""
    return _structure_audit_failures(audit, check_overall=True)


def _csv_rows(audit):
    rows = []

    def add(record_type, values):
        row = {field: "" for field in _AUDIT_FIELDS}
        row.update(values)
        row["record_type"] = record_type
        if isinstance(row["violations"], list):
            row["violations"] = json.dumps(row["violations"], separators=(",", ":"))
        rows.append(row)

    add("parameters", {**audit["parameters"], "passed": audit["parameters"]["n"] >= AUDIT_MIN_N})
    for row in audit["root_checks"]:
        add("root", row)
    for row in audit["leaf_checks"]:
        add("leaf", row)
    for row in audit["reconstruction_checks"]:
        add("reconstruction", row)
    for row in audit["prohibited_form_checks"]:
        add("prohibited_form", row)
    add("gate", {"check": "publication_bound_structure_audit", "passed": audit["passed"]})
    return rows


def write_structure_audit(audit, output_dir) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "structure_checks.json"
    json_temporary = json_path.with_suffix(".json.tmp")
    with json_temporary.open("w", encoding="utf-8") as handle:
        json.dump(audit, handle, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(json_temporary, json_path)

    csv_path = output_dir / "structure_checks.csv"
    csv_temporary = csv_path.with_suffix(".csv.tmp")
    with csv_temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_AUDIT_FIELDS)
        writer.writeheader()
        writer.writerows(_csv_rows(audit))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(csv_temporary, csv_path)
