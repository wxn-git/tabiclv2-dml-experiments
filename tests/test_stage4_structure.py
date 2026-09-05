import csv
import json

import numpy as np
import pytest

from scripts import check_stage4_tree_structures
import tabdml.stage4_structure as structure
from tabdml.stage4_structure import (
    audit_tree_structures,
    split_gain,
    write_structure_audit,
)


AUDIT_MIN_N = 200_000
GAIN_TOLERANCE = 1e-3


@pytest.fixture(scope="module")
def full_audit():
    return audit_tree_structures(n=AUDIT_MIN_N, seed=20260903)


def test_split_gain_detects_signal_and_rejects_xor_root():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(200_000, 2))
    stump = (X[:, 0] > 0).astype(float)
    xor = (X[:, 0] * X[:, 1] > 0).astype(float)
    assert split_gain(stump, X[:, 0]) > 0.24
    assert split_gain(xor, X[:, 0]) < 1e-4
    assert split_gain(xor, X[:, 1]) < 1e-4


def test_audit_binds_publication_sample_size_seed_and_tolerances(full_audit):
    assert full_audit["schema"] == "stage4_structure_audit_v2"
    assert full_audit["parameters"] == {
        "n": 200_000,
        "seed": 20260903,
        "p": 10,
        "gain_tolerance": 1e-3,
        "mse_tolerance": 1e-24,
    }
    assert full_audit["passed"] is True
    assert structure.structure_audit_failures(full_audit) == []


def test_all_declared_roots_report_theory_and_positive_monte_carlo_gain(full_audit):
    roots = full_audit["root_checks"]
    assert len(roots) == 12
    assert {row["scenario"] for row in roots} == {
        "tree_stumps",
        "tree_hierarchical",
        "tree_forest_sum",
    }
    assert all(row["theoretical_split_gain"] > GAIN_TOLERANCE for row in roots)
    assert all(row["monte_carlo_split_gain"] > GAIN_TOLERANCE for row in roots)
    assert all(0.45 < row["monte_carlo_left_probability"] < 0.55 for row in roots)
    assert all(0.45 < row["monte_carlo_right_probability"] < 0.55 for row in roots)
    for row in roots:
        assert row["passed"] is True
        assert row["theoretical_left_mean"] == pytest.approx(
            row["monte_carlo_left_mean"], abs=0.015
        )
        assert row["theoretical_right_mean"] == pytest.approx(
            row["monte_carlo_right_mean"], abs=0.015
        )
        assert row["theoretical_split_gain"] == pytest.approx(
            row["monte_carlo_split_gain"], abs=0.015
        )


def test_each_truth_tree_reports_all_leaf_probabilities(full_audit):
    leaves = full_audit["leaf_checks"]
    groups = {}
    for row in leaves:
        key = (row["scenario"], row["target"], row["tree_index"])
        groups.setdefault(key, []).append(row)
        assert row["passed"] is True
        assert row["theoretical_probability"] == pytest.approx(
            row["monte_carlo_probability"], abs=0.01
        )
    assert len(groups) == 12
    assert {len(rows) for rows in groups.values()} == {2, 4}
    for rows in groups.values():
        assert sum(row["theoretical_probability"] for row in rows) == pytest.approx(1.0)
        assert sum(row["monte_carlo_probability"] for row in rows) == pytest.approx(1.0)


def test_s2_and_s3_truth_tree_reconstruction_is_numerically_exact(full_audit):
    checks = {
        (row["scenario"], row["target"]): row
        for row in full_audit["reconstruction_checks"]
    }
    assert checks[("tree_hierarchical", "m")]["reconstruction_kind"] == "depth_two_truth_tree"
    assert checks[("tree_hierarchical", "g")]["reconstruction_kind"] == "depth_two_truth_tree"
    assert checks[("tree_forest_sum", "m")]["reconstruction_kind"] == "sum_of_two_depth_two_truth_trees"
    assert checks[("tree_forest_sum", "g")]["reconstruction_kind"] == "sum_of_two_depth_two_truth_trees"
    for key in (
        ("tree_hierarchical", "m"),
        ("tree_hierarchical", "g"),
        ("tree_forest_sum", "m"),
        ("tree_forest_sum", "g"),
    ):
        assert checks[key]["raw_reconstruction_mse"] <= full_audit["parameters"]["mse_tolerance"]
        assert checks[key]["passed"] is True


def test_prohibited_form_and_code_audit_is_explicit(full_audit):
    checks = {row["check"]: row for row in full_audit["prohibited_form_checks"]}
    assert set(checks) == {
        "product_threshold_absent",
        "sum_threshold_absent",
        "pure_xor_absent",
        "exact_branch_mean_cancellation_absent",
    }
    assert all(
        row == {"check": name, "violations": [], "passed": True}
        for name, row in checks.items()
    )


@pytest.mark.parametrize(
    "mutate, expected",
    [
        (lambda audit: audit["parameters"].update(n=AUDIT_MIN_N - 1), "n"),
        (lambda audit: audit["parameters"].update(seed=True), "seed"),
        (lambda audit: audit["parameters"].update(p=10.0), "p"),
        (lambda audit: audit["root_checks"][0].update(root_variable=False), "root"),
        (
            lambda audit: audit["root_checks"][0].update(
                monte_carlo_split_gain=GAIN_TOLERANCE
            ),
            "root",
        ),
        (
            lambda audit: audit["reconstruction_checks"][0].update(raw_reconstruction_mse=1.0),
            "reconstruction",
        ),
        (
            lambda audit: audit["prohibited_form_checks"][0].update(
                passed=False, violations=["X0*X1 > 0"]
            ),
            "prohibited",
        ),
        (lambda audit: audit["leaf_checks"][0].update(monte_carlo_probability=0.0), "leaf"),
    ],
)
def test_gate_recomputes_every_component_even_if_passed_flag_is_forged(
    full_audit, mutate, expected
):
    audit = json.loads(json.dumps(full_audit))
    mutate(audit)
    audit["passed"] = True
    assert any(expected in failure for failure in structure.structure_audit_failures(audit))


def test_write_structure_audit_writes_json_and_csv_atomically(tmp_path, full_audit):
    assert write_structure_audit(full_audit, tmp_path) is None

    json_path = tmp_path / "structure_checks.json"
    csv_path = tmp_path / "structure_checks.csv"
    assert json.loads(json_path.read_text(encoding="utf-8")) == full_audit
    with csv_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert {row["record_type"] for row in rows} == {
        "parameters",
        "root",
        "leaf",
        "reconstruction",
        "prohibited_form",
        "gate",
    }
    assert any(
        row["record_type"] == "parameters" and row["n"] == "200000"
        for row in rows
    )
    assert not list(tmp_path.glob("*.tmp"))


def test_structure_audit_cli_writes_full_audit_and_returns_zero(
    tmp_path, capsys, full_audit, monkeypatch
):
    monkeypatch.setattr(
        check_stage4_tree_structures,
        "audit_tree_structures",
        lambda n, seed: full_audit,
    )
    exit_code = check_stage4_tree_structures.main(
        ["--n", "200000", "--seed", "20260903", "--output-dir", str(tmp_path)]
    )

    assert exit_code == 0
    saved = json.loads((tmp_path / "structure_checks.json").read_text())
    assert saved["parameters"]["n"] == 200_000
    assert capsys.readouterr().out.count("monte_carlo_gain=") == 12


def test_structure_audit_cli_returns_nonzero_for_any_failed_gate(
    monkeypatch, tmp_path, capsys
):
    audit = audit_tree_structures(n=20_000, seed=20260903)
    assert audit["passed"] is False
    monkeypatch.setattr(
        check_stage4_tree_structures,
        "audit_tree_structures",
        lambda n, seed: audit,
    )

    exit_code = check_stage4_tree_structures.main(["--output-dir", str(tmp_path)])

    assert exit_code == 1
    assert "FAILED: sample-size" in capsys.readouterr().out
    assert json.loads((tmp_path / "structure_checks.json").read_text()) == audit
