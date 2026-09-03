import csv
import json

import numpy as np

from scripts import check_stage4_tree_structures
from tabdml.stage4_structure import (
    audit_tree_structures,
    split_gain,
    write_structure_audit,
)


def test_split_gain_detects_signal_and_rejects_xor_root():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(200_000, 2))
    stump = (X[:, 0] > 0).astype(float)
    xor = (X[:, 0] * X[:, 1] > 0).astype(float)
    assert split_gain(stump, X[:, 0]) > 0.24
    assert split_gain(xor, X[:, 0]) < 1e-4
    assert split_gain(xor, X[:, 1]) < 1e-4


def test_all_declared_stage4_roots_have_positive_gain():
    rows = audit_tree_structures(n=200_000, seed=20260903)
    assert len(rows) == 12
    assert {row["scenario"] for row in rows} == {
        "tree_stumps",
        "tree_hierarchical",
        "tree_forest_sum",
    }
    assert all(row["split_gain"] > 1e-3 for row in rows)
    assert all(0.45 < row["left_probability"] < 0.55 for row in rows)


def test_write_structure_audit_writes_json_and_csv_atomically(tmp_path):
    records = [
        {
            "scenario": "tree_stumps",
            "target": "m",
            "root_variable": 0,
            "threshold": 0.0,
            "split_gain": 0.25,
            "left_probability": 0.5,
            "left_mean": -0.5,
            "right_mean": 0.5,
        }
    ]

    assert write_structure_audit(records, tmp_path) is None

    json_path = tmp_path / "structure_checks.json"
    csv_path = tmp_path / "structure_checks.csv"
    assert json.loads(json_path.read_text(encoding="utf-8")) == records
    with csv_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows == [
        {
            "scenario": "tree_stumps",
            "target": "m",
            "root_variable": "0",
            "threshold": "0.0",
            "split_gain": "0.25",
            "left_probability": "0.5",
            "left_mean": "-0.5",
            "right_mean": "0.5",
        }
    ]
    assert not list(tmp_path.glob("*.tmp"))


def test_structure_audit_cli_writes_all_records_and_returns_zero(tmp_path, capsys):
    exit_code = check_stage4_tree_structures.main(
        ["--n", "20000", "--seed", "20260903", "--output-dir", str(tmp_path)]
    )

    assert exit_code == 0
    assert len(json.loads((tmp_path / "structure_checks.json").read_text())) == 12
    assert capsys.readouterr().out.count("split_gain=") == 12


def test_structure_audit_cli_returns_nonzero_for_zero_gain(
    monkeypatch, tmp_path, capsys
):
    records = [
        {
            "scenario": "tree_stumps",
            "target": "m",
            "root_variable": 0,
            "threshold": 0.0,
            "split_gain": 1e-3,
            "left_probability": 0.5,
            "left_mean": 0.0,
            "right_mean": 0.0,
        }
    ]
    monkeypatch.setattr(
        check_stage4_tree_structures,
        "audit_tree_structures",
        lambda n, seed: records,
    )

    exit_code = check_stage4_tree_structures.main(["--output-dir", str(tmp_path)])

    assert exit_code == 1
    assert "split_gain=0.001" in capsys.readouterr().out
    assert json.loads((tmp_path / "structure_checks.json").read_text()) == records
