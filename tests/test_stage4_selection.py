from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from scripts import select_stage4_confirmation
from tabdml.config import derive_seed
from tabdml.stage4_config import iter_tree_cells, load_stage4_config
from tabdml.stage4_experiment import (
    STAGE4_SELECTION_RULE,
    Stage4PairSpec,
    stage4_configuration_fingerprint,
    validate_stage4_selection,
)
from tabdml.stage4_selection import (
    paired_squared_error_advantage,
    select_confirmation_cells,
    write_confirmation_cells,
)
from tabdml.stage4_tuning import iter_tuning_tasks


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "stage4_tree_benchmark.yaml"
SELECTION_FIELDS = {
    "execution_profile",
    "screening_stage",
    "screening_seed_namespace",
    "expected_screening_replications",
    "selection_rule",
    "config_fingerprint",
    "screening_ranking",
    "cells",
}
RANKING_FIELDS = {
    "panel",
    "scenario",
    "n",
    "p",
    "mean_paired_squared_error_difference",
    "selection_rule",
}


@pytest.fixture
def config():
    return load_stage4_config(CONFIG_PATH)


def _allowed_tuned_hashes(config, execution_profile):
    hashes = {}
    for task in iter_tuning_tasks(
        config,
        replications=1,
        fast=execution_profile == "fast",
    ):
        key = (task.panel, task.scenario, task.n, task.p, task.target)
        hashes.setdefault(key, task.config_hash)
    return hashes


def _screen_record(
    config,
    cell,
    replication,
    method,
    theta,
    execution_profile="fast",
    tuned_hashes=None,
):
    if method == "tabiclv2_1":
        l_hash = m_hash = "default"
    else:
        tuned_hashes = tuned_hashes or _allowed_tuned_hashes(
            config, execution_profile
        )
        base = (cell.panel, cell.scenario, cell.n, cell.p)
        l_hash = tuned_hashes[(*base, "l")]
        m_hash = tuned_hashes[(*base, "m")]
    pair = Stage4PairSpec(
        stage=config["screening"]["stage"],
        seed_namespace=config["screening"]["seed_namespace"],
        panel=cell.panel,
        scenario=cell.scenario,
        n=cell.n,
        p=cell.p,
        replication=replication,
        learner_l=method,
        learner_m=method,
        folds_count=config["folds"],
        theta0=float(config["theta0"]),
        learner_l_config_hash=l_hash,
        learner_m_config_hash=m_hash,
        execution_profile=execution_profile,
    )
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
        "status": "success",
        "theta": theta,
        "standard_error": 0.1,
        "ci_lower": theta - 0.2,
        "ci_upper": theta + 0.2,
        "l_mse": 0.3,
        "m_mse": 0.2,
        "nuisance_error_product": 0.06,
        "lm_error_cross": 0.01,
        "residual_d_variance": 1.0,
        "bias_numerator_proxy": 0.01,
        "theta_proxy": theta,
        "proxy_error": theta - pair.theta0,
        "runtime_seconds": 0.5,
        "l_fold_seconds": [0.01] * pair.folds_count,
        "m_fold_seconds": [0.01] * pair.folds_count,
        "peak_gpu_mb": None,
    }


def _records(config, execution_profile="fast", tab_multiplier=0.8):
    replications = (
        1
        if execution_profile == "fast"
        else config["screening"]["replications"]
    )
    tuned_hashes = _allowed_tuned_hashes(config, execution_profile)
    records = []
    for cell_index, cell in enumerate(iter_tree_cells(config)):
        for replication in range(replications):
            xgb_error = 0.02 + 0.002 * (cell_index % 4) + 0.001 * replication
            records.append(
                _screen_record(
                    config,
                    cell,
                    replication,
                    "xgboost_tuned",
                    config["theta0"] + xgb_error,
                    execution_profile,
                    tuned_hashes,
                )
            )
            records.append(
                _screen_record(
                    config,
                    cell,
                    replication,
                    "tabiclv2_1",
                    config["theta0"] + tab_multiplier * xgb_error,
                    execution_profile,
                    tuned_hashes,
                )
            )
    return records


def _select_fast(config, records=None):
    return select_confirmation_cells(
        _records(config) if records is None else records,
        config,
        execution_profile="fast",
    )


def test_paired_squared_error_advantage_uses_declared_estimand():
    assert paired_squared_error_advantage(1.2, 1.1, 1.0) == pytest.approx(0.03)


@pytest.mark.parametrize("value", [True, np.nan, np.inf, "1.0"])
def test_paired_squared_error_advantage_rejects_nonfinite_or_nonnumeric(value):
    with pytest.raises(ValueError, match="finite numeric"):
        paired_squared_error_advantage(value, 1.1, 1.0)


def test_paired_squared_error_advantage_rejects_overflowed_result():
    with pytest.raises(ValueError, match="finite numeric"):
        paired_squared_error_advantage(1e308, 1e308, -1e308)


def test_selector_emits_exact_task5_contract_and_all_24_scores(config):
    selected = _select_fast(config)

    assert set(selected) == SELECTION_FIELDS
    assert selected["execution_profile"] == "fast"
    assert selected["screening_stage"] == config["screening"]["stage"]
    assert (
        selected["screening_seed_namespace"]
        == config["screening"]["seed_namespace"]
    )
    assert selected["expected_screening_replications"] == 1
    assert selected["selection_rule"] == STAGE4_SELECTION_RULE
    assert selected["config_fingerprint"] == stage4_configuration_fingerprint(
        config
    )
    assert len(selected["screening_ranking"]) == 24
    assert all(set(row) == RANKING_FIELDS for row in selected["screening_ranking"])
    assert len(selected["cells"]) == 6
    assert {(row["panel"], row["scenario"]) for row in selected["cells"]} == {
        (panel, scenario)
        for panel in config["panels"]
        for scenario in config["structures"]
    }
    assert validate_stage4_selection(config, selected, "fast")


def test_selector_freezes_six_cells_when_every_tab_score_is_positive(config):
    selected = _select_fast(config, _records(config, tab_multiplier=1.2))

    assert len(selected["cells"]) == 6
    assert all(
        row["mean_paired_squared_error_difference"] > 0
        for row in selected["cells"]
    )


def test_selector_breaks_score_ties_by_n_then_p(config):
    records = _records(config, tab_multiplier=1.0)
    selected = _select_fast(config, records)

    expected = {
        ("standard", scenario, 1000, 10) for scenario in config["structures"]
    } | {
        ("small_n_high_p", scenario, 300, 50)
        for scenario in config["structures"]
    }
    assert {
        (row["panel"], row["scenario"], row["n"], row["p"])
        for row in selected["cells"]
    } == expected


def test_selector_ignores_records_for_unselected_stage4_methods(config):
    records = _records(config)
    records.append(
        {
            "task_key": "irrelevant-failure",
            "status": "failed",
            "learner_l": "extra_trees",
            "learner_m": "extra_trees",
        }
    )

    assert len(_select_fast(config, records)["screening_ranking"]) == 24


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda rows: rows.pop(), "complete paired replications"),
        (lambda rows: rows.append(deepcopy(rows[0])), "duplicate"),
        (lambda rows: rows[0].update(status="failed"), "status"),
        (lambda rows: rows[0].update(theta=np.nan), "finite"),
        (lambda rows: rows[0].update(stage="stale-stage"), "stage mismatch"),
        (
            lambda rows: rows[0].update(seed_namespace="foreign-namespace"),
            "seed_namespace mismatch",
        ),
        (lambda rows: rows[0].update(theta0=2.0), "theta0 mismatch"),
        (lambda rows: rows[0].update(execution_profile="full"), "execution_profile"),
        (lambda rows: rows[0].update(replication=1), "replication"),
        (lambda rows: rows[0].update(n=999), "configured cell"),
        (
            lambda rows: rows[0].update(learner_m="tabiclv2_1"),
            "asymmetric",
        ),
        (
            lambda rows: rows[0].update(learner_l_config_hash="stale-hash"),
            "config_hash",
        ),
    ],
)
def test_selector_rejects_invalid_relevant_record_universes(
    config, mutation, message
):
    records = _records(config)
    mutation(records)

    with pytest.raises(ValueError, match=message):
        _select_fast(config, records)


def test_selector_rejects_a_stale_but_self_consistent_tuned_hash(config):
    records = _records(config)
    record = records[0]
    pair = Stage4PairSpec(
        stage=record["stage"],
        seed_namespace=config["screening"]["seed_namespace"],
        panel=record["panel"],
        scenario=record["scenario"],
        n=record["n"],
        p=record["p"],
        replication=record["replication"],
        learner_l=record["learner_l"],
        learner_m=record["learner_m"],
        folds_count=record["folds_count"],
        theta0=record["theta0"],
        learner_l_config_hash="111111111111",
        learner_m_config_hash="222222222222",
        execution_profile=record["execution_profile"],
    )
    record.update(
        task_key=pair.key,
        learner_l_config_hash=pair.learner_l_config_hash,
        learner_m_config_hash=pair.learner_m_config_hash,
    )

    with pytest.raises(ValueError, match="config_hash"):
        _select_fast(config, records)


def test_full_profile_cannot_self_declare_one_replication(config):
    one_replication_full = [
        row
        for row in _records(config, execution_profile="full")
        if row["replication"] == 0
    ]

    with pytest.raises(ValueError, match="full profile contract"):
        select_confirmation_cells(
            one_replication_full,
            config,
            expected_replications=1,
            execution_profile="full",
        )


def test_fast_profile_rejects_any_replication_count_other_than_one(config):
    with pytest.raises(ValueError, match="fast profile contract"):
        select_confirmation_cells(
            _records(config),
            config,
            expected_replications=2,
            execution_profile="fast",
        )


def test_write_confirmation_cells_validates_then_writes_atomically(
    config, tmp_path, monkeypatch
):
    output = tmp_path / "nested" / "selection.json"
    replace_calls = []

    def observed_replace(source, destination):
        assert Path(source).name == "selection.json.tmp"
        assert Path(source).exists()
        replace_calls.append((Path(source), Path(destination)))
        Path(source).rename(destination)

    monkeypatch.setattr("tabdml.stage4_selection.os.replace", observed_replace)
    selected = write_confirmation_cells(
        _records(config),
        output,
        config,
        execution_profile="fast",
    )

    assert replace_calls == [(output.with_suffix(".json.tmp"), output)]
    assert json.loads(output.read_text(encoding="utf-8")) == selected
    validate_stage4_selection(config, selected, "fast")


def test_write_confirmation_cells_does_not_touch_output_on_invalid_input(
    config, tmp_path
):
    output = tmp_path / "selection.json"
    output.write_text('{"sentinel": true}', encoding="utf-8")
    records = _records(config)
    records.pop()

    with pytest.raises(ValueError, match="complete paired replications"):
        write_confirmation_cells(
            records,
            output,
            config,
            execution_profile="fast",
        )

    assert json.loads(output.read_text(encoding="utf-8")) == {"sentinel": True}


@pytest.mark.parametrize("profile_flag", [("--fast",), ("--profile", "fast")])
def test_selection_cli_resolves_all_paths_from_repository_root(
    config, tmp_path, monkeypatch, profile_flag
):
    config_path = tmp_path / "configs" / "stage4.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(CONFIG_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    screening_root = tmp_path / "raw"
    screening_root.mkdir()
    for index, record in enumerate(_records(config)):
        (screening_root / f"record-{index}.json").write_text(
            json.dumps(record), encoding="utf-8"
        )
    outside = tmp_path / "outside"
    outside.mkdir()
    monkeypatch.chdir(outside)
    monkeypatch.setattr(select_stage4_confirmation, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "select_stage4_confirmation.py",
            "--config",
            "configs/stage4.yaml",
            "--screening-root",
            "raw",
            "--output",
            "selected/cells.json",
            *profile_flag,
        ],
    )

    assert select_stage4_confirmation.main() == 0
    selected = json.loads(
        (tmp_path / "selected" / "cells.json").read_text(encoding="utf-8")
    )
    validate_stage4_selection(config, selected, "fast")


def test_selection_cli_rejects_self_declared_under_replication(
    config, tmp_path, monkeypatch
):
    config_path = tmp_path / "stage4.yaml"
    config_path.write_text(CONFIG_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    raw = tmp_path / "raw"
    raw.mkdir()
    monkeypatch.setattr(select_stage4_confirmation, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "select_stage4_confirmation.py",
            "--config",
            "stage4.yaml",
            "--screening-root",
            "raw",
            "--output",
            "selection.json",
            "--expected-replications",
            "1",
        ],
    )

    with pytest.raises(ValueError, match="full profile contract"):
        select_stage4_confirmation.main()
