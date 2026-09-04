from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from scripts import select_stage4_confirmation
from tabdml.config import derive_seed
from tabdml.stage3b_screen import _params_hash
from tabdml.stage4_config import iter_tree_cells, load_stage4_config
from tabdml.stage4_experiment import (
    STAGE4_SELECTION_RULE,
    Stage4PairSpec,
    iter_stage4_pairs,
    stage4_configuration_fingerprint,
    validate_stage4_selection,
)
from tabdml.stage4_selection import (
    paired_squared_error_advantage,
    select_confirmation_cells,
    write_confirmation_cells,
)
from tabdml.stage4_tuning import tuning_run_fingerprint


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


def _frozen_tuning(config, execution_profile="fast", candidate_index=0):
    expected_replications = (
        1
        if execution_profile == "fast"
        else config["tuning"]["replications"]
    )
    candidate = config["tuning"]["xgboost_candidates"][candidate_index]
    nominal_params = dict(candidate["params"])
    params = dict(nominal_params)
    if execution_profile == "fast":
        params["n_estimators"] = 20
    cells = {}
    for cell in iter_tree_cells(config):
        cells[cell.key] = {
            target: {
                "candidate": candidate["name"],
                "learner_kind": "xgboost",
                "execution_profile": execution_profile,
                "nominal_params": nominal_params,
                "nominal_config_hash": _params_hash(nominal_params),
                "params": params,
                "config_hash": _params_hash(params),
                "replications": expected_replications,
                "mean_validation_observed_mse": 1.0,
                "mean_validation_truth_mse_diagnostic": 1.0,
                "selection_metric": (
                    "mean_validation_y_mse"
                    if target == "l"
                    else "mean_validation_d_mse"
                ),
            }
            for target in ("l", "m")
        }
    return {
        "tuning_stage": config["tuning"]["stage"],
        "tuning_seed_namespace": config["tuning"]["seed_namespace"],
        "tuning_run_fingerprint": tuning_run_fingerprint(
            config,
            expected_replications,
            execution_profile,
        ),
        "theta0": config["theta0"],
        "execution_profile": execution_profile,
        "selection_metric_l": "mean_validation_y_mse",
        "selection_metric_m": "mean_validation_d_mse",
        "expected_replications": expected_replications,
        "cells": cells,
    }


@pytest.fixture
def frozen(config):
    return _frozen_tuning(config)


def _screen_record(
    pair,
    theta,
):
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


def _records(
    config,
    frozen_tuning,
    execution_profile="fast",
    tab_multiplier=0.8,
):
    replications = (
        1
        if execution_profile == "fast"
        else config["screening"]["replications"]
    )
    cell_indices = {
        cell.key: index for index, cell in enumerate(iter_tree_cells(config))
    }
    records = []
    for pair in iter_stage4_pairs(
        config,
        "screening",
        frozen_tuning,
        replications=replications,
        fast=execution_profile == "fast",
    ):
        cell_key = f"{pair.panel}__{pair.scenario}__n{pair.n}__p{pair.p}"
        xgb_error = (
            0.02
            + 0.002 * (cell_indices[cell_key] % 4)
            + 0.001 * pair.replication
        )
        if pair.learner_l == pair.learner_m == "xgboost_tuned":
            theta = config["theta0"] + xgb_error
        elif pair.learner_l == pair.learner_m == "tabiclv2_1":
            theta = config["theta0"] + tab_multiplier * xgb_error
        else:
            theta = config["theta0"] + 0.01
        records.append(_screen_record(pair, theta))
    return records


def _select_fast(config, frozen_tuning, records=None):
    return select_confirmation_cells(
        (
            _records(config, frozen_tuning)
            if records is None
            else records
        ),
        config,
        frozen_tuning,
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


def test_selector_requires_frozen_tuning(config, frozen):
    primary_records = [
        record
        for record in _records(config, frozen)
        if record["learner_l"] == record["learner_m"]
        and record["learner_l"] in {"tabiclv2_1", "xgboost_tuned"}
    ]

    with pytest.raises(TypeError, match="frozen_tuning"):
        select_confirmation_cells(
            primary_records,
            config,
            execution_profile="fast",
        )


def test_selector_accepts_exact_task5_universe_and_emits_contract(config, frozen):
    records = _records(config, frozen)
    first = iter_tree_cells(config)[0]
    first_pairs = {
        (record["learner_l"], record["learner_m"])
        for record in records
        if (
            record["panel"],
            record["scenario"],
            record["n"],
            record["p"],
            record["replication"],
        )
        == (first.panel, first.scenario, first.n, first.p, 0)
    }
    assert len(records) == 24 * 10
    assert first_pairs == {
        ("tabiclv2_1", "tabiclv2_1"),
        ("tabiclv2_8", "tabiclv2_8"),
        ("xgboost", "xgboost"),
        ("xgboost_tuned", "xgboost_tuned"),
        ("extra_trees", "extra_trees"),
        ("oracle", "oracle"),
        ("oracle", "xgboost_tuned"),
        ("xgboost_tuned", "oracle"),
        ("oracle", "tabiclv2_1"),
        ("tabiclv2_1", "oracle"),
    }

    selected = _select_fast(config, frozen, records)

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


def test_selector_freezes_six_cells_when_every_tab_score_is_positive(
    config, frozen
):
    selected = _select_fast(
        config,
        frozen,
        _records(config, frozen, tab_multiplier=1.2),
    )

    assert len(selected["cells"]) == 6
    assert all(
        row["mean_paired_squared_error_difference"] > 0
        for row in selected["cells"]
    )


def test_selector_breaks_score_ties_by_n_then_p(config, frozen):
    records = _records(config, frozen, tab_multiplier=1.0)
    selected = _select_fast(config, frozen, records)

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


def test_selector_rejects_malformed_failed_nonprimary_record(config, frozen):
    records = _records(config, frozen)
    extra_trees = next(
        record
        for record in records
        if record["learner_l"] == record["learner_m"] == "extra_trees"
    )
    extra_trees["status"] = "failed"
    extra_trees.pop("theta")

    with pytest.raises(ValueError, match="status mismatch"):
        _select_fast(config, frozen, records)


def test_selector_rejects_missing_nonprimary_oracle_diagnostic(config, frozen):
    records = _records(config, frozen)
    records.pop()

    with pytest.raises(ValueError, match="complete screening task universe"):
        _select_fast(config, frozen, records)


def test_selector_rejects_failed_oracle_diagnostic(config, frozen):
    records = _records(config, frozen)
    oracle_diagnostic = next(
        record
        for record in records
        if record["learner_l"] == "oracle"
        and record["learner_m"] == "xgboost_tuned"
    )
    oracle_diagnostic["status"] = "failed"

    with pytest.raises(ValueError, match="status mismatch"):
        _select_fast(config, frozen, records)


def test_selector_rejects_unknown_method_record(config, frozen):
    records = _records(config, frozen)
    unknown = deepcopy(records[0])
    unknown.update(
        task_key="unknown-method-task",
        learner_l="unknown",
        learner_m="unknown",
    )
    records.append(unknown)

    with pytest.raises(ValueError, match="unexpected task_key"):
        _select_fast(config, frozen, records)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda rows: rows.pop(), "complete screening task universe"),
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
        (lambda rows: rows[0].update(n=999), "n mismatch"),
        (
            lambda rows: rows[0].update(learner_m="xgboost_tuned"),
            "learner_m mismatch",
        ),
        (
            lambda rows: rows[0].update(learner_l_config_hash="stale-hash"),
            "config_hash",
        ),
    ],
)
def test_selector_rejects_invalid_relevant_record_universes(
    config, frozen, mutation, message
):
    records = _records(config, frozen)
    mutation(records)

    with pytest.raises(ValueError, match=message):
        _select_fast(config, frozen, records)


def test_selector_rejects_alternative_valid_candidate_not_frozen(config, frozen):
    records = _records(config, frozen)
    record = next(
        value
        for value in records
        if value["learner_l"] == value["learner_m"] == "xgboost_tuned"
    )
    alternate = _frozen_tuning(config, candidate_index=1)
    cell_key = (
        f"{record['panel']}__{record['scenario']}"
        f"__n{record['n']}__p{record['p']}"
    )
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
        learner_l_config_hash=alternate["cells"][cell_key]["l"]["config_hash"],
        learner_m_config_hash=alternate["cells"][cell_key]["m"]["config_hash"],
        execution_profile=record["execution_profile"],
    )
    record.update(
        task_key=pair.key,
        learner_l_config_hash=pair.learner_l_config_hash,
        learner_m_config_hash=pair.learner_m_config_hash,
    )

    with pytest.raises(ValueError, match="unexpected task_key"):
        _select_fast(config, frozen, records)


def test_full_profile_cannot_self_declare_one_replication(config):
    frozen = _frozen_tuning(config, execution_profile="full")
    one_replication_full = [
        row
        for row in _records(config, frozen, execution_profile="full")
        if row["replication"] == 0
    ]

    with pytest.raises(ValueError, match="full profile contract"):
        select_confirmation_cells(
            one_replication_full,
            config,
            frozen,
            expected_replications=1,
            execution_profile="full",
        )


def test_fast_profile_rejects_any_replication_count_other_than_one(config, frozen):
    with pytest.raises(ValueError, match="fast profile contract"):
        select_confirmation_cells(
            _records(config, frozen),
            config,
            frozen,
            expected_replications=2,
            execution_profile="fast",
        )


def test_write_confirmation_cells_validates_then_writes_atomically(
    config, frozen, tmp_path, monkeypatch
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
        _records(config, frozen),
        output,
        config,
        frozen,
        execution_profile="fast",
    )

    assert replace_calls == [(output.with_suffix(".json.tmp"), output)]
    assert json.loads(output.read_text(encoding="utf-8")) == selected
    validate_stage4_selection(config, selected, "fast")


def test_write_confirmation_cells_does_not_touch_output_on_invalid_input(
    config, frozen, tmp_path
):
    output = tmp_path / "selection.json"
    output.write_text('{"sentinel": true}', encoding="utf-8")
    records = _records(config, frozen)
    records.pop()

    with pytest.raises(ValueError, match="complete screening task universe"):
        write_confirmation_cells(
            records,
            output,
            config,
            frozen,
            execution_profile="fast",
        )

    assert json.loads(output.read_text(encoding="utf-8")) == {"sentinel": True}


@pytest.mark.parametrize("profile_flag", [("--fast",), ("--profile", "fast")])
def test_selection_cli_resolves_all_paths_from_repository_root(
    config, frozen, tmp_path, monkeypatch, profile_flag
):
    config_path = tmp_path / "configs" / "stage4.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(CONFIG_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    screening_root = tmp_path / "raw"
    screening_root.mkdir()
    for index, record in enumerate(_records(config, frozen)):
        (screening_root / f"record-{index}.json").write_text(
            json.dumps(record), encoding="utf-8"
        )
    outside = tmp_path / "outside"
    outside.mkdir()
    tuned_models = tmp_path / "tuning" / "selected.json"
    tuned_models.parent.mkdir()
    tuned_models.write_text(json.dumps(frozen), encoding="utf-8")
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
            "--tuned-models",
            "tuning/selected.json",
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
    frozen = _frozen_tuning(config, execution_profile="full")
    tuned_models = tmp_path / "selected.json"
    tuned_models.write_text(json.dumps(frozen), encoding="utf-8")
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
            "--tuned-models",
            "selected.json",
            "--output",
            "selection.json",
            "--expected-replications",
            "1",
        ],
    )

    with pytest.raises(ValueError, match="full profile contract"):
        select_stage4_confirmation.main()
