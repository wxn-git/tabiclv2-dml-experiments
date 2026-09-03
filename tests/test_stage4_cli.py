import json
from pathlib import Path
import sys

import numpy as np
import pytest
import yaml

from scripts import compose_stage4_dml, run_stage4_cache, run_stage4_tuning
from tabdml.nuisance_cache import NuisanceCache
from tabdml.stage3b_screen import _params_hash
from tabdml.stage4_config import load_stage4_config
from tabdml.stage4_experiment import Stage4PairSpec, build_stage4_nuisance_spec
from tabdml.stage4_tuning import derive_tuning_seeds, iter_tuning_tasks
from tabdml.storage import ResultStore


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG = PROJECT_ROOT / "configs" / "stage4_tree_benchmark.yaml"


def _frozen_tuning(config, execution_profile="full"):
    candidate = config["tuning"]["xgboost_candidates"][0]
    nominal = dict(candidate["params"])
    effective = dict(nominal)
    if execution_profile == "fast":
        effective["n_estimators"] = 20
    cells = {}
    for panel, panel_config in config["panels"].items():
        for scenario in config["structures"]:
            for n in panel_config["sample_sizes"]:
                for p in panel_config["dimensions"]:
                    key = f"{panel}__{scenario}__n{n}__p{p}"
                    cells[key] = {
                        target: {
                            "candidate": candidate["name"],
                            "learner_kind": "xgboost",
                            "execution_profile": execution_profile,
                            "nominal_params": nominal,
                            "nominal_config_hash": _params_hash(nominal),
                            "params": effective,
                            "config_hash": _params_hash(effective),
                            "replications": 1,
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
        "execution_profile": execution_profile,
        "selection_metric_l": "mean_validation_y_mse",
        "selection_metric_m": "mean_validation_d_mse",
        "expected_replications": 1,
        "cells": cells,
    }


def _pair(learner_l, learner_m, execution_profile="full"):
    return Stage4PairSpec(
        stage="stage4_tree_screening",
        seed_namespace="stage4_tree_screening",
        panel="standard",
        scenario="tree_stumps",
        n=20,
        p=10,
        replication=0,
        learner_l=learner_l,
        learner_m=learner_m,
        folds_count=2,
        theta0=1.0,
        execution_profile=execution_profile,
    )


def _write_frozen(tmp_path, config, execution_profile="full"):
    path = tmp_path / f"selected-{execution_profile}.json"
    path.write_text(
        json.dumps(_frozen_tuning(config, execution_profile)),
        encoding="utf-8",
    )
    return path


def test_stage4_tuning_cli_fast_run_uses_valid_config_and_narrowed_tasks(
    monkeypatch, tmp_path
):
    config = load_stage4_config(CONFIG)
    all_tasks = tuple(iter_tuning_tasks(config, replications=1, fast=True))
    first = all_tasks[0]
    tasks = tuple(
        task
        for task in all_tasks
        if task.panel == first.panel
        and task.scenario == first.scenario
        and task.n == first.n
        and task.p == first.p
        and task.candidate == first.candidate
    )
    assert {task.target for task in tasks} == {"l", "m"}

    output_root = tmp_path / "raw"
    selected_output = tmp_path / "selected" / "selected_xgboost.json"
    monkeypatch.setattr(
        run_stage4_tuning,
        "iter_tuning_tasks",
        lambda config, replications, fast: iter(tasks),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_stage4_tuning.py",
            "--config",
            str(CONFIG),
            "--output-root",
            str(output_root),
            "--selected-output",
            str(selected_output),
            "--replications",
            "1",
            "--fast",
            "--select",
        ],
    )

    assert run_stage4_tuning.main() == 0

    records = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(output_root.glob("*.json"))
    ]
    assert len(records) == 2
    assert all(record["status"] == "success" for record in records)
    assert all("validation_observed_mse" in record for record in records)
    assert all("validation_truth_mse_diagnostic" in record for record in records)
    selected = json.loads(selected_output.read_text(encoding="utf-8"))
    cell_key = f"{first.panel}__{first.scenario}__n{first.n}__p{first.p}"
    winner_l = selected["cells"][cell_key]["l"]
    assert selected["execution_profile"] == "fast"
    assert winner_l["execution_profile"] == "fast"
    assert winner_l["nominal_params"]["n_estimators"] == 800
    assert winner_l["params"]["n_estimators"] == 20
    assert winner_l["config_hash"] == first.config_hash
    assert selected["selection_metric_l"] == "mean_validation_y_mse"
    assert selected["selection_metric_m"] == "mean_validation_d_mse"
    assert "truth" not in selected["selection_metric_l"]
    assert "truth" not in selected["selection_metric_m"]


def test_stage4_tuning_cli_refuses_selection_when_an_expected_cell_is_missing(
    monkeypatch, tmp_path
):
    config = load_stage4_config(CONFIG)
    all_tasks = tuple(iter_tuning_tasks(config, replications=1))
    first = all_tasks[0]
    cell_keys = tuple(
        dict.fromkeys(
            (task.panel, task.scenario, task.n, task.p) for task in all_tasks
        )
    )[:2]
    tasks = tuple(
        task
        for task in all_tasks
        if (task.panel, task.scenario, task.n, task.p) in cell_keys
        and task.candidate == first.candidate
    )
    complete_cell = cell_keys[0]
    output_root = tmp_path / "raw"
    store = ResultStore(output_root)
    for task in tasks:
        if (task.panel, task.scenario, task.n, task.p) != complete_cell:
            continue
        store.write(
            {
                "task_key": task.key,
                "status": "success",
                "stage": task.stage,
                "seed_namespace": task.seed_namespace,
                "panel": task.panel,
                "scenario": task.scenario,
                "n": task.n,
                "p": task.p,
                "replication": task.replication,
                "target": task.target,
                "candidate": task.candidate,
                "learner_kind": "xgboost",
                "execution_profile": task.execution_profile,
                "nominal_params": task.params,
                "nominal_config_hash": task.nominal_config_hash,
                "params": task.effective_params,
                "config_hash": task.config_hash,
                "validation_fraction": task.validation_fraction,
                **derive_tuning_seeds(task),
                "validation_observed_mse": 1.0,
                "validation_truth_mse_diagnostic": 2.0,
            }
        )

    selected_output = tmp_path / "selected_xgboost.json"
    monkeypatch.setattr(
        run_stage4_tuning,
        "iter_tuning_tasks",
        lambda config, replications, fast: iter(tasks),
    )
    monkeypatch.setattr(
        run_stage4_tuning,
        "run_tuning_task",
        lambda task, **kwargs: {"task_key": task.key, "status": "skipped"},
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_stage4_tuning.py",
            "--config",
            str(CONFIG),
            "--output-root",
            str(output_root),
            "--selected-output",
            str(selected_output),
            "--replications",
            "1",
            "--select",
        ],
    )

    with pytest.raises(ValueError, match="Incomplete tuning records"):
        run_stage4_tuning.main()

    assert not selected_output.exists()


def test_stage4_tuning_cli_rejects_reversed_targets(monkeypatch, tmp_path):
    config = load_stage4_config(CONFIG)
    config["tuning"]["targets"] = ["m", "l"]
    config_path = tmp_path / "stage4-reversed-targets.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    monkeypatch.setattr(
        run_stage4_tuning,
        "run_tuning_task",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("invalid targets must be rejected before execution")
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_stage4_tuning.py",
            "--config",
            str(config_path),
            "--output-root",
            str(tmp_path / "raw"),
            "--replications",
            "1",
        ],
    )

    with pytest.raises(ValueError, match="exact ordered targets"):
        run_stage4_tuning.main()


@pytest.mark.parametrize(
    ("device_group", "expected_methods"),
    [
        ("gpu", {"tabiclv2_1", "tabiclv2_8"}),
        ("cpu", {"xgboost", "extra_trees", "oracle"}),
    ],
)
def test_stage4_cache_cli_isolates_device_groups(
    monkeypatch, tmp_path, device_group, expected_methods
):
    config = load_stage4_config(CONFIG)
    selected = _write_frozen(tmp_path, config)
    pairs = (
        _pair("tabiclv2_1", "tabiclv2_1"),
        _pair("tabiclv2_8", "tabiclv2_8"),
        _pair("xgboost", "xgboost"),
        _pair("extra_trees", "extra_trees"),
        _pair("oracle", "oracle"),
    )
    calls = []
    monkeypatch.setattr(
        run_stage4_cache,
        "iter_stage4_pairs",
        lambda *args, **kwargs: iter(pairs),
    )
    monkeypatch.setattr(
        run_stage4_cache,
        "fit_stage4_nuisance",
        lambda pair, target, *args, **kwargs: calls.append(
            pair.learner_l if target == "l" else pair.learner_m
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_stage4_cache.py",
            "--config",
            str(CONFIG),
            "--phase",
            "screening",
            "--device-group",
            device_group,
            "--tuned-models",
            str(selected),
            "--cache-root",
            str(tmp_path / "cache"),
            "--replications",
            "1",
        ],
    )

    assert run_stage4_cache.main() == 0
    assert set(calls) == expected_methods
    assert len(calls) == 2 * len(expected_methods)


def test_stage4_cache_cli_shards_exact_nuisance_task_keys(monkeypatch, tmp_path):
    config = load_stage4_config(CONFIG)
    selected = _write_frozen(tmp_path, config)
    pairs = (
        _pair("xgboost", "xgboost"),
        _pair("extra_trees", "extra_trees"),
        _pair("oracle", "oracle"),
    )
    seen = []
    monkeypatch.setattr(
        run_stage4_cache,
        "iter_stage4_pairs",
        lambda *args, **kwargs: iter(pairs),
    )
    monkeypatch.setattr(
        run_stage4_cache,
        "fit_stage4_nuisance",
        lambda pair, target, *args, **kwargs: seen.append(
            build_stage4_nuisance_spec(pair, target).key
        ),
    )
    for shard_index in range(2):
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "run_stage4_cache.py",
                "--config",
                str(CONFIG),
                "--phase",
                "screening",
                "--device-group",
                "cpu",
                "--tuned-models",
                str(selected),
                "--cache-root",
                str(tmp_path / "cache"),
                "--replications",
                "1",
                "--num-shards",
                "2",
                "--shard-index",
                str(shard_index),
            ],
        )
        assert run_stage4_cache.main() == 0

    assert len(seen) == len(set(seen)) == 6


def test_stage4_compose_cli_refuses_missing_cache_before_writing(
    monkeypatch, tmp_path
):
    config = load_stage4_config(CONFIG)
    selected = _write_frozen(tmp_path, config)
    pair = _pair("oracle", "oracle")
    monkeypatch.setattr(
        compose_stage4_dml,
        "iter_stage4_pairs",
        lambda *args, **kwargs: iter((pair,)),
    )
    output_root = tmp_path / "output"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compose_stage4_dml.py",
            "--config",
            str(CONFIG),
            "--phase",
            "screening",
            "--tuned-models",
            str(selected),
            "--cache-root",
            str(tmp_path / "cache"),
            "--output-root",
            str(output_root),
            "--replications",
            "1",
        ],
    )

    with pytest.raises(FileNotFoundError, match="Missing nuisance cache"):
        compose_stage4_dml.main()

    assert not tuple(output_root.glob("*.json"))


def test_stage4_compose_cli_writes_atomically_and_rejects_forged_resume(
    monkeypatch, tmp_path
):
    config = load_stage4_config(CONFIG)
    selected = _write_frozen(tmp_path, config)
    pair = _pair("oracle", "oracle")
    monkeypatch.setattr(
        compose_stage4_dml,
        "iter_stage4_pairs",
        lambda *args, **kwargs: iter((pair,)),
    )
    cache_root = tmp_path / "cache"
    cache = NuisanceCache(cache_root)
    for target in ("l", "m"):
        task = build_stage4_nuisance_spec(pair, target)
        cache.write(
            task,
            np.zeros(pair.n),
            (0.0, 0.0),
            None,
            None,
        )
    output_root = tmp_path / "output"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compose_stage4_dml.py",
            "--config",
            str(CONFIG),
            "--phase",
            "screening",
            "--tuned-models",
            str(selected),
            "--cache-root",
            str(cache_root),
            "--output-root",
            str(output_root),
            "--replications",
            "1",
        ],
    )

    assert compose_stage4_dml.main() == 0
    output_path = output_root / f"{pair.key}.json"
    record = json.loads(output_path.read_text(encoding="utf-8"))
    assert record["status"] == "success"
    assert not output_path.with_suffix(".json.tmp").exists()

    record["panel"] = "forged-panel"
    output_path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="panel mismatch"):
        compose_stage4_dml.main()


def test_stage4_compose_cli_retries_only_failed_records(monkeypatch, tmp_path):
    config = load_stage4_config(CONFIG)
    selected = _write_frozen(tmp_path, config)
    pair = _pair("oracle", "oracle")
    monkeypatch.setattr(
        compose_stage4_dml,
        "iter_stage4_pairs",
        lambda *args, **kwargs: iter((pair,)),
    )
    cache = NuisanceCache(tmp_path / "cache")
    for target in ("l", "m"):
        cache.write(
            build_stage4_nuisance_spec(pair, target),
            np.zeros(pair.n),
            (0.0, 0.0),
            None,
            None,
        )
    output_root = tmp_path / "output"
    ResultStore(output_root).write({"task_key": pair.key, "status": "failed"})
    base_argv = [
        "compose_stage4_dml.py",
        "--config",
        str(CONFIG),
        "--phase",
        "screening",
        "--tuned-models",
        str(selected),
        "--cache-root",
        str(tmp_path / "cache"),
        "--output-root",
        str(output_root),
        "--replications",
        "1",
    ]
    monkeypatch.setattr(sys, "argv", base_argv)

    assert compose_stage4_dml.main() == 0
    assert (
        json.loads((output_root / f"{pair.key}.json").read_text())["status"]
        == "failed"
    )

    monkeypatch.setattr(sys, "argv", base_argv + ["--retry-failed"])
    assert compose_stage4_dml.main() == 0
    assert (
        json.loads((output_root / f"{pair.key}.json").read_text())["status"]
        == "success"
    )


def test_stage4_compose_cli_rejects_forged_failed_resume(
    monkeypatch, tmp_path
):
    config = load_stage4_config(CONFIG)
    selected = _write_frozen(tmp_path, config)
    pair = _pair("oracle", "oracle")
    monkeypatch.setattr(
        compose_stage4_dml,
        "iter_stage4_pairs",
        lambda *args, **kwargs: iter((pair,)),
    )
    cache = NuisanceCache(tmp_path / "cache")
    for target in ("l", "m"):
        cache.write(
            build_stage4_nuisance_spec(pair, target),
            np.zeros(pair.n),
            (0.0, 0.0),
            None,
            None,
        )
    output_root = tmp_path / "output"
    ResultStore(output_root).write(
        {"task_key": pair.key, "status": "failed"}
    )
    output_path = output_root / f"{pair.key}.json"
    failed = json.loads(output_path.read_text(encoding="utf-8"))
    failed["task_key"] = "forged-task"
    output_path.write_text(json.dumps(failed), encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compose_stage4_dml.py",
            "--config",
            str(CONFIG),
            "--phase",
            "screening",
            "--tuned-models",
            str(selected),
            "--cache-root",
            str(tmp_path / "cache"),
            "--output-root",
            str(output_root),
            "--replications",
            "1",
        ],
    )

    with pytest.raises(ValueError, match="task_key mismatch"):
        compose_stage4_dml.main()
