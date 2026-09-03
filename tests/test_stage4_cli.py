import json
from pathlib import Path
import sys

import pytest
import yaml

from scripts import run_stage4_tuning
from tabdml.stage4_config import load_stage4_config
from tabdml.stage4_tuning import iter_tuning_tasks
from tabdml.storage import ResultStore


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG = PROJECT_ROOT / "configs" / "stage4_tree_benchmark.yaml"


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
