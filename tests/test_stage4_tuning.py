import json
from pathlib import Path

import numpy as np
import pytest

from tabdml.dgp import SimulatedData
from tabdml.sharding import belongs_to_shard
from tabdml.stage4_config import TreeBenchmarkCell, load_stage4_config
from tabdml.stage4_tuning import (
    Stage4TuningTask,
    iter_tuning_tasks,
    run_tuning_task,
    select_tuned_xgboost,
    write_tuned_xgboost,
)
from tabdml.storage import ResultStore


CONFIG = Path("configs/stage4_tree_benchmark.yaml")
CELL = TreeBenchmarkCell("standard", "tree_stumps", 1000, 10)
CELL_KEY = CELL.key


@pytest.fixture
def config():
    return load_stage4_config(CONFIG)


def _record(target, candidate, observed, diagnostic, replication=0):
    return {
        "status": "success",
        "stage": "stage4_tree_tuning",
        "panel": CELL.panel,
        "scenario": CELL.scenario,
        "n": CELL.n,
        "p": CELL.p,
        "replication": replication,
        "target": target,
        "candidate": candidate,
        "params": {"max_depth": 1 if candidate == "a" else 2},
        "config_hash": f"{candidate}-hash",
        "validation_observed_mse": observed,
        "validation_truth_mse_diagnostic": diagnostic,
    }


def _task(target):
    return Stage4TuningTask(
        stage="stage4_tree_tuning",
        seed_namespace="stage4_tree_tuning",
        panel="standard",
        scenario="tree_stumps",
        n=8,
        p=2,
        replication=0,
        target=target,
        candidate="tiny",
        params={"n_estimators": 2, "max_depth": 1},
        validation_fraction=0.25,
    )


def _data():
    index = np.arange(8, dtype=float)
    return SimulatedData(
        X=np.column_stack((index, -index)),
        y=100.0 + index,
        d=200.0 + index,
        l0=300.0 + index,
        m0=400.0 + index,
        g0=500.0 + index,
        theta0=1.0,
        categorical_indices=(),
    )


class _RecordingModel:
    def __init__(self, fitted_targets):
        self.fitted_targets = fitted_targets

    def fit(self, X, target):
        del X
        self.fitted_targets.append(np.asarray(target).copy())
        return self

    def predict(self, X):
        return np.zeros(len(X), dtype=float)


def test_tuning_enumerates_cell_target_candidate_replication_product(config):
    tasks = tuple(iter_tuning_tasks(config, replications=2))

    assert len(tasks) == 24 * 2 * 6 * 2
    assert len({task.key for task in tasks}) == len(tasks)
    assert {task.target for task in tasks} == {"l", "m"}
    assert all(task.stage in task.key for task in tasks)
    assert all(task.panel in task.key for task in tasks)
    assert all(f"target-{task.target}" in task.key for task in tasks)
    assert all(task.candidate in task.key for task in tasks)
    assert all(task.config_hash in task.key for task in tasks)


def test_tuning_shards_the_deterministic_task_keys(config):
    all_tasks = tuple(iter_tuning_tasks(config, replications=1))
    shards = [
        tuple(
            iter_tuning_tasks(
                config,
                replications=1,
                num_shards=4,
                shard_index=shard_index,
            )
        )
        for shard_index in range(4)
    ]

    assert {task.key for shard in shards for task in shard} == {
        task.key for task in all_tasks
    }
    assert sum(len(shard) for shard in shards) == len(all_tasks)
    for shard_index, shard in enumerate(shards):
        assert all(belongs_to_shard(task.key, 4, shard_index) for task in shard)


@pytest.mark.parametrize(
    ("target", "expected_minimum", "metric"),
    [("l", 100.0, "validation_y_mse"), ("m", 200.0, "validation_d_mse")],
)
def test_tuning_fit_uses_only_the_observable_target(
    monkeypatch, tmp_path, target, expected_minimum, metric
):
    fitted_targets = []
    factory_calls = []
    monkeypatch.setattr(
        "tabdml.stage4_tuning.simulate_plr", lambda *args, **kwargs: _data()
    )

    def make_model(kind, params, seed, fast):
        factory_calls.append((kind, params, seed, fast))
        return _RecordingModel(fitted_targets)

    monkeypatch.setattr(
        "tabdml.stage4_tuning.make_configured_tree_learner",
        make_model,
    )

    record = run_tuning_task(_task(target), output_root=tmp_path, fast=True)

    assert record["status"] == "success"
    assert len(fitted_targets) == 1
    assert len(fitted_targets[0]) == 6
    assert fitted_targets[0].min() >= expected_minimum
    assert fitted_targets[0].max() < expected_minimum + 8.0
    assert factory_calls[0][0] == "xgboost"
    assert factory_calls[0][3] is True
    assert record["selection_metric"] == metric
    assert record["validation_observed_mse"] > 0
    assert record["validation_truth_mse_diagnostic"] > 0


def test_l_and_m_winners_use_observable_losses_independently():
    records = [
        _record("l", "a", observed=1.0, diagnostic=9.0),
        _record("l", "b", observed=2.0, diagnostic=0.0),
        _record("m", "a", observed=3.0, diagnostic=0.0),
        _record("m", "b", observed=1.5, diagnostic=9.0),
    ]

    selected = select_tuned_xgboost(records, expected_replications=1)

    assert selected["cells"][CELL_KEY]["l"]["candidate"] == "a"
    assert selected["cells"][CELL_KEY]["m"]["candidate"] == "b"
    assert selected["selection_metric_l"] == "mean_validation_y_mse"
    assert selected["selection_metric_m"] == "mean_validation_d_mse"


def test_selection_rejects_incomplete_candidate_replication_groups():
    records = [
        _record("l", "a", observed=1.0, diagnostic=2.0),
        _record("m", "a", observed=1.0, diagnostic=2.0),
    ]

    with pytest.raises(ValueError, match="Incomplete tuning records"):
        select_tuned_xgboost(records, expected_replications=2)


def test_selection_rejects_an_expected_candidate_with_no_records():
    records = [
        _record("l", "a", observed=1.0, diagnostic=2.0),
        _record("m", "a", observed=1.0, diagnostic=2.0),
    ]

    with pytest.raises(ValueError, match="Incomplete tuning records"):
        select_tuned_xgboost(
            records,
            expected_replications=1,
            expected_candidates=("a", "b"),
            expected_cells=(CELL,),
        )


def test_selection_rejects_mixed_configurations_within_a_candidate_group():
    records = [
        _record(target, "a", observed=1.0, diagnostic=2.0, replication=replication)
        for target in ("l", "m")
        for replication in (0, 1)
    ]
    records[1]["config_hash"] = "different-hash"

    with pytest.raises(ValueError, match="Inconsistent tuning configuration"):
        select_tuned_xgboost(records, expected_replications=2)


def test_failed_result_is_resumed_only_with_retry_failed(monkeypatch, tmp_path):
    task = _task("l")
    ResultStore(tmp_path).write({"task_key": task.key, "status": "failed"})
    fitted_targets = []
    monkeypatch.setattr(
        "tabdml.stage4_tuning.simulate_plr", lambda *args, **kwargs: _data()
    )
    monkeypatch.setattr(
        "tabdml.stage4_tuning.make_configured_tree_learner",
        lambda *args, **kwargs: _RecordingModel(fitted_targets),
    )

    skipped = run_tuning_task(task, output_root=tmp_path)
    retried = run_tuning_task(task, output_root=tmp_path, retry_failed=True)

    assert skipped == {"task_key": task.key, "status": "skipped"}
    assert retried["status"] == "success"
    assert len(fitted_targets) == 1


def test_write_tuned_xgboost_atomically_freezes_complete_groups(tmp_path):
    records = [
        _record(target, candidate, observed=1.0, diagnostic=2.0)
        for target in ("l", "m")
        for candidate in ("a", "b")
    ]
    output = tmp_path / "selected" / "selected_xgboost.json"

    selected = write_tuned_xgboost(
        records,
        output,
        expected_replications=1,
        expected_candidates=("a", "b"),
        expected_cells=(CELL,),
    )

    assert json.loads(output.read_text(encoding="utf-8")) == selected
    assert not output.with_suffix(".json.tmp").exists()


def test_write_tuned_xgboost_does_not_create_output_for_incomplete_groups(tmp_path):
    records = [
        _record("l", "a", observed=1.0, diagnostic=2.0),
        _record("m", "a", observed=1.0, diagnostic=2.0),
    ]
    output = tmp_path / "selected_xgboost.json"

    with pytest.raises(ValueError, match="Incomplete tuning records"):
        write_tuned_xgboost(
            records,
            output,
            expected_replications=1,
            expected_candidates=("a", "b"),
            expected_cells=(CELL,),
        )

    assert not output.exists()
