import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from tabdml.dgp import SimulatedData
from tabdml.sharding import belongs_to_shard
from tabdml.stage4_config import TreeBenchmarkCell, load_stage4_config
from tabdml.stage4_tuning import (
    Stage4TuningTask,
    derive_tuning_seeds,
    iter_tuning_tasks,
    run_tuning_task,
    select_tuned_xgboost,
    tuning_run_fingerprint,
    tuning_task_universe_fingerprint,
    write_tuned_xgboost,
)
from tabdml.storage import ResultStore


CONFIG = Path("configs/stage4_tree_benchmark.yaml")
CELL = TreeBenchmarkCell("standard", "tree_stumps", 1000, 10)
CELL_KEY = CELL.key


@pytest.fixture
def config():
    return load_stage4_config(CONFIG)


def _candidate_params(candidate):
    return {
        "n_estimators": 800,
        "max_depth": 1 if candidate == "a" else 2,
    }


def _selection_task(
    target,
    candidate,
    replication=0,
    execution_profile="full",
    cell=CELL,
):
    return Stage4TuningTask(
        stage="stage4_tree_tuning",
        seed_namespace="stage4_tree_tuning",
        panel=cell.panel,
        scenario=cell.scenario,
        n=cell.n,
        p=cell.p,
        replication=replication,
        target=target,
        candidate=candidate,
        params=_candidate_params(candidate),
        validation_fraction=0.25,
        execution_profile=execution_profile,
    )


def _expected_tasks(
    candidates=("a", "b"),
    replications=1,
    execution_profile="full",
    cells=(CELL,),
):
    return tuple(
        _selection_task(target, candidate, replication, execution_profile, cell)
        for cell in cells
        for target in ("l", "m")
        for candidate in candidates
        for replication in range(replications)
    )


def _record(
    target,
    candidate,
    observed,
    diagnostic,
    replication=0,
    execution_profile="full",
):
    task = _selection_task(
        target,
        candidate,
        replication=replication,
        execution_profile=execution_profile,
    )
    return {
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
        "validation_observed_mse": observed,
        "validation_truth_mse_diagnostic": diagnostic,
    }


def _task(target, execution_profile="full"):
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
        execution_profile=execution_profile,
    )


def _data(n=8, p=2):
    index = np.arange(n, dtype=float)
    return SimulatedData(
        X=np.tile(index[:, None], (1, p)),
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


@pytest.mark.parametrize(
    "targets",
    [("l",), ("l", "l"), ("m", "l"), ("l", "m", "g"), "lm"],
)
def test_tuning_rejects_any_target_sequence_except_ordered_l_m(config, targets):
    mutated = deepcopy(config)
    mutated["tuning"]["targets"] = (
        targets if isinstance(targets, str) else list(targets)
    )

    with pytest.raises(ValueError, match="exact ordered targets"):
        tuple(iter_tuning_tasks(mutated, replications=1))


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


def test_fast_and_full_tasks_have_distinct_effective_identity(config):
    full = next(iter_tuning_tasks(config, replications=1, fast=False))
    smoke = next(iter_tuning_tasks(config, replications=1, fast=True))

    assert full.key != smoke.key
    assert full.execution_profile == "full"
    assert smoke.execution_profile == "fast"
    assert full.effective_params["n_estimators"] == 800
    assert smoke.effective_params["n_estimators"] == 20
    assert full.config_hash != smoke.config_hash


def test_tuning_fingerprint_is_canonical_and_binds_run_provenance(config):
    full_tasks = tuple(iter_tuning_tasks(config, replications=1))
    reversed_tasks = tuple(reversed(full_tasks))
    fast_tasks = tuple(iter_tuning_tasks(config, replications=1, fast=True))
    changed_stage = deepcopy(config)
    changed_stage["tuning"]["stage"] = "stage4_tree_tuning_v2"
    changed_namespace = deepcopy(config)
    changed_namespace["tuning"]["seed_namespace"] = "stage4_tree_tuning_v2"

    full_fingerprint = tuning_task_universe_fingerprint(full_tasks, 1)

    assert full_fingerprint == tuning_task_universe_fingerprint(reversed_tasks, 1)
    assert full_fingerprint == tuning_run_fingerprint(
        config,
        replications=1,
        execution_profile="full",
    )
    assert full_fingerprint != tuning_task_universe_fingerprint(fast_tasks, 1)
    assert full_fingerprint != tuning_run_fingerprint(
        changed_stage,
        replications=1,
        execution_profile="full",
    )
    assert full_fingerprint != tuning_run_fingerprint(
        changed_namespace,
        replications=1,
        execution_profile="full",
    )


def test_fast_smoke_record_cannot_resume_or_skip_the_full_task(
    monkeypatch, tmp_path, config
):
    full = next(iter_tuning_tasks(config, replications=1, fast=False))
    smoke = next(iter_tuning_tasks(config, replications=1, fast=True))
    fitted_targets = []
    monkeypatch.setattr(
        "tabdml.stage4_tuning.simulate_plr",
        lambda scenario, n, p, seed, theta0: _data(n, p),
    )
    monkeypatch.setattr(
        "tabdml.stage4_tuning.make_configured_tree_learner",
        lambda *args, **kwargs: _RecordingModel(fitted_targets),
    )

    smoke_record = run_tuning_task(smoke, output_root=tmp_path)
    full_record = run_tuning_task(full, output_root=tmp_path)

    assert smoke_record["status"] == "success"
    assert full_record["status"] == "success"
    assert len(fitted_targets) == 2
    assert len(tuple(tmp_path.glob("*.json"))) == 2
    assert smoke_record["execution_profile"] == "fast"
    assert smoke_record["params"]["n_estimators"] == 20
    assert smoke_record["nominal_params"]["n_estimators"] == 800
    assert smoke_record["config_hash"] == smoke.config_hash
    assert full_record["execution_profile"] == "full"
    assert full_record["params"]["n_estimators"] == 800
    assert full_record["config_hash"] == full.config_hash


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

    record = run_tuning_task(
        _task(target, execution_profile="fast"),
        output_root=tmp_path,
        fast=True,
    )

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

    selected = select_tuned_xgboost(
        records,
        expected_replications=1,
        expected_tasks=_expected_tasks(),
    )

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
        select_tuned_xgboost(
            records,
            expected_replications=2,
            expected_tasks=_expected_tasks(candidates=("a",), replications=2),
        )


def test_selection_rejects_an_expected_candidate_with_no_records():
    records = [
        _record("l", "a", observed=1.0, diagnostic=2.0),
        _record("m", "a", observed=1.0, diagnostic=2.0),
    ]

    with pytest.raises(ValueError, match="Incomplete tuning records"):
        select_tuned_xgboost(
            records,
            expected_replications=1,
            expected_tasks=_expected_tasks(),
        )


def test_selection_rejects_mixed_configurations_within_a_candidate_group():
    records = [
        _record(target, "a", observed=1.0, diagnostic=2.0, replication=replication)
        for target in ("l", "m")
        for replication in (0, 1)
    ]
    records[1]["config_hash"] = "different-hash"

    with pytest.raises(ValueError, match="Invalid tuning record"):
        select_tuned_xgboost(
            records,
            expected_replications=2,
            expected_tasks=_expected_tasks(candidates=("a",), replications=2),
        )


@pytest.mark.parametrize("field", ["stage", "task_key", "config_hash"])
def test_selection_rejects_forged_expected_record_fields(field):
    records = [
        _record(target, "a", observed=1.0, diagnostic=2.0)
        for target in ("l", "m")
    ]
    records[0][field] = f"forged-{field}"

    with pytest.raises(ValueError, match="Invalid tuning record"):
        select_tuned_xgboost(
            records,
            expected_replications=1,
            expected_tasks=_expected_tasks(candidates=("a",)),
        )


@pytest.mark.parametrize("field", ["data_seed", "split_seed", "learner_seed"])
@pytest.mark.parametrize("mutation", ["missing", "forged"])
def test_selection_rejects_missing_or_forged_seed_provenance(field, mutation):
    records = [
        _record(target, "a", observed=1.0, diagnostic=2.0)
        for target in ("l", "m")
    ]
    if mutation == "missing":
        records[0].pop(field)
    else:
        records[0][field] += 1

    with pytest.raises(ValueError, match=rf"{field} mismatch"):
        select_tuned_xgboost(
            records,
            expected_replications=1,
            expected_tasks=_expected_tasks(candidates=("a",)),
        )


def test_selection_rejects_failed_record_from_the_expected_profile():
    records = [
        _record(target, "a", observed=1.0, diagnostic=2.0)
        for target in ("l", "m")
    ]
    records[0]["status"] = "failed"

    with pytest.raises(ValueError, match="Failed tuning record"):
        select_tuned_xgboost(
            records,
            expected_replications=1,
            expected_tasks=_expected_tasks(candidates=("a",)),
        )


def test_selection_isolates_exact_fast_and_full_task_universes():
    full_records = [
        _record(
            target,
            candidate,
            observed=1.0 if candidate == "a" else 2.0,
            diagnostic=9.0 if candidate == "a" else 0.0,
            replication=replication,
        )
        for target in ("l", "m")
        for candidate in ("a", "b")
        for replication in (0, 1)
    ]
    fast_records = [
        _record(
            target,
            candidate,
            observed=3.0 if candidate == "a" else 0.5,
            diagnostic=0.0 if candidate == "a" else 9.0,
            execution_profile="fast",
        )
        for target in ("l", "m")
        for candidate in ("a", "b")
    ]
    mixed_store = full_records + fast_records

    selected_full = select_tuned_xgboost(
        mixed_store,
        expected_replications=2,
        expected_tasks=_expected_tasks(replications=2),
    )
    selected_fast = select_tuned_xgboost(
        mixed_store,
        expected_replications=1,
        expected_tasks=_expected_tasks(execution_profile="fast"),
    )

    assert selected_full["execution_profile"] == "full"
    assert selected_fast["execution_profile"] == "fast"
    assert selected_full["cells"][CELL_KEY]["l"]["candidate"] == "a"
    assert selected_fast["cells"][CELL_KEY]["l"]["candidate"] == "b"
    assert selected_full["cells"][CELL_KEY]["l"]["params"]["n_estimators"] == 800
    assert selected_fast["cells"][CELL_KEY]["l"]["params"]["n_estimators"] == 20


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


def test_successful_result_resume_rejects_forged_seed_provenance(
    monkeypatch, tmp_path
):
    task = _task("l")
    fitted_targets = []
    monkeypatch.setattr(
        "tabdml.stage4_tuning.simulate_plr", lambda *args, **kwargs: _data()
    )
    monkeypatch.setattr(
        "tabdml.stage4_tuning.make_configured_tree_learner",
        lambda *args, **kwargs: _RecordingModel(fitted_targets),
    )
    successful = run_tuning_task(task, output_root=tmp_path)
    successful["learner_seed"] += 1
    ResultStore(tmp_path).write(successful)

    with pytest.raises(ValueError, match="learner_seed mismatch"):
        run_tuning_task(task, output_root=tmp_path, retry_failed=True)

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
        expected_tasks=_expected_tasks(),
    )

    assert json.loads(output.read_text(encoding="utf-8")) == selected
    assert selected["tuning_stage"] == "stage4_tree_tuning"
    assert selected["tuning_seed_namespace"] == "stage4_tree_tuning"
    assert selected["tuning_run_fingerprint"] == (
        tuning_task_universe_fingerprint(_expected_tasks(), 1)
    )
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
            expected_tasks=_expected_tasks(),
        )

    assert not output.exists()
