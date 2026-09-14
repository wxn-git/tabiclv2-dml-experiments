import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from tabdml.dgp import SimulatedData
from tabdml.stage5_config import load_stage5_config, stage5_config_fingerprint
from tabdml.stage5_tuning import (
    Stage5TuningTask,
    derive_stage5_tuning_seeds,
    iter_stage5_tuning_tasks,
    load_stage5_tuning,
    run_stage5_tuning_task,
    select_stage5_tuning,
    stage5_tuning_run_fingerprint,
    write_stage5_tuning,
)
from tabdml.storage import ResultStore


CONFIG = Path("configs/stage5_sensitivity.yaml")


@pytest.fixture
def config():
    return load_stage5_config(CONFIG)


def _task(
    target="l",
    candidate="z-first",
    replication=0,
    profile="full",
    config_fingerprint="fingerprint",
):
    return Stage5TuningTask(
        stage="stage5_tuning",
        seed_namespace="stage5_tuning_v1",
        scenario="linear",
        n=8,
        p=2,
        replication=replication,
        target=target,
        candidate=candidate,
        params={
            "n_estimators": 30 if candidate == "z-first" else 40,
            "max_depth": 1 if candidate == "z-first" else 2,
        },
        validation_fraction=0.25,
        theta0=1.0,
        execution_profile=profile,
        config_fingerprint=config_fingerprint,
    )


def _data():
    index = np.arange(8, dtype=float)
    return SimulatedData(
        X=np.column_stack((index, index)),
        y=index + 10.0,
        d=index + 20.0,
        l0=index + 30.0,
        m0=index + 40.0,
        g0=index,
        theta0=1.0,
        categorical_indices=(),
    )


class _ZeroModel:
    def __init__(self, fitted):
        self.fitted = fitted

    def fit(self, X, y):
        self.fitted.append(np.asarray(y))
        return self

    def predict(self, X):
        return np.zeros(len(X))


def _tiny_config(config):
    value = deepcopy(config)
    value["scenarios"] = ["linear"]
    value["tuning"]["center"] = {"n": 8, "p": 2}
    value["tuning"]["replications"] = 1
    value["tuning"]["xgboost_candidates"] = [
        {"name": "z-first", "params": {"n_estimators": 30, "max_depth": 1}},
        {"name": "a-second", "params": {"n_estimators": 40, "max_depth": 2}},
    ]
    return value


def _record(task, observed=1.0, truth=2.0, **updates):
    record = {
        "task_key": task.key,
        "status": "success",
        "stage": task.stage,
        "seed_namespace": task.seed_namespace,
        "scenario": task.scenario,
        "n": task.n,
        "p": task.p,
        "replication": task.replication,
        "target": task.target,
        "candidate": task.candidate,
        "theta0": task.theta0,
        "learner_kind": "xgboost",
        "execution_profile": task.execution_profile,
        "config_fingerprint": task.config_fingerprint,
        "nominal_params": task.params,
        "nominal_config_hash": task.nominal_config_hash,
        "params": task.effective_params,
        "config_hash": task.config_hash,
        "validation_fraction": task.validation_fraction,
        **derive_stage5_tuning_seeds(task),
        "selection_metric": "validation_y_mse" if task.target == "l" else "validation_d_mse",
        "validation_observed_mse": observed,
        "validation_truth_mse_diagnostic": truth,
        "runtime_seconds": 0.01,
    }
    record.update(updates)
    return record


def test_exact_full_and_fast_universes_are_centered_ordered_and_unique(config):
    full = tuple(iter_stage5_tuning_tasks(config))
    fast = tuple(iter_stage5_tuning_tasks(config, execution_profile="fast"))

    assert len(full) == len({task.key for task in full}) == 720
    assert len(fast) == len({task.key for task in fast}) == 72
    assert {(task.n, task.p) for task in full + fast} == {(1000, 50)}
    assert list(dict.fromkeys(task.scenario for task in full)) == config["scenarios"]
    assert list(dict.fromkeys(task.target for task in full)) == ["l", "m"]
    assert {task.replication for task in full} == set(range(10))
    assert {task.replication for task in fast} == {0}


def test_sharding_is_deterministic_complete_and_disjoint(config):
    all_keys = {task.key for task in iter_stage5_tuning_tasks(config)}
    shards = [
        {task.key for task in iter_stage5_tuning_tasks(config, num_shards=3, shard_index=i)}
        for i in range(3)
    ]
    assert set.union(*shards) == all_keys
    assert sum(map(len, shards)) == len(all_keys)
    assert shards[1] == {
        task.key for task in iter_stage5_tuning_tasks(config, num_shards=3, shard_index=1)
    }


def test_candidates_and_targets_share_data_and_split_but_not_learner_seed(config):
    tasks = tuple(iter_stage5_tuning_tasks(config))
    same_rep = [
        task
        for task in tasks
        if task.scenario == "linear" and task.replication == 0
    ]
    seeds = [derive_stage5_tuning_seeds(task) for task in same_rep]
    assert len({seed["data_seed"] for seed in seeds}) == 1
    assert len({seed["split_seed"] for seed in seeds}) == 1
    assert len({seed["learner_seed"] for seed in seeds}) == len(same_rep)


def test_fast_and_full_identity_caps_only_effective_estimators(config):
    full = next(iter_stage5_tuning_tasks(config))
    fast = next(iter_stage5_tuning_tasks(config, execution_profile="fast"))
    assert full.nominal_config_hash == fast.nominal_config_hash
    assert full.config_hash != fast.config_hash
    assert full.key != fast.key
    assert full.effective_params["n_estimators"] == 800
    assert fast.effective_params["n_estimators"] == 20
    assert full.execution_profile == "full" and fast.execution_profile == "fast"


@pytest.mark.parametrize(("target", "response_offset", "truth_offset", "metric"), [
    ("l", 10.0, 30.0, "validation_y_mse"),
    ("m", 20.0, 40.0, "validation_d_mse"),
])
def test_fit_uses_observed_target_and_records_observed_and_truth_metrics(
    monkeypatch, tmp_path, target, response_offset, truth_offset, metric
):
    fitted = []
    monkeypatch.setattr("tabdml.stage5_tuning.simulate_plr", lambda *a, **k: _data())
    monkeypatch.setattr(
        "tabdml.stage5_tuning.make_configured_tree_learner",
        lambda *a, **k: _ZeroModel(fitted),
    )
    task = _task(target)
    record = run_stage5_tuning_task(task, tmp_path)
    validation = np.setdiff1d(np.arange(8), np.flatnonzero(np.isin(np.arange(8) + response_offset, fitted[0])))
    expected_observed = np.mean((np.arange(8)[validation] + response_offset) ** 2)
    expected_truth = np.mean((np.arange(8)[validation] + truth_offset) ** 2)
    assert record["status"] == "success"
    assert record["selection_metric"] == metric
    assert record["validation_observed_mse"] == pytest.approx(expected_observed)
    assert record["validation_truth_mse_diagnostic"] == pytest.approx(expected_truth)
    assert np.isfinite(record["runtime_seconds"])


@pytest.mark.parametrize("terminal_status", ["failed", "oom"])
def test_resume_terminal_failures_require_retry(
    monkeypatch, tmp_path, terminal_status
):
    task = _task()
    ResultStore(tmp_path).write({**_record(task), "status": terminal_status})
    assert run_stage5_tuning_task(task, tmp_path)["status"] == "skipped"
    monkeypatch.setattr("tabdml.stage5_tuning.simulate_plr", lambda *a, **k: _data())
    monkeypatch.setattr("tabdml.stage5_tuning.make_configured_tree_learner", lambda *a, **k: _ZeroModel([]))
    assert run_stage5_tuning_task(task, tmp_path, retry_failed=True)["status"] == "success"


def test_resume_validates_success_and_rejects_unknown_status(tmp_path):
    task = _task()
    ResultStore(tmp_path).write(_record(task))
    assert run_stage5_tuning_task(task, tmp_path) == {"task_key": task.key, "status": "skipped"}

    forged = _record(task, data_seed=derive_stage5_tuning_seeds(task)["data_seed"] + 1)
    ResultStore(tmp_path).write(forged)
    with pytest.raises(ValueError, match="data_seed mismatch"):
        run_stage5_tuning_task(task, tmp_path)

    ResultStore(tmp_path).write({**_record(task), "status": "mystery"})
    with pytest.raises(ValueError, match="status"):
        run_stage5_tuning_task(task, tmp_path, retry_failed=True)


@pytest.mark.parametrize("replications", [1, 9, 11])
def test_full_profile_rejects_non_contract_replication_overrides(config, replications):
    with pytest.raises(ValueError, match="full.*exactly 10"):
        tuple(iter_stage5_tuning_tasks(config, replications=replications))


def test_selector_uses_observed_loss_and_configured_order_for_ties(config):
    tiny = _tiny_config(config)
    fingerprint = stage5_config_fingerprint(tiny)
    tasks = tuple(
        _task(target, candidate, config_fingerprint=fingerprint)
        for target in ("l", "m")
        for candidate in ("z-first", "a-second")
    )
    records = [
        _record(task, observed=1.0, truth=0.0 if task.candidate == "a-second" else 9.0)
        for task in tasks
    ]
    selected = select_stage5_tuning(records, tiny, "full")
    assert selected["scenarios"]["linear"]["l"]["candidate"] == "z-first"
    assert selected["scenarios"]["linear"]["m"]["candidate"] == "z-first"
    assert [item["candidate"] for item in selected["candidate_ranking"]["linear"]["l"]] == ["z-first", "a-second"]


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda records: records.pop(), "missing"),
        (lambda records: records.append(dict(records[0])), "duplicate"),
        (lambda records: records[0].update(status="failed"), "Failed"),
        (lambda records: records[0].update(validation_observed_mse=float("nan")), "validation_observed_mse"),
        (lambda records: records[0].update(config_fingerprint="wrong"), "config_fingerprint"),
        (lambda records: records[0].update(execution_profile="fast"), "execution_profile"),
        (lambda records: records[0].update(data_seed=records[0]["data_seed"] + 1), "data_seed"),
    ],
)
def test_selector_fails_closed_on_invalid_universe(config, mutation, message):
    tiny = _tiny_config(config)
    fingerprint = stage5_config_fingerprint(tiny)
    tasks = [
        _task(target, candidate, config_fingerprint=fingerprint)
        for target in ("l", "m")
        for candidate in ("z-first", "a-second")
    ]
    records = [_record(task) for task in tasks]
    mutation(records)
    with pytest.raises(ValueError, match=message):
        select_stage5_tuning(records, tiny, "full")


@pytest.mark.parametrize(
    ("field", "alias"),
    [("theta0", True), ("n", 8.0), ("replication", False)],
)
def test_selector_metadata_comparisons_are_type_strict(config, field, alias):
    tiny = _tiny_config(config)
    fingerprint = stage5_config_fingerprint(tiny)
    tasks = [
        _task(target, candidate, config_fingerprint=fingerprint)
        for target in ("l", "m")
        for candidate in ("z-first", "a-second")
    ]
    records = [_record(task) for task in tasks]
    records[0][field] = alias
    with pytest.raises(ValueError, match=rf"{field} mismatch"):
        select_stage5_tuning(records, tiny, "full")


def test_run_fingerprint_is_deterministic_and_profile_sensitive(config):
    assert stage5_tuning_run_fingerprint(config, 10, "full") == stage5_tuning_run_fingerprint(config, 10, "full")
    assert stage5_tuning_run_fingerprint(config, 10, "full") != stage5_tuning_run_fingerprint(config, 1, "fast")


def test_write_is_atomic_and_contains_required_frozen_metadata(monkeypatch, tmp_path, config):
    tiny = _tiny_config(config)
    fingerprint = stage5_config_fingerprint(tiny)
    tasks = [_task(t, c, config_fingerprint=fingerprint) for t in ("l", "m") for c in ("z-first", "a-second")]
    replacements = []
    monkeypatch.setattr("tabdml.stage5_tuning.os.replace", lambda src, dst: (replacements.append((Path(src), Path(dst))), Path(dst).write_bytes(Path(src).read_bytes()), Path(src).unlink()))
    output = tmp_path / "frozen" / "selected.json"
    selected = write_stage5_tuning([_record(task) for task in tasks], tiny, output, "full")
    assert replacements == [(output.with_suffix(".json.tmp"), output)]
    assert json.loads(output.read_text()) == selected
    assert selected["schema_version"] == "stage5_tuning_v1"
    assert selected["config_fingerprint"] == fingerprint
    assert selected["theta0"] == 1.0
    assert selected["expected_replications"] == 1


def test_frozen_loader_validates_schema_and_provenance(tmp_path, config):
    tiny = _tiny_config(config)
    fingerprint = stage5_config_fingerprint(tiny)
    tasks = [_task(t, c, config_fingerprint=fingerprint) for t in ("l", "m") for c in ("z-first", "a-second")]
    output = tmp_path / "selected.json"
    frozen = write_stage5_tuning([_record(task) for task in tasks], tiny, output, "full")
    assert load_stage5_tuning(output, tiny, "full") == frozen
    frozen["tuning_run_fingerprint"] = "forged"
    output.write_text(json.dumps(frozen), encoding="utf-8")
    with pytest.raises(ValueError, match="tuning_run_fingerprint"):
        load_stage5_tuning(output, tiny, "full")


def test_frozen_loader_rejects_boolean_replication_alias(tmp_path, config):
    tiny = _tiny_config(config)
    fingerprint = stage5_config_fingerprint(tiny)
    tasks = [_task(t, c, config_fingerprint=fingerprint) for t in ("l", "m") for c in ("z-first", "a-second")]
    output = tmp_path / "selected.json"
    frozen = write_stage5_tuning([_record(task) for task in tasks], tiny, output, "full")
    frozen["expected_replications"] = True
    output.write_text(json.dumps(frozen), encoding="utf-8")
    with pytest.raises(ValueError, match="expected_replications"):
        load_stage5_tuning(output, tiny, "full")
