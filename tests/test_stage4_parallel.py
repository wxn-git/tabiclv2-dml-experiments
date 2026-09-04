import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from scripts import run_stage4_cache, run_stage4_tuning
from tabdml import stage4_parallel as parallel
from tabdml.nuisance_cache import NuisanceCache
from tabdml.sharding import belongs_to_shard
from tabdml.stage4_config import load_stage4_config
from tabdml.stage4_experiment import (
    build_stage4_nuisance_spec,
    compose_stage4_record,
    iter_stage4_pairs,
)
from tabdml.stage4_tuning import derive_tuning_seeds, iter_tuning_tasks
from tabdml.storage import ResultStore
from test_stage4_experiment import (
    failure_record,
    frozen_for_config,
    selection_for_config,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/stage4_tree_benchmark.yaml"


def option(command, flag):
    return command.argv[command.argv.index(flag) + 1]


@pytest.fixture
def inputs(tmp_path):
    config = load_stage4_config(CONFIG)
    frozen = frozen_for_config(config)
    tuned = tmp_path / "tuned.json"
    tuned.write_text(json.dumps(frozen), encoding="utf-8")
    selected = tmp_path / "selected.json"
    selected.write_text(json.dumps(selection_for_config(config)), encoding="utf-8")
    return dict(
        python_executable=sys.executable,
        project_root=ROOT,
        config_path=CONFIG,
        tuned_models=tuned,
        selected_cells=selected,
        cache_root=tmp_path / "cache",
        output_root=tmp_path / "raw",
        log_dir=tmp_path / "logs",
        replications=1,
    )


@pytest.fixture(autouse=True)
def no_processes(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("A test attempted to start real experiment processes")
    monkeypatch.setattr(parallel, "run_workers", forbidden)


def test_tuning_uses_only_eight_cpu_workers(inputs):
    commands = parallel.build_stage4_tuning_commands(
        sys.executable, ROOT, CONFIG, inputs["output_root"],
        cpu_workers=8, replications=1, fast=True, retry_failed=True,
    )
    assert len(commands) == 8
    for index, command in enumerate(commands):
        assert command.name == f"cpu_stage4_tuning_{index:02d}"
        assert option(command, "--num-shards") == "8"
        assert option(command, "--shard-index") == str(index)
        assert "--fast" in command.argv and "--retry-failed" in command.argv
        assert "--select" not in command.argv


@pytest.mark.parametrize("phase", ["screening", "confirmation"])
def test_cache_uses_one_unsharded_gpu_and_eight_cpu_workers(inputs, phase):
    commands = parallel.build_stage4_cache_commands(
        sys.executable, ROOT, CONFIG, inputs["tuned_models"], inputs["cache_root"],
        phase=phase, cpu_workers=8, replications=5,
        selected_cells=inputs["selected_cells"], retry_failed=True,
    )
    assert len(commands) == 9
    assert commands[0].name == f"gpu_stage4_{phase}"
    assert option(commands[0], "--device-group") == "gpu"
    assert "--num-shards" not in commands[0].argv
    for index, command in enumerate(commands[1:]):
        assert option(command, "--device-group") == "cpu"
        assert option(command, "--num-shards") == "8"
        assert option(command, "--shard-index") == str(index)
        assert "tabiclv2" not in " ".join(command.argv)
    for command in commands:
        assert option(command, "--replications") == "5"
        assert "--fast" not in command.argv
        assert "--retry-failed" in command.argv
        if phase == "confirmation":
            assert option(command, "--selected-cells") == str(inputs["selected_cells"])


def test_confirmation_requires_frozen_cells_path(inputs):
    with pytest.raises(ValueError, match="selected confirmation cells"):
        parallel.build_stage4_cache_commands(
            sys.executable, ROOT, CONFIG, inputs["tuned_models"], inputs["cache_root"],
            phase="confirmation", selected_cells=None,
        )


@pytest.mark.parametrize("cpu_workers", [0, 9, -1, True, 1.5])
def test_worker_limit_rejected(inputs, cpu_workers):
    with pytest.raises(ValueError, match="cpu_workers"):
        parallel.run_stage4_phase(phase="tuning", cpu_workers=cpu_workers, **inputs)


def test_eight_tuning_shards_cover_unique_complete_keys(inputs, monkeypatch):
    config = load_stage4_config(CONFIG)
    expected = {task.key for task in iter_tuning_tasks(config, 1)}
    seen = []
    monkeypatch.setattr(run_stage4_tuning, "run_tuning_task", lambda task, **kw: (
        seen.append(task.key) or {"status": "success"}
    ))
    commands = parallel.build_stage4_tuning_commands(
        sys.executable, ROOT, CONFIG, inputs["output_root"], replications=1,
    )
    for command in commands:
        start = len(seen)
        monkeypatch.setattr(sys, "argv", list(command.argv[1:]))
        assert run_stage4_tuning.main() == 0
        shard = int(option(command, "--shard-index"))
        assert all(belongs_to_shard(key, 8, shard) for key in seen[start:])
    assert len(seen) == len(set(seen)) == len(expected) == 288
    assert set(seen) == expected


def test_cache_cli_shards_unique_nuisances_not_pairs(inputs, monkeypatch):
    config = load_stage4_config(CONFIG)
    pairs = tuple(iter_stage4_pairs(config, "screening", frozen_for_config(config), replications=1))
    expected = {build_stage4_nuisance_spec(pair, target).key
                for pair in pairs for target in ("l", "m")}
    assert len(expected) < 2 * len(pairs)  # Oracle diagnostics share fits.
    seen = []
    monkeypatch.setattr(run_stage4_cache, "fit_stage4_nuisance", lambda pair, target, *a, **kw: (
        seen.append(build_stage4_nuisance_spec(pair, target).key)
    ))
    commands = parallel.build_stage4_cache_commands(
        sys.executable, ROOT, CONFIG, inputs["tuned_models"], inputs["cache_root"],
        phase="screening", replications=1,
    )
    cpu_keys = set()
    gpu_keys = set()
    for command in commands:
        start = len(seen)
        monkeypatch.setattr(sys, "argv", list(command.argv[1:]))
        assert run_stage4_cache.main() == 0
        assigned = seen[start:]
        if option(command, "--device-group") == "cpu":
            shard = int(option(command, "--shard-index"))
            assert all(belongs_to_shard(key, 8, shard) for key in assigned)
            cpu_keys.update(assigned)
        else:
            gpu_keys.update(assigned)
    assert not cpu_keys & gpu_keys
    assert len(seen) == len(set(seen)) == len(expected) == 288
    assert set(seen) == expected


def small_pairs(monkeypatch, inputs):
    config = load_stage4_config(CONFIG)
    pairs = tuple(iter_stage4_pairs(config, "screening", frozen_for_config(config), replications=1))
    pairs = tuple(p for p in pairs[:10] if p.learner_l in {"oracle", "xgboost_tuned"}
                  and p.learner_m in {"oracle", "xgboost_tuned"})
    monkeypatch.setattr(parallel, "iter_stage4_pairs", lambda *a, **kw: iter(pairs))
    return pairs


def write_cache(inputs, pairs):
    cache = NuisanceCache(inputs["cache_root"])
    tasks = {build_stage4_nuisance_spec(p, t).key: build_stage4_nuisance_spec(p, t)
             for p in pairs for t in ("l", "m")}
    for task in tasks.values():
        cache.write(task, np.zeros(task.n), (0.0,) * task.folds_count, None, None)
    return len(tasks)


def write_records(inputs, pairs):
    cache = NuisanceCache(inputs["cache_root"])
    store = ResultStore(inputs["output_root"])
    for pair in pairs:
        store.write(compose_stage4_record(pair, *(
            cache.read(build_stage4_nuisance_spec(pair, t), pair.n) for t in ("l", "m")
        )))


def progress(inputs):
    return json.loads((inputs["log_dir"] / "progress.json").read_text(encoding="utf-8"))


def test_cache_then_compose_progress_restart_and_atomic_writes(inputs, monkeypatch):
    pairs = small_pairs(monkeypatch, inputs)
    stages = []
    snapshots = []
    real_replace = parallel.os.replace
    def replace(source, destination):
        assert Path(source) != Path(destination)
        if Path(destination).name == "progress.json":
            snapshots.append(json.loads(Path(source).read_text(encoding="utf-8")))
        real_replace(source, destination)
    monkeypatch.setattr(parallel.os, "replace", replace)
    def run(commands, cwd, log_dir):
        stages.append(commands)
        before = progress(inputs)
        assert before["status"] == "running"
        if len(commands) == 9:
            assert before["successful_tasks"] == (0 if len(stages) == 1 else 8)
            write_cache(inputs, pairs)
        else:
            assert len(commands) == 1
            assert Path(commands[0].argv[1]).name == "compose_stage4_dml.py"
            assert before["successful_tasks"] >= 4
            assert "--retry-failed" in commands[0].argv
            write_records(inputs, pairs)
        return {c.name: 0 for c in commands}
    monkeypatch.setattr(parallel, "run_workers", run)
    for _ in range(2):
        assert parallel.run_stage4_phase(phase="screening", retry_failed=True, **inputs) == 0
        state = progress(inputs)
        assert state["planned_tasks"] == state["successful_tasks"] == 8
        assert state["failed_tasks"] == 0
        assert state["status"] == "completed"
        assert state["started_at"] <= state["updated_at"]
    assert len(stages) == 4
    assert len(snapshots) >= 8
    assert not list(inputs["log_dir"].glob("*.tmp"))


@pytest.mark.parametrize("outcome", ["worker_failed", "missing_worker", "missing_cache", "invalid_cache", "exception"])
def test_failed_cache_stage_never_chains(inputs, monkeypatch, outcome):
    pairs = small_pairs(monkeypatch, inputs)
    calls = []
    def run(commands, **kwargs):
        calls.append(commands)
        if outcome not in {"missing_cache", "exception"}:
            write_cache(inputs, pairs)
        if outcome == "invalid_cache":
            next(inputs["cache_root"].glob("*.npz")).write_bytes(b"corrupt")
        if outcome == "exception":
            raise OSError("launch failed")
        codes = {c.name: 0 for c in commands}
        if outcome == "worker_failed":
            codes[commands[0].name] = 7
        if outcome == "missing_worker":
            codes.pop(commands[-1].name)
        return codes
    monkeypatch.setattr(parallel, "run_workers", run)
    assert parallel.run_stage4_phase(phase="screening", **inputs) != 0
    assert len(calls) == 1
    state = progress(inputs)
    assert state["status"] == "failed"
    expected = 4 if outcome in {"worker_failed", "missing_worker"} else 3 if outcome == "invalid_cache" else 0
    assert state["successful_tasks"] == expected
    assert state["failed_tasks"] == 4 - expected


@pytest.mark.parametrize("outcome", ["missing", "failed", "forged", "exit_failed"])
def test_compose_exit_zero_is_not_artifact_success(inputs, monkeypatch, outcome):
    pairs = small_pairs(monkeypatch, inputs)
    def run(commands, **kwargs):
        if len(commands) == 9:
            write_cache(inputs, pairs)
        elif outcome != "missing":
            write_records(inputs, pairs)
            if outcome == "failed":
                ResultStore(inputs["output_root"]).write(failure_record(pairs[0]))
            if outcome == "forged":
                path = inputs["output_root"] / f"{pairs[0].key}.json"
                data = json.loads(path.read_text())
                data["panel"] = "forged"
                path.write_text(json.dumps(data))
        return {c.name: (6 if len(commands) == 1 and outcome == "exit_failed" else 0) for c in commands}
    monkeypatch.setattr(parallel, "run_workers", run)
    assert parallel.run_stage4_phase(phase="screening", **inputs) != 0
    state = progress(inputs)
    expected = {"missing": 4, "failed": 7, "forged": 7, "exit_failed": 8}[outcome]
    assert state["successful_tasks"] == expected
    assert state["failed_tasks"] == 8 - expected
    assert state["status"] == "failed"


def tuning_record(task, status="success"):
    return {
        **asdict(task), "task_key": task.key, "status": status,
        "learner_kind": "xgboost", "nominal_params": task.params,
        "nominal_config_hash": task.nominal_config_hash,
        "params": task.effective_params, "config_hash": task.config_hash,
        **derive_tuning_seeds(task), "validation_observed_mse": 1.0,
        "validation_truth_mse_diagnostic": 2.0,
    }


@pytest.mark.parametrize("status", ["success", "failed", "oom"])
def test_tuning_counts_expected_records_not_exit_codes(inputs, monkeypatch, status):
    tasks = tuple(iter_tuning_tasks(load_stage4_config(CONFIG), 1))[:2]
    monkeypatch.setattr(parallel, "iter_tuning_tasks", lambda *a, **kw: iter(tasks))
    store = ResultStore(inputs["output_root"])
    store.write(tuning_record(tasks[0], status))
    # Duplicate content under another filename must not inflate counts.
    (inputs["output_root"] / "duplicate.json").write_text(json.dumps(tuning_record(tasks[0])))
    def run(commands, **kwargs):
        assert len(commands) == 8
        store.write(tuning_record(tasks[1]))
        return {c.name: 0 for c in commands}
    monkeypatch.setattr(parallel, "run_workers", run)
    result = parallel.run_stage4_phase(phase="tuning", **inputs)
    assert (result == 0) == (status == "success")
    state = progress(inputs)
    assert state["planned_tasks"] == 2
    assert state["successful_tasks"] == (2 if status == "success" else 1)
    assert state["failed_tasks"] == (0 if status == "success" else 1)


@pytest.mark.parametrize("bad", ["config", "tuned", "selection", "profile", "output_file", "ancestor_file", "historical", "overlap", "script", "python"])
def test_invalid_inputs_rejected_before_processes_or_progress(inputs, bad, tmp_path):
    if bad == "config":
        inputs["config_path"] = tmp_path / "missing.yaml"
    elif bad == "tuned":
        inputs["tuned_models"].write_text("{}")
    elif bad == "selection":
        inputs["selected_cells"].write_text("{}")
    elif bad == "profile":
        inputs["fast"] = True
    elif bad == "output_file":
        inputs["output_root"].write_text("occupied")
    elif bad == "ancestor_file":
        file = tmp_path / "file"
        file.write_text("occupied")
        inputs["cache_root"] = file / "cache"
    elif bad == "historical":
        inputs["output_root"] = ROOT / "results/stage3b_tree_simple_confirmation_raw"
    elif bad == "overlap":
        inputs["output_root"] = inputs["cache_root"]
    elif bad == "script":
        inputs["project_root"] = tmp_path
    elif bad == "python":
        inputs["python_executable"] = tmp_path / "missing-python.exe"
    with pytest.raises((ValueError, FileNotFoundError)):
        parallel.run_stage4_phase(phase="confirmation", **inputs)
    assert not inputs["log_dir"].exists()


@pytest.mark.parametrize("kind", ["cache", "record", "tuning"])
def test_existing_invalid_artifacts_rejected_prelaunch(inputs, monkeypatch, kind):
    pairs = small_pairs(monkeypatch, inputs)
    phase = "screening"
    if kind == "cache":
        write_cache(inputs, pairs)
        next(inputs["cache_root"].glob("*.npz")).write_bytes(b"invalid")
    elif kind == "record":
        record = failure_record(pairs[0])
        record["panel"] = "forged"
        ResultStore(inputs["output_root"]).write(record)
    else:
        phase = "tuning"
        task = next(iter_tuning_tasks(load_stage4_config(CONFIG), 1))
        record = tuning_record(task)
        record["validation_observed_mse"] = float("nan")
        ResultStore(inputs["output_root"]).write(record)
    with pytest.raises(ValueError):
        parallel.run_stage4_phase(phase=phase, **inputs)
    assert not inputs["log_dir"].exists()


def test_retry_repairs_invalid_cache_and_failed_composition(inputs, monkeypatch):
    pairs = small_pairs(monkeypatch, inputs)
    write_cache(inputs, pairs)
    next(inputs["cache_root"].glob("*.npz")).write_bytes(b"invalid")
    ResultStore(inputs["output_root"]).write(failure_record(pairs[0]))
    def run(commands, **kwargs):
        assert all("--retry-failed" in c.argv for c in commands)
        if len(commands) == 9:
            write_cache(inputs, pairs)
        else:
            write_records(inputs, pairs)
        return {c.name: 0 for c in commands}
    monkeypatch.setattr(parallel, "run_workers", run)
    assert parallel.run_stage4_phase(phase="screening", retry_failed=True, **inputs) == 0


def test_readonly_worker_log_is_rejected_before_launch(inputs, monkeypatch):
    log = inputs["log_dir"] / "tuning/cpu_stage4_tuning_00.stdout.log"
    log.parent.mkdir(parents=True)
    log.write_text("old log", encoding="utf-8")
    access = parallel.os.access
    monkeypatch.setattr(parallel.os, "access", lambda path, mode: (
        False if Path(path) == log else access(path, mode)
    ))
    with pytest.raises(ValueError, match="writable"):
        parallel.run_stage4_phase(phase="tuning", **inputs)
    assert not (inputs["log_dir"] / "progress.json").exists()


def test_relative_executable_resolves_against_project_not_cwd(inputs, monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)
    inputs["python_executable"] = ".venv/Scripts/python.exe"
    inputs["config_path"] = "configs/stage4_tree_benchmark.yaml"
    assert parallel.run_stage4_phase(phase="tuning", dry_run=True, **inputs) == 0
    assert str(ROOT / ".venv/Scripts/python.exe") in capsys.readouterr().out
    assert not inputs["log_dir"].exists()


@pytest.mark.parametrize("builder", ["tuning", "cache"])
def test_fast_command_builders_default_to_one_replication(inputs, builder):
    if builder == "tuning":
        commands = parallel.build_stage4_tuning_commands(
            sys.executable, ROOT, CONFIG, inputs["output_root"], fast=True,
        )
    else:
        commands = parallel.build_stage4_cache_commands(
            sys.executable, ROOT, CONFIG, inputs["tuned_models"], inputs["cache_root"],
            phase="confirmation", selected_cells=inputs["selected_cells"], fast=True,
        )
    assert all(option(command, "--replications") == "1" for command in commands)


@pytest.mark.parametrize("replications", [0, -1, True, 1.5])
def test_invalid_replications_rejected_before_launch(inputs, replications):
    inputs["replications"] = replications
    with pytest.raises(ValueError, match="replications"):
        parallel.run_stage4_phase(phase="tuning", **inputs)


def test_failed_record_skipped_by_real_composer_stays_failed(inputs, monkeypatch):
    from scripts import compose_stage4_dml

    pairs = small_pairs(monkeypatch, inputs)
    write_cache(inputs, pairs)
    write_records(inputs, pairs)
    ResultStore(inputs["output_root"]).write(failure_record(pairs[0]))
    monkeypatch.setattr(compose_stage4_dml, "iter_stage4_pairs", lambda *a, **kw: iter(pairs))
    def run(commands, **kwargs):
        if len(commands) == 1:
            monkeypatch.setattr(sys, "argv", list(commands[0].argv[1:]))
            assert compose_stage4_dml.main() == 0  # CLI skips a previous failure.
        return {c.name: 0 for c in commands}
    monkeypatch.setattr(parallel, "run_workers", run)
    assert parallel.run_stage4_phase(phase="screening", **inputs) == 1
    assert progress(inputs)["successful_tasks"] == 7
    assert progress(inputs)["failed_tasks"] == 1


def test_parent_progress_does_not_trust_stale_completed_state(inputs, monkeypatch):
    tasks = tuple(iter_tuning_tasks(load_stage4_config(CONFIG), 1))[:2]
    monkeypatch.setattr(parallel, "iter_tuning_tasks", lambda *a, **kw: iter(tasks))
    inputs["log_dir"].mkdir()
    (inputs["log_dir"] / "progress.json").write_text(json.dumps({
        "status": "completed", "successful_tasks": 9999,
    }))
    def run(commands, **kwargs):
        assert progress(inputs)["successful_tasks"] == 0
        return {c.name: 0 for c in commands}
    monkeypatch.setattr(parallel, "run_workers", run)
    assert parallel.run_stage4_phase(phase="tuning", **inputs) == 1
    assert progress(inputs)["successful_tasks"] == 0
    assert progress(inputs)["failed_tasks"] == 2


def test_atomic_progress_replace_failure_preserves_previous_snapshot(tmp_path, monkeypatch):
    path = tmp_path / "progress.json"
    path.write_text('{"status": "old"}')
    def fail_replace(*args):
        raise OSError("replace denied")
    monkeypatch.setattr(parallel.os, "replace", fail_replace)
    with pytest.raises(OSError, match="replace denied"):
        parallel._write_progress(path, {"status": "new"})
    assert json.loads(path.read_text()) == {"status": "old"}


def test_interrupted_workers_leave_truthful_atomic_progress(inputs, monkeypatch):
    pairs = small_pairs(monkeypatch, inputs)
    def run(commands, **kwargs):
        write_cache(inputs, pairs)
        raise KeyboardInterrupt
    monkeypatch.setattr(parallel, "run_workers", run)
    with pytest.raises(KeyboardInterrupt):
        parallel.run_stage4_phase(phase="screening", **inputs)
    state = progress(inputs)
    assert state["status"] == "interrupted"
    assert state["successful_tasks"] == 4
    assert state["pending_tasks"] == 4


def test_tuning_overflowing_metric_is_invalid_artifact_not_unhandled_error(inputs, monkeypatch):
    task = next(iter_tuning_tasks(load_stage4_config(CONFIG), 1))
    monkeypatch.setattr(parallel, "iter_tuning_tasks", lambda *a, **kw: iter((task,)))
    def run(commands, **kwargs):
        record = tuning_record(task)
        record["validation_observed_mse"] = 10 ** 400
        ResultStore(inputs["output_root"]).write(record)
        return {c.name: 0 for c in commands}
    monkeypatch.setattr(parallel, "run_workers", run)
    assert parallel.run_stage4_phase(phase="tuning", **inputs) == 1
    assert progress(inputs)["successful_tasks"] == 0
    assert progress(inputs)["failed_tasks"] == 1
