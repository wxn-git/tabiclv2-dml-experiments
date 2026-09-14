from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from tabdml import stage5_parallel as parallel
from tabdml.parallel import WorkerCommand


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/stage5_sensitivity.yaml"


def option(command, name):
    index = command.argv.index(name)
    return command.argv[index + 1]


def test_command_batches_have_one_gpu_cpu_shards_ensemble_then_compose(tmp_path):
    batches = parallel.build_stage5_cache_commands(
        sys.executable, ROOT, CONFIG, tmp_path / "frozen.json",
        tmp_path / "cache", "smoke", cpu_workers=5, ensemble_workers=2,
    )
    assert len(batches.concurrent) == 6
    assert len(batches.ensemble) == 2
    assert len(batches.compose) == 1
    gpu = batches.concurrent[0]
    assert option(gpu, "--device-group") == "gpu"
    assert "--num-shards" not in gpu.argv
    assert all(option(c, "--device-group") == "cpu" for c in batches.concurrent[1:])
    assert all(option(c, "--device-group") == "ensemble" for c in batches.ensemble)
    assert all("--num-shards" in c.argv for c in (*batches.concurrent[1:], *batches.ensemble))


@pytest.mark.parametrize(("cpu", "ensemble"), [(4, 2), (9, 2), (5, 1), (5, 5), (True, 2)])
def test_invalid_worker_counts(cpu, ensemble, tmp_path):
    with pytest.raises(ValueError):
        parallel.build_stage5_cache_commands(
            sys.executable, ROOT, CONFIG, tmp_path / "f.json",
            tmp_path / "cache", "smoke", cpu_workers=cpu,
            ensemble_workers=ensemble,
        )


def test_progress_write_is_atomic_and_strict_json(tmp_path):
    path = tmp_path / "logs/progress.json"
    parallel.write_stage5_progress(path, {"status": "running", "count": 1})
    assert json.loads(path.read_text(encoding="utf-8")) == {
        "status": "running", "count": 1,
    }
    assert not path.with_suffix(".json.tmp").exists()


def exact_preflight_summary(config_fingerprint="c" * 64, tuning_fingerprint="t" * 64):
    return {
        "stage": "stage5_preflight",
        "profile": "preflight",
        "seed_namespace": "stage5_preflight_v1",
        "config_fingerprint": config_fingerprint,
        "tuning_fingerprint": tuning_fingerprint,
        "expected_records": 900,
        "successful_records": 900,
        "failed": 0,
        "oom": 0,
        "fallback": 0,
        "missing": 0,
        "duplicate": 0,
        "provenance": 0,
        "non_finite": 0,
        "paired_seed_check": True,
        "center_dedup_check": True,
    }


def test_formal_gate_accepts_exact_summary():
    parallel.validate_stage5_gate(
        "formal", exact_preflight_summary(), True,
        config_fingerprint="c" * 64, tuning_fingerprint="t" * 64,
    )


@pytest.mark.parametrize("field", [
    "failed", "oom", "fallback", "missing", "duplicate", "provenance", "non_finite",
])
def test_formal_gate_rejects_each_violation(field):
    summary = exact_preflight_summary()
    summary[field] = 1
    with pytest.raises(ValueError):
        parallel.validate_stage5_gate(
            "formal", summary, True,
            config_fingerprint="c" * 64, tuning_fingerprint="t" * 64,
        )


@pytest.mark.parametrize("bad", ["no_approval", "no_summary", "bool_count", "stale", "nan"])
def test_formal_gate_fails_closed(bad):
    summary = exact_preflight_summary()
    approved = True
    if bad == "no_approval":
        approved = False
    elif bad == "no_summary":
        summary = None
    elif bad == "bool_count":
        summary["successful_records"] = True
    elif bad == "stale":
        summary["config_fingerprint"] = "x" * 64
    elif bad == "nan":
        summary["successful_records"] = float("nan")
    with pytest.raises(ValueError):
        parallel.validate_stage5_gate(
            "formal", summary, approved,
            config_fingerprint="c" * 64, tuning_fingerprint="t" * 64,
        )


def test_nonformal_rejects_formal_approval():
    with pytest.raises(ValueError):
        parallel.validate_stage5_gate("smoke", None, True)


def _controller_fixture(monkeypatch, tmp_path):
    methods = ["tabiclv2_1", "tabiclv2_8", "xgboost_tuned", "extra_trees", "lasso", "ensemble"]
    config = {"methods": methods}
    frozen = {"tuning_run_fingerprint": "f" * 64}
    monkeypatch.setattr(parallel, "load_stage5_config", lambda path: config)
    monkeypatch.setattr(parallel, "resolve_stage5_profile", lambda *args: None)
    monkeypatch.setattr(parallel, "load_stage5_tuning", lambda *args: frozen)
    monkeypatch.setattr(parallel, "stage5_config_fingerprint", lambda value: "c" * 64)
    monkeypatch.setattr(parallel, "_validate_destinations", lambda *args: None)
    monkeypatch.setattr(parallel, "_universe", lambda *args: ({str(i): object() for i in range(180)}, {}, {}))
    monkeypatch.setattr(parallel.shutil, "which", lambda value: str(Path(sys.executable).resolve()))
    script = str((ROOT / "scripts/run_stage5_cache.py").resolve())
    compose = str((ROOT / "scripts/compose_stage5_dml.py").resolve())
    batches = parallel.Stage5CommandBatches(
        (WorkerCommand("gpu", (sys.executable, script)), WorkerCommand("cpu", (sys.executable, script))),
        (WorkerCommand("ensemble", (sys.executable, script)),),
        (WorkerCommand("compose", (sys.executable, compose)),),
    )
    monkeypatch.setattr(parallel, "build_stage5_cache_commands", lambda *args, **kwargs: batches)
    kwargs = dict(
        python_executable=sys.executable, project_root=ROOT, profile="smoke",
        frozen_tuning=tmp_path / "frozen.json", cache_root=tmp_path / "cache",
        output_root=tmp_path / "raw", log_dir=tmp_path / "log",
    )
    return methods, batches, kwargs


def test_controller_runs_batches_in_required_order(monkeypatch, tmp_path):
    methods, batches, kwargs = _controller_fixture(monkeypatch, tmp_path)
    completed = 0
    calls = []

    def scan(*args):
        cache_done = 0 if completed == 0 else 10 if completed == 1 else 12
        results_done = 180 if completed == 3 else 0
        return {
            "expected_results": 180, "expected_caches": 12,
            "completed_results": results_done, "completed_caches": cache_done,
            **{field: 0 if field != "missing" else 192 - cache_done - results_done for field in parallel._VIOLATIONS},
            "by_method": {method: {
                "expected_caches": 2, "completed_caches": (2 if completed >= (2 if method == "ensemble" else 1) else 0)
            } for method in methods},
        }

    def run(commands, **unused):
        nonlocal completed
        calls.append(tuple(command.name for command in commands))
        completed += 1
        return {command.name: 0 for command in commands}

    monkeypatch.setattr(parallel, "_scan", scan)
    monkeypatch.setattr(parallel, "run_workers", run)
    assert parallel.run_stage5_parallel(**kwargs) == 0
    assert calls == [
        tuple(c.name for c in batches.concurrent),
        tuple(c.name for c in batches.ensemble),
        tuple(c.name for c in batches.compose),
    ]
    assert json.loads((tmp_path / "log/progress.json").read_text())["status"] == "completed"


def test_controller_stops_after_failed_concurrent_worker(monkeypatch, tmp_path):
    methods, _, kwargs = _controller_fixture(monkeypatch, tmp_path)
    calls = []
    monkeypatch.setattr(parallel, "_scan", lambda *args: {
        "expected_results": 180, "expected_caches": 12,
        "completed_results": 0, "completed_caches": 0,
        **{field: 0 if field != "missing" else 192 for field in parallel._VIOLATIONS},
        "by_method": {method: {"expected_caches": 2, "completed_caches": 0} for method in methods},
    })

    def run(commands, **unused):
        calls.append(commands)
        return {command.name: (7 if index == 0 else 0) for index, command in enumerate(commands)}

    monkeypatch.setattr(parallel, "run_workers", run)
    assert parallel.run_stage5_parallel(**kwargs) == 7
    assert len(calls) == 1
