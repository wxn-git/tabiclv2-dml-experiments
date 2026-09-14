import json
import inspect
import subprocess
import sys
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import (
    analyze_stage5, compose_stage5_dml, run_stage5_cache, run_stage5_parallel,
    run_stage5_tuning, select_stage5_tuning,
)
from tabdml.stage5_config import load_stage5_config, stage5_config_fingerprint
from tabdml.stage5_tuning import derive_stage5_tuning_seeds, iter_stage5_tuning_tasks
from tabdml.storage import ResultStore
from tabdml.nuisance_cache import NuisanceCache
from tabdml.stage5_experiment import (
    Stage5NuisanceResult,
    build_stage5_nuisance_spec,
    compose_stage5_record,
    iter_stage5_pairs,
    resolve_stage5_method,
    stage5_nuisance_metadata_path,
)


CONFIG = Path("configs/stage5_sensitivity.yaml")
FIVE_METHOD_CONFIG = "configs/stage5_sensitivity_five.yaml"


@pytest.mark.parametrize(
    ("module", "arguments"),
    [
        (run_stage5_parallel, ["--profile", "smoke", "--frozen-tuning", "f", "--cache-root", "c", "--output-root", "o", "--log-dir", "l"]),
        (analyze_stage5, ["--profile", "smoke", "--input", "i", "--output-root", "o"]),
        (run_stage5_cache, ["--device-group", "cpu"]),
        (compose_stage5_dml, ["--profile", "smoke", "--frozen-tuning", "f", "--cache-root", "c", "--output", "o"]),
        (run_stage5_tuning, []),
        (select_stage5_tuning, []),
    ],
)
def test_stage5_clis_default_to_five_method_protocol(monkeypatch, module, arguments):
    monkeypatch.setattr(sys, "argv", [module.__name__, *arguments])
    assert module.parse_args().config == FIVE_METHOD_CONFIG


def test_stage5_controller_api_defaults_to_five_method_protocol():
    parameters = inspect.signature(run_stage5_parallel.run_stage5_parallel).parameters
    assert parameters["config_path"].default == FIVE_METHOD_CONFIG
    assert parameters["frozen_tuning"].default == "results/stage5_five/tuning/frozen-fast.json"
    assert parameters["cache_root"].default == "results/stage5_five/smoke/cache"
    assert parameters["output_root"].default == "results/stage5_five/smoke/raw"
    assert parameters["log_dir"].default == "results/stage5_five/smoke/log"


def test_stage5_worker_defaults_keep_five_method_artifacts_separate(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["run_stage5_cache.py", "--device-group", "cpu"])
    cache = run_stage5_cache.parse_args()
    assert cache.frozen_tuning == "results/stage5_five/tuning/frozen-fast.json"
    assert cache.cache_root == "results/stage5_five/smoke/cache"

    monkeypatch.setattr(sys, "argv", ["run_stage5_tuning.py"])
    assert run_stage5_tuning.parse_args().output_root == "results/stage5_five/tuning/raw"

    monkeypatch.setattr(sys, "argv", ["select_stage5_tuning.py"])
    assert select_stage5_tuning.parse_args().output == (
        "results/stage5_five/tuning/selected_xgboost.json"
    )


def _record(task):
    return {
        "task_key": task.key, "status": "success", "stage": task.stage,
        "seed_namespace": task.seed_namespace, "scenario": task.scenario,
        "n": task.n, "p": task.p, "replication": task.replication,
        "target": task.target, "candidate": task.candidate, "theta0": task.theta0,
        "learner_kind": "xgboost", "execution_profile": task.execution_profile,
        "config_fingerprint": task.config_fingerprint, "nominal_params": task.params,
        "nominal_config_hash": task.nominal_config_hash, "params": task.effective_params,
        "config_hash": task.config_hash, "validation_fraction": task.validation_fraction,
        **derive_stage5_tuning_seeds(task),
        "selection_metric": "validation_y_mse" if task.target == "l" else "validation_d_mse",
        "validation_observed_mse": 1.0, "validation_truth_mse_diagnostic": 2.0,
        "runtime_seconds": 0.01,
    }


def test_cli_modules_import_from_foreign_current_directory(tmp_path):
    root = Path(__file__).resolve().parents[1]
    code = "import runpy; runpy.run_path(r'%s', run_name='stage5_import_test')" % (root / "scripts" / "run_stage5_tuning.py")
    result = subprocess.run([sys.executable, "-c", code], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("script", ["run_stage5_cache.py", "compose_stage5_dml.py"])
def test_experiment_cli_modules_import_from_foreign_current_directory(tmp_path, script):
    root = Path(__file__).resolve().parents[1]
    code = "import runpy; runpy.run_path(r'%s', run_name='stage5_import_test')" % (root / "scripts" / script)
    result = subprocess.run([sys.executable, "-c", code], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_cache_cli_rejects_sharded_gpu_before_loading_files(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["run_stage5_cache.py", "--device-group", "gpu", "--num-shards", "2", "--shard-index", "0"])
    with pytest.raises(ValueError, match="unsharded"):
        run_stage5_cache.main()


def test_composition_cli_returns_nonzero_for_incomplete_cache(monkeypatch, tmp_path):
    root = Path(__file__).resolve().parents[1]
    config = load_stage5_config(CONFIG)
    from tests.test_stage5_experiment import _frozen

    frozen_path = tmp_path / "frozen.json"
    frozen_path.write_text(json.dumps(_frozen(config)), encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compose_stage5_dml.py", "--config", str(CONFIG.resolve()), "--profile", "smoke",
            "--frozen-tuning", str(frozen_path), "--cache-root", str(tmp_path / "empty"),
            "--output", str(tmp_path / "out"),
        ],
    )
    assert compose_stage5_dml.main() != 0


def _compose_cli_fixture(monkeypatch, tmp_path, retry_failed=False):
    from tests.test_stage5_experiment import _frozen

    config = load_stage5_config(CONFIG)
    frozen = _frozen(config)
    pair = next(
        p for p in iter_stage5_pairs(config, frozen, "smoke") if p.method == "lasso"
    )
    cache_root = tmp_path / "cache"
    output_root = tmp_path / "output"
    nuisance = Stage5NuisanceResult(
        np.zeros(pair.n), (0.0,) * pair.folds_count, None, None,
        0.5, 0.75, "cpu", "cpu",
    )
    for target in ("l", "m"):
        resolved = resolve_stage5_method(pair, target, config, frozen)
        task = build_stage5_nuisance_spec(pair, target, resolved)
        cache = NuisanceCache(cache_root)
        cache.path(task).touch()
        stage5_nuisance_metadata_path(cache, task).touch()
    monkeypatch.setattr(compose_stage5_dml, "load_stage5_config", lambda path: config)
    monkeypatch.setattr(compose_stage5_dml, "load_stage5_tuning", lambda *a: frozen)
    monkeypatch.setattr(compose_stage5_dml, "iter_stage5_pairs", lambda *a: iter((pair,)))
    monkeypatch.setattr(compose_stage5_dml, "read_stage5_nuisance", lambda *a, **k: nuisance)
    monkeypatch.setattr(
        compose_stage5_dml,
        "parse_args",
        lambda: SimpleNamespace(
            config=str(CONFIG), profile="smoke", frozen_tuning="frozen.json",
            cache_root=str(cache_root), output=str(output_root),
            retry_failed=retry_failed,
        ),
    )
    return pair, nuisance, output_root


@pytest.mark.parametrize("status", ["failed", "oom"])
def test_composition_resume_reports_matching_failure_unless_retrying(
    monkeypatch, tmp_path, status
):
    pair, nuisance, output_root = _compose_cli_fixture(monkeypatch, tmp_path)
    monkeypatch.setattr(
        "tabdml.stage5_experiment.simulate_plr",
        lambda *a, **k: SimpleNamespace(
            y=np.linspace(0.0, 2.0, pair.n), d=np.linspace(-1.0, 1.0, pair.n),
            l0=np.zeros(pair.n), m0=np.zeros(pair.n),
        ),
    )
    record = compose_stage5_record(pair, nuisance, nuisance)
    record["status"] = status
    ResultStore(output_root).write(record)
    assert compose_stage5_dml.main() == 1
    assert json.loads((output_root / f"{pair.key}.json").read_text())["status"] == status

    monkeypatch.setattr(compose_stage5_dml, "parse_args", lambda: SimpleNamespace(
        config=str(CONFIG), profile="smoke", frozen_tuning="frozen.json",
        cache_root=str(tmp_path / "cache"), output=str(output_root), retry_failed=True,
    ))
    assert compose_stage5_dml.main() == 0
    assert json.loads((output_root / f"{pair.key}.json").read_text())["status"] == "success"


def test_composition_retry_rejects_corrupt_record_and_reports_diagnostics(
    monkeypatch, tmp_path, capsys
):
    pair, nuisance, output_root = _compose_cli_fixture(monkeypatch, tmp_path)
    output_root.mkdir(parents=True)
    path = output_root / f"{pair.key}.json"
    path.write_text("not json", encoding="utf-8")
    assert compose_stage5_dml.main() == 1
    assert "Invalid existing Stage 5 record" in capsys.readouterr().err

    monkeypatch.setattr(compose_stage5_dml, "parse_args", lambda: SimpleNamespace(
        config=str(CONFIG), profile="smoke", frozen_tuning="frozen.json",
        cache_root=str(tmp_path / "cache"), output=str(output_root), retry_failed=True,
    ))
    assert compose_stage5_dml.main() == 1


def test_composition_unknown_status_fails_even_with_retry(monkeypatch, tmp_path):
    pair, _, output_root = _compose_cli_fixture(monkeypatch, tmp_path, retry_failed=True)
    output_root.mkdir(parents=True)
    (output_root / f"{pair.key}.json").write_text(
        json.dumps({"task_key": pair.key, "status": "mystery"}), encoding="utf-8"
    )
    assert compose_stage5_dml.main() == 1


@pytest.mark.parametrize("status", ["failed", "oom"])
def test_cache_cli_reports_matching_failure_unless_retrying(
    monkeypatch, tmp_path, status
):
    from tests.test_stage5_experiment import _frozen

    config = load_stage5_config(CONFIG)
    frozen = _frozen(config)
    pair = next(
        p for p in iter_stage5_pairs(config, frozen, "smoke") if p.method == "lasso"
    )
    resolved = resolve_stage5_method(pair, "l", config, frozen)
    task = build_stage5_nuisance_spec(pair, "l", resolved)
    cache_root = tmp_path / "cache"
    failure = {
        "task_key": task.key, "status": status, "error_type": "RuntimeError",
        "error_message": "boom", "traceback": "trace", "pair_key": pair.key,
        "profile": pair.profile, "config_fingerprint": pair.config_fingerprint,
        "tuning_fingerprint": pair.tuning_fingerprint, "task": asdict(task),
    }
    ResultStore(cache_root.parent / "failures").write(failure)
    calls = []
    monkeypatch.setattr(run_stage5_cache, "load_stage5_config", lambda path: config)
    monkeypatch.setattr(run_stage5_cache, "load_stage5_tuning", lambda *a: frozen)
    monkeypatch.setattr(run_stage5_cache, "resolve_stage5_profile", lambda *a: SimpleNamespace(full_settings=False))
    monkeypatch.setattr(run_stage5_cache, "iter_stage5_pairs", lambda *a: iter((pair,)))
    monkeypatch.setattr(run_stage5_cache, "methods_for_device", lambda group: (pair.method,))
    monkeypatch.setattr(run_stage5_cache, "resolve_stage5_method", lambda *a: resolved)
    monkeypatch.setattr(run_stage5_cache, "build_stage5_nuisance_spec", lambda *a: task)
    monkeypatch.setattr(run_stage5_cache, "fit_stage5_nuisance", lambda *a, **k: (
        calls.append(k), Stage5NuisanceResult(np.zeros(pair.n), (0.0,) * 5, None, None, 0.0, 0.0, "cpu", "cpu")
    )[1])
    args = dict(config=str(CONFIG), profile="smoke", frozen_tuning="frozen.json",
                cache_root=str(cache_root), device_group="cpu", num_shards=1,
                shard_index=0, retry_failed=False)
    monkeypatch.setattr(run_stage5_cache, "parse_args", lambda: SimpleNamespace(**args))
    assert run_stage5_cache.main() == 1
    assert calls == []

    args["retry_failed"] = True
    assert run_stage5_cache.main() == 0
    assert len(calls) == 1
    assert not (cache_root.parent / "failures" / f"{task.key}.json").exists()


def test_cache_cli_rejects_unknown_failure_status_and_key_collisions(monkeypatch, tmp_path):
    from tests.test_stage5_experiment import _frozen

    config = load_stage5_config(CONFIG)
    frozen = _frozen(config)
    pair = next(iter_stage5_pairs(config, frozen, "smoke"))
    other = replace(pair, method=config["methods"][1])
    resolved = resolve_stage5_method(pair, "l", config, frozen)
    task = build_stage5_nuisance_spec(pair, "l", resolved)
    cache_root = tmp_path / "cache"
    ResultStore(cache_root.parent / "failures").write({
        "task_key": task.key, "status": "mystery"
    })
    monkeypatch.setattr(run_stage5_cache, "load_stage5_config", lambda path: config)
    monkeypatch.setattr(run_stage5_cache, "load_stage5_tuning", lambda *a: frozen)
    monkeypatch.setattr(run_stage5_cache, "resolve_stage5_profile", lambda *a: SimpleNamespace(full_settings=False))
    monkeypatch.setattr(run_stage5_cache, "methods_for_device", lambda group: (pair.method, other.method))
    monkeypatch.setattr(run_stage5_cache, "iter_stage5_pairs", lambda *a: iter((pair,)))
    monkeypatch.setattr(run_stage5_cache, "resolve_stage5_method", lambda *a: resolved)
    monkeypatch.setattr(run_stage5_cache, "build_stage5_nuisance_spec", lambda *a: task)
    monkeypatch.setattr(run_stage5_cache, "parse_args", lambda: SimpleNamespace(
        config=str(CONFIG), profile="smoke", frozen_tuning="frozen.json",
        cache_root=str(cache_root), device_group="cpu", num_shards=1,
        shard_index=0, retry_failed=True,
    ))
    with pytest.raises(ValueError, match="status"):
        run_stage5_cache.main()

    (cache_root.parent / "failures" / f"{task.key}.json").unlink()
    monkeypatch.setattr(run_stage5_cache, "iter_stage5_pairs", lambda *a: iter((pair, other)))
    with pytest.raises(ValueError, match="collision"):
        run_stage5_cache.main()


def test_runner_resolves_paths_from_repo_root_and_uses_sharded_result_store(monkeypatch, tmp_path):
    config = load_stage5_config(CONFIG)
    tasks = tuple(iter_stage5_tuning_tasks(config, num_shards=2, shard_index=1))[:2]
    seen = []
    monkeypatch.setattr(run_stage5_tuning, "__file__", str(tmp_path / "project" / "scripts" / "run_stage5_tuning.py"))
    monkeypatch.setattr(run_stage5_tuning, "load_stage5_config", lambda path: (seen.append(Path(path)), config)[1])
    monkeypatch.setattr(run_stage5_tuning, "iter_stage5_tuning_tasks", lambda *a, **k: iter(tasks))
    monkeypatch.setattr(run_stage5_tuning, "run_stage5_tuning_task", lambda task, output_root, retry_failed=False: (seen.append(Path(output_root)), {"status": "success"})[1])
    monkeypatch.setattr(sys, "argv", ["run_stage5_tuning.py", "--config", "configs/stage5_sensitivity.yaml", "--output-root", "results/raw", "--num-shards", "2", "--shard-index", "1"])
    assert run_stage5_tuning.main() == 0
    assert seen[0] == tmp_path / "project" / "configs" / "stage5_sensitivity.yaml"
    assert seen[1:] == [tmp_path / "project" / "results" / "raw"] * 2


def test_selector_merges_repeatable_roots_and_glob(monkeypatch, tmp_path):
    config = load_stage5_config(CONFIG)
    tasks = tuple(iter_stage5_tuning_tasks(config, execution_profile="fast"))
    roots = [tmp_path / "shard0", tmp_path / "shard1"]
    extra = tmp_path / "globbed" / "records"
    for index, task in enumerate(tasks):
        root = roots[0] if index < 24 else roots[1] if index < 48 else extra
        ResultStore(root).write(_record(task))
    output = tmp_path / "selected.json"
    monkeypatch.setattr(sys, "argv", ["select_stage5_tuning.py", "--config", str(CONFIG.resolve()), "--input-root", str(roots[0]), "--input-root", str(roots[1]), "--input-glob", str(tmp_path / "globbed" / "*"), "--output", str(output), "--execution-profile", "fast"])
    assert select_stage5_tuning.main() == 0
    assert json.loads(output.read_text())["config_fingerprint"] == stage5_config_fingerprint(config)


def test_selector_cli_rejects_duplicate_keys_across_roots(monkeypatch, tmp_path):
    config = load_stage5_config(CONFIG)
    task = next(iter_stage5_tuning_tasks(config, execution_profile="fast"))
    for name in ("one", "two"):
        ResultStore(tmp_path / name).write(_record(task))
    monkeypatch.setattr(sys, "argv", ["select_stage5_tuning.py", "--config", str(CONFIG.resolve()), "--input-root", str(tmp_path / "one"), "--input-root", str(tmp_path / "two"), "--output", str(tmp_path / "out.json"), "--execution-profile", "fast"])
    with pytest.raises(ValueError, match="duplicate"):
        select_stage5_tuning.main()


@pytest.mark.parametrize("replications", [1, 9, 11])
def test_runner_cli_cannot_weaken_full_replication_contract(
    monkeypatch, tmp_path, replications
):
    def forbidden(*args, **kwargs):
        pytest.fail("invalid replication count must fail before task execution")

    monkeypatch.setattr(run_stage5_tuning, "run_stage5_tuning_task", forbidden)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_stage5_tuning.py",
            "--config",
            str(CONFIG.resolve()),
            "--output-root",
            str(tmp_path / "raw"),
            "--replications",
            str(replications),
        ],
    )
    with pytest.raises(ValueError, match="full.*exactly 10"):
        run_stage5_tuning.main()
    assert not (tmp_path / "raw").exists()
