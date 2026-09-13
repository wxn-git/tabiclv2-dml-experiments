import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts import run_stage5_tuning, select_stage5_tuning
from tabdml.stage5_config import load_stage5_config, stage5_config_fingerprint
from tabdml.stage5_tuning import derive_stage5_tuning_seeds, iter_stage5_tuning_tasks
from tabdml.storage import ResultStore


CONFIG = Path("configs/stage5_sensitivity.yaml")


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


def test_runner_resolves_paths_from_repo_root_and_uses_sharded_result_store(monkeypatch, tmp_path):
    config = load_stage5_config(CONFIG)
    tasks = tuple(iter_stage5_tuning_tasks(config, replications=1, num_shards=2, shard_index=1))[:2]
    seen = []
    monkeypatch.setattr(run_stage5_tuning, "__file__", str(tmp_path / "project" / "scripts" / "run_stage5_tuning.py"))
    monkeypatch.setattr(run_stage5_tuning, "load_stage5_config", lambda path: (seen.append(Path(path)), config)[1])
    monkeypatch.setattr(run_stage5_tuning, "iter_stage5_tuning_tasks", lambda *a, **k: iter(tasks))
    monkeypatch.setattr(run_stage5_tuning, "run_stage5_tuning_task", lambda task, output_root, retry_failed=False: (seen.append(Path(output_root)), {"status": "success"})[1])
    monkeypatch.setattr(sys, "argv", ["run_stage5_tuning.py", "--config", "configs/stage5_sensitivity.yaml", "--output-root", "results/raw", "--replications", "1", "--num-shards", "2", "--shard-index", "1"])
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
