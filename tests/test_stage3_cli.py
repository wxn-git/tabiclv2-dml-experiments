import os
from pathlib import Path
import subprocess
import sys

import yaml

from tabdml.stage3 import iter_stage3_tasks


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_stage3_diagnosis_config_builds_45_unique_smoke_tasks():
    with (PROJECT_ROOT / "configs" / "stage3_tree_diagnosis.yaml").open(
        encoding="utf-8"
    ) as handle:
        config = yaml.safe_load(handle)

    tasks = list(
        iter_stage3_tasks(
            config,
            replications=5,
            pair_names=None,
            num_shards=1,
            shard_index=0,
        )
    )

    assert len(tasks) == 45
    assert len({task.key for task in tasks}) == 45
    assert len({(task.learner_l, task.learner_m) for task in tasks}) == 9


def _run_stage3(config: Path, output_root: Path, shard_index: int):
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(PROJECT_ROOT / "src")
    return subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "run_stage3.py"),
            "--config",
            str(config),
            "--output-root",
            str(output_root),
            "--fast",
            "--num-shards",
            "2",
            "--shard-index",
            str(shard_index),
        ],
        cwd=PROJECT_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )


def test_stage3_cli_partitions_replications_between_shards(tmp_path):
    config = tmp_path / "stage3_test.yaml"
    config.write_text(
        """\
stage: stage3_test
seed_namespace: stage3_test
selected_configurations:
  - {scenario: linear, n: 60, p: 5}
learner_pairs:
  - {name: oracle_oracle, learner_l: oracle, learner_m: oracle}
folds: 2
replications: 2
theta0: 1.0
tabicl_estimators: 1
""",
        encoding="utf-8",
    )
    output_root = tmp_path / "results"

    first = _run_stage3(config, output_root, 0)
    second = _run_stage3(config, output_root, 1)

    assert first.returncode == 0, first.stderr
    assert second.returncode == 0, second.stderr
    paths = sorted(output_root.glob("stage3_test__*.json"))
    assert len(paths) == 2
    assert "r000" in first.stdout and "r001" not in first.stdout
    assert "r001" in second.stdout and "r000" not in second.stdout
