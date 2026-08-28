import os
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _run_stage2(config: Path, output_root: Path, shard_index: int):
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(PROJECT_ROOT / "src")
    return subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "run_stage2.py"),
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


def test_stage2_cli_partitions_replications_between_shards(tmp_path):
    config = tmp_path / "stage2_test.yaml"
    config.write_text(
        """\
stage: stage2
selected_configurations:
  - {scenario: linear, n: 60, p: 5}
learners: [lasso]
folds: 2
replications: 2
theta0: 1.0
""",
        encoding="utf-8",
    )
    output_root = tmp_path / "results"

    first = _run_stage2(config, output_root, 0)
    second = _run_stage2(config, output_root, 1)

    assert first.returncode == 0, first.stderr
    assert second.returncode == 0, second.stderr
    assert sorted(path.stem for path in output_root.glob("stage2__*.json")) == [
        "stage2__linear__n60__p5__r000__lasso__e0",
        "stage2__linear__n60__p5__r001__lasso__e0",
    ]
    assert "r000" in first.stdout and "r001" not in first.stdout
    assert "r001" in second.stdout and "r000" not in second.stdout
