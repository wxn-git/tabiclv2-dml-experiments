from pathlib import Path
import subprocess
import sys

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_tree_simple_config_is_isolated_and_keeps_publication_counts():
    config_path = PROJECT_ROOT / "configs" / "stage3b_tree_simple.yaml"
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    assert config["selected_configuration"] == {
        "scenario": "tree_simple",
        "n": 2000,
        "p": 10,
    }
    assert config["screening"]["stage"] == "stage3b_tree_simple_screening"
    assert (
        config["screening"]["seed_namespace"]
        == "stage3b_tree_simple_screening"
    )
    assert config["screening"]["replications"] == 10
    assert len(config["screening"]["candidates"]) == 17
    assert config["confirmation"]["stage"] == "stage3b_tree_simple_confirmation"
    assert (
        config["confirmation"]["seed_namespace"]
        == "stage3b_tree_simple_confirmation"
    )
    assert config["confirmation"]["replications"] == 50


def test_stage3b_aggregator_exposes_isolated_root_arguments():
    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / "aggregate_stage3b.py"), "--help"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    for option in (
        "--batch-a-root",
        "--screening-root",
        "--confirmation-root",
        "--output-root",
        "--title",
        "--baseline-confirmation-summary",
    ):
        assert option in result.stdout
