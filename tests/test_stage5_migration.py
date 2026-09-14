from __future__ import annotations

import json
from pathlib import Path

import pytest

from tabdml.stage3b_screen import _params_hash
from tabdml.stage5_config import load_stage5_config, stage5_config_fingerprint
from tabdml.stage5_migration import (
    _assert_migration_paths,
    _assert_disjoint_paths,
    _snapshot_tree,
    rebind_stage5_tuning,
)
from tabdml.stage5_tuning import load_stage5_tuning, stage5_tuning_run_fingerprint


def _write_frozen_tuning(path: Path, config: dict) -> dict:
    candidates = config["tuning"]["xgboost_candidates"]
    rankings = {}
    winners = {}
    for scenario in config["scenarios"]:
        rankings[scenario] = {}
        winners[scenario] = {}
        for target in ("l", "m"):
            values = []
            for rank, candidate in enumerate(candidates):
                params = dict(candidate["params"])
                values.append({
                    "candidate": candidate["name"],
                    "learner_kind": "xgboost",
                    "execution_profile": "full",
                    "nominal_params": params,
                    "nominal_config_hash": _params_hash(params),
                    "effective_params": params,
                    "effective_config_hash": _params_hash(params),
                    "replications": config["tuning"]["replications"],
                    "mean_validation_observed_mse": float(rank + 1),
                    "mean_validation_truth_mse_diagnostic": float(rank + 2),
                    "selection_metric": (
                        "mean_validation_y_mse" if target == "l"
                        else "mean_validation_d_mse"
                    ),
                })
            rankings[scenario][target] = values
            winners[scenario][target] = dict(values[0])
    frozen = {
        "schema_version": "stage5_tuning_v1",
        "config_fingerprint": stage5_config_fingerprint(config),
        "tuning_stage": config["tuning"]["stage"],
        "tuning_seed_namespace": config["tuning"]["seed_namespace"],
        "tuning_run_fingerprint": stage5_tuning_run_fingerprint(
            config, config["tuning"]["replications"], "full"
        ),
        "theta0": config["theta0"],
        "execution_profile": "full",
        "expected_replications": config["tuning"]["replications"],
        "selection_metric_names": {
            "l": "mean_validation_y_mse",
            "m": "mean_validation_d_mse",
        },
        "candidate_ranking": rankings,
        "scenarios": winners,
    }
    path.write_text(json.dumps(frozen), encoding="utf-8")
    return frozen


def test_rebind_tuning_changes_only_config_derived_provenance(tmp_path):
    old = load_stage5_config(Path("configs/stage5_sensitivity.yaml"))
    new = load_stage5_config(Path("configs/stage5_sensitivity_five.yaml"))
    source = tmp_path / "legacy-frozen-full.json"
    original = _write_frozen_tuning(source, old)
    destination = tmp_path / "frozen-full.json"
    rebound = rebind_stage5_tuning(old, new, source, destination, "full")
    validated = load_stage5_tuning(destination, new, "full")
    assert rebound == validated
    assert validated["config_fingerprint"] == stage5_config_fingerprint(new)
    assert validated["scenarios"] == original["scenarios"]
    assert validated["candidate_ranking"] == original["candidate_ranking"]


def test_cache_migration_paths_must_be_disjoint(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    with pytest.raises(ValueError, match="disjoint"):
        _assert_disjoint_paths(source, source)
    with pytest.raises(ValueError, match="disjoint"):
        _assert_disjoint_paths(source, source / "nested")
    with pytest.raises(ValueError, match="disjoint"):
        _assert_disjoint_paths(source / "nested", source)
    _assert_disjoint_paths(source, tmp_path / "destination")


def test_source_tree_snapshot_detects_file_mutation(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    path = source / "cache.npz"
    path.write_bytes(b"before")
    before = _snapshot_tree(source)
    assert before["file_count"] == 1
    path.write_bytes(b"after")
    after = _snapshot_tree(source)
    assert before["tree_sha256"] != after["tree_sha256"]


@pytest.mark.parametrize("output_name", ["destination-tuning.json", "manifest.json"])
def test_migration_rejects_every_output_inside_source_cache(tmp_path, output_name):
    source_cache = tmp_path / "source-cache"
    source_cache.mkdir()
    source_tuning = tmp_path / "source-tuning.json"
    destination_cache = tmp_path / "destination-cache"
    destination_tuning = tmp_path / "destination-tuning.json"
    manifest = tmp_path / "manifest.json"
    if output_name == "destination-tuning.json":
        destination_tuning = source_cache / output_name
    else:
        manifest = source_cache / output_name
    with pytest.raises(ValueError, match="disjoint"):
        _assert_migration_paths(
            source_tuning, source_cache, destination_tuning,
            destination_cache, manifest,
        )
