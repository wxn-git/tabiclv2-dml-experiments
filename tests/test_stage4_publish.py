import hashlib
import json
import sys

import pytest

from tabdml.stage4_publish import publish_stage4, validate_stage4_publication
from tabdml.stage4_analysis import write_stage4_analysis
from tabdml.stage4_config import load_stage4_config
from tabdml.stage4_selection import select_confirmation_cells
from tabdml.stage4_structure import audit_tree_structures, write_structure_audit
from tabdml.stage4_tuning import derive_tuning_seeds, iter_tuning_tasks, select_tuned_xgboost
from test_stage4_analysis import CONFIG_PATH, _phase_records


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture(scope="module")
def formal(tmp_path_factory):
    """Synthetic full-universe records; no models/workers, NOT experiment evidence."""
    root = tmp_path_factory.mktemp("pub")
    config = load_stage4_config(CONFIG_PATH)
    tasks = tuple(iter_tuning_tasks(config, 10))
    tuning = []
    for task in tasks:
        record = {name: getattr(task, name) for name in (
            "stage", "seed_namespace", "panel", "scenario", "n", "p", "replication",
            "target", "candidate", "theta0", "execution_profile", "validation_fraction",
        )}
        record.update(task_key=task.key, status="success", learner_kind="xgboost",
                      nominal_params=task.params, nominal_config_hash=task.nominal_config_hash,
                      params=task.effective_params, config_hash=task.config_hash,
                      validation_observed_mse=1.0, validation_truth_mse_diagnostic=1.0,
                      **derive_tuning_seeds(task))
        tuning.append(record)
    frozen = select_tuned_xgboost(tuning, 10, expected_tasks=tasks)
    screening = _phase_records(config, frozen, "screening", execution_profile="full")
    selected = select_confirmation_cells(screening, config, frozen)
    confirmation = _phase_records(config, frozen, "confirmation", selected, execution_profile="full")
    for phase, records in (("tuning", tuning), ("screening", screening), ("confirmation", confirmation)):
        for index, record in enumerate(records):
            write_json(root / f"stage4_tree_{phase}_raw/{index:05}.json", record)
    write_json(root / "stage4_tree_tuning/selected_xgboost.json", frozen)
    write_json(root / "stage4_tree_screening/selected_confirmation_cells.json", selected)
    write_structure_audit(audit_tree_structures(n=1000), root / "stage4_tree_structure_checks")
    analysis = root / "stage4_tree_confirmation"
    write_stage4_analysis(screening, confirmation, config, frozen, selected, analysis)
    packages = {name: "test" for name in ("numpy", "pandas", "scipy", "scikit-learn",
        "xgboost", "torch", "tabicl", "doubleml")}
    write_json(analysis / "environment.json", {"python": "synthetic-test", "platform": "test",
        "packages": packages, "cuda": packages["torch"], "gpu": "Test GPU, 1 MiB, driver"})
    return root


@pytest.fixture
def edit():
    originals = {}
    def change(path, transform=None):
        originals.setdefault(path, path.read_bytes())
        if transform is None:
            path.unlink()
        else:
            path.write_bytes(transform(path.read_bytes()))
    yield change
    for path, data in originals.items():
        path.write_bytes(data)


def change_json(edit, path, **fields):
    edit(path, lambda data: json.dumps(dict(json.loads(data), **fields)).encode())


def test_publisher_rejects_incomplete_results(tmp_path):
    with pytest.raises(ValueError, match="Stage 4 publication is incomplete"):
        validate_stage4_publication(tmp_path, expected_replications=100)


def test_failed_validation_does_not_create_destination(tmp_path):
    destination = tmp_path / "published"
    with pytest.raises(ValueError, match="Stage 4 publication is incomplete"):
        publish_stage4(tmp_path / "source", destination, expected_replications=100)
    assert not destination.exists()


@pytest.mark.parametrize("replications", [1, 5, 99, 101, True, 100.0])
def test_cannot_relax_formal_replication_gate(tmp_path, replications):
    with pytest.raises(ValueError, match="100"):
        validate_stage4_publication(tmp_path, expected_replications=replications)


def test_full_bundle_manifest_and_compact_copy(formal, tmp_path):
    manifest = validate_stage4_publication(formal)
    assert manifest["counts"] == {"tuning_entries": 48, "tuning_records": 2880,
        "screening_cells": 24, "screening_records": 4800, "confirmation_cells": 6,
        "confirmation_records": 6000, "confirmation_replications": 100,
        "primary_comparisons": 6}
    destination = tmp_path / "published"
    publish_stage4(formal, destination)
    saved = json.loads((destination / "manifest.json").read_text())
    assert saved == manifest
    assert len(saved["files"]) == 16  # 10 analysis + structure pair + tuning + cells + env + config
    assert "screening_cell_ranking.csv" in saved["files"]
    for name, entry in saved["files"].items():
        assert hashlib.sha256((destination / name).read_bytes()).hexdigest() == entry["sha256"]
    assert len(list(destination.rglob("*.*"))) == 17


@pytest.mark.parametrize("name", ["screening_summary.csv", "screening_cell_ranking.csv",
    "confirmation_summary.csv", "primary_paired_comparisons.csv", "coverage_diagnostics.csv",
    "nuisance_diagnostics.csv", "analysis_report_zh.md", "figures/dml_rmse_by_panel.png",
    "figures/nuisance_mse_by_panel.png", "figures/coverage_by_panel.png", "environment.json"])
def test_each_required_analysis_artifact_is_mandatory(formal, edit, tmp_path, name):
    edit(formal / "stage4_tree_confirmation" / name)
    destination = tmp_path / "publication"
    with pytest.raises(ValueError, match="Stage 4 publication is incomplete"):
        publish_stage4(formal, destination)
    assert not destination.exists()


@pytest.mark.parametrize("field,value", [("execution_profile", "fast"),
    ("stage", "stage4_tree_confirmation_preflight"), ("theta", float("nan")),
    ("data_seed", 0), ("learner_l_config_hash", "stale"), ("status", "failed"),
    ("fallback_reason", "OOM"), ("replication", 100)])
def test_rejects_invalid_raw_identity_not_just_counts(formal, edit, field, value):
    change_json(edit, formal / "stage4_tree_confirmation_raw/00000.json", **{field: value})
    with pytest.raises(ValueError):
        validate_stage4_publication(formal)


@pytest.mark.parametrize("phase", ["tuning", "screening", "confirmation"])
def test_missing_and_duplicate_raw_records_rejected(formal, edit, phase):
    first = formal / f"stage4_tree_{phase}_raw/00000.json"
    second = first.with_name("00001.json")
    edit(second, lambda _: first.read_bytes())
    with pytest.raises(ValueError):
        validate_stage4_publication(formal)
    edit(second)
    with pytest.raises(ValueError):
        validate_stage4_publication(formal)


@pytest.mark.parametrize("artifact", ["confirmation_summary.csv", "screening_cell_ranking.csv",
    "primary_paired_comparisons.csv", "analysis_report_zh.md", "figures/coverage_by_panel.png"])
def test_rejects_stale_analysis_including_report_and_figures(formal, edit, artifact):
    edit(formal / "stage4_tree_confirmation" / artifact, lambda data: data + b"stale")
    with pytest.raises(ValueError, match="stale"):
        validate_stage4_publication(formal)


def test_tuning_winners_recomputed_from_raw(formal, edit):
    path = formal / "stage4_tree_tuning/selected_xgboost.json"
    def wrong_metric(data):
        value = json.loads(data)
        next(iter(value["cells"].values()))["l"]["mean_validation_observed_mse"] = 99
        return json.dumps(value).encode()
    edit(path, wrong_metric)
    with pytest.raises(ValueError, match="tuning"):
        validate_stage4_publication(formal)


def test_existing_destination_requires_explicit_replace(formal, tmp_path):
    destination = tmp_path / "publication"
    destination.mkdir()
    (destination / "historical.txt").write_text("untouched")
    with pytest.raises(ValueError, match="replace"):
        publish_stage4(formal, destination)
    with pytest.raises(ValueError, match="Stage 4"):
        publish_stage4(formal, destination, replace=True)
    assert (destination / "historical.txt").read_text() == "untouched"


def test_atomic_replace_rolls_back_on_install_failure(formal, tmp_path, monkeypatch):
    import tabdml.stage4_publish as publisher
    destination = tmp_path / "publication"
    publish_stage4(formal, destination)
    before = (destination / "manifest.json").read_bytes()
    real_replace = publisher.os.replace
    def fail_install(source, target):
        if ".stage-" in str(source) and target == destination:
            raise OSError("injected install failure")
        return real_replace(source, target)
    monkeypatch.setattr(publisher.os, "replace", fail_install)
    with pytest.raises(OSError, match="injected"):
        publish_stage4(formal, destination, replace=True)
    assert (destination / "manifest.json").read_bytes() == before
    assert list(tmp_path.iterdir()) == [destination]


def test_explicit_replace_updates_valid_publication(formal, tmp_path):
    destination = tmp_path / "publication"
    publish_stage4(formal, destination)
    publish_stage4(formal, destination, replace=True)
    assert (destination / "manifest.json").is_file()


def test_source_destination_overlap_rejected(formal):
    with pytest.raises(ValueError, match="overlap"):
        publish_stage4(formal, formal, replace=True)


def test_cli_rejects_incomplete_without_writing(tmp_path, monkeypatch):
    from scripts import publish_stage4 as cli
    destination = tmp_path / "pub"
    monkeypatch.setattr(sys, "argv", ["publish_stage4.py", "--results-root", str(tmp_path),
        "--destination", str(destination)])
    with pytest.raises(ValueError):
        cli.main()
    assert not destination.exists()


@pytest.mark.parametrize("relative", ["stage4_tree_structure_checks/structure_checks.json",
    "stage4_tree_structure_checks/structure_checks.csv", "stage4_tree_tuning/selected_xgboost.json",
    "stage4_tree_screening/selected_confirmation_cells.json"])
def test_other_required_artifacts(formal, edit, relative):
    edit(formal / relative)
    with pytest.raises(ValueError, match="Stage 4 publication is incomplete"):
        validate_stage4_publication(formal)


def test_rejects_partial_same_count_screening_and_stale_selection(formal, edit):
    change_json(edit, formal / "stage4_tree_screening_raw/00000.json", data_seed=0)
    with pytest.raises(ValueError):
        validate_stage4_publication(formal)


def test_rejects_stale_selection_binding(formal, edit):
    change_json(edit, formal / "stage4_tree_screening/selected_confirmation_cells.json",
                config_fingerprint="stale")
    with pytest.raises(ValueError):
        validate_stage4_publication(formal)


def test_different_config_rejected(formal, tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(CONFIG_PATH.read_text().replace("theta0: 1.0", "theta0: 2.0"))
    with pytest.raises(ValueError):
        validate_stage4_publication(formal, config_path=path)


def test_renamed_smoke_comparisons_rejected(formal, edit):
    edit(formal / "stage4_tree_confirmation/primary_paired_comparisons.csv",
         lambda data: data.replace(b"paired_inference", b"implementation_smoke"))
    # Status text varies only if the writer changes; enforce a real mutation.
    path = formal / "stage4_tree_confirmation/primary_paired_comparisons.csv"
    import pandas as pd
    frame = pd.read_csv(path)
    frame["inference_status"] = "implementation_smoke"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="stale"):
        validate_stage4_publication(formal)


@pytest.mark.parametrize("malformation", ["missing-package", "nonfinite"])
def test_invalid_environment_rejected(formal, edit, malformation):
    change_json(edit, formal / "stage4_tree_confirmation/environment.json",
                **({"packages": {"numpy": "test"}} if malformation == "missing-package" else {"gpu": float("inf")}))
    with pytest.raises(ValueError):
        validate_stage4_publication(formal)


def test_rejects_forged_replace_manifest_even_when_directory_empty(tmp_path):
    destination = tmp_path / "pub"
    write_json(destination / "manifest.json", {"schema": "stage4_publication_v1",
        "execution_profile": "full", "files": {}})
    with pytest.raises(ValueError, match="intact Stage 4 publication"):
        publish_stage4(tmp_path / "missing-source", destination, replace=True)


def test_copy_failure_preserves_destination(formal, tmp_path, monkeypatch):
    import tabdml.stage4_publish as publisher
    destination = tmp_path / "pub"
    publish_stage4(formal, destination)
    before = (destination / "manifest.json").read_bytes()
    def fail_copy(*args, **kwargs):
        raise OSError("injected copy failure")
    monkeypatch.setattr(publisher.shutil, "copyfile", fail_copy)
    with pytest.raises(OSError, match="injected copy"):
        publish_stage4(formal, destination, replace=True)
    assert (destination / "manifest.json").read_bytes() == before
    assert list(tmp_path.iterdir()) == [destination]


def test_source_mutation_after_validation_prevents_publication(formal, edit, tmp_path, monkeypatch):
    import tabdml.stage4_publish as publisher
    original = publisher.validate_stage4_publication
    def validate_then_mutate(*args, **kwargs):
        result = original(*args, **kwargs)
        change_json(edit, formal / "stage4_tree_confirmation_raw/00000.json", theta=2.0)
        return result
    monkeypatch.setattr(publisher, "validate_stage4_publication", validate_then_mutate)
    destination = tmp_path / "pub"
    with pytest.raises(ValueError, match="changed"):
        publish_stage4(formal, destination)
    assert list(tmp_path.iterdir()) == []


def test_cli_explicit_short_layout(formal, tmp_path, monkeypatch):
    from scripts import publish_stage4 as cli
    # Rename only our synthetic temporary fixture; never access a real run root.
    renames = {"stage4_tree_tuning_raw": "tr", "stage4_tree_screening_raw": "sr",
        "stage4_tree_confirmation_raw": "cr", "stage4_tree_confirmation": "an",
        "stage4_tree_structure_checks": "st"}
    destination = tmp_path / "pub"
    try:
        for old, new in renames.items():
            (formal / old).rename(formal / new)
        monkeypatch.setattr(sys, "argv", ["publish_stage4.py", "--results-root", str(formal),
            "--config", str(CONFIG_PATH), "--destination", str(destination),
            "--structure-dir", str(formal / "st"), "--analysis-dir", str(formal / "an"),
            "--tuned-models", str(formal / "stage4_tree_tuning/selected_xgboost.json"),
            "--selected-cells", str(formal / "stage4_tree_screening/selected_confirmation_cells.json"),
            "--tuning-root", str(formal / "tr"), "--screening-root", str(formal / "sr"),
            "--confirmation-root", str(formal / "cr"), "--expected-replications", "100"])
        assert cli.main() == 0
        assert (destination / "screening_cell_ranking.csv").is_file()
    finally:
        for old, new in renames.items():
            if (formal / new).exists():
                (formal / new).rename(formal / old)


def test_structure_must_pass_same_threshold_as_audit_cli(formal, edit):
    import pandas as pd
    json_path = formal / "stage4_tree_structure_checks/structure_checks.json"
    rows = json.loads(json_path.read_bytes())
    rows[0]["split_gain"] = 0.0001  # Positive but CLI explicitly fails <= 1e-3.
    edit(json_path, lambda _: json.dumps(rows).encode())
    csv_path = formal / "stage4_tree_structure_checks/structure_checks.csv"
    edit(csv_path, lambda _: pd.DataFrame(rows).to_csv(index=False).encode())
    with pytest.raises(ValueError, match="structure audit did not pass"):
        validate_stage4_publication(formal)


def test_replace_rejects_unexpected_empty_directory(formal, tmp_path):
    destination = tmp_path / "pub"
    publish_stage4(formal, destination)
    (destination / "unrelated").mkdir()
    with pytest.raises(ValueError, match="intact Stage 4 publication"):
        publish_stage4(formal, destination, replace=True)
    assert (destination / "unrelated").is_dir()
