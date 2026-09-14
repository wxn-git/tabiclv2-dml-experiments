"""Validation-first publication of the fixed, full-profile Stage 4 experiment.

No learners are run. Analysis is regenerated in an isolated temporary directory
and compared byte-for-byte, including plots, before any destination is touched.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

import pandas as pd

from .stage4_analysis import build_stage4_analysis, _write_analysis_bundle
from .stage4_config import load_stage4_config
from .stage4_structure import (
    _AUDIT_FIELDS,
    _csv_rows,
    audit_tree_structures,
    structure_audit_failures,
)
from .stage4_tuning import iter_tuning_tasks, select_tuned_xgboost


SCHEMA = "stage4_publication_v1"
DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs/stage4_tree_benchmark.yaml"
ANALYSIS_FILES = (
    "screening_summary.csv", "screening_cell_ranking.csv", "confirmation_summary.csv",
    "primary_paired_comparisons.csv", "coverage_diagnostics.csv", "nuisance_diagnostics.csv",
    "analysis_report_zh.md", "figures/dml_rmse_by_panel.png",
    "figures/nuisance_mse_by_panel.png", "figures/coverage_by_panel.png",
)
COMPACT_FILES = {*ANALYSIS_FILES, "structure_checks.json", "structure_checks.csv",
                 "selected_xgboost.json", "selected_confirmation_cells.json",
                 "environment.json", "stage4_tree_benchmark.yaml"}
ENVIRONMENT_PACKAGES = {"numpy", "pandas", "scipy", "scikit-learn", "xgboost",
                        "torch", "tabicl", "doubleml"}


def _layout(results_root, *, config_path=None, structure_dir=None, tuned_models=None,
            selected_cells=None, analysis_dir=None, tuning_root=None,
            screening_root=None, confirmation_root=None):
    root = Path(results_root).resolve()
    def path(value, default):
        return Path(value if value is not None else default).resolve()
    analysis = path(analysis_dir, root / "stage4_tree_confirmation")
    structure = path(structure_dir, root / "stage4_tree_structure_checks")
    files = {name: analysis / name for name in ANALYSIS_FILES}
    files.update({
        "structure_checks.json": structure / "structure_checks.json",
        "structure_checks.csv": structure / "structure_checks.csv",
        "selected_xgboost.json": path(tuned_models, root / "stage4_tree_tuning/selected_xgboost.json"),
        "selected_confirmation_cells.json": path(selected_cells, root / "stage4_tree_screening/selected_confirmation_cells.json"),
        "environment.json": analysis / "environment.json",
        "stage4_tree_benchmark.yaml": path(config_path, DEFAULT_CONFIG),
    })
    raw = {phase: path(value, root / f"stage4_tree_{phase}_raw") for phase, value in (
        ("tuning", tuning_root), ("screening", screening_root), ("confirmation", confirmation_root))}
    return root, files, raw


def _sha(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _file_entry(path):
    if not path.is_file() or path.is_symlink() or path.stat().st_size == 0:
        raise ValueError(f"missing, empty or linked artifact: {path}")
    return {"sha256": _sha(path), "bytes": path.stat().st_size}


def _json(path):
    value = json.loads(path.read_text(encoding="utf-8"))
    def finite(item):
        if isinstance(item, float) and not math.isfinite(item):
            raise ValueError(f"nonfinite JSON value: {path}")
        if isinstance(item, dict):
            for child in item.values():
                finite(child)
        elif isinstance(item, list):
            for child in item:
                finite(child)
    finite(value)
    return value


def _snapshot(files, raw):
    artifacts = {name: _file_entry(path) for name, path in files.items()}
    inputs = {}
    for phase, root in raw.items():
        if not root.is_dir() or root.is_symlink():
            raise ValueError(f"missing or linked {phase} raw directory")
        entries = {p.name: _file_entry(p) for p in sorted(root.glob("*.json"))}
        if not entries:
            raise ValueError(f"missing {phase} raw records")
        payload = json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()
        inputs[phase] = {"records": len(entries), "sha256": hashlib.sha256(payload).hexdigest()}
    return {"files": artifacts, "raw_inputs": inputs}


def _structure(files):
    audit = _json(files["structure_checks.json"])
    failures = structure_audit_failures(audit)
    if failures:
        raise ValueError(f"structure audit did not pass: {'; '.join(failures)}")

    parameters = audit["parameters"]
    expected = audit_tree_structures(n=parameters["n"], seed=parameters["seed"])
    if json.dumps(audit, sort_keys=True, separators=(",", ":")) != json.dumps(
        expected, sort_keys=True, separators=(",", ":")
    ):
        raise ValueError("structure audit did not pass: diagnostics are not reproducible")

    with files["structure_checks.csv"].open(encoding="utf-8", newline="") as handle:
        csv_rows = list(csv.DictReader(handle))
    expected_csv = [
        {field: "" if row[field] == "" else str(row[field]) for field in _AUDIT_FIELDS}
        for row in _csv_rows(audit)
    ]
    if csv_rows != expected_csv:
        raise ValueError("structure audit did not pass: CSV disagrees with JSON")


def validate_stage4_publication(results_root, expected_replications=100, **paths) -> dict[str, Any]:
    """Return SHA-256 manifest only for complete, reproducible formal artifacts.

    Path overrides are explicit paths relative to cwd, not to results_root.
    Optional absent GPU telemetry remains allowed by the analysis schema; NaN or
    infinity in raw JSON, inferential metrics or other required fields is not.
    """
    if type(expected_replications) is not int or expected_replications != 100:
        raise ValueError("Stage 4 publication requires exactly 100 formal replications")
    try:
        _, files, raw = _layout(results_root, **paths)
        before = _snapshot(files, raw)
        config = load_stage4_config(files["stage4_tree_benchmark.yaml"])
        _structure(files)
        environment = _json(files["environment.json"])
        packages = environment.get("packages") if isinstance(environment, dict) else None
        if (not isinstance(environment, dict) or not environment.get("python")
                or not environment.get("platform") or not isinstance(packages, dict)
                or set(packages) != ENVIRONMENT_PACKAGES
                or any(not isinstance(value, str) or not value for value in packages.values())
                or environment.get("cuda") != packages["torch"]
                or (environment.get("gpu") is not None
                    and (not isinstance(environment["gpu"], str) or not environment["gpu"]))):
            raise ValueError("invalid environment report")
        tuning = _json(files["selected_xgboost.json"])
        selection = _json(files["selected_confirmation_cells.json"])
        records = {phase: [_json(p) for p in sorted(directory.glob("*.json"))]
                   for phase, directory in raw.items()}
        for phase, values in records.items():
            for record in values:
                if (not isinstance(record, dict) or record.get("execution_profile") != "full"
                        or record.get("status") != "success" or record.get("fallback_reason")):
                    raise ValueError(f"{phase} requires full-profile successful records without fallback")
        expected_tuning = select_tuned_xgboost(records["tuning"], 10,
                                              expected_tasks=tuple(iter_tuning_tasks(config, 10)))
        if tuning != expected_tuning:
            raise ValueError("stale frozen tuning: not reproduced by raw tuning records")
        analysis = build_stage4_analysis(records["screening"], records["confirmation"],
                                         config, tuning, selection, execution_profile="full")
        primary = analysis["primary_paired_comparisons"]
        if not primary["paired_count"].eq(100).all() or primary["inference_status"].eq("implementation_smoke").any():
            raise ValueError("primary comparisons require complete formal 100-rep inference")
        # Canonical writer gives exact schemas, NA rules, ordering and plots.
        # No destination directory is created during validation.
        with tempfile.TemporaryDirectory(prefix="stage4-verify-") as directory:
            regenerated = Path(directory)
            _write_analysis_bundle(analysis, regenerated)
            for name in ANALYSIS_FILES:
                if _sha(regenerated / name) != before["files"][name]["sha256"]:
                    raise ValueError(f"stale analysis artifact: {name}; regenerate with analyze_stage4.py")
        if _snapshot(files, raw) != before:
            raise ValueError("source artifacts changed during validation")
        return {"schema": SCHEMA, "execution_profile": "full", "counts": {
            "tuning_entries": 48, "tuning_records": len(records["tuning"]),
            "screening_cells": 24, "screening_records": len(records["screening"]),
            "confirmation_cells": 6, "confirmation_records": len(records["confirmation"]),
            "confirmation_replications": 100, "primary_comparisons": 6,
        }, **before}
    except (ValueError, TypeError, KeyError, OSError, AssertionError) as error:
        raise ValueError(f"Stage 4 publication is incomplete or invalid: {error}") from error


def _replaceable(destination):
    """Refuse arbitrary or modified historical directories even with --replace."""
    try:
        if not destination.is_dir() or destination.is_symlink():
            raise ValueError("linked or non-directory destination")
        manifest = _json(destination / "manifest.json")
        if manifest["schema"] != SCHEMA or manifest["execution_profile"] != "full":
            raise ValueError("wrong publication schema")
        if set(manifest["files"]) != COMPACT_FILES:
            raise ValueError("wrong publication file set")
        entries = list(destination.rglob("*"))
        if any(p.is_symlink() or (hasattr(p, "is_junction") and p.is_junction()) for p in entries):
            raise ValueError("linked publication content")
        directories = {p.relative_to(destination).as_posix() for p in entries if p.is_dir()}
        if directories != {"figures"}:
            raise ValueError("unexpected directories")
        actual = {p.relative_to(destination).as_posix() for p in entries if p.is_file()}
        if actual != set(manifest["files"]) | {"manifest.json"}:
            raise ValueError("unexpected files")
        for name, entry in manifest["files"].items():
            path = destination / name
            if not path.resolve().is_relative_to(destination) or _file_entry(path) != entry:
                raise ValueError("modified publication file")
    except (ValueError, KeyError, TypeError, OSError) as error:
        raise ValueError("--replace requires an intact Stage 4 publication destination") from error


def publish_stage4(results_root, destination, expected_replications=100, *, replace=False, **paths):
    root, files, raw = _layout(results_root, **paths)
    target = Path(destination)
    if target.is_symlink():
        raise ValueError("destination must not be a symlink")
    target = target.resolve()
    sources = [*files.values(), *raw.values()]
    if target == root or root.is_relative_to(target) or any(
        source == target or source.is_relative_to(target) or target.is_relative_to(source)
        for source in sources
    ):
        raise ValueError("publication destination and source paths overlap")
    if target.exists():
        if not replace:
            raise ValueError("destination exists; explicit --replace required")
        _replaceable(target)
    manifest = validate_stage4_publication(results_root, expected_replications, **paths)
    target.parent.mkdir(parents=True, exist_ok=True)
    lock = target.with_name(f".{target.name}.publish.lock")
    # Serialize cooperating publishers; a stale lock needs operator inspection.
    with lock.open("x"):
        pass
    staging = None
    backup = None
    try:
        staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.stage-", dir=target.parent))
        for name, source in files.items():
            output = staging / name
            output.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, output)
            if _file_entry(output) != manifest["files"][name]:
                raise ValueError("source artifact changed while copying")
        if _snapshot(files, raw) != {key: manifest[key] for key in ("files", "raw_inputs")}:
            raise ValueError("source artifacts changed after validation")
        with (staging / "manifest.json").open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        if target.exists():
            if not replace:
                raise ValueError("destination appeared; explicit --replace required")
            _replaceable(target)
            backup = staging.with_name(staging.name.replace(".stage-", ".backup-"))
            os.replace(target, backup)
        try:
            os.replace(staging, target)
        except Exception:
            if backup is not None:
                os.replace(backup, target)
            raise
        if backup is not None:
            shutil.rmtree(backup)
        return manifest
    finally:
        if staging is not None and staging.exists():
            shutil.rmtree(staging)
        # Never remove a backup if rollback itself failed.
        lock.unlink()
