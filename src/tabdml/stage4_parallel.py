"""Stage 4 process orchestration, with artifact-validated phase boundaries.

Tuning selection and screening selection/analysis remain explicit CLI steps.
Counts include each expected tuning JSON, nuisance NPZ, and composed JSON once;
worker exits are reported separately and never stand in for artifact success.
"""
from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .nuisance_cache import NuisanceCache
from .parallel import WorkerCommand, run_workers
from .stage4_config import load_stage4_config
from .stage4_experiment import (
    build_stage4_nuisance_spec,
    iter_stage4_pairs,
    validate_stage4_preflight,
    validate_stage4_cached_result,
    validate_stage4_resume_record,
)
from .stage4_tuning import _validate_record_metadata, iter_tuning_tasks


PathArg = str | os.PathLike[str]


def _validate_workers(cpu_workers: int, replications: int | None) -> None:
    if type(cpu_workers) is not int or not 1 <= cpu_workers <= 8:
        raise ValueError("cpu_workers must be an integer between 1 and 8")
    if replications is not None and (
        type(replications) is not int or replications < 1
    ):
        raise ValueError("replications must be a positive integer")


def _resolve(root: PathArg, value: PathArg) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else Path(root) / path).resolve()


def _flags(
    replications: int | None, fast: bool, retry_failed: bool, preflight: bool = False,
) -> tuple[str, ...]:
    if replications is None and fast:
        replications = 1
    return (
        *(("--replications", str(replications)) if replications is not None else ()),
        *(("--fast",) if fast else ()),
        *(("--preflight",) if preflight else ()),
        *(("--retry-failed",) if retry_failed else ()),
    )


def build_stage4_tuning_commands(
    python_executable: PathArg,
    project_root: PathArg,
    config_path: PathArg,
    output_root: PathArg,
    cpu_workers: int = 8,
    replications: int | None = None,
    *,
    fast: bool = False,
    retry_failed: bool = False,
    preflight: bool = False,
) -> tuple[WorkerCommand, ...]:
    if preflight:
        raise ValueError("preflight is only supported for full-model confirmation")
    _validate_workers(cpu_workers, replications)
    common = (
        str(python_executable),
        str(_resolve(project_root, "scripts/run_stage4_tuning.py")),
        "--config", str(_resolve(project_root, config_path)),
        "--output-root", str(_resolve(project_root, output_root)),
        *_flags(replications, fast, retry_failed),
        "--num-shards", str(cpu_workers),
    )
    return tuple(
        WorkerCommand(
            f"cpu_stage4_tuning_{index:02d}",
            (*common, "--shard-index", str(index)),
        )
        for index in range(cpu_workers)
    )


def build_stage4_cache_commands(
    python_executable: PathArg,
    project_root: PathArg,
    config_path: PathArg,
    tuned_models: PathArg,
    cache_root: PathArg,
    *,
    phase: str,
    cpu_workers: int = 8,
    replications: int | None = None,
    selected_cells: PathArg | None = None,
    fast: bool = False,
    retry_failed: bool = False,
    preflight: bool = False,
) -> tuple[WorkerCommand, ...]:
    _validate_workers(cpu_workers, replications)
    if preflight:
        replications = validate_stage4_preflight(
            load_stage4_config(_resolve(project_root, config_path)), phase,
            replications, fast=fast, preflight=True,
        )
    if phase not in {"screening", "confirmation"}:
        raise ValueError("cache phase must be screening or confirmation")
    if phase == "confirmation" and not selected_cells:
        raise ValueError("confirmation requires selected confirmation cells")
    common = (
        str(python_executable),
        str(_resolve(project_root, "scripts/run_stage4_cache.py")),
        "--config", str(_resolve(project_root, config_path)),
        "--phase", phase,
        "--tuned-models", str(_resolve(project_root, tuned_models)),
        "--cache-root", str(_resolve(project_root, cache_root)),
        *(("--selected-cells", str(_resolve(project_root, selected_cells)))
          if phase == "confirmation" else ()),
        *_flags(replications, fast, retry_failed, preflight),
    )
    # The child CLI restricts methods by device and deduplicates nuisance keys
    # BEFORE applying tabdml.sharding.belongs_to_shard. Never shard pair keys.
    return (
        WorkerCommand(f"gpu_stage4_{phase}", (*common, "--device-group", "gpu")),
        *(WorkerCommand(
            f"cpu_stage4_{phase}_{index:02d}",
            (*common, "--device-group", "cpu", "--num-shards", str(cpu_workers),
             "--shard-index", str(index)),
        ) for index in range(cpu_workers)),
    )


def _read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ValueError(f"Invalid JSON artifact: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _tuning_status(path, task):
    record = _read_json(path)
    _validate_record_metadata(record, task)
    status = record.get("status")
    if status not in {"success", "failed", "oom"}:
        raise ValueError(f"Invalid tuning status: {path}")
    if status == "success":
        for metric in ("validation_observed_mse", "validation_truth_mse_diagnostic"):
            value = record.get(metric)
            if type(value) not in {int, float} or not math.isfinite(value) or value < 0:
                raise ValueError(f"Invalid tuning metric {metric}: {path}")
    return status


def _cache_status(path, task, pair, target):
    result = NuisanceCache(path.parent).read(task, expected_length=pair.n)
    validate_stage4_cached_result(pair, result, target)
    return "success"


@dataclass(frozen=True)
class _Artifact:
    path: Path
    validate: Callable[[Path], str]
    repairable: bool = False


@dataclass(frozen=True)
class _Stage:
    name: str
    commands: tuple[WorkerCommand, ...]
    artifacts: tuple[_Artifact, ...]


def _check_destination(path: Path, *, directory: bool) -> None:
    if path.exists() and path.is_dir() != directory:
        raise ValueError(f"Invalid {'directory' if directory else 'file'} path: {path}")
    if path.exists() and not os.access(path, os.W_OK):
        raise ValueError(f"Output path is not writable: {path}")
    ancestor = path if directory else path.parent
    while not ancestor.exists():
        ancestor = ancestor.parent
    if not ancestor.is_dir() or not os.access(ancestor, os.W_OK):
        raise ValueError(f"Output ancestor is not a writable directory: {ancestor}")


def _validate_paths(root, commands, inputs, directories, artifacts):
    if not root.is_dir():
        raise ValueError(f"Invalid project root: {root}")
    for command in commands:
        if not Path(command.argv[1]).is_file():
            raise FileNotFoundError(f"Missing child script: {command.argv[1]}")
    for path in inputs:
        if not path.is_file():
            raise FileNotFoundError(f"Missing input file: {path}")
    for index, directory in enumerate(directories):
        # Reject historical destinations even when explicitly supplied. Resolve
        # first so traversal and symlink aliases cannot bypass this guard.
        if any(
            re.search(r"stage[123]|tree[-_]simple", part, re.I)
            for part in directory.parts
        ):
            raise ValueError(f"Historical artifact destination is forbidden: {directory}")
        _check_destination(directory, directory=True)
        for other in (*inputs, *directories[index + 1:]):
            if (
                other == directory
                or directory in other.parents
                or other in directory.parents
            ):
                raise ValueError(f"Overlapping input/output paths: {directory}, {other}")
    for artifact in artifacts:
        _check_destination(artifact.path, directory=False)
        _check_destination(
            artifact.path.with_suffix(artifact.path.suffix + ".tmp"),
            directory=False,
        )


def _scan(stages, attempted, *, strict=False, retry_failed=False):
    counts = {}
    for stage in stages:
        successful = failed = 0
        errors = []
        for artifact in stage.artifacts:
            if not artifact.path.exists():
                failed += stage.name in attempted
                continue
            try:
                status = artifact.validate(artifact.path)
            except (ValueError, OSError, TypeError, KeyError, OverflowError) as error:
                if strict and not (retry_failed and artifact.repairable):
                    raise ValueError(f"Invalid existing artifact {artifact.path}: {error}") from error
                status = "failed"
                if len(errors) < 20:
                    errors.append(str(error))
            successful += status == "success"
            failed += status != "success"
        planned = len(stage.artifacts)
        counts[stage.name] = {
            "planned_tasks": planned, "successful_tasks": successful,
            "failed_tasks": failed, "pending_tasks": planned - successful - failed,
            "artifact_errors": errors,
        }
    return counts


def _write_progress(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(".json.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def run_stage4_phase(
    python_executable: PathArg,
    project_root: PathArg,
    config_path: PathArg = "configs/stage4_tree_benchmark.yaml",
    *,
    phase: str,
    tuned_models: PathArg = "results/stage4_tree_tuning/selected_xgboost.json",
    selected_cells: PathArg | None = None,
    cache_root: PathArg = "results/stage4_tree_cache",
    output_root: PathArg | None = None,
    log_dir: PathArg | None = None,
    cpu_workers: int = 8,
    replications: int | None = None,
    fast: bool = False,
    retry_failed: bool = False,
    dry_run: bool = False,
    preflight: bool = False,
) -> int:
    """Validate before launching; run cache then composition, or tuning alone.

    Fast defaults to ONE implementation-smoke replication. Five-rep full-model
    preflight requires ``phase='confirmation', preflight=True``; the
    default full confirmation uses the configured 100 replications.
    """
    if phase not in {"tuning", "screening", "confirmation"}:
        raise ValueError("phase must be tuning, screening or confirmation")
    _validate_workers(cpu_workers, replications)
    root = Path(project_root).resolve()
    config_path = _resolve(root, config_path)
    config = load_stage4_config(config_path)
    replications = validate_stage4_preflight(
        config, phase, replications, fast=fast, preflight=preflight,
    )
    replications = replications if replications is not None else (
        1 if fast else int(config[phase]["replications"])
    )
    executable_path = Path(python_executable)
    executable = shutil.which(str(
        _resolve(root, executable_path)
        if executable_path.parent != Path(".") else executable_path
    ))
    if executable is None:
        raise FileNotFoundError(f"Missing Python executable: {python_executable}")
    output = _resolve(root, output_root or f"results/stage4_tree_{phase}_raw")
    logs = _resolve(root, log_dir or f"results/logs/stage4_tree/{phase}")
    inputs = [config_path]
    directories = [output, logs]
    common = dict(
        cpu_workers=cpu_workers, replications=replications,
        fast=fast, retry_failed=retry_failed, preflight=preflight,
    )
    if phase == "tuning":
        tasks = {task.key: task for task in iter_tuning_tasks(config, replications, fast=fast)}
        commands = build_stage4_tuning_commands(executable, root, config_path, output, **common)
        stages = (_Stage("tuning", commands, tuple(
            _Artifact(output / f"{task.key}.json", lambda path, task=task: _tuning_status(path, task))
            for task in tasks.values()
        )),)
    else:
        tuned_path = _resolve(root, tuned_models)
        frozen = _read_json(tuned_path)
        inputs.append(tuned_path)
        selected_path = None
        selection = None
        if phase == "confirmation":
            if not selected_cells:
                raise ValueError("confirmation requires selected confirmation cells")
            selected_path = _resolve(root, selected_cells)
            selection = _read_json(selected_path)
            inputs.append(selected_path)
        pairs = tuple(iter_stage4_pairs(
            config, phase, frozen, selected_confirmation=selection,
            replications=replications, fast=fast,
            preflight=preflight,
        ))
        cache = _resolve(root, cache_root)
        directories.append(cache)
        requests = {}
        for pair in pairs:
            for target in ("l", "m"):
                task = build_stage4_nuisance_spec(pair, target)
                requests.setdefault(
                    task.key,
                    _Artifact(
                        cache / f"{task.key}.npz",
                        lambda path, task=task, pair=pair, target=target:
                            _cache_status(path, task, pair, target),
                        repairable=True,
                    ),
                )
        commands = build_stage4_cache_commands(
            executable, root, config_path, tuned_path, cache,
            phase=phase, selected_cells=selected_path, **common,
        )
        compose = WorkerCommand(f"compose_stage4_{phase}", (
            executable, str(root / "scripts/compose_stage4_dml.py"),
            "--config", str(config_path), "--phase", phase,
            "--tuned-models", str(tuned_path), "--cache-root", str(cache),
            "--output-root", str(output),
            *(("--selected-cells", str(selected_path)) if selected_path else ()),
            *_flags(replications, fast, retry_failed, preflight),
        ))
        stages = (
            _Stage("cache", commands, tuple(requests.values())),
            _Stage("compose", (compose,), tuple(
                _Artifact(output / f"{pair.key}.json", lambda path, pair=pair:
                          validate_stage4_resume_record(_read_json(path), pair))
                for pair in {pair.key: pair for pair in pairs}.values()
            )),
        )
    commands = tuple(command for stage in stages for command in stage.commands)
    artifacts = tuple(artifact for stage in stages for artifact in stage.artifacts)
    _validate_paths(root, commands, inputs, directories, artifacts)
    # Separate state logs retain both cache and composition histories.
    for stage in stages:
        stage_logs = logs / stage.name
        _check_destination(stage_logs, directory=True)
        for name in ("state.json", "state.json.tmp", *(f"{c.name}.{stream}.log"
                     for c in stage.commands for stream in ("stdout", "stderr"))):
            _check_destination(stage_logs / name, directory=False)
    progress_path = logs / "progress.json"
    for path in (progress_path, progress_path.with_suffix(".json.tmp")):
        _check_destination(path, directory=False)
    _scan(stages, set(), strict=True, retry_failed=retry_failed)
    if dry_run:
        for stage in stages:
            print(f"[{phase}/{stage.name}]")
            for command in stage.commands:
                print(subprocess.list2cmdline(command.argv))
        return 0

    logs.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now(timezone.utc).isoformat()
    attempted = set()
    exits = {}

    def update(status, child_stage, error=None):
        counts = _scan(stages, attempted)
        payload = {
            "phase": phase, "child_stage": child_stage, "status": status,
            "execution_profile": "fast" if fast else "full", "replications": replications,
            "preflight": preflight,
            "started_at": started_at, "updated_at": datetime.now(timezone.utc).isoformat(),
            "stages": counts, "worker_exit_codes": dict(exits), "error": error,
            **{field: sum(count[field] for count in counts.values()) for field in (
                "planned_tasks", "successful_tasks", "failed_tasks", "pending_tasks")},
        }
        _write_progress(progress_path, payload)
        return counts

    for index, stage in enumerate(stages):
        update("running", stage.name)
        attempted.add(stage.name)
        try:
            codes = run_workers(stage.commands, cwd=root, log_dir=logs / stage.name)
        except Exception as error:
            update("failed", stage.name, f"{type(error).__name__}: {error}")
            return 1
        except BaseException as error:
            update("interrupted", stage.name, type(error).__name__)
            raise
        exits.update(codes)
        worker_ok = set(codes) == {c.name for c in stage.commands} and all(
            type(code) is int and code == 0 for code in codes.values()
        )
        counts = _scan(stages, attempted)
        artifacts_ok = all(
            counts[s.name]["successful_tasks"] == len(s.artifacts)
            for s in stages[:index + 1]
        )
        if not worker_ok or not artifacts_ok:
            update("failed", stage.name, "Workers or required artifacts did not all succeed")
            return next(
                (code for code in codes.values() if type(code) is int and code != 0),
                1,
            )
        update("completed" if index == len(stages) - 1 else "running", stage.name)
    return 0
