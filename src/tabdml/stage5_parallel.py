"""Fail-closed, resumable orchestration for the Stage 5 sensitivity study."""
from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .nuisance_cache import NuisanceCache
from .parallel import WorkerCommand, run_workers
from .stage5_config import load_stage5_config, resolve_stage5_profile, stage5_config_fingerprint
from .stage5_experiment import (
    build_stage5_nuisance_spec,
    iter_stage5_pairs,
    read_stage5_nuisance,
    resolve_stage5_method,
    stage5_nuisance_metadata_path,
    validate_stage5_resume_record,
)
from .stage5_tuning import load_stage5_tuning


PathArg = str | os.PathLike[str]
_EXPECTED = {"smoke": 180, "preflight": 900, "formal": 18000}
_VIOLATIONS = ("failed", "oom", "fallback", "missing", "duplicate", "provenance", "non_finite")


@dataclass(frozen=True)
class Stage5CommandBatches:
    concurrent: tuple[WorkerCommand, ...]
    ensemble: tuple[WorkerCommand, ...]
    compose: tuple[WorkerCommand, ...]


def _native_range(value: int, low: int, high: int, name: str) -> None:
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{name} must be a native integer between {low} and {high}")


def _resolve(root: PathArg, value: PathArg) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else Path(root) / path).resolve()


def build_stage5_cache_commands(
    python_executable: PathArg,
    project_root: PathArg,
    config_path: PathArg,
    frozen_tuning: PathArg,
    cache_root: PathArg,
    profile: str,
    cpu_workers: int = 5,
    ensemble_workers: int = 2,
    retry_failed: bool = False,
    *,
    output_root: PathArg | None = None,
) -> Stage5CommandBatches:
    _native_range(cpu_workers, 5, 8, "cpu_workers")
    _native_range(ensemble_workers, 2, 4, "ensemble_workers")
    if profile not in _EXPECTED:
        raise ValueError("profile must be smoke, preflight, or formal")
    root = Path(project_root).resolve()
    python = _resolve(root, python_executable) if Path(python_executable).parent != Path(".") else Path(python_executable)
    config = _resolve(root, config_path)
    tuning = _resolve(root, frozen_tuning)
    cache = _resolve(root, cache_root)
    output = _resolve(root, output_root or cache.parent / "raw")
    script = _resolve(root, "scripts/run_stage5_cache.py")
    active_methods = tuple(load_stage5_config(config)["methods"])
    common = (
        str(python), str(script), "--config", str(config), "--profile", profile,
        "--frozen-tuning", str(tuning), "--cache-root", str(cache),
        *(("--retry-failed",) if retry_failed else ()),
    )
    concurrent = (
        WorkerCommand("gpu_stage5", (*common, "--device-group", "gpu")),
        *(WorkerCommand(
            f"cpu_stage5_{index:02d}",
            (*common, "--device-group", "cpu", "--num-shards", str(cpu_workers),
             "--shard-index", str(index)),
        ) for index in range(cpu_workers)),
    )
    ensemble = tuple(WorkerCommand(
        f"ensemble_stage5_{index:02d}",
        (*common, "--device-group", "ensemble", "--num-shards", str(ensemble_workers),
         "--shard-index", str(index)),
    ) for index in range(ensemble_workers)) if "ensemble" in active_methods else ()
    compose_argv = (
        str(python), str(_resolve(root, "scripts/compose_stage5_dml.py")),
        "--config", str(config), "--profile", profile,
        "--frozen-tuning", str(tuning), "--cache-root", str(cache),
        "--output", str(output), *(("--retry-failed",) if retry_failed else ()),
    )
    return Stage5CommandBatches(concurrent, ensemble, (WorkerCommand("compose_stage5", compose_argv),))


def write_stage5_progress(path: PathArg, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(dict(payload), handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def validate_stage5_gate(
    profile: str,
    preflight_summary: Mapping[str, Any] | None = None,
    formal_approved: bool = False,
    *,
    config_fingerprint: str | None = None,
    tuning_fingerprint: str | None = None,
    expected_preflight_records: int = 900,
) -> None:
    if profile not in _EXPECTED:
        raise ValueError("profile must be smoke, preflight, or formal")
    if type(formal_approved) is not bool:
        raise ValueError("formal_approved must be boolean")
    if profile != "formal":
        if formal_approved:
            raise ValueError("formal approval is invalid outside the formal profile")
        return
    if type(expected_preflight_records) is not int or expected_preflight_records < 1:
        raise ValueError("expected_preflight_records must be a positive native integer")
    if not formal_approved or not isinstance(preflight_summary, Mapping):
        raise ValueError("formal profile requires explicit approval and a preflight summary")
    expected = {
        "stage": "stage5_preflight", "profile": "preflight",
        "seed_namespace": "stage5_preflight_v1",
        "config_fingerprint": config_fingerprint,
        "tuning_fingerprint": tuning_fingerprint,
        "expected_records": expected_preflight_records,
        "successful_records": expected_preflight_records,
        **{field: 0 for field in _VIOLATIONS},
        "paired_seed_check": True, "center_dedup_check": True,
    }
    if set(preflight_summary) != set(expected):
        raise ValueError("preflight summary schema mismatch")
    for field, wanted in expected.items():
        actual = preflight_summary.get(field)
        if type(actual) is not type(wanted) or actual != wanted:
            raise ValueError(f"preflight summary {field} mismatch")
        if isinstance(actual, float) and not math.isfinite(actual):
            raise ValueError(f"preflight summary {field} must be finite")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as error:
        raise ValueError(f"Invalid JSON artifact: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _contains_non_finite(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, float):
        return not math.isfinite(value)
    if isinstance(value, Mapping):
        return any(_contains_non_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_non_finite(item) for item in value)
    return False


def _validate_destinations(root: Path, inputs: tuple[Path, ...], outputs: tuple[Path, ...]) -> None:
    if not root.is_dir():
        raise ValueError(f"Invalid project root: {root}")
    for path in inputs:
        if not path.is_file():
            raise FileNotFoundError(f"Missing input file: {path}")
    for path in outputs:
        if any(re.fullmatch(r"stage[1-4].*", part, re.I) for part in path.parts):
            raise ValueError(f"Historical Stage 1-4 destination is forbidden: {path}")
        ancestor = path
        while not ancestor.exists():
            ancestor = ancestor.parent
        if not ancestor.is_dir() or not os.access(ancestor, os.W_OK):
            raise ValueError(f"Output ancestor is not writable: {ancestor}")
    for index, path in enumerate(outputs):
        for other in (*inputs, *outputs[index + 1:]):
            if path == other or path in other.parents or other in path.parents:
                raise ValueError(f"Overlapping Stage 5 paths: {path}, {other}")


def _universe(config, frozen, profile):
    pairs_by_key = {}
    tasks_by_key = {}
    task_methods = defaultdict(set)
    for pair in iter_stage5_pairs(config, frozen, profile):
        if pair.key in pairs_by_key:
            raise ValueError(f"Duplicate Stage 5 pair key: {pair.key}")
        pairs_by_key[pair.key] = pair
        for target in ("l", "m"):
            resolved = resolve_stage5_method(pair, target, config, frozen)
            task = build_stage5_nuisance_spec(pair, target, resolved)
            prior = tasks_by_key.get(task.key)
            value = (task, resolved, pair)
            if prior is not None and prior[:2] != value[:2]:
                raise ValueError(f"Stage 5 nuisance key collision: {task.key}")
            tasks_by_key.setdefault(task.key, value)
            task_methods[task.key].add(pair.method)
    return pairs_by_key, tasks_by_key, task_methods


def _scan(config, frozen, profile, cache_root: Path, output_root: Path):
    pairs, tasks, task_methods = _universe(config, frozen, profile)
    totals = Counter({field: 0 for field in _VIOLATIONS})
    by_method = {
        method: Counter({field: 0 for field in (*_VIOLATIONS, "expected_results", "completed_results", "expected_caches", "completed_caches")})
        for method in config["methods"]
    }
    completed_cache = 0
    for key, (task, resolved, _) in tasks.items():
        consumers = task_methods[key]
        for method in consumers:
            by_method[method]["expected_caches"] += 1
        cache_path = cache_root / f"{key}.npz"
        metadata_path = stage5_nuisance_metadata_path(NuisanceCache(cache_root), task)
        if not cache_path.exists() or not metadata_path.exists():
            totals["missing"] += 1
            for method in consumers:
                by_method[method]["missing"] += 1
            continue
        try:
            result = read_stage5_nuisance(task, resolved, cache_root, profile != "smoke")
            if result.fallback_reason:
                raise RuntimeError("fallback")
            completed_cache += 1
            for method in consumers:
                by_method[method]["completed_caches"] += 1
        except FileNotFoundError:
            totals["missing"] += 1
            for method in consumers:
                by_method[method]["missing"] += 1
        except RuntimeError:
            totals["fallback"] += 1
            for method in consumers:
                by_method[method]["fallback"] += 1
        except (ValueError, OSError, KeyError, TypeError, OverflowError):
            totals["provenance"] += 1
            for method in consumers:
                by_method[method]["provenance"] += 1
    failure_root = cache_root.parent / "failures"
    for path in failure_root.glob("*.json"):
        try:
            failure = _read_json(path)
            if path.stem not in tasks:
                raise ValueError("unknown failure task")
            task, _, pair = tasks[path.stem]
            status = failure.get("status")
            if status not in {"failed", "oom"}:
                raise ValueError("invalid failure status")
            expected = {
                "task_key": task.key, "pair_key": pair.key,
                "profile": pair.profile,
                "config_fingerprint": pair.config_fingerprint,
                "tuning_fingerprint": pair.tuning_fingerprint,
                "task": asdict(task),
            }
            if set(failure) != {*expected, "status", "error_type", "error_message", "traceback"}:
                raise ValueError("failure schema mismatch")
            if any(type(failure.get(field)) is not type(value) or failure.get(field) != value for field, value in expected.items()):
                raise ValueError("failure provenance mismatch")
            if any(not isinstance(failure.get(field), str) for field in ("error_type", "error_message", "traceback")):
                raise ValueError("failure diagnostics mismatch")
            totals[status] += 1
            for method in task_methods[task.key]:
                by_method[method][status] += 1
        except (ValueError, OSError, KeyError, TypeError):
            totals["provenance"] += 1
    result_status = Counter()
    seen = set()
    for key, pair in pairs.items():
        by_method[pair.method]["expected_results"] += 1
        path = output_root / f"{key}.json"
        if not path.exists():
            result_status["missing"] += 1
            by_method[pair.method]["missing"] += 1
            continue
        try:
            record = _read_json(path)
            if _contains_non_finite(record):
                result_status["non_finite"] += 1
                by_method[pair.method]["non_finite"] += 1
                continue
            status = validate_stage5_resume_record(record, pair)
            if status == "success":
                seen.add(key)
                by_method[pair.method]["completed_results"] += 1
            else:
                result_status[status] += 1
                by_method[pair.method][status] += 1
        except (ValueError, OSError, KeyError, TypeError, OverflowError):
            result_status["provenance"] += 1
            by_method[pair.method]["provenance"] += 1
    extra_results = [p for p in output_root.glob("*.json") if p.stem not in pairs]
    result_status["duplicate"] += len(extra_results)
    totals.update(result_status)
    return {
        "expected_results": len(pairs), "expected_caches": len(tasks),
        "completed_results": len(seen), "completed_caches": completed_cache,
        **{field: totals[field] for field in _VIOLATIONS},
        "by_method": {method: dict(counts) for method, counts in by_method.items()},
    }


def run_stage5_parallel(
    python_executable: PathArg,
    project_root: PathArg,
    config_path: PathArg = "configs/stage5_sensitivity_five.yaml",
    *,
    profile: str = "smoke",
    frozen_tuning: PathArg = "results/stage5_five/tuning/frozen-fast.json",
    cache_root: PathArg = "results/stage5_five/smoke/cache",
    output_root: PathArg = "results/stage5_five/smoke/raw",
    log_dir: PathArg = "results/stage5_five/smoke/log",
    cpu_workers: int = 5,
    ensemble_workers: int = 2,
    retry_failed: bool = False,
    dry_run: bool = False,
    formal_approved: bool = False,
    preflight_summary: PathArg | Mapping[str, Any] | None = None,
) -> int:
    _native_range(cpu_workers, 5, 8, "cpu_workers")
    _native_range(ensemble_workers, 2, 4, "ensemble_workers")
    root = Path(project_root).resolve()
    config_path = _resolve(root, config_path)
    tuning_path = _resolve(root, frozen_tuning)
    cache = _resolve(root, cache_root)
    output = _resolve(root, output_root)
    logs = _resolve(root, log_dir)
    config = load_stage5_config(config_path)
    resolve_stage5_profile(config, profile)
    tuning_profile = "fast" if profile == "smoke" else "full"
    frozen = load_stage5_tuning(tuning_path, config, tuning_profile)
    summary = preflight_summary
    if summary is not None and not isinstance(summary, Mapping):
        summary = _read_json(_resolve(root, summary))
    validate_stage5_gate(
        profile, summary, formal_approved,
        config_fingerprint=stage5_config_fingerprint(config),
        tuning_fingerprint=frozen["tuning_run_fingerprint"],
        expected_preflight_records=30 * 5 * len(config["methods"]),
    )
    executable_path = Path(python_executable)
    candidate = _resolve(root, executable_path) if executable_path.parent != Path(".") else executable_path
    executable = shutil.which(str(candidate))
    if executable is None:
        raise FileNotFoundError(f"Missing Python executable: {python_executable}")
    batches = build_stage5_cache_commands(
        executable, root, config_path, tuning_path, cache, profile,
        cpu_workers, ensemble_workers, retry_failed, output_root=output,
    )
    all_commands = (*batches.concurrent, *batches.ensemble, *batches.compose)
    for command in all_commands:
        if not Path(command.argv[1]).is_file():
            raise FileNotFoundError(f"Missing child script: {command.argv[1]}")
    _validate_destinations(root, (config_path, tuning_path), (cache, output, logs))
    pairs, tasks, _ = _universe(config, frozen, profile)
    expected_results = 30 * {"smoke": 1, "preflight": 5, "formal": 100}[profile] * len(config["methods"])
    if len(pairs) != expected_results:
        raise ValueError("Stage 5 expected result universe mismatch")
    if dry_run:
        for name, commands in (("concurrent_cache", batches.concurrent), ("ensemble_cache", batches.ensemble), ("compose", batches.compose)):
            print(f"[{profile}/{name}]")
            for command in commands:
                print(subprocess.list2cmdline(command.argv))
        return 0

    initial = _scan(config, frozen, profile, cache, output)
    existing_violations = sum(initial[field] for field in _VIOLATIONS if field != "missing")
    if existing_violations and not retry_failed:
        raise ValueError("Existing Stage 5 artifacts are failed, fallback, corrupt, or stale; use --retry-failed to repair")

    started = datetime.now(timezone.utc).isoformat()
    began = time.perf_counter()
    exits: dict[str, int] = {}
    progress_path = logs / "progress.json"
    stages = (
        ("concurrent_cache", batches.concurrent),
        *(((("ensemble_cache", batches.ensemble),)) if batches.ensemble else ()),
        ("compose", batches.compose),
    )

    def update(status: str, child: str, error: str | None = None):
        counts = _scan(config, frozen, profile, cache, output)
        payload = {
            "profile": profile, "status": status, "child_stage": child,
            "started_at": started, "updated_at": datetime.now(timezone.utc).isoformat(),
            **counts, "worker_exit_codes": dict(exits),
            "elapsed_seconds": float(time.perf_counter() - began), "error": error,
        }
        write_stage5_progress(progress_path, payload)
        return counts

    for child, commands in stages:
        update("running", child)
        try:
            codes = run_workers(commands, cwd=root, log_dir=logs / child)
        except Exception as error:
            update("failed", child, f"{type(error).__name__}: {error}")
            return 1
        except BaseException as error:
            update("interrupted", child, type(error).__name__)
            raise
        exits.update(codes)
        workers_ok = set(codes) == {c.name for c in commands} and all(type(v) is int and v == 0 for v in codes.values())
        counts = update("running", child)
        if child == "concurrent_cache":
            artifacts_ok = all(
                counts["by_method"][method]["completed_caches"]
                == counts["by_method"][method]["expected_caches"]
                for method in config["methods"] if method != "ensemble"
            )
        elif child == "ensemble_cache":
            artifacts_ok = counts["completed_caches"] == counts["expected_caches"]
        else:
            artifacts_ok = counts["completed_results"] == counts["expected_results"]
        if not workers_ok or not artifacts_ok or any(counts[field] for field in _VIOLATIONS if field != "missing"):
            update("failed", child, "Workers or required artifacts did not all succeed")
            return next((v for v in codes.values() if type(v) is int and v != 0), 1)
    update("completed", "compose")
    return 0
