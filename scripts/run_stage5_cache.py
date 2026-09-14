from __future__ import annotations

import argparse
import json
import sys
import traceback
from dataclasses import asdict
from pathlib import Path

from tabdml.runner import classify_failure
from tabdml.sharding import belongs_to_shard, validate_shard
from tabdml.stage5_config import load_stage5_config, resolve_stage5_profile
from tabdml.stage5_experiment import (
    build_stage5_nuisance_spec,
    fit_stage5_nuisance,
    iter_stage5_pairs,
    methods_for_device,
    resolve_stage5_method,
)
from tabdml.stage5_tuning import load_stage5_tuning
from tabdml.storage import ResultStore


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/stage5_sensitivity_five.yaml")
    parser.add_argument("--profile", choices=("smoke", "preflight", "formal"), default="smoke")
    parser.add_argument("--frozen-tuning", default="results/stage5_five/tuning/frozen-fast.json")
    parser.add_argument("--cache-root", default="results/stage5_five/smoke/cache")
    parser.add_argument("--device-group", choices=("gpu", "cpu", "ensemble"), required=True)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--retry-failed", action="store_true")
    return parser.parse_args()


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def main() -> int:
    args = parse_args()
    validate_shard(args.num_shards, args.shard_index)
    if args.device_group == "gpu" and (args.num_shards, args.shard_index) != (1, 0):
        raise ValueError("GPU Stage 5 worker must be unsharded")
    root = Path(__file__).resolve().parents[1]
    config = load_stage5_config(_resolve(root, args.config))
    profile = resolve_stage5_profile(config, args.profile)
    tuning_profile = "fast" if args.profile == "smoke" else "full"
    frozen = load_stage5_tuning(_resolve(root, args.frozen_tuning), config, tuning_profile)
    cache_root = _resolve(root, args.cache_root)
    allowed = set(methods_for_device(args.device_group))
    requests = {}
    for pair in iter_stage5_pairs(config, frozen, args.profile):
        if pair.method not in allowed:
            continue
        for target in ("l", "m"):
            resolved = resolve_stage5_method(pair, target, config, frozen)
            task = build_stage5_nuisance_spec(pair, target, resolved)
            request = (pair, task, resolved)
            if task.key in requests and requests[task.key] != request:
                raise ValueError(f"Stage 5 nuisance task key collision: {task.key}")
            requests[task.key] = request
    failures = ResultStore(cache_root.parent / "failures")
    failed = False
    for key, (pair, task, resolved) in requests.items():
        if not belongs_to_shard(key, args.num_shards, args.shard_index):
            continue
        failure_path = failures.root / f"{key}.json"
        if failure_path.exists():
            try:
                previous = json.loads(failure_path.read_text(encoding="utf-8"))
            except Exception as error:
                raise ValueError(f"Invalid Stage 5 failure record: {failure_path}") from error
            status = previous.get("status") if isinstance(previous, dict) else None
            if status not in {"failed", "oom"}:
                raise ValueError("Invalid Stage 5 failure record status")
            expected = {
                "task_key": key,
                "pair_key": pair.key,
                "profile": pair.profile,
                "config_fingerprint": pair.config_fingerprint,
                "tuning_fingerprint": pair.tuning_fingerprint,
                "task": asdict(task),
            }
            if set(previous) != {
                *expected, "status", "error_type", "error_message", "traceback"
            }:
                raise ValueError("Invalid Stage 5 failure record schema")
            if any(
                previous.get(field) != value
                or type(previous.get(field)) is not type(value)
                for field, value in expected.items()
            ):
                raise ValueError("Invalid Stage 5 failure record provenance")
            if any(
                not isinstance(previous.get(field), str)
                for field in ("error_type", "error_message", "traceback")
            ):
                raise ValueError("Invalid Stage 5 failure record diagnostics")
            if not args.retry_failed:
                failed = True
                print(key, "skipped-failure", flush=True)
                continue
        try:
            result = fit_stage5_nuisance(
                task,
                resolved,
                cache_root,
                pair.theta0,
                profile.full_settings,
                retry_failed=args.retry_failed,
            )
            if result.fallback_reason:
                raise RuntimeError(f"silent fallback: {result.fallback_reason}")
            failure_path.unlink(missing_ok=True)
            print(key, "success", flush=True)
        except Exception as error:
            failed = True
            failures.write({
                "task_key": key,
                "status": classify_failure(error),
                "error_type": type(error).__name__,
                "error_message": str(error)[:1000],
                "traceback": traceback.format_exc(limit=8),
                "pair_key": pair.key,
                "profile": pair.profile,
                "config_fingerprint": pair.config_fingerprint,
                "tuning_fingerprint": pair.tuning_fingerprint,
                "task": asdict(task),
            })
            print(key, classify_failure(error), file=sys.stderr, flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
