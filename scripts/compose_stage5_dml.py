from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from pathlib import Path

from tabdml.nuisance_cache import NuisanceCache
from tabdml.stage5_config import load_stage5_config
from tabdml.stage5_experiment import (
    build_stage5_nuisance_spec,
    compose_stage5_record,
    iter_stage5_pairs,
    read_stage5_nuisance,
    resolve_stage5_method,
    stage5_nuisance_metadata_path,
    validate_stage5_record,
    validate_stage5_resume_record,
)
from tabdml.stage5_tuning import load_stage5_tuning
from tabdml.storage import ResultStore


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/stage5_sensitivity_five.yaml")
    parser.add_argument("--profile", choices=("smoke", "preflight", "formal"), required=True)
    parser.add_argument("--frozen-tuning", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--retry-failed", action="store_true")
    return parser.parse_args()


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _existing(path: Path) -> Mapping:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as error:
        raise ValueError(f"Invalid existing Stage 5 record: {path}") from error
    if not isinstance(value, Mapping):
        raise ValueError(f"Invalid existing Stage 5 record: {path}")
    return value


def _validate_existing_failure(record: Mapping, pair) -> None:
    expected = {
        "task_key": pair.key,
        "stage": pair.stage,
        "profile": pair.profile,
        "seed_namespace": pair.seed_namespace,
        "scenario": pair.scenario,
        "n": pair.n,
        "p": pair.p,
        "replication": pair.replication,
        "method": pair.method,
        "config_fingerprint": pair.config_fingerprint,
        "tuning_fingerprint": pair.tuning_fingerprint,
    }
    if any(
        record.get(field) != value or type(record.get(field)) is not type(value)
        for field, value in expected.items()
    ):
        raise ValueError("Stage 5 existing failure provenance mismatch")


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    config = load_stage5_config(_resolve(root, args.config))
    tuning_profile = "fast" if args.profile == "smoke" else "full"
    frozen = load_stage5_tuning(_resolve(root, args.frozen_tuning), config, tuning_profile)
    cache_root = _resolve(root, args.cache_root)
    output_root = _resolve(root, args.output)
    pairs = tuple(iter_stage5_pairs(config, frozen, args.profile))
    expected_output_paths = {output_root / f"{pair.key}.json" for pair in pairs}
    prepared = []
    expected_cache_paths = set()
    expected_metadata_paths = set()
    encountered_failure = False
    try:
        for pair in pairs:
            results = []
            for target in ("l", "m"):
                resolved = resolve_stage5_method(pair, target, config, frozen)
                task = build_stage5_nuisance_spec(pair, target, resolved)
                cache = NuisanceCache(cache_root)
                expected_cache_paths.add(cache.path(task))
                expected_metadata_paths.add(stage5_nuisance_metadata_path(cache, task))
                results.append(
                    read_stage5_nuisance(
                        task,
                        resolved,
                        cache_root,
                        full_settings=args.profile != "smoke",
                    )
                )
            output_path = output_root / f"{pair.key}.json"
            skip = False
            if output_path.exists():
                old = _existing(output_path)
                status = validate_stage5_resume_record(old, pair)
                if status == "success":
                    skip = True
                elif status in {"failed", "oom"} and not args.retry_failed:
                    encountered_failure = True
                    skip = True
                elif status == "fallback" and not args.retry_failed:
                    raise ValueError("Stage 5 existing result contains fallback")
            prepared.append((pair, results[0], results[1], skip))
        if set(cache_root.glob("*.npz")) != expected_cache_paths:
            raise ValueError("Stage 5 nuisance cache universe is not exact")
        if set(cache_root.glob("stage5-meta-*.json")) != expected_metadata_paths:
            raise ValueError("Stage 5 nuisance metadata universe is not exact")
        failure_root = cache_root.parent / "failures"
        if list(failure_root.glob("*.json")):
            raise ValueError("Stage 5 nuisance cache contains failed tasks")
        unexpected_outputs = set(output_root.glob("*.json")).difference(
            expected_output_paths
        )
        if unexpected_outputs:
            raise ValueError("Stage 5 result store contains unexpected task keys")
        if encountered_failure:
            raise ValueError("Stage 5 existing failed/OOM result was skipped")
    except (ValueError, OSError, KeyError, TypeError) as error:
        print(str(error), file=sys.stderr)
        return 1
    output = ResultStore(output_root)
    try:
        for pair, l_result, m_result, skip in prepared:
            if skip:
                continue
            record = compose_stage5_record(pair, l_result, m_result)
            if record["status"] != "success":
                print(
                    f"Stage 5 composition rejected {pair.key}: {record['status']}",
                    file=sys.stderr,
                )
                return 1
            validate_stage5_record(record, pair)
            output.write(record)
    except (ValueError, OSError, KeyError, TypeError) as error:
        print(str(error), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
