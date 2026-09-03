from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path

from tabdml.nuisance_cache import NuisanceCache
from tabdml.sharding import validate_shard
from tabdml.stage4_config import load_stage4_config
from tabdml.stage4_experiment import (
    build_stage4_nuisance_spec,
    compose_stage4_record,
    iter_stage4_pairs,
    validate_frozen_tuning,
    validate_stage4_cached_result,
    validate_stage4_record,
)
from tabdml.storage import ResultStore


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/stage4_tree_benchmark.yaml")
    parser.add_argument("--phase", choices=("screening", "confirmation"), required=True)
    parser.add_argument(
        "--tuned-models",
        default="results/stage4_tree_tuning/selected_xgboost.json",
    )
    parser.add_argument("--selected-cells")
    parser.add_argument("--cache-root", default="results/stage4_tree_cache")
    parser.add_argument("--output-root")
    parser.add_argument("--replications", type=int)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--fast", action="store_true")
    return parser.parse_args()


def _read_json(path: str | Path, label: str) -> Mapping:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception as error:
        raise ValueError(f"Invalid {label} JSON: {path}") from error
    if not isinstance(value, Mapping):
        raise ValueError(f"Invalid {label} JSON: expected an object")
    return value


def _read_existing(path: Path) -> Mapping:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as error:
        raise ValueError(f"Invalid existing Stage 4 record: {path}") from error
    if not isinstance(value, Mapping):
        raise ValueError(f"Invalid existing Stage 4 record: {path}")
    return value


def main() -> int:
    args = parse_args()
    validate_shard(args.num_shards, args.shard_index)
    project_root = Path(__file__).resolve().parents[1]
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = project_root / config_path
    config = load_stage4_config(config_path)
    frozen_tuning = _read_json(args.tuned_models, "frozen tuning")
    profile = "fast" if args.fast else "full"
    validate_frozen_tuning(config, frozen_tuning, profile)
    selected_confirmation = None
    if args.phase == "confirmation":
        if not args.selected_cells:
            raise ValueError("confirmation requires selected confirmation cells")
        selected_confirmation = _read_json(
            args.selected_cells, "selected confirmation cells"
        )
    pairs = tuple(
        iter_stage4_pairs(
            config,
            args.phase,
            frozen_tuning,
            selected_confirmation=selected_confirmation,
            replications=args.replications,
            num_shards=args.num_shards,
            shard_index=args.shard_index,
            fast=args.fast,
        )
    )
    output_root = Path(
        args.output_root or f"results/stage4_tree_{args.phase}_raw"
    )
    cache = NuisanceCache(args.cache_root)

    cached_results = {}
    pair_tasks = {}
    for pair in pairs:
        tasks = {}
        for target in ("l", "m"):
            task = build_stage4_nuisance_spec(pair, target)
            path = cache.path(task)
            if not path.exists():
                raise FileNotFoundError(f"Missing nuisance cache: {path}")
            if task.key not in cached_results:
                result = cache.read(task, expected_length=pair.n)
                validate_stage4_cached_result(pair, result, target)
                cached_results[task.key] = result
            tasks[target] = task
        pair_tasks[pair.key] = tasks

    actions = []
    for pair in pairs:
        output_path = output_root / f"{pair.key}.json"
        if output_path.exists():
            previous = _read_existing(output_path)
            if previous.get("task_key") != pair.key:
                raise ValueError(
                    f"Invalid Stage 4 record {pair.key}: task_key mismatch"
                )
            if previous.get("status") == "success":
                validate_stage4_record(previous, pair)
                actions.append((pair, "skip"))
                continue
            if not args.retry_failed:
                actions.append((pair, "skip"))
                continue
        actions.append((pair, "compose"))

    output = ResultStore(output_root)
    for pair, action in actions:
        if action == "skip":
            print(pair.key, "skipped", flush=True)
            continue
        tasks = pair_tasks[pair.key]
        record = compose_stage4_record(
            pair,
            cached_results[tasks["l"].key],
            cached_results[tasks["m"].key],
        )
        validate_stage4_record(record, pair)
        output.write(record)
        print(pair.key, record["status"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
