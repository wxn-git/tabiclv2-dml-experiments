from __future__ import annotations

import argparse
from pathlib import Path

from tabdml.sharding import belongs_to_shard, validate_shard
from tabdml.stage4_config import load_stage4_config
from tabdml.stage4_tuning import (
    iter_tuning_tasks,
    run_tuning_task,
    write_tuned_xgboost,
)
from tabdml.storage import ResultStore


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/stage4_tree_benchmark.yaml")
    parser.add_argument("--output-root", default="results/stage4_tree_tuning_raw")
    parser.add_argument(
        "--selected-output",
        default="results/stage4_tree_tuning/selected_xgboost.json",
    )
    parser.add_argument("--replications", type=int)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--select", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    validate_shard(args.num_shards, args.shard_index)
    project_root = Path(__file__).resolve().parents[1]
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = project_root / config_path
    config = load_stage4_config(config_path)
    replications = (
        int(config["tuning"]["replications"])
        if args.replications is None
        else args.replications
    )
    all_tasks = tuple(iter_tuning_tasks(config, replications, fast=args.fast))

    for task in all_tasks:
        if not belongs_to_shard(task.key, args.num_shards, args.shard_index):
            continue
        record = run_tuning_task(
            task,
            output_root=args.output_root,
            retry_failed=args.retry_failed,
            fast=args.fast,
        )
        print(task.key, record["status"], flush=True)

    if args.select:
        write_tuned_xgboost(
            ResultStore(args.output_root).read_all(),
            args.selected_output,
            expected_replications=replications,
            expected_tasks=all_tasks,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
