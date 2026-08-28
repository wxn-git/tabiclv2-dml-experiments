from __future__ import annotations

import argparse

from tabdml.config import load_config
from tabdml.runner import run_task
from tabdml.sharding import belongs_to_shard, validate_shard
from tabdml.stages import enumerate_stage1_tasks


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/stage1.yaml")
    parser.add_argument("--scenarios", nargs="+")
    parser.add_argument("--sample-sizes", nargs="+", type=int)
    parser.add_argument("--dimensions", nargs="+", type=int)
    parser.add_argument("--replications", type=int)
    parser.add_argument("--learners", nargs="+")
    parser.add_argument("--output-root", default="results/raw")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    validate_shard(args.num_shards, args.shard_index)
    config = load_config(args.config)
    completed = 0
    for task in enumerate_stage1_tasks(config):
        if args.scenarios and task.scenario not in args.scenarios:
            continue
        if args.sample_sizes and task.n not in args.sample_sizes:
            continue
        if args.dimensions and task.p not in args.dimensions:
            continue
        if args.replications is not None and task.replication >= args.replications:
            continue
        if args.learners and task.learner not in args.learners:
            continue
        if not belongs_to_shard(task.key, args.num_shards, args.shard_index):
            continue
        record = run_task(
            task,
            folds_count=config.folds,
            theta0=config.theta0,
            output_root=args.output_root,
            retry_failed=args.retry_failed,
            fast=args.fast,
        )
        completed += 1
        print(task.key, record["status"], flush=True)
    print(f"Processed {completed} tasks.")


if __name__ == "__main__":
    main()
