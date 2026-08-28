from __future__ import annotations

import argparse

import yaml

from tabdml.stage3 import iter_stage3_tasks, run_stage3_task


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/stage3_tree_diagnosis.yaml")
    parser.add_argument("--replications", type=int)
    parser.add_argument("--pair-names", nargs="+")
    parser.add_argument(
        "--output-root", default="results/stage3_tree_diagnosis_raw"
    )
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    with open(args.config, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    replications = args.replications or int(config["replications"])
    for task in iter_stage3_tasks(
        config,
        replications=replications,
        pair_names=set(args.pair_names) if args.pair_names else None,
        num_shards=args.num_shards,
        shard_index=args.shard_index,
    ):
        result = run_stage3_task(
            task,
            folds_count=int(config["folds"]),
            theta0=float(config["theta0"]),
            output_root=args.output_root,
            retry_failed=args.retry_failed,
            fast=args.fast,
        )
        print(task.key, result["status"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
