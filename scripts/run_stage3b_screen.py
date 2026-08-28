from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from tabdml.stage3b_screen import (
    iter_screening_tasks,
    run_screening_task,
    write_screening_winners,
)
from tabdml.storage import ResultStore


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/stage3b_tree_publication.yaml")
    parser.add_argument("--output-root", default="results/stage3b_screening_raw")
    parser.add_argument("--selected-output", default="results/stage3b_screening/selected_models.json")
    parser.add_argument("--replications", type=int)
    parser.add_argument("--candidate-groups", nargs="+")
    parser.add_argument("--candidates", nargs="+")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--select", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = project_root / config_path
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    replications = args.replications or int(config["screening"]["replications"])
    for task in iter_screening_tasks(
        config,
        replications,
        candidate_groups=set(args.candidate_groups) if args.candidate_groups else None,
        candidate_names=set(args.candidates) if args.candidates else None,
        num_shards=args.num_shards,
        shard_index=args.shard_index,
    ):
        record = run_screening_task(
            task,
            output_root=args.output_root,
            retry_failed=args.retry_failed,
            fast=args.fast,
        )
        print(task.key, record["status"], flush=True)
    if args.select:
        write_screening_winners(
            ResultStore(args.output_root).read_all(),
            args.selected_output,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
