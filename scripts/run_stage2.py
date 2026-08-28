from __future__ import annotations

import argparse

import yaml

from tabdml.config import TaskSpec
from tabdml.runner import run_task
from tabdml.sharding import replication_belongs_to_shard, validate_shard


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/stage2_selected.yaml")
    parser.add_argument("--replications", type=int)
    parser.add_argument("--learners", nargs="+")
    parser.add_argument("--output-root", default="results/raw")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    args = parser.parse_args()
    validate_shard(args.num_shards, args.shard_index)
    with open(args.config, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    repetitions = args.replications or int(config["replications"])
    for selected in config["selected_configurations"]:
        for replication in range(repetitions):
            if not replication_belongs_to_shard(
                replication,
                args.num_shards,
                args.shard_index,
            ):
                continue
            for learner in config["learners"]:
                if args.learners and learner not in args.learners:
                    continue
                estimators = 8 if learner == "tabiclv2_8" else 1 if learner == "tabiclv2_1" else 0
                task = TaskSpec(
                    "stage2",
                    selected["scenario"],
                    int(selected["n"]),
                    int(selected["p"]),
                    replication,
                    learner,
                    estimators,
                )
                result = run_task(
                    task,
                    int(config["folds"]),
                    float(config["theta0"]),
                    args.output_root,
                    args.retry_failed,
                    args.fast,
                )
                print(task.key, result["status"], flush=True)


if __name__ == "__main__":
    main()
