from __future__ import annotations

import argparse
import json
from pathlib import Path

from tabdml.sharding import replication_belongs_to_shard, validate_shard
from tabdml.stage3b import Stage3BPairSpec, build_nuisance_spec, fit_cached_nuisance


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", default="results/stage3b_cache_batch_a")
    parser.add_argument("--stage", default="stage3b_batch_a")
    parser.add_argument("--seed-namespace", default="stage3_tree_diagnosis")
    parser.add_argument("--scenario", default="tree")
    parser.add_argument("--n", type=int, default=2000)
    parser.add_argument("--p", type=int, default=10)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--theta0", type=float, default=1.0)
    parser.add_argument("--replications", type=int, default=50)
    parser.add_argument("--learners", nargs="+", default=["tabiclv2_1", "xgboost", "oracle"])
    parser.add_argument("--targets", nargs="+", choices=["l", "m"], default=["l", "m"])
    parser.add_argument("--learner-targets", nargs="+")
    parser.add_argument("--selected-models")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--fast", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    validate_shard(args.num_shards, args.shard_index)
    selected = {}
    if args.selected_models:
        selected = json.loads(Path(args.selected_models).read_text(encoding="utf-8"))
    if args.learner_targets:
        requests = []
        for value in args.learner_targets:
            target, learner = value.split(":", maxsplit=1)
            if target not in {"l", "m"}:
                raise ValueError(f"Invalid learner target: {value}")
            requests.append((target, learner))
    else:
        requests = [(target, learner) for learner in args.learners for target in args.targets]
    for replication in range(args.replications):
        if not replication_belongs_to_shard(
            replication, args.num_shards, args.shard_index
        ):
            continue
        for target, learner in requests:
            selected_model = selected.get(learner, {})
            config_hash = str(selected_model.get("config_hash") or "default")
            pair = Stage3BPairSpec(
                stage=args.stage,
                seed_namespace=args.seed_namespace,
                scenario=args.scenario,
                n=args.n,
                p=args.p,
                replication=replication,
                learner_l=learner,
                learner_m=learner,
                folds_count=args.folds,
                theta0=args.theta0,
                learner_l_config_hash=config_hash,
                learner_m_config_hash=config_hash,
            )
            task = build_nuisance_spec(pair, target)
            fit_cached_nuisance(
                task,
                cache_root=args.cache_root,
                theta0=args.theta0,
                fast=args.fast,
                learner_kind=selected_model.get("learner_kind"),
                learner_params=selected_model.get("params"),
            )
            print(task.key, "success", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
