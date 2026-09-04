from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path

from tabdml.sharding import belongs_to_shard, validate_shard
from tabdml.stage4_config import load_stage4_config
from tabdml.stage4_experiment import (
    build_stage4_nuisance_spec,
    fit_stage4_nuisance,
    iter_stage4_pairs,
    resolve_method,
    validate_frozen_tuning,
)


_DEVICE_METHODS = {
    "gpu": frozenset({"tabiclv2_1", "tabiclv2_8"}),
    "cpu": frozenset({"xgboost", "xgboost_tuned", "extra_trees", "oracle"}),
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/stage4_tree_benchmark.yaml")
    parser.add_argument(
        "--phase", choices=("screening", "confirmation"), required=True
    )
    parser.add_argument("--device-group", choices=("gpu", "cpu"), required=True)
    parser.add_argument(
        "--tuned-models",
        default="results/stage4_tree_tuning/selected_xgboost.json",
    )
    parser.add_argument("--selected-cells")
    parser.add_argument("--cache-root", default="results/stage4_tree_cache")
    parser.add_argument("--replications", type=int)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--fast", action="store_true")
    parser.add_argument(
        "--preflight", action="store_true",
        help="Use with --phase confirmation: independent five-rep full-model preflight",
    )
    return parser.parse_args()


def _read_json(path: str | Path, label: str) -> Mapping:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception as error:
        raise ValueError(f"Invalid {label} JSON: {path}") from error
    if not isinstance(value, Mapping):
        raise ValueError(f"Invalid {label} JSON: expected an object")
    return value


def _resolve(project_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def main() -> int:
    args = parse_args()
    validate_shard(args.num_shards, args.shard_index)
    project_root = Path(__file__).resolve().parents[1]
    config_path = _resolve(project_root, args.config)
    config = load_stage4_config(config_path)
    tuned_models_path = _resolve(project_root, args.tuned_models)
    frozen_tuning = _read_json(tuned_models_path, "frozen tuning")
    cache_root = _resolve(project_root, args.cache_root)
    profile = "fast" if args.fast else "full"
    validate_frozen_tuning(config, frozen_tuning, profile)
    selected_confirmation = None
    if args.phase == "confirmation":
        if not args.selected_cells:
            raise ValueError("confirmation requires selected confirmation cells")
        selected_cells_path = _resolve(project_root, args.selected_cells)
        selected_confirmation = _read_json(
            selected_cells_path, "selected confirmation cells"
        )

    pairs = iter_stage4_pairs(
        config,
        args.phase,
        frozen_tuning,
        selected_confirmation=selected_confirmation,
        replications=args.replications,
        fast=args.fast,
        preflight=args.preflight,
    )
    allowed_methods = _DEVICE_METHODS[args.device_group]
    requests = {}
    extra_trees_params = config["extra_trees"]["params"]
    for pair in pairs:
        for target in ("l", "m"):
            learner = pair.learner_l if target == "l" else pair.learner_m
            if learner not in allowed_methods:
                continue
            resolved = resolve_method(
                pair, target, frozen_tuning, extra_trees_params
            )
            task = build_stage4_nuisance_spec(pair, target, resolved)
            requests.setdefault(task.key, (pair, target))

    for task_key, (pair, target) in requests.items():
        if not belongs_to_shard(task_key, args.num_shards, args.shard_index):
            continue
        fit_stage4_nuisance(
            pair,
            target,
            frozen_tuning,
            extra_trees_params,
            cache_root,
            fast=args.fast,
            retry_failed=args.retry_failed,
        )
        print(task_key, "success", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
