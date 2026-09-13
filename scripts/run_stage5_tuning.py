from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tabdml.stage5_config import load_stage5_config
from tabdml.stage5_tuning import iter_stage5_tuning_tasks, run_stage5_tuning_task


def _project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else Path(__file__).resolve().parents[1] / path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/stage5_sensitivity.yaml")
    parser.add_argument("--output-root", default="results/stage5_tuning_raw")
    parser.add_argument("--replications", type=int)
    parser.add_argument("--execution-profile", choices=("full", "fast"), default="full")
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--retry-failed", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    profile = "fast" if args.fast else args.execution_profile
    config = load_stage5_config(_project_path(args.config))
    output_root = _project_path(args.output_root)
    tasks = iter_stage5_tuning_tasks(
        config,
        args.replications,
        profile,
        args.num_shards,
        args.shard_index,
    )
    for task in tasks:
        record = run_stage5_tuning_task(task, output_root, args.retry_failed)
        print(task.key, record["status"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
