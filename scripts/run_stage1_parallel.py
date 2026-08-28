from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tabdml.parallel import build_worker_commands, run_workers


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/stage1.yaml")
    parser.add_argument("--cpu-workers", type=int, default=8)
    parser.add_argument("--output-root", default="results/raw")
    parser.add_argument("--log-dir", default="results/logs/stage1_parallel")
    parser.add_argument("--scenarios", nargs="+")
    parser.add_argument("--sample-sizes", nargs="+", type=int)
    parser.add_argument("--dimensions", nargs="+", type=int)
    parser.add_argument("--replications", type=int)
    parser.add_argument("--fast", action="store_true")
    return parser.parse_args()


def _extra_stage1_args(args) -> tuple[str, ...]:
    extra = ["--config", args.config]
    if args.scenarios:
        extra.extend(("--scenarios", *args.scenarios))
    if args.sample_sizes:
        extra.extend(("--sample-sizes", *map(str, args.sample_sizes)))
    if args.dimensions:
        extra.extend(("--dimensions", *map(str, args.dimensions)))
    if args.replications is not None:
        extra.extend(("--replications", str(args.replications)))
    if args.fast:
        extra.append("--fast")
    return tuple(extra)


def main() -> int:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    stage1_script = project_root / "scripts" / "run_stage1.py"
    log_dir = Path(args.log_dir)
    if not log_dir.is_absolute():
        log_dir = project_root / log_dir
    commands = build_worker_commands(
        sys.executable,
        stage1_script,
        args.output_root,
        args.cpu_workers,
        _extra_stage1_args(args),
    )
    print(f"Launching {len(commands)} Stage 1 workers.", flush=True)
    for command in commands:
        print(f"{command.name}: {' '.join(command.argv)}", flush=True)
    exit_codes = run_workers(commands, cwd=project_root, log_dir=log_dir)
    failed = {name: code for name, code in exit_codes.items() if code != 0}
    if failed:
        print(f"Workers failed: {failed}", flush=True)
        return 1
    print("All Stage 1 workers completed.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
