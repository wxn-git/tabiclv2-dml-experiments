from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from tabdml.parallel import run_workers
from tabdml.stage3b_parallel import build_stage3b_batch_a_commands


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--replications", type=int, default=50)
    parser.add_argument("--cpu-workers", type=int, default=8)
    parser.add_argument("--cache-root", default="results/stage3b_cache_batch_a")
    parser.add_argument("--output-root", default="results/stage3b_batch_a_raw")
    parser.add_argument("--log-dir", default="results/logs/stage3b_batch_a")
    parser.add_argument("--stage", default="stage3b_batch_a")
    parser.add_argument("--seed-namespace", default="stage3_tree_diagnosis")
    parser.add_argument("--scenario", default="tree")
    return parser.parse_args()


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def main() -> int:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    cache_root = _resolve(project_root, args.cache_root)
    commands = build_stage3b_batch_a_commands(
        sys.executable,
        project_root,
        cache_root,
        args.cpu_workers,
        args.replications,
        stage=args.stage,
        seed_namespace=args.seed_namespace,
        scenario=args.scenario,
    )
    print(f"Launching {len(commands)} Stage 3B Batch A workers.", flush=True)
    exit_codes = run_workers(
        commands,
        cwd=project_root,
        log_dir=_resolve(project_root, args.log_dir),
    )
    if any(code != 0 for code in exit_codes.values()):
        return 1
    compose = subprocess.run(
        (
            sys.executable,
            str(project_root / "scripts" / "compose_stage3b_batch_a.py"),
            "--cache-root",
            str(cache_root),
            "--output-root",
            str(_resolve(project_root, args.output_root)),
            "--replications",
            str(args.replications),
            "--stage",
            args.stage,
            "--seed-namespace",
            args.seed_namespace,
            "--scenario",
            args.scenario,
        ),
        cwd=project_root,
        check=False,
    )
    return compose.returncode


if __name__ == "__main__":
    raise SystemExit(main())
