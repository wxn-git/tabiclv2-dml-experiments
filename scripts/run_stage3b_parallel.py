from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from tabdml.parallel import run_workers
from tabdml.stage3b_parallel import build_stage3b_confirmation_commands


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--replications", type=int, default=5)
    parser.add_argument("--cpu-workers", type=int, default=8)
    parser.add_argument("--cache-root", default="results/stage3b_cache_confirmation")
    parser.add_argument("--output-root", default="results/stage3b_confirmation_raw")
    parser.add_argument("--selected-models", default="results/stage3b_screening/selected_models.json")
    parser.add_argument("--log-dir", default="results/logs/stage3b_confirmation")
    return parser.parse_args()


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def main() -> int:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    cache_root = _resolve(project_root, args.cache_root)
    selected_models = _resolve(project_root, args.selected_models)
    commands = build_stage3b_confirmation_commands(
        sys.executable,
        project_root,
        cache_root,
        selected_models,
        args.cpu_workers,
        args.replications,
    )
    print(f"Launching {len(commands)} Stage 3B cache workers.", flush=True)
    exit_codes = run_workers(
        commands,
        cwd=project_root,
        log_dir=_resolve(project_root, args.log_dir),
    )
    failed = {name: code for name, code in exit_codes.items() if code != 0}
    if failed:
        print(f"Stage 3B cache workers failed: {failed}", flush=True)
        return 1
    compose = subprocess.run(
        (
            sys.executable,
            str(project_root / "scripts" / "compose_stage3b_confirmation.py"),
            "--selected-models",
            str(selected_models),
            "--cache-root",
            str(cache_root),
            "--output-root",
            str(_resolve(project_root, args.output_root)),
            "--replications",
            str(args.replications),
        ),
        cwd=project_root,
        check=False,
    )
    if compose.returncode != 0:
        return compose.returncode
    print("Stage 3B confirmation cache and composition completed.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
