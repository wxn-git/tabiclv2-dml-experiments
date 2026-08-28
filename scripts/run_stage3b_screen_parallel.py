from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from tabdml.parallel import run_workers
from tabdml.stage3b_parallel import build_stage3b_screening_commands


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--replications", type=int, default=10)
    parser.add_argument("--cpu-workers", type=int, default=8)
    parser.add_argument("--output-root", default="results/stage3b_screening_raw")
    parser.add_argument("--selected-output", default="results/stage3b_screening/selected_models.json")
    parser.add_argument("--log-dir", default="results/logs/stage3b_screening")
    return parser.parse_args()


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def main() -> int:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    output_root = _resolve(project_root, args.output_root)
    commands = build_stage3b_screening_commands(
        sys.executable,
        project_root,
        output_root,
        args.cpu_workers,
        args.replications,
    )
    print(f"Launching {len(commands)} Stage 3B screening workers.", flush=True)
    exit_codes = run_workers(
        commands,
        cwd=project_root,
        log_dir=_resolve(project_root, args.log_dir),
    )
    if any(code != 0 for code in exit_codes.values()):
        return 1
    select = subprocess.run(
        (
            sys.executable,
            str(project_root / "scripts" / "run_stage3b_screen.py"),
            "--output-root",
            str(output_root),
            "--selected-output",
            str(_resolve(project_root, args.selected_output)),
            "--replications",
            str(args.replications),
            "--select",
        ),
        cwd=project_root,
        check=False,
    )
    return select.returncode


if __name__ == "__main__":
    raise SystemExit(main())
