from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tabdml.parallel import build_stage3_worker_commands, run_workers


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/stage3_tree_diagnosis.yaml")
    parser.add_argument(
        "--output-root", default="results/stage3_tree_diagnosis_raw"
    )
    parser.add_argument(
        "--log-dir", default="results/logs/stage3_tree_diagnosis_smoke"
    )
    parser.add_argument("--cpu-workers", type=int, default=8)
    parser.add_argument("--replications", type=int, default=5)
    return parser.parse_args()


def _resolve(project_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def main() -> int:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    commands = build_stage3_worker_commands(
        sys.executable,
        project_root / "scripts" / "run_stage3.py",
        _resolve(project_root, args.output_root),
        _resolve(project_root, args.config),
        args.cpu_workers,
        args.replications,
    )
    log_dir = _resolve(project_root, args.log_dir)
    print(f"Launching {len(commands)} Stage 3 workers.", flush=True)
    for command in commands:
        print(f"{command.name}: {' '.join(command.argv)}", flush=True)
    exit_codes = run_workers(commands, cwd=project_root, log_dir=log_dir)
    failed = {name: code for name, code in exit_codes.items() if code != 0}
    if failed:
        print(f"Workers failed: {failed}", flush=True)
        return 1
    print("All Stage 3 workers completed.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
