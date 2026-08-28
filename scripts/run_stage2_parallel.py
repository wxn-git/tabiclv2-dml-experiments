from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tabdml.parallel import build_stage2_worker_commands, run_workers


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/stage2_selected.yaml")
    parser.add_argument(
        "--partition-config-dir",
        default="configs/stage2_parallel",
    )
    parser.add_argument("--output-root", default="results/raw")
    parser.add_argument("--log-dir", default="results/logs/stage2_parallel")
    return parser.parse_args()


def _resolve(project_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def main() -> int:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    commands = build_stage2_worker_commands(
        sys.executable,
        project_root / "scripts" / "run_stage2.py",
        _resolve(project_root, args.output_root),
        _resolve(project_root, args.config),
        _resolve(project_root, args.partition_config_dir),
    )
    log_dir = _resolve(project_root, args.log_dir)
    print(f"Launching {len(commands)} Stage 2 workers.", flush=True)
    for command in commands:
        print(f"{command.name}: {' '.join(command.argv)}", flush=True)
    exit_codes = run_workers(commands, cwd=project_root, log_dir=log_dir)
    failed = {name: code for name, code in exit_codes.items() if code != 0}
    if failed:
        print(f"Workers failed: {failed}", flush=True)
        return 1
    print("All Stage 2 workers completed.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
