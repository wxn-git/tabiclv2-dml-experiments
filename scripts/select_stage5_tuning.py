from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tabdml.stage5_config import load_stage5_config
from tabdml.stage5_tuning import write_stage5_tuning


def _project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else Path(__file__).resolve().parents[1] / path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/stage5_sensitivity_five.yaml")
    parser.add_argument("--input-root", action="append", default=[])
    parser.add_argument("--input-glob", action="append", default=[])
    parser.add_argument("--output", default="results/stage5_five/tuning/selected_xgboost.json")
    parser.add_argument("--execution-profile", choices=("full", "fast"), default="full")
    return parser.parse_args()


def _records(args) -> list[dict]:
    roots = [_project_path(value) for value in args.input_root]
    for pattern in args.input_glob:
        resolved = pattern if Path(pattern).is_absolute() else str(Path(__file__).resolve().parents[1] / pattern)
        roots.extend(Path(value) for value in sorted(glob.glob(resolved)))
    if not roots:
        raise ValueError("At least one --input-root or --input-glob is required")
    records = []
    seen = set()
    for root in roots:
        if root.is_file():
            paths = (root,)
        elif root.is_dir():
            paths = tuple(sorted(root.glob("*.json")))
        else:
            raise FileNotFoundError(f"Stage 5 tuning input does not exist: {root}")
        for path in paths:
            with path.open("r", encoding="utf-8") as handle:
                record = json.load(handle)
            key = record.get("task_key")
            if key in seen:
                raise ValueError(f"duplicate task_key across input roots: {key}")
            seen.add(key)
            records.append(record)
    return records


def main() -> int:
    args = parse_args()
    config = load_stage5_config(_project_path(args.config))
    write_stage5_tuning(
        _records(args), config, _project_path(args.output), args.execution_profile
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
