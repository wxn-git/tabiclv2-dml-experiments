from __future__ import annotations

import argparse
from collections.abc import Mapping
import json
from pathlib import Path

from tabdml.stage4_config import load_stage4_config
from tabdml.stage4_selection import write_confirmation_cells


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/stage4_tree_benchmark.yaml")
    parser.add_argument(
        "--screening-root", default="results/stage4_tree_screening_raw"
    )
    parser.add_argument("--tuned-models", required=True)
    parser.add_argument(
        "--output",
        default="results/stage4_tree_screening/selected_confirmation_cells.json",
    )
    parser.add_argument("--expected-replications", type=int)
    profile = parser.add_mutually_exclusive_group()
    profile.add_argument("--profile", choices=("full", "fast"))
    profile.add_argument("--fast", action="store_true")
    return parser.parse_args()


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _read_screening_records(root: Path) -> list[Mapping]:
    if not root.is_dir():
        raise ValueError(f"Screening root is not a directory: {root}")
    records = []
    for path in sorted(root.glob("*.json")):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except Exception as error:
            raise ValueError(f"Invalid screening JSON: {path}") from error
        if not isinstance(value, Mapping):
            raise ValueError(f"Invalid screening JSON object: {path}")
        records.append(value)
    return records


def _read_json_object(path: Path, label: str) -> Mapping:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as error:
        raise ValueError(f"Invalid {label} JSON: {path}") from error
    if not isinstance(value, Mapping):
        raise ValueError(f"Invalid {label} JSON object: {path}")
    return value


def main() -> int:
    args = parse_args()
    execution_profile = "fast" if args.fast else (args.profile or "full")
    config = load_stage4_config(_resolve(args.config))
    frozen_tuning = _read_json_object(
        _resolve(args.tuned_models), "frozen tuning"
    )
    records = _read_screening_records(_resolve(args.screening_root))
    write_confirmation_cells(
        records,
        _resolve(args.output),
        config,
        frozen_tuning,
        expected_replications=args.expected_replications,
        execution_profile=execution_profile,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
