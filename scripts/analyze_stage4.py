from __future__ import annotations

import argparse
from collections.abc import Mapping
import json
from pathlib import Path

from tabdml.stage4_analysis import write_stage4_analysis
from tabdml.stage4_config import load_stage4_config


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/stage4_tree_benchmark.yaml")
    parser.add_argument(
        "--screening-root", default="results/stage4_tree_screening_raw"
    )
    parser.add_argument(
        "--confirmation-root", default="results/stage4_tree_confirmation_raw"
    )
    parser.add_argument(
        "--tuned-models",
        default="results/stage4_tree_tuning/selected_xgboost.json",
    )
    parser.add_argument(
        "--selected-cells",
        default="results/stage4_tree_screening/selected_confirmation_cells.json",
    )
    parser.add_argument(
        "--output-dir", default="results/stage4_tree_confirmation"
    )
    parser.add_argument("--alpha", type=float, default=0.05)
    profile = parser.add_mutually_exclusive_group()
    profile.add_argument("--profile", choices=("full", "fast"))
    profile.add_argument("--fast", action="store_true")
    return parser.parse_args()


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else _project_root() / path


def _read_json_object(path: Path, label: str) -> Mapping:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as error:
        raise ValueError(f"Invalid {label} JSON: {path}") from error
    if not isinstance(value, Mapping):
        raise ValueError(f"Invalid {label} JSON object: {path}")
    return value


def _read_record_directory(root: Path, label: str) -> list[Mapping]:
    if not root.is_dir():
        raise ValueError(f"{label} root is not a directory: {root}")
    records = []
    for path in sorted(root.glob("*.json")):
        records.append(_read_json_object(path, f"{label} record"))
    return records


def main() -> int:
    args = parse_args()
    execution_profile = "fast" if args.fast else (args.profile or "full")
    config = load_stage4_config(_resolve(args.config))
    frozen_tuning = _read_json_object(
        _resolve(args.tuned_models), "frozen tuning"
    )
    selected_confirmation = _read_json_object(
        _resolve(args.selected_cells), "selected confirmation"
    )
    screening_records = _read_record_directory(
        _resolve(args.screening_root), "screening"
    )
    confirmation_records = _read_record_directory(
        _resolve(args.confirmation_root), "confirmation"
    )
    write_stage4_analysis(
        screening_records,
        confirmation_records,
        config,
        frozen_tuning,
        selected_confirmation,
        _resolve(args.output_dir),
        execution_profile=execution_profile,
        alpha=args.alpha,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
