from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tabdml.stage4_parallel import run_stage4_phase


def parse_args():
    parser = argparse.ArgumentParser(
        description="Resumable Stage 4: at most one GPU and eight CPU workers.",
        epilog=(
            "--fast defaults to one implementation-smoke replication. "
            "Independent full-model preflight: --phase confirmation --preflight "
            "(exactly five replications, no --fast). "
            "--replications 5 alone is a formal subset, not preflight. "
            "Use a separate --output-root and --log-dir for preflight; "
            "formal analysis rejects preflight records. "
            "Full defaults: tuning 10, screening 20, confirmation 100. "
            "Tuning selection, confirmation-cell selection and analysis are separate steps."
        ),
    )
    parser.add_argument("--phase", choices=("tuning", "screening", "confirmation"), required=True)
    parser.add_argument("--replications", type=int)
    parser.add_argument("--cpu-workers", type=int, default=8)
    parser.add_argument("--config", default="configs/stage4_tree_benchmark.yaml")
    parser.add_argument("--tuned-models", default="results/stage4_tree_tuning/selected_xgboost.json")
    parser.add_argument("--selected-cells")
    parser.add_argument("--cache-root", default="results/stage4_tree_cache")
    parser.add_argument("--output-root", help="Defaults to results/stage4_tree_<phase>_raw")
    parser.add_argument("--log-dir", help="Defaults to results/logs/stage4_tree/<phase>")
    parser.add_argument("--fast", action="store_true")
    parser.add_argument(
        "--preflight", action="store_true",
        help="Independent five-rep full-model confirmation preflight",
    )
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Validate and print commands; start no processes")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return run_stage4_phase(
        sys.executable,
        Path(__file__).resolve().parents[1],
        args.config,
        phase=args.phase,
        tuned_models=args.tuned_models,
        selected_cells=args.selected_cells,
        cache_root=args.cache_root,
        output_root=args.output_root,
        log_dir=args.log_dir,
        cpu_workers=args.cpu_workers,
        replications=args.replications,
        fast=args.fast,
        retry_failed=args.retry_failed,
        dry_run=args.dry_run,
        preflight=args.preflight,
    )


if __name__ == "__main__":
    raise SystemExit(main())
