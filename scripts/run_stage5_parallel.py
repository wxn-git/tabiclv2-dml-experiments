from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tabdml.stage5_parallel import run_stage5_parallel


def parse_args():
    parser = argparse.ArgumentParser(description="Resumable Stage 5 sensitivity benchmark")
    parser.add_argument("--config", default="configs/stage5_sensitivity_five.yaml")
    parser.add_argument("--profile", choices=("smoke", "preflight", "formal"), required=True)
    parser.add_argument("--frozen-tuning", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--log-dir", required=True)
    parser.add_argument("--cpu-workers", type=int, default=5)
    parser.add_argument("--ensemble-workers", type=int, default=2)
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--formal-approved", action="store_true")
    parser.add_argument("--preflight-summary")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    return run_stage5_parallel(
        sys.executable, root, args.config, profile=args.profile,
        frozen_tuning=args.frozen_tuning, cache_root=args.cache_root,
        output_root=args.output_root, log_dir=args.log_dir,
        cpu_workers=args.cpu_workers, ensemble_workers=args.ensemble_workers,
        retry_failed=args.retry_failed, dry_run=args.dry_run,
        formal_approved=args.formal_approved,
        preflight_summary=args.preflight_summary,
    )


if __name__ == "__main__":
    raise SystemExit(main())
