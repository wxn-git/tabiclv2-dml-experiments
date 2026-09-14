from __future__ import annotations

import argparse
from pathlib import Path

from tabdml.stage5_config import load_stage5_config
from tabdml.stage5_migration import migrate_stage5_cache


def parse_args():
    parser = argparse.ArgumentParser(description="Audit and migrate Stage 5 nuisance caches")
    parser.add_argument("--source-config", required=True)
    parser.add_argument("--destination-config", required=True)
    parser.add_argument("--profile", choices=("preflight",), default="preflight")
    parser.add_argument("--source-tuning", required=True)
    parser.add_argument("--destination-tuning", required=True)
    parser.add_argument("--source-cache", required=True)
    parser.add_argument("--destination-cache", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    resolve = lambda value: Path(value) if Path(value).is_absolute() else root / value
    source_config = load_stage5_config(resolve(args.source_config))
    destination_config = load_stage5_config(resolve(args.destination_config))
    manifest = migrate_stage5_cache(
        source_config, destination_config, resolve(args.source_tuning),
        resolve(args.destination_tuning), resolve(args.source_cache),
        resolve(args.destination_cache), resolve(args.manifest),
        profile=args.profile, resume=args.resume,
    )
    print(f"migrated={manifest['migrated_caches']} violations={manifest['violations']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
