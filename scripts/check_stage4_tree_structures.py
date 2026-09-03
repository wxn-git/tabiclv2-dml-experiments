from __future__ import annotations

import argparse

from tabdml.stage4_structure import audit_tree_structures, write_structure_audit


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument(
        "--output-dir",
        default="results/stage4_tree_structure_checks",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    records = audit_tree_structures(n=args.n, seed=args.seed)
    write_structure_audit(records, args.output_dir)
    for row in records:
        print(
            f"{row['scenario']} target={row['target']} "
            f"root=X{row['root_variable']} threshold={row['threshold']:g} "
            f"split_gain={row['split_gain']:.12g} "
            f"left_probability={row['left_probability']:.6f}",
            flush=True,
        )
    return int(any(row["split_gain"] <= 1e-3 for row in records))


if __name__ == "__main__":
    raise SystemExit(main())
