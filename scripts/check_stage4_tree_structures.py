from __future__ import annotations

import argparse

from tabdml.stage4_structure import (
    audit_tree_structures,
    structure_audit_failures,
    write_structure_audit,
)


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
    audit = audit_tree_structures(n=args.n, seed=args.seed)
    write_structure_audit(audit, args.output_dir)
    for row in audit["root_checks"]:
        print(
            f"{row['scenario']} target={row['target']} "
            f"root=X{row['root_variable']} threshold={row['threshold']:g} "
            f"theoretical_gain={row['theoretical_split_gain']:.12g} "
            f"monte_carlo_gain={row['monte_carlo_split_gain']:.12g} "
            f"left_probability={row['monte_carlo_left_probability']:.6f}",
            flush=True,
        )
    failures = structure_audit_failures(audit)
    for failure in failures:
        print(f"FAILED: {failure}", flush=True)
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
