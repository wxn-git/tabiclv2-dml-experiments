from __future__ import annotations

import argparse
from pathlib import Path

from tabdml.aggregate import summarize
from tabdml.storage import ResultStore


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="results/raw")
    parser.add_argument("--output", default="results/summary.csv")
    args = parser.parse_args()
    records = ResultStore(args.input).read_all()
    summary = summarize(records)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output, index=False)
    print(f"Wrote {len(summary)} summary rows to {output}.")


if __name__ == "__main__":
    main()

