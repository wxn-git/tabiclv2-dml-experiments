from __future__ import annotations

import argparse

import pandas as pd

from tabdml.figures import make_accuracy_cost_figure


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", default="results/summary.csv")
    parser.add_argument("--output-dir", default="results/figures")
    args = parser.parse_args()
    frame = pd.read_csv(args.summary)
    try:
        path = make_accuracy_cost_figure(frame, args.output_dir)
    except ValueError:
        print("No successful summary rows to plot.")
        return
    print(f"Wrote {path}.")


if __name__ == "__main__":
    main()
