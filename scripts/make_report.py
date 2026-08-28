from __future__ import annotations

import argparse

import pandas as pd

from tabdml.report import write_chinese_report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", default="results/summary.csv")
    parser.add_argument("--output", default="results/report_zh.md")
    args = parser.parse_args()
    summary = pd.read_csv(args.summary)
    output = write_chinese_report(summary, args.output)
    print(f"Wrote {output}.")


if __name__ == "__main__":
    main()

