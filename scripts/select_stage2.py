from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

from tabdml.stages import select_stage2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", default="results/summary_stage1.csv")
    parser.add_argument("--output", default="configs/stage2_selected.yaml")
    args = parser.parse_args()
    selected = select_stage2(pd.read_csv(args.summary))
    payload = {
        "stage": "stage2",
        "selected_configurations": selected,
        "learners": [
            "lasso",
            "random_forest",
            "xgboost",
            "mlp",
            "ensemble",
            "tabiclv2_1",
            "tabiclv2_8",
        ],
        "folds": 5,
        "replications": 100,
        "theta0": 1.0,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    print(f"Wrote {len(selected)} selected configurations to {output}.")


if __name__ == "__main__":
    main()

