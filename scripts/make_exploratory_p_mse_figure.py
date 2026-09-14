from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

_matplotlib_cache = Path(tempfile.gettempdir()) / "tabdml-matplotlib"
_matplotlib_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_matplotlib_cache))

import pandas as pd

from tabdml.figures import (
    make_treatment_effect_p_trend_figure,
    prepare_treatment_effect_p_trend_data,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create the exploratory treatment-effect MSE versus p figure."
    )
    parser.add_argument(
        "--stage2-summary",
        type=Path,
        default=Path("results/published/stage2/summary_stage2.csv"),
    )
    parser.add_argument(
        "--stage4-screening",
        type=Path,
        default=Path("results/published/stage4_tree_benchmark/screening_summary.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/published/figures"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stage2 = pd.read_csv(args.stage2_summary)
    stage4 = pd.read_csv(args.stage4_screening)
    plot_data = prepare_treatment_effect_p_trend_data(stage2, stage4)
    outputs = make_treatment_effect_p_trend_figure(plot_data, args.output_dir)
    for kind, path in outputs.items():
        print(f"{kind}: {path}")


if __name__ == "__main__":
    main()
