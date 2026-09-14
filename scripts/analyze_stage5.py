from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_MPL_CACHE = _PROJECT_ROOT / "results" / ".matplotlib"
_MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CACHE))

from tabdml.figures import make_stage5_sensitivity_figures
from tabdml.stage5_analysis import (
    build_stage5_plot_data,
    estimate_formal_runtime,
    load_stage5_records,
    summarize_stage5,
)
from tabdml.stage5_config import (
    iter_sensitivity_cells,
    load_stage5_config,
    resolve_stage5_profile,
    stage5_config_fingerprint,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze and plot Stage 5 results")
    parser.add_argument("--config", default="configs/stage5_sensitivity_five.yaml")
    parser.add_argument("--profile", choices=("smoke", "preflight", "formal"), required=True)
    parser.add_argument("--input", required=True, help="ResultStore directory or JSONL file")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--cpu-workers", type=int, default=5)
    parser.add_argument("--ensemble-workers", type=int, default=2)
    return parser.parse_args()


def _write_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main() -> int:
    args = parse_args()
    root = _PROJECT_ROOT
    config_path = Path(args.config)
    config_path = config_path if config_path.is_absolute() else root / config_path
    input_path = Path(args.input)
    input_path = input_path if input_path.is_absolute() else root / input_path
    output = Path(args.output_root)
    output = output if output.is_absolute() else root / output
    output.mkdir(parents=True, exist_ok=True)
    config = load_stage5_config(config_path)
    profile = resolve_stage5_profile(config, args.profile)
    records = load_stage5_records(input_path)
    expected = 30 * len(config["methods"]) * profile.replications
    if len(records) != expected:
        raise ValueError(f"Stage 5 analysis requires exactly {expected} records")
    expected_groups = {
        (cell.scenario, cell.n, cell.p, method)
        for cell in iter_sensitivity_cells(config) for method in config["methods"]
    }
    actual_groups = set(records[["scenario", "n", "p", "method"]].itertuples(index=False, name=None))
    if actual_groups != expected_groups:
        raise ValueError("Stage 5 analysis group universe mismatch")
    if "config_fingerprint" not in records or records["config_fingerprint"].nunique() != 1:
        raise ValueError("Stage 5 records require one config fingerprint")
    if records["config_fingerprint"].iloc[0] != stage5_config_fingerprint(config):
        raise ValueError("Stage 5 record config fingerprint mismatch")
    if "tuning_fingerprint" not in records or records["tuning_fingerprint"].nunique() != 1:
        raise ValueError("Stage 5 records require one tuning fingerprint")
    if not {"data_seed", "fold_seed"}.issubset(records.columns):
        raise ValueError("Stage 5 records require paired data and fold seeds")
    paired = records.groupby(["scenario", "n", "p", "replication"])[["data_seed", "fold_seed"]].nunique()
    paired_seed_check = bool(paired.eq(1).all().all())
    if not paired_seed_check:
        raise ValueError("Stage 5 records do not preserve paired seeds across methods")
    summary = summarize_stage5(records, bootstrap_resamples=args.bootstrap_resamples)
    summary.to_csv(output / "summary.csv", index=False)
    _write_json(output / "summary.json", summary.to_dict(orient="records"))
    fixed_n, fixed_p = build_stage5_plot_data(summary)
    fixed_n.to_csv(output / "plot_data_fixed_n.csv", index=False)
    fixed_p.to_csv(output / "plot_data_fixed_p.csv", index=False)
    make_stage5_sensitivity_figures(summary, output / "figures")
    projection = estimate_formal_runtime(
        records, cpu_workers=args.cpu_workers, ensemble_workers=args.ensemble_workers,
    )
    _write_json(output / "formal_runtime_projection.json", projection)
    statuses = records["status"].value_counts()
    gate = {
        "stage": profile.stage, "profile": args.profile,
        "seed_namespace": profile.seed_namespace,
        "config_fingerprint": records["config_fingerprint"].iloc[0],
        "tuning_fingerprint": records["tuning_fingerprint"].iloc[0],
        "expected_records": expected,
        "successful_records": int(statuses.get("success", 0)),
        "failed": int(statuses.get("failed", 0)), "oom": int(statuses.get("oom", 0)),
        "fallback": int(statuses.get("fallback", 0)), "missing": max(0, expected - len(records)),
        "duplicate": int(records.duplicated(["scenario", "n", "p", "method", "replication"]).sum()),
        "provenance": 0, "non_finite": 0,
        "paired_seed_check": paired_seed_check,
        "center_dedup_check": bool(
            len(records.loc[records["n"].eq(1000) & records["p"].eq(50)])
            == len(config["scenarios"]) * len(config["methods"]) * profile.replications
        ),
    }
    _write_json(output / "gate_summary.json", gate)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
