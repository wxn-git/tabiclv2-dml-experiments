from __future__ import annotations

import argparse
import json
from pathlib import Path

from tabdml.nuisance_cache import NuisanceCache
from tabdml.stage3 import Stage3TaskSpec
from tabdml.stage3b import Stage3BPairSpec, build_nuisance_spec, compose_dml_record
from tabdml.storage import ResultStore


PAIRS = (
    ("tabiclv2_1", "tabiclv2_1"),
    ("tabiclv2_1", "xgboost"),
    ("xgboost", "tabiclv2_1"),
    ("xgboost", "xgboost"),
    ("oracle", "tabiclv2_1"),
    ("tabiclv2_1", "oracle"),
    ("oracle", "xgboost"),
    ("xgboost", "oracle"),
    ("oracle", "oracle"),
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", default="results/stage3b_cache_batch_a")
    parser.add_argument("--output-root", default="results/stage3b_batch_a_raw")
    parser.add_argument("--stage3a-root", default="results/stage3_tree_diagnosis_raw")
    parser.add_argument("--replications", type=int, default=50)
    parser.add_argument("--n", type=int, default=2000)
    parser.add_argument("--p", type=int, default=10)
    parser.add_argument("--folds", type=int, default=5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cache = NuisanceCache(args.cache_root)
    output = ResultStore(args.output_root)
    stage3a_root = Path(args.stage3a_root)
    for replication in range(args.replications):
        for learner_l, learner_m in PAIRS:
            pair = Stage3BPairSpec(
                stage="stage3b_batch_a",
                seed_namespace="stage3_tree_diagnosis",
                scenario="tree",
                n=args.n,
                p=args.p,
                replication=replication,
                learner_l=learner_l,
                learner_m=learner_m,
                folds_count=args.folds,
                theta0=1.0,
            )
            if output.exists(pair):
                print(pair.key, "skipped", flush=True)
                continue
            l_task = build_nuisance_spec(pair, "l")
            m_task = build_nuisance_spec(pair, "m")
            record = compose_dml_record(
                pair,
                cache.read(l_task, expected_length=pair.n),
                cache.read(m_task, expected_length=pair.n),
            )
            legacy = Stage3TaskSpec(
                stage="stage3_tree_diagnosis",
                seed_namespace="stage3_tree_diagnosis",
                scenario="tree",
                n=args.n,
                p=args.p,
                replication=replication,
                learner_l=learner_l,
                learner_m=learner_m,
                tabicl_estimators=1,
            )
            legacy_path = stage3a_root / f"{legacy.key}.json"
            if legacy_path.exists():
                old = json.loads(legacy_path.read_text(encoding="utf-8"))
                record["stage3a_theta"] = old.get("theta")
                record["stage3a_theta_difference"] = record["theta"] - float(old["theta"])
            output.write(record)
            print(pair.key, record["status"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
