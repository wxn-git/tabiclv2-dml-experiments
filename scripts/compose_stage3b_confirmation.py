from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from tabdml.nuisance_cache import NuisanceCache
from tabdml.stage3b import Stage3BPairSpec, build_nuisance_spec, compose_dml_record
from tabdml.storage import ResultStore


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/stage3b_tree_publication.yaml")
    parser.add_argument("--selected-models", default="results/stage3b_screening/selected_models.json")
    parser.add_argument("--cache-root", default="results/stage3b_cache_confirmation")
    parser.add_argument("--output-root", default="results/stage3b_confirmation_raw")
    parser.add_argument("--replications", type=int, default=50)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    with Path(args.config).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    selected = json.loads(Path(args.selected_models).read_text(encoding="utf-8"))
    chosen = config["selected_configuration"]
    confirmation = config["confirmation"]
    cache = NuisanceCache(args.cache_root)
    output = ResultStore(args.output_root)

    def config_hash(learner: str) -> str:
        return str(selected.get(learner, {}).get("config_hash") or "default")

    for replication in range(args.replications):
        for learner_l in confirmation["learner_l"]:
            for learner_m in confirmation["learner_m"]:
                pair = Stage3BPairSpec(
                    stage=str(confirmation["stage"]),
                    seed_namespace=str(confirmation["seed_namespace"]),
                    scenario=str(chosen["scenario"]),
                    n=int(chosen["n"]),
                    p=int(chosen["p"]),
                    replication=replication,
                    learner_l=str(learner_l),
                    learner_m=str(learner_m),
                    folds_count=int(config["folds"]),
                    theta0=float(config["theta0"]),
                    learner_l_config_hash=config_hash(str(learner_l)),
                    learner_m_config_hash=config_hash(str(learner_m)),
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
                output.write(record)
                print(pair.key, record["status"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
