from __future__ import annotations

import argparse

from tabdml.stage4_publish import publish_stage4


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate and publish only full 10/20/100 Stage 4 results.")
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--destination", required=True)
    parser.add_argument("--expected-replications", type=int, default=100)
    parser.add_argument("--replace", action="store_true", help="Replace only an intact Stage 4 publication")
    parser.add_argument("--config", dest="config_path")
    for name in ("structure-dir", "tuned-models", "selected-cells", "analysis-dir",
                 "tuning-root", "screening-root", "confirmation-root"):
        parser.add_argument(f"--{name}", help="Explicit path, relative to current working directory")
    args = vars(parser.parse_args())
    publish_stage4(**args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
