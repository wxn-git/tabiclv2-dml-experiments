from __future__ import annotations

import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path


def collect_environment() -> dict:
    packages = {}
    for name in ("numpy", "pandas", "scipy", "scikit-learn", "xgboost", "torch", "tabicl", "doubleml"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    gpu = None
    try:
        gpu = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            text=True,
            timeout=10,
        ).strip()
    except Exception:
        pass
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
        "cuda": packages.get("torch"),
        "gpu": gpu,
    }


def main():
    output = Path("results/environment.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(collect_environment(), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {output}.")


if __name__ == "__main__":
    main()

