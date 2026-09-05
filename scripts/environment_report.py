from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import tempfile
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/environment.json")
    output = Path(parser.parse_args().output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output.parent,
            prefix=f".{output.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(
                collect_environment(),
                handle,
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    print(f"Wrote {output}.")


if __name__ == "__main__":
    main()
