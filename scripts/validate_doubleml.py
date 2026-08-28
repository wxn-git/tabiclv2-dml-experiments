from __future__ import annotations

import json
from pathlib import Path

from tabdml.validation import compare_with_doubleml


def main():
    result = compare_with_doubleml()
    output = Path("results/doubleml_validation.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
