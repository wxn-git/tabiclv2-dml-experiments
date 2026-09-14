from __future__ import annotations

import json
from pathlib import Path

from tabdml.stage5_config import load_stage5_config, stage5_config_fingerprint
from tabdml.stage5_migration import rebind_stage5_tuning
from tabdml.stage5_tuning import load_stage5_tuning


def test_rebind_tuning_changes_only_config_derived_provenance(tmp_path):
    old = load_stage5_config(Path("configs/stage5_sensitivity.yaml"))
    new = load_stage5_config(Path("configs/stage5_sensitivity_five.yaml"))
    source = Path("results/stage5/tuning/frozen-full.json")
    destination = tmp_path / "frozen-full.json"
    rebound = rebind_stage5_tuning(old, new, source, destination, "full")
    validated = load_stage5_tuning(destination, new, "full")
    assert rebound == validated
    assert validated["config_fingerprint"] == stage5_config_fingerprint(new)
    original = json.loads(source.read_text(encoding="utf-8"))
    assert validated["scenarios"] == original["scenarios"]
    assert validated["candidate_ranking"] == original["candidate_ranking"]
