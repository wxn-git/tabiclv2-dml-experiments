import json
import os
import sys

import pytest

from scripts import environment_report


def test_explicit_output_preserves_default(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    default = tmp_path / "results/environment.json"
    default.parent.mkdir()
    default.write_text("historical", encoding="utf-8")
    output = tmp_path / "isolated/env.json"
    monkeypatch.setattr(environment_report, "collect_environment", lambda: {"python": "test"})
    monkeypatch.setattr(sys, "argv", ["environment_report.py", "--output", str(output)])
    environment_report.main()
    assert json.loads(output.read_text()) == {"python": "test"}
    assert default.read_text() == "historical"


def test_default_output_unchanged(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(environment_report, "collect_environment", lambda: {"python": "test"})
    monkeypatch.setattr(sys, "argv", ["environment_report.py"])
    environment_report.main()
    assert json.loads((tmp_path / "results/environment.json").read_text()) == {"python": "test"}


def test_replace_failure_preserves_existing_environment_report(monkeypatch, tmp_path):
    output = tmp_path / "environment.json"
    output.write_text("historical", encoding="utf-8")
    monkeypatch.setattr(environment_report, "collect_environment", lambda: {"python": "new"})
    monkeypatch.setattr(sys, "argv", ["environment_report.py", "--output", str(output)])
    monkeypatch.setattr(os, "replace", lambda *_: (_ for _ in ()).throw(OSError("injected")))

    with pytest.raises(OSError, match="injected"):
        environment_report.main()

    assert output.read_text(encoding="utf-8") == "historical"
    assert list(tmp_path.iterdir()) == [output]
