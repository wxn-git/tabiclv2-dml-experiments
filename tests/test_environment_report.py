import json
import sys

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
