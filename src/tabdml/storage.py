from __future__ import annotations

import json
import os
from pathlib import Path

from .config import TaskSpec


class ResultStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, task_or_key: TaskSpec | str | object) -> Path:
        key = task_or_key if isinstance(task_or_key, str) else task_or_key.key
        return self.root / f"{key}.json"

    def exists(self, task: TaskSpec | object) -> bool:
        path = self._path(task)
        if not path.exists():
            return False
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle).get("status") == "success"

    def write(self, record: dict) -> Path:
        path = self._path(str(record["task_key"]))
        temporary = path.with_suffix(".json.tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(record, handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        return path

    def read_all(self) -> list[dict]:
        records = []
        for path in sorted(self.root.glob("*.json")):
            with path.open("r", encoding="utf-8") as handle:
                records.append(json.load(handle))
        return records
