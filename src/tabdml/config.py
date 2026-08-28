from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class TaskSpec:
    stage: str
    scenario: str
    n: int
    p: int
    replication: int
    learner: str
    tabicl_estimators: int

    @property
    def key(self) -> str:
        return (
            f"{self.stage}__{self.scenario}__n{self.n}__p{self.p}"
            f"__r{self.replication:03d}__{self.learner}__e{self.tabicl_estimators}"
        )


@dataclass(frozen=True)
class ExperimentConfig:
    stage: str
    scenarios: tuple[str, ...]
    sample_sizes: tuple[int, ...]
    dimensions: tuple[int, ...]
    learners: tuple[str, ...]
    folds: int
    replications: int
    theta0: float
    tabicl_estimators: int


def derive_seed(*parts: object) -> int:
    payload = "|".join(map(str, parts)).encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, "little") % (2**32 - 1)


def load_config(path: str | Path) -> ExperimentConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw: dict[str, Any] = yaml.safe_load(handle)
    return ExperimentConfig(
        stage=str(raw["stage"]),
        scenarios=tuple(raw["scenarios"]),
        sample_sizes=tuple(int(x) for x in raw["sample_sizes"]),
        dimensions=tuple(int(x) for x in raw["dimensions"]),
        learners=tuple(raw["learners"]),
        folds=int(raw["folds"]),
        replications=int(raw["replications"]),
        theta0=float(raw["theta0"]),
        tabicl_estimators=int(raw["tabicl_estimators"]),
    )

