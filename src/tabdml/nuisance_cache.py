from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class NuisanceTaskSpec:
    seed_namespace: str
    scenario: str
    n: int
    p: int
    replication: int
    target: str
    learner: str
    tabicl_estimators: int
    folds_count: int
    learner_seed: int
    learner_config_hash: str = "default"

    def __post_init__(self):
        if self.target not in {"l", "m"}:
            raise ValueError("target must be 'l' or 'm'.")

    @property
    def key(self) -> str:
        return (
            f"{self.seed_namespace}__{self.scenario}__n{self.n}__p{self.p}"
            f"__r{self.replication:03d}__{self.target}__{self.learner}"
            f"__e{self.tabicl_estimators}__k{self.folds_count}"
            f"__s{self.learner_seed}__h{self.learner_config_hash}"
        )


@dataclass(frozen=True)
class CachedNuisanceResult:
    prediction: NDArray[np.float64]
    fold_seconds: tuple[float, ...]
    peak_gpu_mb: float | None
    fallback_reason: str | None


class NuisanceCache:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, task: NuisanceTaskSpec) -> Path:
        return self.root / f"{task.key}.npz"

    def exists(self, task: NuisanceTaskSpec) -> bool:
        return self.path(task).exists()

    def write(
        self,
        task: NuisanceTaskSpec,
        prediction: ArrayLike,
        fold_seconds: tuple[float, ...],
        peak_gpu_mb: float | None,
        fallback_reason: str | None,
    ) -> Path:
        path = self.path(task)
        temporary = path.with_suffix(".npz.tmp")
        metadata = json.dumps(asdict(task), ensure_ascii=False, sort_keys=True)
        with temporary.open("wb") as handle:
            np.savez_compressed(
                handle,
                prediction=np.asarray(prediction, dtype=float).reshape(-1),
                fold_seconds=np.asarray(fold_seconds, dtype=float),
                peak_gpu_mb=np.asarray(
                    np.nan if peak_gpu_mb is None else float(peak_gpu_mb)
                ),
                fallback_reason=np.asarray(fallback_reason or ""),
                metadata=np.asarray(metadata),
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        return path

    def read(
        self,
        task: NuisanceTaskSpec,
        expected_length: int,
    ) -> CachedNuisanceResult:
        path = self.path(task)
        try:
            with np.load(path, allow_pickle=False) as payload:
                metadata = json.loads(str(payload["metadata"].item()))
                prediction = np.asarray(payload["prediction"], dtype=float)
                fold_seconds = tuple(
                    float(value) for value in payload["fold_seconds"].tolist()
                )
                peak_value = float(payload["peak_gpu_mb"].item())
                fallback_value = str(payload["fallback_reason"].item())
        except Exception as error:
            raise ValueError(f"Invalid nuisance cache file: {path}") from error

        if metadata != asdict(task):
            raise ValueError("Nuisance cache metadata does not match the task.")
        if prediction.ndim != 1 or len(prediction) != expected_length:
            raise ValueError("Nuisance cache prediction length is invalid.")
        if not np.isfinite(prediction).all():
            raise ValueError("Nuisance cache predictions must be finite.")
        if not np.isfinite(np.asarray(fold_seconds, dtype=float)).all():
            raise ValueError("Nuisance cache fold times must be finite.")
        return CachedNuisanceResult(
            prediction=prediction,
            fold_seconds=fold_seconds,
            peak_gpu_mb=None if np.isnan(peak_value) else peak_value,
            fallback_reason=fallback_value or None,
        )
