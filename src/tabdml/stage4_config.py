from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Any

import yaml


_KNOWN_STRUCTURES = frozenset(
    {"tree_stumps", "tree_hierarchical", "tree_forest_sum"}
)
_PANEL_ORDER = ("standard", "small_n_high_p")
_REQUIRED_TOP_LEVEL = frozenset(
    {
        "theta0",
        "folds",
        "structures",
        "panels",
        "tuning",
        "screening",
        "confirmation",
        "extra_trees",
    }
)


@dataclass(frozen=True)
class TreeBenchmarkCell:
    panel: str
    scenario: str
    n: int
    p: int

    @property
    def key(self) -> str:
        return f"{self.panel}__{self.scenario}__n{self.n}__p{self.p}"


def _require_mapping(value: Any, location: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{location} must be a mapping")
    return value


def _require_sequence(value: Any, location: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{location} must be a sequence")
    return value


def _require_fields(
    mapping: Mapping[str, Any], required: set[str] | frozenset[str], location: str
) -> None:
    missing = required.difference(mapping)
    if missing:
        names = ", ".join(sorted(missing))
        raise ValueError(f"{location} is missing required fields: {names}")


def _require_positive_int(value: Any, location: str, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{location} must be an integer >= {minimum}")
    return value


def _validate_stage4_config(config: Mapping[str, Any]) -> None:
    _require_fields(config, _REQUIRED_TOP_LEVEL, "config required sections")

    theta0 = config["theta0"]
    if isinstance(theta0, bool) or not isinstance(theta0, Real):
        raise ValueError("theta0 must be numeric")
    _require_positive_int(config["folds"], "folds", minimum=2)

    structures = _require_sequence(config["structures"], "structures")
    if any(not isinstance(structure, str) for structure in structures):
        raise ValueError("unknown structures: structure names must be strings")
    if len(structures) != len(set(structures)):
        raise ValueError("duplicate structures are not allowed")
    unknown = set(structures).difference(_KNOWN_STRUCTURES)
    if unknown:
        raise ValueError(f"unknown structures: {', '.join(sorted(unknown))}")
    if set(structures) != _KNOWN_STRUCTURES:
        raise ValueError("structures must contain all known Stage 4 scenarios")

    panels = _require_mapping(config["panels"], "panels")
    if set(panels) != set(_PANEL_ORDER):
        raise ValueError("panels must contain exactly standard and small_n_high_p")
    panel_values: dict[str, tuple[Sequence[Any], Sequence[Any]]] = {}
    for panel_name in _PANEL_ORDER:
        panel = _require_mapping(panels[panel_name], f"panels.{panel_name}")
        _require_fields(
            panel,
            {"sample_sizes", "dimensions"},
            f"panels.{panel_name}",
        )
        sample_sizes = _require_sequence(
            panel["sample_sizes"], f"panels.{panel_name}.sample_sizes"
        )
        dimensions = _require_sequence(
            panel["dimensions"], f"panels.{panel_name}.dimensions"
        )
        if not sample_sizes or not dimensions:
            raise ValueError(f"panels.{panel_name} grid values must not be empty")
        for n in sample_sizes:
            _require_positive_int(n, f"panels.{panel_name}.n")
        for p in dimensions:
            if isinstance(p, bool) or not isinstance(p, int) or p < 10:
                raise ValueError("Stage 4 dimensions require p >= 10")
        panel_values[panel_name] = (sample_sizes, dimensions)

    tuning = _require_mapping(config["tuning"], "tuning")
    _require_fields(
        tuning,
        {
            "stage",
            "seed_namespace",
            "replications",
            "validation_fraction",
            "targets",
            "xgboost_candidates",
        },
        "tuning",
    )
    _require_positive_int(tuning["replications"], "tuning replications")
    fraction = tuning["validation_fraction"]
    if (
        isinstance(fraction, bool)
        or not isinstance(fraction, Real)
        or not 0 < fraction < 1
    ):
        raise ValueError("validation_fraction must be strictly between 0 and 1")
    candidates = _require_sequence(
        tuning["xgboost_candidates"], "tuning.xgboost_candidates"
    )
    candidate_names = []
    for index, candidate_value in enumerate(candidates):
        candidate = _require_mapping(
            candidate_value, f"tuning.xgboost_candidates[{index}]"
        )
        _require_fields(
            candidate,
            {"name", "params"},
            f"tuning.xgboost_candidates[{index}]",
        )
        name = candidate["name"]
        if not isinstance(name, str) or not name:
            raise ValueError("XGBoost candidate names must be nonempty strings")
        _require_mapping(
            candidate["params"], f"tuning.xgboost_candidates[{index}].params"
        )
        candidate_names.append(name)
    if len(candidate_names) != len(set(candidate_names)):
        raise ValueError("XGBoost candidate names must be unique")

    screening = _require_mapping(config["screening"], "screening")
    _require_fields(
        screening,
        {"stage", "seed_namespace", "replications", "methods"},
        "screening",
    )
    _require_positive_int(screening["replications"], "screening replications")

    confirmation = _require_mapping(config["confirmation"], "confirmation")
    _require_fields(
        confirmation,
        {
            "stage",
            "seed_namespace",
            "smoke_replications",
            "replications",
            "methods",
        },
        "confirmation",
    )
    _require_positive_int(
        confirmation["smoke_replications"], "confirmation smoke replications"
    )
    _require_positive_int(
        confirmation["replications"], "confirmation replications"
    )

    extra_trees = _require_mapping(config["extra_trees"], "extra_trees")
    _require_fields(extra_trees, {"params"}, "extra_trees")
    _require_mapping(extra_trees["params"], "extra_trees.params")

    identities = [
        (panel_name, scenario, n, p)
        for panel_name, (sample_sizes, dimensions) in panel_values.items()
        for scenario in structures
        for n in sample_sizes
        for p in dimensions
    ]
    if len(identities) != len(set(identities)):
        raise ValueError("panel grids produce duplicate cells")


def load_stage4_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    config = _require_mapping(raw, "config")
    _validate_stage4_config(config)
    return dict(config)


def iter_tree_cells(config: Mapping[str, Any]) -> tuple[TreeBenchmarkCell, ...]:
    config = _require_mapping(config, "config")
    _validate_stage4_config(config)
    structures = config["structures"]
    panels = config["panels"]
    return tuple(
        TreeBenchmarkCell(panel_name, scenario, n, p)
        for panel_name in _PANEL_ORDER
        for scenario in structures
        for n in panels[panel_name]["sample_sizes"]
        for p in panels[panel_name]["dimensions"]
    )
