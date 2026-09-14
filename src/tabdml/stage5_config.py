from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


_SCENARIOS = (
    "linear",
    "smooth",
    "tree",
    "tree_stumps",
    "tree_hierarchical",
    "tree_forest_sum",
)
_METHODS = (
    "tabiclv2_1",
    "tabiclv2_8",
    "xgboost_tuned",
    "extra_trees",
    "lasso",
    "ensemble",
)
_FIVE_METHODS = _METHODS[:-1]
_PROFILE_ORDER = ("smoke", "preflight", "formal")
_PROFILE_CONTRACT = {
    "smoke": ("stage5_smoke", "stage5_smoke_v1", 1, False),
    "preflight": ("stage5_preflight", "stage5_preflight_v1", 5, True),
    "formal": ("stage5_formal", "stage5_formal_v1", 100, True),
}
_XGBOOST_CANDIDATES = (
    ("xgb_d1_lr003", {"n_estimators": 800, "max_depth": 1, "learning_rate": 0.03, "min_child_weight": 1, "reg_lambda": 1.0, "subsample": 0.9, "colsample_bytree": 1.0, "tree_method": "hist"}),
    ("xgb_d2_lr003", {"n_estimators": 800, "max_depth": 2, "learning_rate": 0.03, "min_child_weight": 1, "reg_lambda": 1.0, "subsample": 0.9, "colsample_bytree": 1.0, "tree_method": "hist"}),
    ("xgb_d2_lr005", {"n_estimators": 600, "max_depth": 2, "learning_rate": 0.05, "min_child_weight": 5, "reg_lambda": 1.0, "subsample": 0.9, "colsample_bytree": 1.0, "tree_method": "hist"}),
    ("xgb_d3_lr003", {"n_estimators": 800, "max_depth": 3, "learning_rate": 0.03, "min_child_weight": 5, "reg_lambda": 1.0, "subsample": 0.9, "colsample_bytree": 1.0, "tree_method": "hist"}),
    ("xgb_d3_lr005", {"n_estimators": 600, "max_depth": 3, "learning_rate": 0.05, "min_child_weight": 10, "reg_lambda": 2.0, "subsample": 0.9, "colsample_bytree": 1.0, "tree_method": "hist"}),
    ("xgb_d4_lr003", {"n_estimators": 800, "max_depth": 4, "learning_rate": 0.03, "min_child_weight": 5, "reg_lambda": 2.0, "subsample": 0.9, "colsample_bytree": 1.0, "tree_method": "hist"}),
)
_EXTRA_TREES_PARAMS = {
    "n_estimators": 600,
    "max_features": 1.0,
    "min_samples_leaf": 2,
}
_REQUIRED_TOP_LEVEL = frozenset(
    {"theta0", "folds", "scenarios", "sweeps", "methods", "tuning", "profiles", "extra_trees"}
)


@dataclass(frozen=True)
class SensitivityCell:
    scenario: str
    n: int
    p: int

    @property
    def key(self) -> str:
        return f"{self.scenario}__n{self.n}__p{self.p}"


@dataclass(frozen=True)
class Stage5Profile:
    name: str
    stage: str
    seed_namespace: str
    replications: int
    full_settings: bool


def _mapping(value: Any, location: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{location} must be a mapping")
    return value


def _sequence(value: Any, location: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{location} must be a sequence")
    return value


def _fields(mapping: Mapping[str, Any], required: set[str] | frozenset[str], location: str) -> None:
    missing = required.difference(mapping)
    if missing:
        raise ValueError(
            f"{location} is missing required fields: {', '.join(sorted(missing))}"
        )


def _positive_int(value: Any, location: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{location} must be a positive native integer")
    return value


def _exact(actual: Any, expected: Any) -> bool:
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, Mapping):
        return set(actual) == set(expected) and all(
            _exact(actual[key], expected[key]) for key in expected
        )
    if isinstance(expected, (list, tuple)):
        return len(actual) == len(expected) and all(
            _exact(left, right) for left, right in zip(actual, expected)
        )
    return actual == expected


def _validate_exact_ordered_list(value: Any, expected: tuple[str, ...], location: str) -> None:
    values = _sequence(value, location)
    if any(not isinstance(item, str) for item in values):
        raise ValueError(f"{location} values must be strings")
    if len(values) != len(set(values)):
        raise ValueError(f"{location} contains duplicate values")
    if set(values) != set(expected):
        raise ValueError(f"{location} must contain the exact prescribed values")
    if tuple(values) != expected:
        raise ValueError(f"{location} must use the exact prescribed order")


def _validate_grid(config: Mapping[str, Any]) -> tuple[tuple[int, int], ...]:
    sweeps = _mapping(config["sweeps"], "sweeps")
    _fields(sweeps, {"dimension", "sample_size"}, "sweeps")
    dimension = _mapping(sweeps["dimension"], "sweeps.dimension")
    sample_size = _mapping(sweeps["sample_size"], "sweeps.sample_size")
    _fields(dimension, {"n", "p"}, "sweeps.dimension")
    _fields(sample_size, {"n", "p"}, "sweeps.sample_size")

    dimension_n = _positive_int(dimension["n"], "sweeps.dimension.n")
    dimension_p = _sequence(dimension["p"], "sweeps.dimension.p")
    sample_n = _sequence(sample_size["n"], "sweeps.sample_size.n")
    sample_p = _positive_int(sample_size["p"], "sweeps.sample_size.p")
    for value in dimension_p:
        _positive_int(value, "sweeps.dimension.p")
    for value in sample_n:
        _positive_int(value, "sweeps.sample_size.n")
    if len(dimension_p) != len(set(dimension_p)) or len(sample_n) != len(set(sample_n)):
        raise ValueError("sweep grid values must be duplicate-free")
    if (dimension_n, tuple(dimension_p), sample_p, tuple(sample_n)) != (
        1000,
        (10, 50, 100),
        50,
        (500, 1000, 2000),
    ):
        raise ValueError("sweeps must match the exact prescribed grid")

    ordered = [(dimension_n, p) for p in dimension_p]
    ordered.extend((n, sample_p) for n in sample_n if (n, sample_p) not in ordered)
    if len(ordered) != 5:
        raise ValueError("Stage 5 requires exactly five unique cells per scenario")
    return tuple(ordered)


def _validate_tuning(config: Mapping[str, Any]) -> None:
    tuning = _mapping(config["tuning"], "tuning")
    _fields(
        tuning,
        {"stage", "seed_namespace", "center", "replications", "validation_fraction", "targets", "xgboost_candidates"},
        "tuning",
    )
    if tuning["stage"] != "stage5_tuning":
        raise ValueError("tuning.stage must be stage5_tuning")
    if tuning["seed_namespace"] != "stage5_tuning_v1":
        raise ValueError(
            "tuning.seed_namespace must be stage5_tuning_v1 and all namespaces "
            "must differ"
        )
    center = _mapping(tuning["center"], "tuning.center")
    if not _exact(center, {"n": 1000, "p": 50}):
        raise ValueError("tuning center must align with (1000, 50)")
    if _positive_int(tuning["replications"], "tuning.replications") != 10:
        raise ValueError("tuning.replications must be exactly 10")
    fraction = tuning["validation_fraction"]
    if type(fraction) is not float or fraction != 0.25:
        raise ValueError("tuning.validation_fraction must be exactly 0.25")
    targets = _sequence(tuning["targets"], "tuning.targets")
    if any(not isinstance(target, str) for target in targets):
        raise ValueError("tuning.targets values must be strings")
    if len(targets) != len(set(targets)):
        raise ValueError("tuning.targets contains duplicate values")
    if set(targets) != {"l", "m"}:
        raise ValueError("tuning.targets must contain exactly l and m")
    if tuple(targets) != ("l", "m"):
        raise ValueError("tuning.targets must use the exact prescribed order")

    candidates = _sequence(tuning["xgboost_candidates"], "tuning.xgboost_candidates")
    if len(candidates) != 6:
        raise ValueError("XGBoost candidates must contain the exact six candidates")
    for index, (actual_value, (expected_name, expected_params)) in enumerate(zip(candidates, _XGBOOST_CANDIDATES)):
        actual = _mapping(actual_value, f"tuning.xgboost_candidates[{index}]")
        if not _exact(actual, {"name": expected_name, "params": expected_params}):
            raise ValueError(
                f"XGBoost candidate {index} must match the exact Stage 4 name, order, and parameters"
            )


def _validate_profiles(config: Mapping[str, Any]) -> None:
    profiles = _mapping(config["profiles"], "profiles")
    if set(profiles) != set(_PROFILE_ORDER):
        raise ValueError("profiles must contain exactly smoke, preflight, and formal")
    namespaces = []
    for key in _PROFILE_ORDER:
        profile = _mapping(profiles[key], f"profiles.{key}")
        required_fields = {
            "name",
            "stage",
            "seed_namespace",
            "replications",
            "full_settings",
        }
        _fields(profile, required_fields, f"profiles.{key}")
        unexpected = set(profile).difference(required_fields)
        if unexpected:
            raise ValueError(
                f"profiles.{key} has unexpected fields: "
                f"{', '.join(sorted(unexpected))}"
            )
        name, namespace, replications, full_settings = _PROFILE_CONTRACT[key]
        expected = {
            "name": name,
            "stage": name,
            "seed_namespace": namespace,
            "replications": replications,
            "full_settings": full_settings,
        }
        _positive_int(profile["replications"], f"profiles.{key}.replications")
        if not _exact(profile, expected):
            differing = next(field for field in expected if not _exact(profile.get(field), expected[field]))
            raise ValueError(f"profiles.{key}.{differing} must match the exact contract")
        namespaces.append(profile["seed_namespace"])
    namespaces.append(config["tuning"]["seed_namespace"])
    if len(namespaces) != len(set(namespaces)):
        raise ValueError("tuning and profile seed namespaces must all differ")


def _validate_stage5_config(config: Mapping[str, Any]) -> None:
    _fields(config, _REQUIRED_TOP_LEVEL, "config required sections")
    theta0 = config["theta0"]
    if type(theta0) not in (int, float) or theta0 != 1.0:
        raise ValueError("theta0 must be the native numeric value 1.0")
    if _positive_int(config["folds"], "folds") != 5:
        raise ValueError("folds must be exactly 5")
    _validate_exact_ordered_list(config["scenarios"], _SCENARIOS, "scenarios")
    methods = _sequence(config["methods"], "methods")
    if any(not isinstance(item, str) for item in methods):
        raise ValueError("methods values must be strings")
    if len(methods) != len(set(methods)):
        raise ValueError("methods contains duplicate values")
    if tuple(methods) not in {_METHODS, _FIVE_METHODS}:
        raise ValueError("methods must use the exact prescribed five- or legacy six-method order")
    grid = _validate_grid(config)
    if len(grid) * len(_SCENARIOS) != 30:
        raise ValueError("Stage 5 requires exactly 30 cells overall")
    _validate_tuning(config)
    _validate_profiles(config)
    extra_trees = _mapping(config["extra_trees"], "extra_trees")
    _fields(extra_trees, {"params"}, "extra_trees")
    if not _exact(extra_trees["params"], _EXTRA_TREES_PARAMS):
        raise ValueError("ExtraTrees parameters must match the exact Stage 4 parameters")


def load_stage5_config(path: Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    config = _mapping(raw, "config")
    _validate_stage5_config(config)
    return dict(config)


def iter_sensitivity_cells(config: Mapping[str, Any]) -> tuple[SensitivityCell, ...]:
    config = _mapping(config, "config")
    _validate_stage5_config(config)
    grid = _validate_grid(config)
    return tuple(
        SensitivityCell(scenario, n, p)
        for scenario in config["scenarios"]
        for n, p in grid
    )


def resolve_stage5_profile(
    config: Mapping[str, Any], profile: str, replications: int | None = None
) -> Stage5Profile:
    config = _mapping(config, "config")
    _validate_stage5_config(config)
    if profile not in _PROFILE_ORDER:
        raise ValueError("profile must be smoke, preflight, or formal")
    value = config["profiles"][profile]
    expected = value["replications"]
    if replications is not None and (type(replications) is not int or replications != expected):
        raise ValueError(f"{profile} profile requires exactly {expected} replications")
    return Stage5Profile(
        name=value["name"],
        stage=value["stage"],
        seed_namespace=value["seed_namespace"],
        replications=expected,
        full_settings=value["full_settings"],
    )


def _canonical(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _canonical(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_canonical(item) for item in value]
    return value


def stage5_config_fingerprint(config: Mapping[str, Any]) -> str:
    config = _mapping(config, "config")
    payload = json.dumps(
        _canonical(config),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
