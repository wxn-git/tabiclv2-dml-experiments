from copy import deepcopy
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest
import yaml

from tabdml.stage5_config import (
    SensitivityCell,
    iter_sensitivity_cells,
    load_stage5_config,
    resolve_stage5_profile,
    stage5_config_fingerprint,
)


CONFIG = Path("configs/stage5_sensitivity.yaml")


def _write_config(tmp_path, config):
    path = tmp_path / "stage5.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def test_stage5_config_enumerates_thirty_unique_cells_with_six_centers():
    cells = iter_sensitivity_cells(load_stage5_config(CONFIG))

    assert len(cells) == 30
    assert len({cell.key for cell in cells}) == 30
    assert sum((cell.n, cell.p) == (1000, 50) for cell in cells) == 6


def test_five_method_config_has_exact_protocol_and_counts():
    five = load_stage5_config(Path("configs/stage5_sensitivity_five.yaml"))
    assert five["methods"] == [
        "tabiclv2_1", "tabiclv2_8", "xgboost_tuned", "extra_trees", "lasso"
    ]
    assert len(iter_sensitivity_cells(five)) == 30


def test_linear_scenario_has_exact_cross_shaped_sensitivity_grid():
    cells = iter_sensitivity_cells(load_stage5_config(CONFIG))

    assert {(cell.n, cell.p) for cell in cells if cell.scenario == "linear"} == {
        (1000, 10),
        (1000, 50),
        (1000, 100),
        (500, 50),
        (2000, 50),
    }


def test_cells_are_immutable_canonical_and_deterministically_ordered():
    config = load_stage5_config(CONFIG)
    cells = iter_sensitivity_cells(config)

    assert cells == iter_sensitivity_cells(deepcopy(config))
    assert cells[0].key == "linear__n1000__p10"
    assert cells[-1].key == "tree_forest_sum__n2000__p50"
    with pytest.raises(FrozenInstanceError):
        SensitivityCell("linear", 1000, 50).n = 500


def test_profiles_preserve_exact_counts_names_and_unique_namespaces():
    config = load_stage5_config(CONFIG)
    profiles = [
        resolve_stage5_profile(config, name)
        for name in ("smoke", "preflight", "formal")
    ]

    assert [profile.name for profile in profiles] == [
        "stage5_smoke",
        "stage5_preflight",
        "stage5_formal",
    ]
    assert [profile.replications for profile in profiles] == [1, 5, 100]
    assert [profile.full_settings for profile in profiles] == [False, True, True]
    assert len({profile.seed_namespace for profile in profiles}) == 3
    assert len(
        {config["tuning"]["seed_namespace"]}
        | {profile.seed_namespace for profile in profiles}
    ) == 4


def test_profile_with_extra_field_fails_closed_with_useful_value_error(tmp_path):
    config = load_stage5_config(CONFIG)
    config["profiles"]["smoke"]["unexpected"] = "value"

    with pytest.raises(ValueError, match="profiles.smoke.*unexpected fields.*unexpected"):
        load_stage5_config(_write_config(tmp_path, config))


def test_reordered_profile_mapping_keys_preserve_semantic_behavior(tmp_path):
    config = load_stage5_config(CONFIG)
    config["profiles"] = {
        key: config["profiles"][key]
        for key in ("formal", "smoke", "preflight")
    }

    loaded = load_stage5_config(_write_config(tmp_path, config))

    assert [
        resolve_stage5_profile(loaded, name)
        for name in ("smoke", "preflight", "formal")
    ] == [
        resolve_stage5_profile(load_stage5_config(CONFIG), name)
        for name in ("smoke", "preflight", "formal")
    ]
    assert stage5_config_fingerprint(loaded) == stage5_config_fingerprint(
        load_stage5_config(CONFIG)
    )


@pytest.mark.parametrize("profile", ["smoke", "preflight", "formal"])
def test_profile_override_may_only_equal_configured_contract(profile):
    config = load_stage5_config(CONFIG)
    expected = config["profiles"][profile]["replications"]

    assert resolve_stage5_profile(config, profile, expected).replications == expected


@pytest.mark.parametrize(
    ("profile", "override"),
    [("smoke", 2), ("preflight", 1), ("preflight", 4), ("formal", 1),
     ("formal", 99), ("formal", True), ("formal", 100.0)],
)
def test_replication_overrides_fail_closed(profile, override):
    config = load_stage5_config(CONFIG)

    with pytest.raises(ValueError, match=f"{profile}.*exactly"):
        resolve_stage5_profile(config, profile, override)


def test_unknown_profile_fails_closed():
    with pytest.raises(ValueError, match="profile.*smoke.*preflight.*formal"):
        resolve_stage5_profile(load_stage5_config(CONFIG), "quick")


@pytest.mark.parametrize(
    "section",
    ["theta0", "folds", "scenarios", "sweeps", "methods", "tuning", "profiles", "extra_trees"],
)
def test_missing_required_sections_fail_closed(tmp_path, section):
    config = load_stage5_config(CONFIG)
    del config[section]

    with pytest.raises(ValueError, match="required sections"):
        load_stage5_config(_write_config(tmp_path, config))


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("theta0",), True, "theta0"),
        (("folds",), True, "folds"),
        (("folds",), 0, "folds"),
        (("sweeps", "dimension", "n"), True, "dimension.n"),
        (("sweeps", "dimension", "p"), [10, 0, 100], "dimension.p"),
        (("tuning", "replications"), True, "tuning.replications"),
        (("tuning", "validation_fraction"), True, "validation_fraction"),
        (("profiles", "formal", "replications"), 0, "formal.replications"),
    ],
)
def test_native_numeric_types_and_positive_values_are_required(tmp_path, path, value, message):
    config = load_stage5_config(CONFIG)
    target = config
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value

    with pytest.raises(ValueError, match=message):
        load_stage5_config(_write_config(tmp_path, config))


@pytest.mark.parametrize(
    ("field", "mutation", "message"),
    [
        ("scenarios", lambda values: values.__setitem__(0, "unknown"), "scenarios.*exact"),
        ("scenarios", lambda values: values.append(values[0]), "scenarios.*duplicate"),
        ("scenarios", lambda values: values.reverse(), "scenarios.*order"),
        ("methods", lambda values: values.__setitem__(0, "oracle"), "methods.*exact"),
        ("methods", lambda values: values.append(values[0]), "methods.*duplicate"),
        ("methods", lambda values: values.reverse(), "methods.*order"),
    ],
)
def test_scenarios_and_methods_require_exact_duplicate_free_order(tmp_path, field, mutation, message):
    config = load_stage5_config(CONFIG)
    mutation(config[field])

    with pytest.raises(ValueError, match=message):
        load_stage5_config(_write_config(tmp_path, config))


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("scenarios", 0), {"linear": True}, "scenarios.*strings"),
        (("methods", 0), ["tabiclv2_1"], "methods.*strings"),
        (("sweeps", "dimension", "p", 0), [10], "dimension.p"),
    ],
)
def test_unhashable_values_fail_closed_with_value_error(tmp_path, path, value, message):
    config = load_stage5_config(CONFIG)
    target = config
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value

    with pytest.raises(ValueError, match=message):
        load_stage5_config(_write_config(tmp_path, config))


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("sweeps", "dimension", "n"), 999, "exact prescribed grid"),
        (("sweeps", "dimension", "p"), [10, 50, 90], "exact prescribed grid"),
        (("sweeps", "sample_size", "p"), 40, "exact prescribed grid"),
        (("sweeps", "sample_size", "n"), [500, 1000, 1500], "exact prescribed grid"),
        (("sweeps", "dimension", "p"), [10, 50, 50], "duplicate"),
        (("tuning", "center"), {"n": 500, "p": 50}, "center.*1000.*50"),
    ],
)
def test_grids_and_tuning_center_are_frozen(tmp_path, path, value, message):
    config = load_stage5_config(CONFIG)
    target = config
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value

    with pytest.raises(ValueError, match=message):
        load_stage5_config(_write_config(tmp_path, config))


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("tuning", "stage"), "wrong", "tuning.stage"),
        (("tuning", "seed_namespace"), "stage5_smoke_v1", "namespaces.*differ"),
        (("tuning", "replications"), 9, "tuning.replications"),
        (("tuning", "validation_fraction"), 0.2, "validation_fraction"),
        (("tuning", "targets"), ["m", "l"], "targets.*order"),
        (("profiles", "smoke", "name"), "smoke", "smoke.name"),
        (("profiles", "preflight", "full_settings"), False, "preflight.full_settings"),
        (("profiles", "formal", "replications"), 99, "formal.replications"),
    ],
)
def test_tuning_and_profile_contract_values_are_frozen(tmp_path, path, value, message):
    config = load_stage5_config(CONFIG)
    target = config
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value

    with pytest.raises(ValueError, match=message):
        load_stage5_config(_write_config(tmp_path, config))


@pytest.mark.parametrize("mutation", ["name", "order", "value", "missing", "extra"])
def test_xgboost_candidates_match_exact_stage4_contract(tmp_path, mutation):
    config = load_stage5_config(CONFIG)
    candidates = config["tuning"]["xgboost_candidates"]
    if mutation == "name":
        candidates[0]["name"] = "wrong"
    elif mutation == "order":
        candidates[0], candidates[1] = candidates[1], candidates[0]
    elif mutation == "value":
        candidates[0]["params"]["max_depth"] = 9
    elif mutation == "missing":
        del candidates[0]["params"]["tree_method"]
    else:
        candidates[0]["params"]["gamma"] = 0

    with pytest.raises(ValueError, match="XGBoost candidate"):
        load_stage5_config(_write_config(tmp_path, config))


@pytest.mark.parametrize("mutation", ["value", "missing", "extra"])
def test_extra_trees_parameters_match_exact_stage4_contract(tmp_path, mutation):
    config = load_stage5_config(CONFIG)
    params = config["extra_trees"]["params"]
    if mutation == "value":
        params["n_estimators"] = 500
    elif mutation == "missing":
        del params["max_features"]
    else:
        params["max_depth"] = None

    with pytest.raises(ValueError, match="ExtraTrees.*exact"):
        load_stage5_config(_write_config(tmp_path, config))


def test_fingerprint_is_mapping_order_independent_and_sensitive_to_values():
    config = load_stage5_config(CONFIG)
    reordered = {
        key: config[key]
        for key in reversed(tuple(config))
    }
    reordered["tuning"] = {
        key: config["tuning"][key]
        for key in reversed(tuple(config["tuning"]))
    }
    changed = deepcopy(config)
    changed["theta0"] = 2.0

    assert stage5_config_fingerprint(config) == stage5_config_fingerprint(reordered)
    assert stage5_config_fingerprint(config) != stage5_config_fingerprint(changed)
