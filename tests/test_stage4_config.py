from copy import deepcopy
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest
import yaml

from tabdml.stage4_config import TreeBenchmarkCell, iter_tree_cells, load_stage4_config


CONFIG = Path("configs/stage4_tree_benchmark.yaml")


def _write_config(tmp_path, config):
    path = tmp_path / "stage4.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def test_stage4_config_enumerates_exactly_two_twelve_cell_panels():
    cells = iter_tree_cells(load_stage4_config(CONFIG))
    assert len(cells) == 24
    assert len({cell.key for cell in cells}) == 24
    assert sum(cell.panel == "standard" for cell in cells) == 12
    assert sum(cell.panel == "small_n_high_p" for cell in cells) == 12
    assert {cell.scenario for cell in cells} == {
        "tree_stumps",
        "tree_hierarchical",
        "tree_forest_sum",
    }


def test_stage4_config_keeps_panel_ranges_disjoint():
    cells = iter_tree_cells(load_stage4_config(CONFIG))
    assert {(cell.n, cell.p) for cell in cells if cell.panel == "standard"} == {
        (1000, 10), (1000, 50), (2000, 10), (2000, 50)
    }
    assert {(cell.n, cell.p) for cell in cells if cell.panel == "small_n_high_p"} == {
        (300, 50), (300, 100), (500, 50), (500, 100)
    }


def test_invalid_duplicate_or_low_dimension_cell_is_rejected(tmp_path):
    config = load_stage4_config(CONFIG)
    config["panels"]["standard"]["dimensions"] = [9]
    with pytest.raises(ValueError, match="p >= 10"):
        iter_tree_cells(config)


def test_tree_benchmark_cell_is_immutable_and_has_canonical_key():
    cell = TreeBenchmarkCell("standard", "tree_stumps", 1000, 10)

    assert cell.key == "standard__tree_stumps__n1000__p10"
    with pytest.raises(FrozenInstanceError):
        cell.n = 2000


def test_tree_cell_enumeration_is_deterministic():
    config = load_stage4_config(CONFIG)

    first = iter_tree_cells(config)
    second = iter_tree_cells(deepcopy(config))

    assert first == second
    assert first[0].key == "standard__tree_stumps__n1000__p10"
    assert first[-1].key == "small_n_high_p__tree_forest_sum__n500__p100"


@pytest.mark.parametrize(
    "section",
    [
        "theta0",
        "folds",
        "structures",
        "panels",
        "tuning",
        "screening",
        "confirmation",
        "extra_trees",
    ],
)
def test_load_rejects_missing_top_level_sections(tmp_path, section):
    config = load_stage4_config(CONFIG)
    del config[section]

    with pytest.raises(ValueError, match="required sections"):
        load_stage4_config(_write_config(tmp_path, config))


def test_load_rejects_missing_nested_sections(tmp_path):
    config = load_stage4_config(CONFIG)
    del config["tuning"]["xgboost_candidates"]

    with pytest.raises(ValueError, match="tuning.*required fields"):
        load_stage4_config(_write_config(tmp_path, config))


def test_duplicate_structures_are_rejected():
    config = load_stage4_config(CONFIG)
    config["structures"].append("tree_stumps")

    with pytest.raises(ValueError, match="duplicate structures"):
        iter_tree_cells(config)


def test_unknown_structures_are_rejected():
    config = load_stage4_config(CONFIG)
    config["structures"][0] = "xor"

    with pytest.raises(ValueError, match="unknown structures"):
        iter_tree_cells(config)


def test_nonstring_structures_are_rejected_as_unknown():
    config = load_stage4_config(CONFIG)
    config["structures"][0] = {"tree_stumps": True}

    with pytest.raises(ValueError, match="unknown structures"):
        iter_tree_cells(config)


def test_duplicate_xgboost_candidate_names_are_rejected():
    config = load_stage4_config(CONFIG)
    config["tuning"]["xgboost_candidates"][1]["name"] = "xgb_d1_lr003"

    with pytest.raises(ValueError, match="candidate names"):
        iter_tree_cells(config)


@pytest.mark.parametrize("folds", [1, 0, -1, True])
def test_invalid_folds_are_rejected(folds):
    config = load_stage4_config(CONFIG)
    config["folds"] = folds

    with pytest.raises(ValueError, match="folds"):
        iter_tree_cells(config)


@pytest.mark.parametrize(
    ("section", "field"),
    [
        ("tuning", "replications"),
        ("screening", "replications"),
        ("confirmation", "smoke_replications"),
        ("confirmation", "replications"),
    ],
)
@pytest.mark.parametrize("value", [0, -1, True])
def test_nonpositive_replications_are_rejected(section, field, value):
    config = load_stage4_config(CONFIG)
    config[section][field] = value

    with pytest.raises(ValueError, match="replications"):
        iter_tree_cells(config)


@pytest.mark.parametrize("fraction", [0, 1, -0.1, 1.1, True])
def test_invalid_validation_fractions_are_rejected(fraction):
    config = load_stage4_config(CONFIG)
    config["tuning"]["validation_fraction"] = fraction

    with pytest.raises(ValueError, match="validation_fraction"):
        iter_tree_cells(config)


@pytest.mark.parametrize("field", ["sample_sizes", "dimensions"])
def test_duplicate_panel_values_are_rejected(field):
    config = load_stage4_config(CONFIG)
    values = config["panels"]["standard"][field]
    values.append(values[0])

    with pytest.raises(ValueError, match="duplicate cells"):
        iter_tree_cells(config)
