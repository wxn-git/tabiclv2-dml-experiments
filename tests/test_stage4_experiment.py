from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from tabdml.nuisance_cache import CachedNuisanceResult, NuisanceCache
from tabdml.sharding import belongs_to_shard
from tabdml.stage3b_screen import _params_hash
from tabdml.stage4_config import load_stage4_config
from tabdml.stage4_experiment import (
    Stage4PairSpec,
    build_stage4_nuisance_spec,
    compose_stage4_record,
    fit_stage4_nuisance,
    iter_stage4_pairs,
    resolve_method,
    validate_frozen_tuning,
)


CONFIG = Path("configs/stage4_tree_benchmark.yaml")


def make_pair(
    panel="standard",
    learner_l="xgboost",
    learner_m="xgboost",
    execution_profile="full",
):
    return Stage4PairSpec(
        stage="stage4_tree_screening",
        seed_namespace="stage4_tree_screening",
        panel=panel,
        scenario="tree_stumps",
        n=80,
        p=10,
        replication=0,
        learner_l=learner_l,
        learner_m=learner_m,
        folds_count=2,
        theta0=1.0,
        execution_profile=execution_profile,
    )


@pytest.fixture
def frozen():
    return {
        "cells": {
            "standard__tree_stumps__n80__p10": {
                "l": {
                    "learner_kind": "xgboost",
                    "params": {"max_depth": 1, "n_estimators": 20},
                    "config_hash": "l-hash",
                },
                "m": {
                    "learner_kind": "xgboost",
                    "params": {"max_depth": 2, "n_estimators": 20},
                    "config_hash": "m-hash",
                },
            }
        }
    }


@pytest.fixture
def config():
    return load_stage4_config(CONFIG)


def frozen_for_config(config, execution_profile="full"):
    candidate = config["tuning"]["xgboost_candidates"][0]
    nominal_params = dict(candidate["params"])
    params = dict(nominal_params)
    if execution_profile == "fast":
        params["n_estimators"] = 20
    cells = {}
    for panel, panel_config in config["panels"].items():
        for scenario in config["structures"]:
            for n in panel_config["sample_sizes"]:
                for p in panel_config["dimensions"]:
                    cell_key = f"{panel}__{scenario}__n{n}__p{p}"
                    cells[cell_key] = {}
                    for target in ("l", "m"):
                        cells[cell_key][target] = {
                            "candidate": candidate["name"],
                            "learner_kind": "xgboost",
                            "execution_profile": execution_profile,
                            "nominal_params": nominal_params,
                            "nominal_config_hash": _params_hash(nominal_params),
                            "params": params,
                            "config_hash": _params_hash(params),
                            "replications": 1,
                            "mean_validation_observed_mse": 1.0,
                            "mean_validation_truth_mse_diagnostic": 1.0,
                            "selection_metric": (
                                "mean_validation_y_mse"
                                if target == "l"
                                else "mean_validation_d_mse"
                            ),
                        }
    return {
        "execution_profile": execution_profile,
        "selection_metric_l": "mean_validation_y_mse",
        "selection_metric_m": "mean_validation_d_mse",
        "expected_replications": 1,
        "cells": cells,
    }


def fail_if_called(*args, **kwargs):
    raise AssertionError("a complete nuisance cache entry must be reused")


def test_stage4_pair_key_contains_panel_profile_and_ordered_methods():
    pair = make_pair(
        panel="small_n_high_p",
        learner_l="tabiclv2_1",
        learner_m="xgboost_tuned",
        execution_profile="fast",
    )

    assert "small_n_high_p" in pair.key
    assert "__ltabiclv2_1__mxgboost_tuned__" in pair.key
    assert "profile-fast" in pair.key
    assert pair.effective_seed_namespace == (
        "stage4_tree_screening__small_n_high_p"
    )


def test_tuned_xgboost_resolves_separate_l_and_m_hashes(frozen):
    pair = make_pair(learner_l="xgboost_tuned", learner_m="xgboost_tuned")

    l_method = resolve_method(pair, "l", frozen, extra_trees_params={})
    m_method = resolve_method(pair, "m", frozen, extra_trees_params={})

    assert l_method.config_hash == "l-hash"
    assert m_method.config_hash == "m-hash"
    assert l_method.params["max_depth"] == 1
    assert m_method.params["max_depth"] == 2


def test_fast_and_full_nuisance_specs_have_distinct_cache_identity():
    full = build_stage4_nuisance_spec(make_pair(), "l")
    fast = build_stage4_nuisance_spec(
        make_pair(execution_profile="fast"), "l"
    )

    assert full.key != fast.key
    assert full.seed_namespace == "stage4_tree_screening__standard"
    assert fast.seed_namespace == full.seed_namespace


def test_cached_stage4_nuisance_is_reused(monkeypatch, tmp_path, frozen):
    pair = make_pair(learner_l="xgboost", learner_m="oracle")
    first = fit_stage4_nuisance(pair, "l", frozen, {}, tmp_path, fast=True)
    monkeypatch.setattr("tabdml.stage3b.crossfit_single_nuisance", fail_if_called)

    second = fit_stage4_nuisance(
        pair, "l", frozen, {}, tmp_path, fast=True
    )

    np.testing.assert_array_equal(first.prediction, second.prediction)


def test_frozen_tuning_validation_accepts_exact_effective_provenance(config):
    selected = frozen_for_config(config, execution_profile="fast")

    validated = validate_frozen_tuning(
        config, selected, execution_profile="fast"
    )

    assert validated is selected


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("profile", "execution_profile"),
        ("missing_cell", "cell keys"),
        ("wrong_target", "targets"),
        ("forged_hash", "config_hash"),
        ("stale_candidate", "candidate"),
    ],
)
def test_frozen_tuning_validation_rejects_contaminated_inputs(
    config, mutation, message
):
    selected = frozen_for_config(config)
    cell_key = next(iter(selected["cells"]))
    if mutation == "profile":
        selected["execution_profile"] = "fast"
    elif mutation == "missing_cell":
        selected["cells"].pop(cell_key)
    elif mutation == "wrong_target":
        selected["cells"][cell_key]["g"] = selected["cells"][cell_key].pop("l")
    elif mutation == "forged_hash":
        selected["cells"][cell_key]["l"]["config_hash"] = "forged"
    else:
        selected["cells"][cell_key]["l"]["candidate"] = "retired-candidate"

    with pytest.raises(ValueError, match=message):
        validate_frozen_tuning(config, selected, execution_profile="full")


def test_resolve_method_rejects_wrong_profile_frozen_selection(config):
    pair = make_pair(learner_l="xgboost_tuned")
    selected = frozen_for_config(config, execution_profile="fast")
    selected["cells"] = {
        "standard__tree_stumps__n80__p10": next(iter(selected["cells"].values()))
    }

    with pytest.raises(ValueError, match="execution_profile"):
        resolve_method(pair, "l", selected, extra_trees_params={})


def test_resolve_method_rejects_forged_effective_hash_with_provenance(config):
    selected = frozen_for_config(config)
    pair = next(
        pair
        for pair in iter_stage4_pairs(
            config, "screening", selected, replications=1
        )
        if pair.learner_l == pair.learner_m == "xgboost_tuned"
    )
    selected["cells"][
        f"{pair.panel}__{pair.scenario}__n{pair.n}__p{pair.p}"
    ]["l"]["config_hash"] = "forged"

    with pytest.raises(ValueError, match="config_hash"):
        resolve_method(pair, "l", selected, extra_trees_params={})


def test_corrupt_stage4_cache_is_repaired_only_when_retry_is_explicit(
    tmp_path,
):
    pair = make_pair(learner_l="oracle", learner_m="oracle")
    task = build_stage4_nuisance_spec(pair, "l")
    cache = NuisanceCache(tmp_path)
    cache.path(task).write_bytes(b"not-an-npz")

    with pytest.raises(ValueError, match="Invalid nuisance cache file"):
        fit_stage4_nuisance(pair, "l", {}, {}, tmp_path)

    repaired = fit_stage4_nuisance(
        pair,
        "l",
        {},
        {},
        tmp_path,
        retry_failed=True,
    )

    assert np.isfinite(repaired.prediction).all()
    assert len(repaired.prediction) == pair.n


def test_fast_extra_trees_resolution_hashes_effective_parameters():
    pair = make_pair(learner_l="extra_trees", execution_profile="fast")

    method = resolve_method(
        pair,
        "l",
        {},
        extra_trees_params={"n_estimators": 600, "min_samples_leaf": 2},
    )

    assert method.params["n_estimators"] == 20
    assert method.config_hash == _params_hash(method.params)


def test_stage4_pairs_enumerate_same_methods_and_four_oracle_diagnostics(config):
    selected = frozen_for_config(config)

    pairs = tuple(
        iter_stage4_pairs(
            config,
            "screening",
            selected,
            replications=1,
        )
    )

    assert len(pairs) == 24 * 10
    assert len({pair.key for pair in pairs}) == len(pairs)
    first_cell_pairs = [
        pair
        for pair in pairs
        if (pair.panel, pair.scenario, pair.n, pair.p)
        == ("standard", "tree_stumps", 1000, 10)
    ]
    assert [(pair.learner_l, pair.learner_m) for pair in first_cell_pairs] == [
        ("tabiclv2_1", "tabiclv2_1"),
        ("tabiclv2_8", "tabiclv2_8"),
        ("xgboost", "xgboost"),
        ("xgboost_tuned", "xgboost_tuned"),
        ("extra_trees", "extra_trees"),
        ("oracle", "oracle"),
        ("oracle", "xgboost_tuned"),
        ("xgboost_tuned", "oracle"),
        ("oracle", "tabiclv2_1"),
        ("tabiclv2_1", "oracle"),
    ]


def test_stage4_pair_shards_partition_exact_pair_keys(config):
    selected = frozen_for_config(config)
    all_pairs = tuple(
        iter_stage4_pairs(config, "screening", selected, replications=1)
    )
    shards = [
        tuple(
            iter_stage4_pairs(
                config,
                "screening",
                selected,
                replications=1,
                num_shards=3,
                shard_index=index,
            )
        )
        for index in range(3)
    ]

    assert {pair.key for shard in shards for pair in shard} == {
        pair.key for pair in all_pairs
    }
    assert sum(map(len, shards)) == len(all_pairs)
    for index, shard in enumerate(shards):
        assert all(belongs_to_shard(pair.key, 3, index) for pair in shard)


def test_confirmation_pairs_require_six_exact_frozen_cells(config):
    selected = frozen_for_config(config)
    chosen = {
        "cells": [
            {
                "panel": panel,
                "scenario": scenario,
                "n": config["panels"][panel]["sample_sizes"][0],
                "p": config["panels"][panel]["dimensions"][0],
            }
            for panel in ("standard", "small_n_high_p")
            for scenario in config["structures"]
        ]
    }

    pairs = tuple(
        iter_stage4_pairs(
            config,
            "confirmation",
            selected,
            selected_confirmation=chosen,
            replications=1,
        )
    )

    assert len(pairs) == 6 * 10
    incomplete = deepcopy(chosen)
    incomplete["cells"].pop()
    with pytest.raises(ValueError, match="six selected confirmation cells"):
        tuple(
            iter_stage4_pairs(
                config,
                "confirmation",
                selected,
                selected_confirmation=incomplete,
                replications=1,
            )
        )


def test_tuned_pair_carries_different_frozen_l_and_m_hashes(config):
    selected = frozen_for_config(config)
    first_cell = next(iter(selected["cells"].values()))
    candidate = config["tuning"]["xgboost_candidates"][1]
    params = dict(candidate["params"])
    first_cell["m"].update(
        {
            "candidate": candidate["name"],
            "nominal_params": params,
            "nominal_config_hash": _params_hash(params),
            "params": params,
            "config_hash": _params_hash(params),
        }
    )

    pair = next(
        pair
        for pair in iter_stage4_pairs(
            config, "screening", selected, replications=1
        )
        if pair.learner_l == pair.learner_m == "xgboost_tuned"
    )

    assert pair.learner_l_config_hash != pair.learner_m_config_hash


def test_compose_stage4_record_preserves_panel_seed_pairing_and_finite_values(
    tmp_path,
):
    pair = make_pair(learner_l="oracle", learner_m="oracle")
    l_result = fit_stage4_nuisance(pair, "l", {}, {}, tmp_path)
    m_result = fit_stage4_nuisance(pair, "m", {}, {}, tmp_path)

    record = compose_stage4_record(pair, l_result, m_result)

    assert record["task_key"] == pair.key
    assert record["panel"] == pair.panel
    assert record["seed_namespace"] == pair.effective_seed_namespace
    assert record["execution_profile"] == "full"
    assert record["l_mse"] == 0.0
    assert record["m_mse"] == 0.0
    assert record["lm_error_cross"] == 0.0
    assert record["theta_proxy"] == 1.0
    assert record["ci_lower"] <= record["theta"] <= record["ci_upper"]


def test_compose_stage4_record_rejects_nonfinite_cached_values():
    pair = make_pair(learner_l="oracle", learner_m="oracle")
    invalid = CachedNuisanceResult(
        prediction=np.full(pair.n, np.nan),
        fold_seconds=(0.0, 0.0),
        peak_gpu_mb=None,
        fallback_reason=None,
    )

    with pytest.raises(ValueError, match="finite"):
        compose_stage4_record(pair, invalid, invalid)
