from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from tabdml.nuisance_cache import CachedNuisanceResult, NuisanceCache
from tabdml.config import derive_seed
from tabdml.sharding import belongs_to_shard
from tabdml.stage3b_screen import _params_hash
from tabdml.stage4_config import load_stage4_config
from tabdml.stage4_experiment import (
    STAGE4_SELECTION_RULE,
    Stage4PairSpec,
    build_stage4_nuisance_spec,
    compose_stage4_record,
    fit_stage4_nuisance,
    iter_stage4_pairs,
    resolve_method,
    stage4_configuration_fingerprint,
    validate_frozen_tuning,
    validate_stage4_selection,
    validate_stage4_resume_record,
)
from tabdml.stage4_tuning import tuning_run_fingerprint


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
    expected_replications = (
        1
        if execution_profile == "fast"
        else config["tuning"]["replications"]
    )
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
                            "replications": (
                                expected_replications
                            ),
                            "mean_validation_observed_mse": 1.0,
                            "mean_validation_truth_mse_diagnostic": 1.0,
                            "selection_metric": (
                                "mean_validation_y_mse"
                                if target == "l"
                                else "mean_validation_d_mse"
                            ),
                        }
    return {
        "tuning_stage": config["tuning"]["stage"],
        "tuning_seed_namespace": config["tuning"]["seed_namespace"],
        "tuning_run_fingerprint": tuning_run_fingerprint(
            config,
            expected_replications,
            execution_profile,
        ),
        "theta0": config["theta0"],
        "execution_profile": execution_profile,
        "selection_metric_l": "mean_validation_y_mse",
        "selection_metric_m": "mean_validation_d_mse",
        "expected_replications": expected_replications,
        "cells": cells,
    }


def selection_for_config(config, execution_profile="full"):
    ranking = []
    for panel, panel_config in config["panels"].items():
        for scenario in config["structures"]:
            for n in panel_config["sample_sizes"]:
                for p in panel_config["dimensions"]:
                    ranking.append(
                        {
                            "panel": panel,
                            "scenario": scenario,
                            "n": n,
                            "p": p,
                            "mean_paired_squared_error_difference": float(
                                n + p / 1000
                            ),
                            "selection_rule": (
                                "minimum_mean_tab_minus_xgb_squared_error"
                            ),
                        }
                    )
    groups = {
        (panel, scenario)
        for panel in config["panels"]
        for scenario in config["structures"]
    }
    cells = [
        min(
            (
                row
                for row in ranking
                if (row["panel"], row["scenario"]) == group
            ),
            key=lambda row: (
                row["mean_paired_squared_error_difference"],
                row["n"],
                row["p"],
            ),
        )
        for group in sorted(groups)
    ]
    return {
        "execution_profile": execution_profile,
        "screening_stage": config["screening"]["stage"],
        "screening_seed_namespace": config["screening"]["seed_namespace"],
        "expected_screening_replications": (
            1
            if execution_profile == "fast"
            else config["screening"]["replications"]
        ),
        "selection_rule": "minimum_mean_tab_minus_xgb_squared_error",
        "config_fingerprint": stage4_configuration_fingerprint(config),
        "screening_ranking": ranking,
        "cells": cells,
    }


def failure_record(pair, status="failed"):
    return {
        "task_key": pair.key,
        "stage": pair.stage,
        "seed_namespace": pair.effective_seed_namespace,
        "panel": pair.panel,
        "scenario": pair.scenario,
        "n": pair.n,
        "p": pair.p,
        "replication": pair.replication,
        "learner_l": pair.learner_l,
        "learner_m": pair.learner_m,
        "learner_l_config_hash": pair.learner_l_config_hash,
        "learner_m_config_hash": pair.learner_m_config_hash,
        "folds_count": pair.folds_count,
        "theta0": pair.theta0,
        "execution_profile": pair.execution_profile,
        "data_seed": derive_seed(
            pair.effective_seed_namespace,
            pair.scenario,
            pair.n,
            pair.p,
            pair.replication,
            "data",
        ),
        "fold_seed": derive_seed(
            pair.effective_seed_namespace,
            pair.scenario,
            pair.n,
            pair.p,
            pair.replication,
            "folds",
        ),
        "status": status,
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


def test_selection_contract_exports_the_fixed_task6_rule():
    assert (
        STAGE4_SELECTION_RULE
        == "minimum_mean_tab_minus_xgb_squared_error"
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


@pytest.mark.parametrize("execution_profile", ["full", "fast"])
def test_frozen_tuning_validation_accepts_exact_task4_run_provenance(
    config, execution_profile
):
    selected = frozen_for_config(config, execution_profile)

    validated = validate_frozen_tuning(
        config, selected, execution_profile=execution_profile
    )

    assert validated is selected


def test_frozen_tuning_accepts_exact_theta0_metadata(config):
    selected = frozen_for_config(config)

    assert selected["theta0"] == config["theta0"]
    assert validate_frozen_tuning(config, selected, "full") is selected


def test_frozen_tuning_rejects_missing_theta0_metadata(config):
    selected = frozen_for_config(config)
    selected.pop("theta0")

    with pytest.raises(ValueError, match="theta0"):
        validate_frozen_tuning(config, selected, "full")


@pytest.mark.parametrize(
    "invalid",
    [True, "1.0", 2.0, np.nan, np.inf, -np.inf],
)
def test_frozen_tuning_rejects_invalid_theta0_metadata(config, invalid):
    selected = frozen_for_config(config)
    selected["theta0"] = invalid

    with pytest.raises(ValueError, match="theta0"):
        validate_frozen_tuning(config, selected, "full")


@pytest.mark.parametrize("field", ["stage", "seed_namespace"])
def test_frozen_tuning_rejects_artifact_after_tuning_identity_change(
    config, field
):
    selected = frozen_for_config(config)
    changed_config = deepcopy(config)
    changed_config["tuning"][field] += "__changed"

    with pytest.raises(ValueError, match=f"tuning_{field}"):
        validate_frozen_tuning(
            changed_config,
            selected,
            execution_profile="full",
        )


@pytest.mark.parametrize(
    ("field", "invalid"),
    [
        ("tuning_stage", "forged-stage"),
        ("tuning_seed_namespace", "forged-namespace"),
        ("tuning_run_fingerprint", "forged-fingerprint"),
    ],
)
def test_frozen_tuning_rejects_forged_task4_run_provenance(
    config, field, invalid
):
    selected = frozen_for_config(config)
    selected[field] = invalid

    with pytest.raises(ValueError, match=field):
        validate_frozen_tuning(config, selected, execution_profile="full")


def test_frozen_tuning_rejects_one_replication_labeled_full(config):
    selected = frozen_for_config(config)
    selected["expected_replications"] = 1
    for targets in selected["cells"].values():
        for winner in targets.values():
            winner["replications"] = 1

    with pytest.raises(ValueError, match="expected_replications"):
        validate_frozen_tuning(config, selected, execution_profile="full")


def test_frozen_tuning_accepts_explicit_one_replication_fast_smoke(config):
    selected = frozen_for_config(config, execution_profile="fast")

    assert (
        validate_frozen_tuning(config, selected, execution_profile="fast")
        is selected
    )


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

    with pytest.raises(ValueError, match="integrity"):
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


@pytest.mark.parametrize(
    ("fold_seconds", "peak_gpu_mb"),
    [
        ((0.0,), None),
        ((0.0, -1.0), None),
        ((0.0, 0.0), np.inf),
    ],
)
def test_fit_rejects_semantically_invalid_cached_nuisance(
    tmp_path, fold_seconds, peak_gpu_mb
):
    pair = make_pair(learner_l="oracle", learner_m="oracle")
    task = build_stage4_nuisance_spec(pair, "l")
    NuisanceCache(tmp_path).write(
        task,
        np.zeros(pair.n),
        fold_seconds,
        peak_gpu_mb,
        None,
    )

    with pytest.raises(ValueError, match="integrity"):
        fit_stage4_nuisance(pair, "l", {}, {}, tmp_path)


def test_retry_repairs_semantically_invalid_cached_nuisance(tmp_path):
    pair = make_pair(learner_l="oracle", learner_m="oracle")
    task = build_stage4_nuisance_spec(pair, "l")
    cache = NuisanceCache(tmp_path)
    cache.write(task, np.zeros(pair.n), (0.0,), None, None)

    repaired = fit_stage4_nuisance(
        pair, "l", {}, {}, tmp_path, retry_failed=True
    )

    assert len(repaired.fold_seconds) == pair.folds_count
    assert all(value >= 0 for value in repaired.fold_seconds)


def test_retry_rejects_and_removes_semantically_invalid_rebuild(
    monkeypatch, tmp_path
):
    pair = make_pair(learner_l="oracle", learner_m="oracle")
    task = build_stage4_nuisance_spec(pair, "l")
    cache = NuisanceCache(tmp_path)
    cache.write(task, np.zeros(pair.n), (0.0,), None, None)
    invalid = CachedNuisanceResult(
        prediction=np.zeros(pair.n),
        fold_seconds=(0.0,),
        peak_gpu_mb=None,
        fallback_reason=None,
    )
    monkeypatch.setattr(
        "tabdml.stage4_experiment.fit_cached_nuisance",
        lambda *args, **kwargs: invalid,
    )

    with pytest.raises(ValueError, match="rebuilt nuisance.*integrity"):
        fit_stage4_nuisance(
            pair, "l", {}, {}, tmp_path, retry_failed=True
        )

    assert not cache.path(task).exists()


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
    chosen = selection_for_config(config)

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


@pytest.mark.parametrize("execution_profile", ["full", "fast"])
def test_selection_contract_accepts_deterministic_profile_artifact(
    config, execution_profile
):
    selected = selection_for_config(config, execution_profile)

    cells = validate_stage4_selection(
        config,
        selected,
        execution_profile=execution_profile,
    )

    assert len(cells) == 6


def test_selection_contract_requires_all_24_screening_rows(config):
    selected = selection_for_config(config)
    selected["screening_ranking"].pop()

    with pytest.raises(ValueError, match="all 24 cells"):
        validate_stage4_selection(config, selected, execution_profile="full")


@pytest.mark.parametrize("field", ["n", "p"])
@pytest.mark.parametrize("invalid", [1000.0, "1000", True])
def test_selection_contract_rejects_non_native_integer_cell_values(
    config, field, invalid
):
    selected = selection_for_config(config)
    selected["screening_ranking"][0][field] = invalid

    with pytest.raises(ValueError, match="native integer"):
        validate_stage4_selection(config, selected, execution_profile="full")


def test_selection_contract_rejects_arbitrary_nonminimum_chosen_cell(config):
    selected = selection_for_config(config)
    first = selected["cells"][0]
    alternatives = [
        row
        for row in selected["screening_ranking"]
        if (row["panel"], row["scenario"])
        == (first["panel"], first["scenario"])
        and row != first
    ]
    selected["cells"][0] = alternatives[-1]

    with pytest.raises(ValueError, match="deterministic minima"):
        validate_stage4_selection(config, selected, execution_profile="full")


@pytest.mark.parametrize(
    ("field", "invalid", "message"),
    [
        ("screening_stage", "stale-stage", "screening_stage"),
        (
            "screening_seed_namespace",
            "stale-namespace",
            "screening_seed_namespace",
        ),
        ("selection_rule", "pick-a-winner", "selection_rule"),
        ("config_fingerprint", "stale-grid", "config_fingerprint"),
        ("expected_screening_replications", 1, "screening replications"),
    ],
)
def test_selection_contract_rejects_stale_or_underreplicated_full_artifact(
    config, field, invalid, message
):
    selected = selection_for_config(config)
    selected[field] = invalid

    with pytest.raises(ValueError, match=message):
        validate_stage4_selection(config, selected, execution_profile="full")


def test_selection_contract_prevents_fast_full_crossing(config):
    selected = selection_for_config(config, execution_profile="fast")

    with pytest.raises(ValueError, match="execution_profile"):
        validate_stage4_selection(config, selected, execution_profile="full")


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


@pytest.mark.parametrize("status", ["failed", "oom"])
def test_resume_accepts_only_recognized_failure_with_exact_identity(status):
    pair = make_pair(learner_l="oracle", learner_m="oracle")

    assert validate_stage4_resume_record(failure_record(pair, status), pair) == status


@pytest.mark.parametrize(
    "field",
    [
        "stage",
        "seed_namespace",
        "panel",
        "scenario",
        "n",
        "p",
        "replication",
        "learner_l",
        "learner_m",
        "learner_l_config_hash",
        "learner_m_config_hash",
        "execution_profile",
        "data_seed",
        "fold_seed",
    ],
)
def test_resume_rejects_stale_failure_provenance(field):
    pair = make_pair(learner_l="oracle", learner_m="oracle")
    record = failure_record(pair)
    record[field] = "stale" if isinstance(record[field], str) else -1

    with pytest.raises(ValueError, match=rf"{field} mismatch"):
        validate_stage4_resume_record(record, pair)


@pytest.mark.parametrize(
    ("field", "invalid"),
    [
        ("replication", False),
        ("replication", 0.0),
        ("n", 80.0),
        ("p", 10.0),
        ("folds_count", 2.0),
    ],
)
def test_resume_rejects_non_native_integer_identity(field, invalid):
    pair = make_pair(learner_l="oracle", learner_m="oracle")
    record = failure_record(pair)
    record[field] = invalid

    with pytest.raises(ValueError, match=rf"{field} mismatch"):
        validate_stage4_resume_record(record, pair)


@pytest.mark.parametrize("invalid", [True, "1.0", np.inf])
def test_resume_rejects_invalid_theta0_identity(invalid):
    pair = make_pair(learner_l="oracle", learner_m="oracle")
    record = failure_record(pair)
    record["theta0"] = invalid

    with pytest.raises(ValueError, match="theta0 mismatch"):
        validate_stage4_resume_record(record, pair)


def test_resume_rejects_non_string_identity_even_if_it_compares_equal():
    pair = make_pair(learner_l="oracle", learner_m="oracle")
    record = failure_record(pair)

    class EqualStage:
        def __eq__(self, other):
            return other == pair.stage

    record["stage"] = EqualStage()

    with pytest.raises(ValueError, match="stage mismatch"):
        validate_stage4_resume_record(record, pair)


def test_preflight_and_formal_complete_universes_are_disjoint(config):
    from dataclasses import replace

    frozen = frozen_for_config(config)
    selection = selection_for_config(config)
    originals = deepcopy((config, frozen, selection))
    formal = tuple(iter_stage4_pairs(config, "confirmation", frozen, selection))
    preflight = tuple(iter_stage4_pairs(
        config, "confirmation", frozen, selection, preflight=True,
    ))
    assert len(formal) == 6000
    assert len(preflight) == 300
    assert not {p.key for p in formal} & {p.key for p in preflight}

    def nuisances(pairs):
        return {build_stage4_nuisance_spec(p, t).key for p in pairs for t in ("l", "m")}

    assert len(nuisances(formal)) == 7200
    assert len(nuisances(preflight)) == 360  # 600 pair-target requests share fits.
    assert not nuisances(formal) & nuisances(preflight)
    for field in ("data_seed", "fold_seed"):
        formal_seeds = {failure_record(p)[field] for p in formal}
        preflight_seeds = {failure_record(p)[field] for p in preflight}
        assert len(formal_seeds) == 600
        assert len(preflight_seeds) == 30
        assert not formal_seeds & preflight_seeds
    for p in preflight:
        assert p.stage == config["confirmation"]["stage"] + "_preflight"
        assert p.seed_namespace == config["confirmation"]["seed_namespace"] + "_preflight"
    restored = {replace(p, stage=config["confirmation"]["stage"],
                        seed_namespace=config["confirmation"]["seed_namespace"]).key
                for p in preflight}
    assert restored == {p.key for p in formal if p.replication < 5}
    assert (config, frozen, selection) == originals


def test_preflight_full_models_and_original_fingerprints(config):
    frozen = frozen_for_config(config)
    selection = selection_for_config(config)
    pairs = tuple(iter_stage4_pairs(
        config, "confirmation", frozen, selection, preflight=True,
    ))
    for pair in pairs:
        assert pair.execution_profile == "full"
        assert pair.folds_count == 5
        for target in ("l", "m"):
            method = resolve_method(pair, target, frozen, config["extra_trees"]["params"])
            if method.params is not None:
                assert method.params["n_estimators"] >= 600
    invalid = deepcopy(selection)
    invalid["config_fingerprint"] = "different-config"
    with pytest.raises(ValueError, match="fingerprint"):
        tuple(iter_stage4_pairs(config, "confirmation", frozen, invalid, preflight=True))


@pytest.mark.parametrize(("phase", "fast", "replications"), [
    ("screening", False, None), ("confirmation", True, None),
    ("confirmation", False, 1), ("confirmation", False, 100),
    ("confirmation", False, 0), ("confirmation", False, True),
    ("confirmation", False, 5.0),
])
def test_preflight_rejects_invalid_protocol(config, phase, fast, replications):
    with pytest.raises(ValueError, match="preflight"):
        tuple(iter_stage4_pairs(
            config, phase, frozen_for_config(config), selection_for_config(config),
            replications=replications, fast=fast, preflight=True,
        ))


@pytest.mark.parametrize("count", [1, 6, True, 5.0])
def test_preflight_requires_five_rep_config(config, count):
    frozen = frozen_for_config(config)
    selection = selection_for_config(config)
    config["confirmation"]["smoke_replications"] = count
    with pytest.raises(ValueError, match="preflight"):
        tuple(iter_stage4_pairs(
            config, "confirmation", frozen, selection, preflight=True,
        ))


def test_preflight_resume_is_separate_in_shared_cache(config, tmp_path, monkeypatch):
    import tabdml.stage4_experiment as experiment

    frozen = frozen_for_config(config)
    selection = selection_for_config(config)
    formal = next(iter_stage4_pairs(config, "confirmation", frozen, selection))
    preflight = next(iter_stage4_pairs(config, "confirmation", frozen, selection, preflight=True))
    fitted = []
    cache = NuisanceCache(tmp_path)

    def fake_fit(task, **kwargs):
        assert kwargs["fast"] is False
        fitted.append(task.key)
        cache.write(task, np.zeros(task.n), (0.0,) * task.folds_count, None, None)
        return cache.read(task, task.n)

    monkeypatch.setattr(experiment, "fit_cached_nuisance", fake_fit)
    for pair in (preflight, formal, preflight, formal):
        fit_stage4_nuisance(pair, "l", frozen, config["extra_trees"]["params"], tmp_path)
    assert len(fitted) == len(set(fitted)) == 2
    for pair in (preflight, formal):
        assert validate_stage4_resume_record(failure_record(pair), pair) == "failed"
    with pytest.raises(ValueError, match="mismatch"):
        validate_stage4_resume_record(failure_record(preflight), formal)


def test_formal_analysis_rejects_preflight_stage_and_namespace(config):
    from tabdml.stage4_analysis import _validate_record_universe

    frozen = frozen_for_config(config)
    selection = selection_for_config(config)
    formal = tuple(iter_stage4_pairs(config, "confirmation", frozen, selection))
    pair = next(iter_stage4_pairs(config, "confirmation", frozen, selection, preflight=True))
    cached = CachedNuisanceResult(np.zeros(pair.n), (0.0,) * pair.folds_count, None, None)
    record = compose_stage4_record(pair, cached, cached)
    with pytest.raises(ValueError, match="foreign confirmation task_key"):
        _validate_record_universe([record], formal, "confirmation")
    for field in ("stage", "seed_namespace"):
        forged = dict(record, task_key=formal[0].key)
        if field == "seed_namespace":
            forged["stage"] = formal[0].stage
        with pytest.raises(ValueError, match=field):
            _validate_record_universe([forged], formal, "confirmation")
