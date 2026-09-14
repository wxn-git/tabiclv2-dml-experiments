from __future__ import annotations

from dataclasses import asdict, replace
import json
from pathlib import Path

import numpy as np
import pytest

from tabdml.nuisance_cache import NuisanceCache
from tabdml.stage3b_screen import _params_hash
from tabdml.stage5_config import load_stage5_config, stage5_config_fingerprint
from tabdml.stage5_tuning import stage5_tuning_run_fingerprint
from tabdml.stage5_experiment import (
    ResolvedStage5Method,
    Stage5NuisanceResult,
    build_stage5_nuisance_spec,
    compose_stage5_record,
    fit_stage5_nuisance,
    iter_stage5_pairs,
    methods_for_device,
    resolve_stage5_method,
    stage5_nuisance_metadata_path,
    validate_stage5_record,
)


CONFIG = Path("configs/stage5_sensitivity.yaml")


def _frozen(config, profile="fast"):
    replications = 1 if profile == "fast" else 10
    candidates = config["tuning"]["xgboost_candidates"]
    rankings = {}
    winners = {}
    for scenario in config["scenarios"]:
        rankings[scenario] = {}
        winners[scenario] = {}
        for target in ("l", "m"):
            ordered = candidates if target == "l" else candidates[1:] + candidates[:1]
            values = []
            for rank, candidate in enumerate(ordered):
                nominal = dict(candidate["params"])
                effective = dict(nominal)
                if profile == "fast":
                    effective["n_estimators"] = min(effective["n_estimators"], 20)
                values.append(
                    {
                        "candidate": candidate["name"],
                        "learner_kind": "xgboost",
                        "execution_profile": profile,
                        "nominal_params": nominal,
                        "nominal_config_hash": _params_hash(nominal),
                        "effective_params": effective,
                        "effective_config_hash": _params_hash(effective),
                        "replications": replications,
                        "mean_validation_observed_mse": float(rank + 1),
                        "mean_validation_truth_mse_diagnostic": float(rank + 2),
                        "selection_metric": "mean_validation_y_mse" if target == "l" else "mean_validation_d_mse",
                    }
                )
            rankings[scenario][target] = values
            winners[scenario][target] = dict(values[0])
    return {
        "schema_version": "stage5_tuning_v1",
        "config_fingerprint": stage5_config_fingerprint(config),
        "tuning_stage": config["tuning"]["stage"],
        "tuning_seed_namespace": config["tuning"]["seed_namespace"],
        "tuning_run_fingerprint": stage5_tuning_run_fingerprint(config, replications, profile),
        "theta0": config["theta0"],
        "execution_profile": profile,
        "expected_replications": replications,
        "selection_metric_names": {"l": "mean_validation_y_mse", "m": "mean_validation_d_mse"},
        "candidate_ranking": rankings,
        "scenarios": winners,
    }


@pytest.fixture
def config():
    return load_stage5_config(CONFIG)


def test_exact_pair_counts_order_center_and_unique_keys(config):
    expected = {"smoke": 180, "preflight": 900, "formal": 18000}
    for profile, count in expected.items():
        frozen = _frozen(config, "fast" if profile == "smoke" else "full")
        pairs = tuple(iter_stage5_pairs(config, frozen, profile))
        assert len(pairs) == len({pair.key for pair in pairs}) == count
        assert [pair.method for pair in pairs[:6]] == config["methods"]
        centers = [pair for pair in pairs if pair.n == 1000 and pair.p == 50]
        assert len(centers) == 6 * 6 * {"smoke": 1, "preflight": 5, "formal": 100}[profile]


def test_pairing_shares_data_and_folds_but_profiles_are_isolated(config):
    smoke = tuple(iter_stage5_pairs(config, _frozen(config), "smoke"))
    group = [p for p in smoke if (p.scenario, p.n, p.p, p.replication) == ("linear", 1000, 10, 0)]
    assert len(group) == 6
    assert len({p.data_seed for p in group}) == len({p.fold_seed for p in group}) == 1
    preflight = next(iter_stage5_pairs(config, _frozen(config, "full"), "preflight"))
    assert (smoke[0].data_seed, smoke[0].fold_seed) != (preflight.data_seed, preflight.fold_seed)


def test_method_groups_are_exact_and_gpu_must_be_unsharded():
    assert methods_for_device("gpu") == ("tabiclv2_1", "tabiclv2_8")
    assert methods_for_device("cpu") == ("xgboost_tuned", "extra_trees", "lasso")
    assert methods_for_device("ensemble") == ("ensemble",)
    with pytest.raises(ValueError, match="device group"):
        methods_for_device("tpu")


def test_resolution_uses_target_specific_tuning_and_exact_devices(config):
    frozen = _frozen(config)
    pairs = tuple(iter_stage5_pairs(config, frozen, "smoke"))
    xgb = next(pair for pair in pairs if pair.method == "xgboost_tuned")
    l_method = resolve_stage5_method(xgb, "l", config, frozen)
    m_method = resolve_stage5_method(xgb, "m", config, frozen)
    assert l_method.params != m_method.params
    assert l_method.config_hash != m_method.config_hash
    assert l_method.requested_device == m_method.requested_device == "cpu"
    tab = next(pair for pair in pairs if pair.method == "tabiclv2_8")
    assert resolve_stage5_method(tab, "l", config, frozen).requested_device == "cuda"


@pytest.mark.parametrize("mutation", ["scenario", "config", "tuning", "profile", "target"])
def test_resolution_fails_closed_on_forged_provenance(config, mutation):
    frozen = _frozen(config)
    pair = next(p for p in iter_stage5_pairs(config, frozen, "smoke") if p.method == "xgboost_tuned")
    target = "l"
    if mutation == "scenario":
        pair = pair.__class__(**{**pair.__dict__, "scenario": "forged"})
    elif mutation == "config":
        pair = pair.__class__(**{**pair.__dict__, "config_fingerprint": "forged"})
    elif mutation == "tuning":
        pair = pair.__class__(**{**pair.__dict__, "tuning_fingerprint": "forged"})
    elif mutation == "profile":
        pair = pair.__class__(**{**pair.__dict__, "profile": "preflight"})
    else:
        target = "g"
    with pytest.raises(ValueError):
        resolve_stage5_method(pair, target, config, frozen)


def test_nuisance_key_contains_target_learner_estimators_folds_seed_and_provenance(config):
    frozen = _frozen(config)
    pair = next(p for p in iter_stage5_pairs(config, frozen, "smoke") if p.method == "tabiclv2_8")
    method = resolve_stage5_method(pair, "m", config, frozen)
    task = build_stage5_nuisance_spec(pair, "m", method)
    assert pair.seed_namespace in task.key
    assert "__m__tabiclv2_8__e8__k5__s" in task.key
    forged = pair.__class__(**{**pair.__dict__, "config_fingerprint": "forged"})
    assert build_stage5_nuisance_spec(forged, "m", method).key != task.key
    assert method.config_hash != task.learner_config_hash


def test_pairs_and_records_retain_full_fingerprints_while_cache_paths_stay_compact(config):
    frozen = _frozen(config)
    pair = next(iter_stage5_pairs(config, frozen, "smoke"))
    resolved = resolve_stage5_method(pair, "l", config, frozen)
    task = build_stage5_nuisance_spec(pair, "l", resolved)

    assert len(pair.config_fingerprint) == 64
    assert len(pair.tuning_fingerprint) == 64
    assert pair.config_fingerprint == stage5_config_fingerprint(config)
    assert pair.tuning_fingerprint == frozen["tuning_run_fingerprint"]
    assert pair.config_fingerprint not in task.key
    assert pair.tuning_fingerprint not in task.key
    assert len(f"{task.key}.npz") < 180


def test_fit_passes_exact_params_and_smoke_flag_and_reuses_valid_cache(monkeypatch, tmp_path, config):
    frozen = _frozen(config)
    pair = next(p for p in iter_stage5_pairs(config, frozen, "smoke") if p.method == "xgboost_tuned")
    resolved = resolve_stage5_method(pair, "l", config, frozen)
    task = build_stage5_nuisance_spec(pair, "l", resolved)
    calls = []

    class Result:
        prediction = np.arange(task.n, dtype=float)
        fold_seconds = (0.1,) * 5
        peak_gpu_mb = None
        fallback_reason = None

    monkeypatch.setattr("tabdml.stage5_experiment.simulate_plr", lambda *a, **k: object())
    monkeypatch.setattr("tabdml.stage5_experiment.make_folds", lambda *a, **k: ((np.array([0]), np.array([1])),) * 5)
    monkeypatch.setattr("tabdml.stage5_experiment.crossfit_single_nuisance", lambda *a, **k: (calls.append(k), Result())[1])
    first = fit_stage5_nuisance(task, resolved, tmp_path, pair.theta0, full_settings=False)
    second = fit_stage5_nuisance(task, resolved, tmp_path, pair.theta0, full_settings=False)
    assert len(calls) == 1
    assert calls[0]["learner_kind"] == "xgboost"
    assert calls[0]["learner_params"] == resolved.params
    assert calls[0]["fast"] is True
    assert np.array_equal(first.prediction, second.prediction)


def test_stage5_sidecar_preserves_exact_task_runtime_and_device_metadata(
    monkeypatch, tmp_path, config
):
    frozen = _frozen(config)
    pair = next(
        p for p in iter_stage5_pairs(config, frozen, "smoke")
        if p.method == "tabiclv2_1"
    )
    resolved = resolve_stage5_method(pair, "l", config, frozen)
    task = build_stage5_nuisance_spec(pair, "l", resolved)

    class Result:
        prediction = np.arange(task.n, dtype=float)
        fold_seconds = (0.25,) * task.folds_count
        peak_gpu_mb = 12.5
        fallback_reason = None

    monkeypatch.setattr("tabdml.stage5_experiment.simulate_plr", lambda *a, **k: object())
    monkeypatch.setattr("tabdml.stage5_experiment.make_folds", lambda *a, **k: ())
    monkeypatch.setattr("tabdml.stage5_experiment.crossfit_single_nuisance", lambda *a, **k: Result())
    first = fit_stage5_nuisance(task, resolved, tmp_path, pair.theta0, False)
    metadata_path = stage5_nuisance_metadata_path(NuisanceCache(tmp_path), task)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    second = fit_stage5_nuisance(task, resolved, tmp_path, pair.theta0, False)

    assert metadata["task"] == asdict(task)
    assert metadata["fit_time"] == first.fit_time == second.fit_time == 1.25
    assert metadata["total_time"] == first.total_time == second.total_time
    assert metadata["requested_device"] == first.requested_device == second.requested_device == "cuda"
    assert metadata["observed_device"] == first.observed_device == second.observed_device == "cuda"


def test_fit_rejects_resolved_configuration_mismatch_before_simulation(
    monkeypatch, tmp_path, config
):
    frozen = _frozen(config)
    pair = next(
        p
        for p in iter_stage5_pairs(config, frozen, "smoke")
        if p.method == "xgboost_tuned"
    )
    resolved = resolve_stage5_method(pair, "l", config, frozen)
    task = build_stage5_nuisance_spec(pair, "l", resolved)
    forged = ResolvedStage5Method(
        resolved.learner,
        resolved.learner_kind,
        resolved.params,
        "forged",
        resolved.requested_device,
    )
    monkeypatch.setattr(
        "tabdml.stage5_experiment.simulate_plr",
        lambda *a, **k: pytest.fail("provenance must fail before simulation"),
    )
    with pytest.raises(ValueError, match="configuration"):
        fit_stage5_nuisance(
            task, forged, tmp_path, pair.theta0, full_settings=False
        )


def test_retry_repairs_non_object_stage5_cache_metadata(
    monkeypatch, tmp_path, config
):
    frozen = _frozen(config)
    pair = next(
        p for p in iter_stage5_pairs(config, frozen, "smoke") if p.method == "lasso"
    )
    resolved = resolve_stage5_method(pair, "l", config, frozen)
    task = build_stage5_nuisance_spec(pair, "l", resolved)
    calls = []

    class Result:
        prediction = np.arange(task.n, dtype=float)
        fold_seconds = (0.0,) * task.folds_count
        peak_gpu_mb = None
        fallback_reason = None

    monkeypatch.setattr("tabdml.stage5_experiment.simulate_plr", lambda *a, **k: object())
    monkeypatch.setattr("tabdml.stage5_experiment.make_folds", lambda *a, **k: ())
    monkeypatch.setattr(
        "tabdml.stage5_experiment.crossfit_single_nuisance",
        lambda *a, **k: (calls.append(1), Result())[1],
    )
    fit_stage5_nuisance(task, resolved, tmp_path, pair.theta0, False)
    metadata = stage5_nuisance_metadata_path(NuisanceCache(tmp_path), task)
    metadata.write_text("[]", encoding="utf-8")
    fit_stage5_nuisance(
        task, resolved, tmp_path, pair.theta0, False, retry_failed=True
    )
    assert len(calls) == 2


def test_composition_fixture_covers_all_formulas_schema_timing_device_and_fallback(monkeypatch, config):
    frozen = _frozen(config)
    pair = next(p for p in iter_stage5_pairs(config, frozen, "smoke") if p.method == "lasso")
    pair = replace(pair, n=4)

    class Data:
        y = np.array([1.0, 3.0, 2.0, 6.0])
        d = np.array([0.0, 1.0, 1.0, 2.0])
        l0 = np.array([1.0, 2.0, 2.0, 4.0])
        m0 = np.array([0.0, 0.5, 1.0, 1.5])

    monkeypatch.setattr("tabdml.stage5_experiment.simulate_plr", lambda *a, **k: Data())
    l = Stage5NuisanceResult(np.array([1.0, 2.0, 2.0, 4.0]), (0.1,) * 5, None, "l warning", 0.5, 0.6, "cpu", "cpu")
    m = Stage5NuisanceResult(np.array([0.0, 0.5, 1.0, 1.5]), (0.2,) * 5, None, None, 1.0, 1.2, "cpu", "cpu")
    record = compose_stage5_record(pair, l, m)
    required = {
        "task_key", "stage", "profile", "seed_namespace", "scenario", "n", "p", "replication",
        "method", "learner_l", "learner_m", "config_fingerprint", "tuning_fingerprint",
        "learner_l_config_hash", "learner_m_config_hash", "folds_count", "theta0", "data_seed",
        "fold_seed", "status", "theta_hat", "theta", "standard_error", "ci_lower", "ci_upper",
        "covered", "squared_error", "l_mse", "m_mse", "nuisance_error_product",
        "lm_error_cross", "residual_d_variance", "bias_numerator_proxy", "theta_proxy",
        "proxy_error", "l_fold_seconds", "m_fold_seconds", "l_fit_time", "m_fit_time",
        "runtime_seconds", "peak_gpu_mb", "requested_device", "observed_device", "fallback_reason",
    }
    assert required <= set(record)
    assert record["theta_hat"] == record["theta"]
    assert record["squared_error"] == pytest.approx((record["theta_hat"] - pair.theta0) ** 2)
    assert record["covered"] == (record["ci_lower"] <= pair.theta0 <= record["ci_upper"])
    assert record["nuisance_error_product"] == pytest.approx(np.sqrt(record["l_mse"] * record["m_mse"]))
    assert record["l_fit_time"] == 0.5 and record["m_fit_time"] == 1.0
    assert record["runtime_seconds"] == pytest.approx(1.8)
    assert record["status"] == "fallback" and record["fallback_reason"] == "l warning"


def test_composition_rejects_non_cuda_tabicl_observation(config):
    frozen = _frozen(config)
    pair = next(p for p in iter_stage5_pairs(config, frozen, "smoke") if p.method == "tabiclv2_1")
    result = Stage5NuisanceResult(np.zeros(pair.n), (0.0,) * 5, None, None, 0.0, 0.0, "cuda", "cpu")
    with pytest.raises(ValueError, match="CUDA"):
        compose_stage5_record(pair, result, result)


def _valid_composed_record(monkeypatch, config):
    frozen = _frozen(config)
    pair = next(
        p for p in iter_stage5_pairs(config, frozen, "smoke") if p.method == "lasso"
    )

    class Data:
        y = np.linspace(0.0, 2.0, pair.n)
        d = np.linspace(-1.0, 1.0, pair.n)
        l0 = np.zeros(pair.n)
        m0 = np.zeros(pair.n)

    monkeypatch.setattr("tabdml.stage5_experiment.simulate_plr", lambda *a, **k: Data())
    nuisance = Stage5NuisanceResult(
        np.zeros(pair.n), (0.1,) * pair.folds_count, None, None,
        0.5, 0.75, "cpu", "cpu",
    )
    return pair, compose_stage5_record(pair, nuisance, nuisance)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda record: record.pop("proxy_error"), "schema"),
        (lambda record: record.__setitem__("unexpected", 1), "schema"),
        (lambda record: record.__setitem__("covered", 1), "coverage"),
        (lambda record: record.__setitem__("nuisance_error_product", 999.0), "product"),
        (lambda record: record.__setitem__("proxy_error", 999.0), "proxy_error"),
        (lambda record: record.__setitem__("runtime_seconds", 0.1), "runtime"),
        (lambda record: record.__setitem__("l_fold_seconds", ["0"] * 5), "fold"),
        (lambda record: record.__setitem__("peak_gpu_mb", float("inf")), "peak"),
        (lambda record: record.__setitem__("requested_device", "cuda"), "device"),
        (lambda record: record.__setitem__("fallback_reason", "hidden"), "fallback"),
    ],
)
def test_existing_success_requires_exact_full_schema_and_derived_values(
    monkeypatch, config, mutation, message
):
    pair, record = _valid_composed_record(monkeypatch, config)
    mutation(record)
    with pytest.raises(ValueError, match=message):
        validate_stage5_record(record, pair)


@pytest.mark.parametrize(
    "field,value",
    [
        ("l_fold_seconds", [0.0] * 4),
        ("m_fold_seconds", [0.0, 0.0, 0.0, 0.0, float("nan")]),
        ("l_fit_time", -0.1),
        ("peak_gpu_mb", -0.1),
    ],
)
def test_composed_numeric_validation_rejects_bad_fold_runtime_and_gpu(
    monkeypatch, config, field, value
):
    pair, record = _valid_composed_record(monkeypatch, config)
    record[field] = value
    with pytest.raises(ValueError):
        validate_stage5_record(record, pair)
