import numpy as np

from tabdml.stage3b import (
    Stage3BPairSpec,
    build_nuisance_spec,
    compose_dml_record,
    fit_cached_nuisance,
)


def _pair(learner_l="oracle", learner_m="oracle"):
    return Stage3BPairSpec(
        stage="stage3b_batch_a",
        seed_namespace="stage3b_test",
        scenario="tree",
        n=80,
        p=10,
        replication=0,
        learner_l=learner_l,
        learner_m=learner_m,
        folds_count=4,
        theta0=1.0,
    )


def test_fit_cached_nuisance_reuses_successful_prediction(monkeypatch, tmp_path):
    task = build_nuisance_spec(_pair("lasso", "oracle"), "l")
    first = fit_cached_nuisance(task, cache_root=tmp_path, theta0=1.0, fast=True)

    def fail_if_refit(*args, **kwargs):
        raise AssertionError("cache miss caused a refit")

    monkeypatch.setattr(
        "tabdml.stage3b.crossfit_single_nuisance",
        fail_if_refit,
    )
    second = fit_cached_nuisance(task, cache_root=tmp_path, theta0=1.0, fast=True)

    np.testing.assert_array_equal(first.prediction, second.prediction)


def test_compose_oracle_record_contains_proxy_error(tmp_path):
    pair = _pair()
    l_task = build_nuisance_spec(pair, "l")
    m_task = build_nuisance_spec(pair, "m")
    l_result = fit_cached_nuisance(l_task, tmp_path, theta0=1.0, fast=True)
    m_result = fit_cached_nuisance(m_task, tmp_path, theta0=1.0, fast=True)

    record = compose_dml_record(pair, l_result, m_result)

    assert record["status"] == "success"
    assert record["lm_error_cross"] == 0.0
    assert record["l_mse"] == 0.0
    assert record["m_mse"] == 0.0
    assert record["theta_proxy"] == 1.0
    assert np.isclose(record["proxy_error"], record["theta"] - 1.0)
    assert record["standard_error"] > 0


def test_pair_key_preserves_ordered_learners():
    forward = _pair("tabiclv2_1", "xgboost")
    reverse = _pair("xgboost", "tabiclv2_1")

    assert forward.key != reverse.key
    assert "__ltabiclv2_1__mxgboost__" in forward.key
