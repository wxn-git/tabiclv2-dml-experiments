import numpy as np

from tabdml.crossfit import (
    crossfit_nuisance_pair,
    crossfit_nuisances,
    crossfit_single_nuisance,
    make_folds,
)
from tabdml.dgp import simulate_plr


def test_folds_are_deterministic_and_partition_all_rows():
    a = make_folds(101, 5, 42)
    b = make_folds(101, 5, 42)
    assert all(np.array_equal(x[1], y[1]) for x, y in zip(a, b))
    assert sorted(np.concatenate([test for _, test in a]).tolist()) == list(range(101))


def test_crossfit_produces_one_prediction_per_row():
    data = simulate_plr("linear", 120, 10, 3)
    result = crossfit_nuisances(
        data, "lasso", make_folds(120, 5, 4), seed=5, tabicl_estimators=0, fast=True
    )
    assert np.isfinite(result.l_hat).all()
    assert np.isfinite(result.m_hat).all()
    assert len(result.fold_seconds) == 5


def test_cpu_crossfit_does_not_initialize_cuda(monkeypatch):
    data = simulate_plr("linear", 60, 10, 3)

    def fail_if_called():
        raise AssertionError("CPU learner initialized CUDA")

    monkeypatch.setattr("tabdml.crossfit._cuda_helpers", fail_if_called)
    result = crossfit_nuisances(
        data, "lasso", make_folds(60, 3, 4), seed=5, tabicl_estimators=0, fast=True
    )

    assert result.peak_gpu_mb is None


def test_oracle_pair_returns_exact_true_nuisances_without_cuda(monkeypatch):
    data = simulate_plr("tree", 80, 10, 31)

    def fail_if_called():
        raise AssertionError("Oracle pair initialized CUDA")

    monkeypatch.setattr("tabdml.crossfit._cuda_helpers", fail_if_called)
    result = crossfit_nuisance_pair(
        data,
        "oracle",
        "oracle",
        make_folds(80, 4, 17),
        seed_l=101,
        seed_m=202,
        tabicl_estimators=0,
        fast=True,
    )

    np.testing.assert_array_equal(result.l_hat, data.l0)
    np.testing.assert_array_equal(result.m_hat, data.m0)
    assert result.peak_gpu_mb is None
    assert len(result.fold_seconds) == 4


def test_homogeneous_pair_matches_legacy_crossfit_exactly():
    data = simulate_plr("linear", 90, 10, 41)
    folds = make_folds(90, 3, 19)
    legacy = crossfit_nuisances(
        data,
        "lasso",
        folds,
        seed=303,
        tabicl_estimators=0,
        fast=True,
    )
    paired = crossfit_nuisance_pair(
        data,
        "lasso",
        "lasso",
        folds,
        seed_l=303,
        seed_m=303,
        tabicl_estimators=0,
        fast=True,
    )

    np.testing.assert_array_equal(paired.l_hat, legacy.l_hat)
    np.testing.assert_array_equal(paired.m_hat, legacy.m_hat)


def test_single_crossfit_matches_pair_sides_exactly():
    data = simulate_plr("linear", 90, 10, 41)
    folds = make_folds(90, 3, 19)
    paired = crossfit_nuisance_pair(
        data,
        "lasso",
        "lasso",
        folds,
        seed_l=303,
        seed_m=303,
        tabicl_estimators=0,
        fast=True,
    )

    l_result = crossfit_single_nuisance(
        data,
        "l",
        "lasso",
        folds,
        seed=303,
        tabicl_estimators=0,
        fast=True,
    )
    m_result = crossfit_single_nuisance(
        data,
        "m",
        "lasso",
        folds,
        seed=303,
        tabicl_estimators=0,
        fast=True,
    )

    np.testing.assert_array_equal(l_result.prediction, paired.l_hat)
    np.testing.assert_array_equal(m_result.prediction, paired.m_hat)


def test_single_oracle_returns_exact_target_without_cuda(monkeypatch):
    data = simulate_plr("tree", 80, 10, 51)

    def fail_if_called():
        raise AssertionError("Oracle initialized CUDA")

    monkeypatch.setattr("tabdml.crossfit._cuda_helpers", fail_if_called)
    result = crossfit_single_nuisance(
        data,
        "m",
        "oracle",
        make_folds(80, 4, 11),
        seed=17,
        tabicl_estimators=0,
        fast=True,
    )

    np.testing.assert_array_equal(result.prediction, data.m0)
    assert result.peak_gpu_mb is None
