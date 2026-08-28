import numpy as np

from tabdml.dgp import simulate_plr
from tabdml.dml import estimate_plr_dml


def test_point_estimate_matches_manual_residual_regression():
    y = np.array([2.0, 4.0, 5.0, 8.0])
    d = np.array([1.0, 2.0, 2.0, 4.0])
    zeros = np.zeros(4)
    result = estimate_plr_dml(y, d, zeros, zeros)
    assert np.isclose(result.theta, d @ y / (d @ d))


def test_oracle_nuisances_recover_theta():
    data = simulate_plr("smooth", 20000, 10, 99)
    result = estimate_plr_dml(data.y, data.d, data.l0, data.m0)
    assert abs(result.theta - 1.0) < 0.04
    assert result.ci_lower < result.theta < result.ci_upper

