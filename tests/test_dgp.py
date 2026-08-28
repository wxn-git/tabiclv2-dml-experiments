import numpy as np
import pytest

from tabdml.dgp import simulate_plr


@pytest.mark.parametrize("scenario", ["linear", "smooth", "tree", "mixed"])
def test_dgp_is_reproducible_and_obeys_plr_identity(scenario):
    a = simulate_plr(scenario, n=300, p=10, seed=7)
    b = simulate_plr(scenario, n=300, p=10, seed=7)
    np.testing.assert_allclose(a.X, b.X)
    np.testing.assert_allclose(a.l0, a.theta0 * a.m0 + a.g0)
    assert a.X.shape == (300, 10)
    assert a.y.shape == a.d.shape == (300,)


def test_mixed_scenario_marks_binary_and_categorical_columns():
    data = simulate_plr("mixed", 300, 10, 11)
    assert len(data.categorical_indices) >= 2

