import numpy as np
import pytest

from tabdml.dgp import simulate_plr


@pytest.mark.parametrize(
    "scenario", ["linear", "smooth", "tree", "tree_simple", "mixed"]
)
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


def test_tree_simple_uses_only_axis_aligned_thresholds():
    data = simulate_plr("tree_simple", n=500, p=10, seed=19)
    x0, x1, x2, x3, x4 = (data.X[:, index] for index in range(5))
    raw_m = 0.9 * (x0 > 0) - 0.7 * (x1 > 0) + 0.5 * (x2 > 0)
    raw_g = 0.8 * (x0 > 0) + 0.6 * (x3 > 0) - 0.5 * (x4 > 0)
    expected_m = (raw_m - raw_m.mean()) / raw_m.std()
    expected_g = (raw_g - raw_g.mean()) / raw_g.std()

    np.testing.assert_allclose(data.m0, expected_m)
    np.testing.assert_allclose(data.g0, expected_g)
    np.testing.assert_allclose(data.l0, data.theta0 * data.m0 + data.g0)
