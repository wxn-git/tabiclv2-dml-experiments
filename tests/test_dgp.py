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


def _scale(values):
    values = np.asarray(values, dtype=float)
    return (values - values.mean()) / values.std()


def _h(X, root, left, right, a, b, c):
    return (
        a * (X[:, root] > 0)
        + b * ((X[:, root] > 0) & (X[:, left] > 0))
        + c * ((X[:, root] <= 0) & (X[:, right] > 0))
    )


@pytest.mark.parametrize(
    "scenario", ["tree_stumps", "tree_hierarchical", "tree_forest_sum"]
)
def test_stage4_tree_dgps_are_reproducible(scenario):
    first = simulate_plr(scenario, n=500, p=10, seed=29)
    second = simulate_plr(scenario, n=500, p=10, seed=29)
    np.testing.assert_array_equal(first.X, second.X)
    np.testing.assert_array_equal(first.m0, second.m0)
    np.testing.assert_array_equal(first.g0, second.g0)


def test_tree_stumps_matches_declared_formula():
    data = simulate_plr("tree_stumps", n=500, p=10, seed=31)
    X = data.X
    raw_m = 0.9 * (X[:, 0] > 0) - 0.7 * (X[:, 1] > 0) + 0.5 * (X[:, 2] > 0)
    raw_g = 0.8 * (X[:, 0] > 0) + 0.6 * (X[:, 3] > 0) - 0.5 * (X[:, 4] > 0)
    np.testing.assert_allclose(data.m0, _scale(raw_m))
    np.testing.assert_allclose(data.g0, _scale(raw_g))


def test_tree_hierarchical_matches_declared_formula():
    data = simulate_plr("tree_hierarchical", n=500, p=10, seed=37)
    np.testing.assert_allclose(data.m0, _scale(_h(data.X, 0, 1, 2, 0.8, 0.6, -0.4)))
    np.testing.assert_allclose(data.g0, _scale(_h(data.X, 0, 3, 4, 0.7, 0.5, -0.4)))


def test_tree_forest_sum_matches_declared_formula():
    data = simulate_plr("tree_forest_sum", n=500, p=10, seed=41)
    raw_m = _h(data.X, 0, 1, 2, 0.55, 0.40, -0.30) + _h(
        data.X, 3, 4, 5, 0.45, -0.35, 0.30
    )
    raw_g = _h(data.X, 0, 6, 7, 0.50, 0.35, -0.25) + _h(
        data.X, 3, 8, 9, 0.40, -0.30, 0.25
    )
    np.testing.assert_allclose(data.m0, _scale(raw_m))
    np.testing.assert_allclose(data.g0, _scale(raw_g))


def test_tree_forest_sum_requires_ten_columns():
    with pytest.raises(ValueError, match="requires p >= 10"):
        simulate_plr("tree_forest_sum", n=500, p=9, seed=43)
