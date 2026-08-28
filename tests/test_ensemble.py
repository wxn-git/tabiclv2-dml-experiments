import numpy as np

from tabdml.ensemble import ConvexOOFEnsemble


def test_ensemble_weights_are_nonnegative_and_sum_to_one():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(80, 6))
    y = X[:, 0] - 0.5 * X[:, 1] + rng.normal(scale=0.2, size=80)
    model = ConvexOOFEnsemble(seed=2, fast=True).fit(X, y)
    assert np.all(model.weights_ >= 0)
    assert np.isclose(model.weights_.sum(), 1.0)
    assert model.predict(X[:5]).shape == (5,)

