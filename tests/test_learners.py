import numpy as np
import pytest
from sklearn.base import clone

from tabdml.learners import make_learner


@pytest.mark.parametrize("name", ["lasso", "random_forest", "xgboost", "mlp"])
def test_traditional_learner_is_cloneable_and_predicts(name):
    rng = np.random.default_rng(1)
    X = rng.normal(size=(40, 6))
    y = rng.normal(size=40)
    model = make_learner(name, seed=1, fast=True)
    fitted = clone(model).fit(X, y)
    assert fitted.predict(X[:3]).shape == (3,)


def test_preprocessing_is_inside_pipeline():
    model = make_learner("lasso", seed=1)
    assert "preprocess" in model.named_steps
