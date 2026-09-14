import sys
import types

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


def test_tabicl_binds_learner_seed_and_cuda_device(monkeypatch):
    calls = []

    class FakeTabICLRegressor:
        def __init__(self, **kwargs):
            calls.append(kwargs)

    monkeypatch.setitem(
        sys.modules, "tabicl", types.SimpleNamespace(TabICLRegressor=FakeTabICLRegressor)
    )
    monkeypatch.setitem(
        sys.modules,
        "torch",
        types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: True)),
    )

    make_learner("tabiclv2_8", seed=314)

    assert calls == [{"n_estimators": 8, "random_state": 314, "device": "cuda"}]


def test_tabicl_hard_fails_when_cuda_is_unavailable(monkeypatch):
    monkeypatch.setitem(
        sys.modules, "tabicl", types.SimpleNamespace(TabICLRegressor=object)
    )
    monkeypatch.setitem(
        sys.modules,
        "torch",
        types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False)),
    )

    with pytest.raises(RuntimeError, match="CUDA"):
        make_learner("tabiclv2_1", seed=2)
