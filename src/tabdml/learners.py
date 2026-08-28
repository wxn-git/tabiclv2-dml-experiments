from __future__ import annotations

from typing import Sequence

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import LassoCV
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def _grid(estimator, grid: dict, fast: bool):
    if fast:
        return estimator
    return GridSearchCV(estimator, grid, cv=3, scoring="neg_mean_squared_error", n_jobs=1)


def make_configured_tree_learner(
    kind: str,
    params: dict,
    seed: int,
    fast: bool = False,
):
    configured = dict(params)
    if fast:
        configured["n_estimators"] = min(int(configured.get("n_estimators", 20)), 20)
    if kind == "xgboost":
        from xgboost import XGBRegressor

        return XGBRegressor(
            objective="reg:squarederror",
            random_state=seed,
            n_jobs=1,
            **configured,
        )
    if kind == "extra_trees":
        return ExtraTreesRegressor(
            random_state=seed,
            n_jobs=1,
            **configured,
        )
    raise ValueError(f"Unknown configured learner: {kind}")


def make_learner(
    name: str,
    seed: int,
    categorical_indices: Sequence[int] = (),
    tabicl_estimators: int = 1,
    fast: bool = False,
):
    del categorical_indices
    if name == "lasso":
        model = LassoCV(
            alphas=np.logspace(-4, 1, 20),
            cv=3,
            random_state=seed,
            max_iter=10000,
        )
        return Pipeline([("preprocess", StandardScaler()), ("model", model)])

    if name == "random_forest":
        forest = RandomForestRegressor(
            n_estimators=20 if fast else 300,
            random_state=seed,
            n_jobs=1,
        )
        return _grid(
            forest,
            {"max_features": [0.5, 1.0], "min_samples_leaf": [2, 10]},
            fast,
        )

    if name == "xgboost":
        from xgboost import XGBRegressor

        model = XGBRegressor(
            n_estimators=20 if fast else 500,
            max_depth=3,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=seed,
            n_jobs=1,
            objective="reg:squarederror",
        )
        return _grid(
            model,
            {"max_depth": [3, 6], "learning_rate": [0.03, 0.1]},
            fast,
        )

    if name == "mlp":
        model = MLPRegressor(
            hidden_layer_sizes=(16,) if fast else (64,),
            max_iter=30 if fast else 500,
            early_stopping=not fast,
            random_state=seed,
        )
        if fast:
            return Pipeline([("preprocess", StandardScaler()), ("model", model)])
        search = GridSearchCV(
            model,
            {"hidden_layer_sizes": [(64,), (128, 64)]},
            cv=3,
            scoring="neg_mean_squared_error",
            n_jobs=1,
        )
        return Pipeline([("preprocess", StandardScaler()), ("model", search)])

    if name in {"tabiclv2", "tabiclv2_1", "tabiclv2_8"}:
        try:
            from tabicl import TabICLRegressor
        except ImportError as exc:
            raise ImportError(
                "TabICLv2 requires the optional 'tabicl' and 'torch' dependencies."
            ) from exc
        estimators = 1 if name == "tabiclv2_1" else 8 if name == "tabiclv2_8" else tabicl_estimators
        return TabICLRegressor(n_estimators=estimators)

    if name == "ensemble":
        from .ensemble import ConvexOOFEnsemble

        return ConvexOOFEnsemble(seed=seed, fast=fast)

    raise ValueError(f"Unknown learner: {name}")
