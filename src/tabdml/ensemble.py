from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_predict


class ConvexOOFEnsemble(RegressorMixin, BaseEstimator):
    def __init__(self, seed: int = 0, fast: bool = False):
        self.seed = seed
        self.fast = fast

    def _models(self):
        from .learners import make_learner

        return [
            make_learner(name, self.seed + index, fast=self.fast)
            for index, name in enumerate(("lasso", "random_forest", "xgboost", "mlp"))
        ]

    def fit(self, X, y):
        X_arr = np.asarray(X)
        y_arr = np.asarray(y, dtype=float)
        cv = KFold(n_splits=3, shuffle=True, random_state=self.seed)
        templates = self._models()
        oof = np.column_stack(
            [cross_val_predict(clone(model), X_arr, y_arr, cv=cv, n_jobs=1) for model in templates]
        )
        initial = np.full(len(templates), 1 / len(templates))
        result = minimize(
            lambda w: np.mean((y_arr - oof @ w) ** 2),
            initial,
            method="SLSQP",
            bounds=[(0.0, 1.0)] * len(templates),
            constraints={"type": "eq", "fun": lambda w: w.sum() - 1.0},
        )
        self.fallback_reason_ = None
        if result.success and np.isfinite(result.x).all():
            weights = np.clip(result.x, 0, None)
            self.weights_ = weights / weights.sum()
        else:
            self.weights_ = initial
            self.fallback_reason_ = result.message
        self.models_ = [clone(model).fit(X_arr, y_arr) for model in templates]
        return self

    def predict(self, X):
        predictions = np.column_stack([model.predict(X) for model in self.models_])
        return predictions @ self.weights_

