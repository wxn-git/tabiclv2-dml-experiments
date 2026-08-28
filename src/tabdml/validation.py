from __future__ import annotations

from doubleml import DoubleMLData, DoubleMLPLR
from sklearn.dummy import DummyRegressor

from .crossfit import make_folds
from .dgp import simulate_plr
from .dml import estimate_plr_dml


def compare_with_doubleml(n: int = 1000, seed: int = 123) -> dict[str, float]:
    data = simulate_plr("linear", n=n, p=10, seed=seed)
    folds = make_folds(n, 5, seed + 1)
    custom = estimate_plr_dml(data.y, data.d, data.l0, data.m0)

    dml_data = DoubleMLData.from_arrays(data.X, data.y, data.d)
    reference = DoubleMLPLR(
        dml_data,
        DummyRegressor(),
        DummyRegressor(),
        n_folds=5,
        draw_sample_splitting=False,
    )
    reference.set_sample_splitting(list(folds))
    external = {
        "d": {
            "ml_l": data.l0[:, None],
            "ml_m": data.m0[:, None],
        }
    }
    reference.fit(external_predictions=external)
    return {
        "custom_theta": custom.theta,
        "doubleml_theta": float(reference.coef[0]),
        "theta_difference": abs(custom.theta - float(reference.coef[0])),
        "custom_standard_error": custom.standard_error,
        "doubleml_standard_error": float(reference.se[0]),
        "standard_error_difference": abs(
            custom.standard_error - float(reference.se[0])
        ),
    }

