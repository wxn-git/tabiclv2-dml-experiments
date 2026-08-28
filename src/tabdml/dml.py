from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import norm


@dataclass(frozen=True)
class DMLResult:
    theta: float
    standard_error: float
    ci_lower: float
    ci_upper: float
    score: NDArray[np.float64]


def estimate_plr_dml(
    y: ArrayLike,
    d: ArrayLike,
    l_hat: ArrayLike,
    m_hat: ArrayLike,
    alpha: float = 0.05,
) -> DMLResult:
    arrays = [np.asarray(x, dtype=float).reshape(-1) for x in (y, d, l_hat, m_hat)]
    if len({len(x) for x in arrays}) != 1:
        raise ValueError("All inputs must have equal length.")
    if not all(np.isfinite(x).all() for x in arrays):
        raise ValueError("All inputs must be finite.")
    y_arr, d_arr, l_arr, m_arr = arrays
    v = d_arr - m_arr
    u = y_arr - l_arr
    denominator = float(np.mean(v * v))
    if denominator < 1e-12:
        raise ValueError("Residualized treatment has near-zero variance.")
    theta = float(np.mean(v * u) / denominator)
    score = v * (u - theta * v)
    standard_error = float(
        np.sqrt(np.mean(score * score) / (len(y_arr) * denominator**2))
    )
    critical = float(norm.ppf(1 - alpha / 2))
    return DMLResult(
        theta=theta,
        standard_error=standard_error,
        ci_lower=theta - critical * standard_error,
        ci_upper=theta + critical * standard_error,
        score=score,
    )

