from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from .dgp import SimulatedData


@dataclass(frozen=True)
class NuisanceDiagnostics:
    l_mse: float
    m_mse: float
    lm_error_cross: float
    residual_d_variance: float
    bias_numerator_proxy: float
    theta_proxy: float


def compute_nuisance_diagnostics(
    data: SimulatedData,
    l_hat: ArrayLike,
    m_hat: ArrayLike,
    theta0: float,
) -> NuisanceDiagnostics:
    l_prediction = np.asarray(l_hat, dtype=float).reshape(-1)
    m_prediction = np.asarray(m_hat, dtype=float).reshape(-1)
    if len(l_prediction) != len(data.y) or len(m_prediction) != len(data.y):
        raise ValueError("Nuisance predictions must match the simulated sample size.")
    if not np.isfinite(l_prediction).all() or not np.isfinite(m_prediction).all():
        raise ValueError("Nuisance predictions must be finite.")

    delta_l = l_prediction - data.l0
    delta_m = m_prediction - data.m0
    l_mse = float(np.mean(delta_l**2))
    m_mse = float(np.mean(delta_m**2))
    lm_error_cross = float(np.mean(delta_l * delta_m))
    theta_value = float(theta0)
    return NuisanceDiagnostics(
        l_mse=l_mse,
        m_mse=m_mse,
        lm_error_cross=lm_error_cross,
        residual_d_variance=float(np.mean((data.d - m_prediction) ** 2)),
        bias_numerator_proxy=lm_error_cross - theta_value * m_mse,
        theta_proxy=(theta_value + lm_error_cross) / (1.0 + m_mse),
    )
