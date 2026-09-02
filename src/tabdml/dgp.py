from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class SimulatedData:
    X: NDArray[np.float64]
    y: NDArray[np.float64]
    d: NDArray[np.float64]
    l0: NDArray[np.float64]
    m0: NDArray[np.float64]
    g0: NDArray[np.float64]
    theta0: float
    categorical_indices: tuple[int, ...]


def _unit_scale(values: NDArray[np.float64]) -> NDArray[np.float64]:
    centered = values - values.mean()
    scale = centered.std()
    if scale < 1e-12:
        raise ValueError("Structural function has zero variance.")
    return centered / scale


def _base_columns(X: NDArray[np.float64]) -> tuple[NDArray[np.float64], ...]:
    return tuple(X[:, i % X.shape[1]] for i in range(6))


def simulate_plr(
    scenario: str,
    n: int,
    p: int,
    seed: int,
    theta0: float = 1.0,
) -> SimulatedData:
    if n < 10 or p < 5:
        raise ValueError("PLR simulations require n >= 10 and p >= 5.")
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    categorical_indices: tuple[int, ...] = ()

    if scenario == "mixed":
        binary_idx = p - 2
        category_idx = p - 1
        X[:, binary_idx] = (X[:, binary_idx] > 0).astype(float)
        X[:, category_idx] = np.digitize(X[:, category_idx], [-0.45, 0.45]).astype(float)
        categorical_indices = (binary_idx, category_idx)

    x0, x1, x2, x3, x4, x5 = _base_columns(X)
    if scenario == "linear":
        raw_m = 0.8 * x0 - 0.6 * x1 + 0.4 * x2
        raw_g = 0.7 * x0 + 0.5 * x1 - 0.4 * x3
    elif scenario == "smooth":
        raw_m = np.sin(x0) + 0.5 * x1**2 - 0.4 * np.exp(-(x2**2))
        raw_g = 0.8 * np.cos(x0) + 0.5 * np.abs(x1) + 0.3 * x2 * x3
    elif scenario == "tree":
        raw_m = 0.9 * (x0 > 0) - 0.7 * (x1 > 0.5) + 0.5 * (x2 * x3 > 0)
        raw_g = (
            0.8 * (x0 + x1 > 0)
            + 0.6 * (x2 > 0) * (x3 < 0)
            - 0.5 * (x4 > 0.5)
        )
    elif scenario == "tree_simple":
        raw_m = 0.9 * (x0 > 0) - 0.7 * (x1 > 0) + 0.5 * (x2 > 0)
        raw_g = 0.8 * (x0 > 0) + 0.6 * (x3 > 0) - 0.5 * (x4 > 0)
    elif scenario == "mixed":
        binary = X[:, p - 2]
        category = X[:, p - 1]
        raw_m = np.sin(x0) + 0.7 * binary - 0.45 * (category == 2) + 0.4 * x1 * binary
        raw_g = 0.6 * x0**2 + 0.5 * (category == 1) - 0.4 * binary * (x2 > 0)
    else:
        raise ValueError(f"Unknown scenario: {scenario}")

    m0 = _unit_scale(np.asarray(raw_m, dtype=float))
    g0 = _unit_scale(np.asarray(raw_g, dtype=float))
    v = rng.normal(size=n)
    epsilon = rng.normal(size=n)
    d = m0 + v
    y = theta0 * d + g0 + epsilon
    l0 = theta0 * m0 + g0
    return SimulatedData(
        X=np.asarray(X, dtype=float),
        y=np.asarray(y, dtype=float),
        d=np.asarray(d, dtype=float),
        l0=np.asarray(l0, dtype=float),
        m0=m0,
        g0=g0,
        theta0=float(theta0),
        categorical_indices=categorical_indices,
    )
