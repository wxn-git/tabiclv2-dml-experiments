import numpy as np

from tabdml.dgp import simulate_plr
from tabdml.diagnostics import compute_nuisance_diagnostics


def test_diagnostics_compute_signed_cross_term_and_proxy():
    data = simulate_plr("tree", 80, 10, 19)
    delta_l = np.linspace(-0.2, 0.2, 80)
    delta_m = np.linspace(0.1, -0.1, 80)

    result = compute_nuisance_diagnostics(
        data,
        data.l0 + delta_l,
        data.m0 + delta_m,
        theta0=1.0,
    )

    assert np.isclose(result.l_mse, np.mean(delta_l**2))
    assert np.isclose(result.m_mse, np.mean(delta_m**2))
    assert np.isclose(result.lm_error_cross, np.mean(delta_l * delta_m))
    assert np.isclose(
        result.theta_proxy,
        (1.0 + np.mean(delta_l * delta_m)) / (1.0 + np.mean(delta_m**2)),
    )
    assert np.isclose(
        result.bias_numerator_proxy,
        np.mean(delta_l * delta_m) - np.mean(delta_m**2),
    )


def test_oracle_diagnostics_are_zero():
    data = simulate_plr("tree", 80, 10, 20)

    result = compute_nuisance_diagnostics(
        data,
        data.l0,
        data.m0,
        theta0=1.0,
    )

    assert result.l_mse == 0.0
    assert result.m_mse == 0.0
    assert result.lm_error_cross == 0.0
    assert result.bias_numerator_proxy == 0.0
    assert result.theta_proxy == 1.0
