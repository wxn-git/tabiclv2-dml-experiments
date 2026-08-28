import numpy as np

from tabdml.aggregate import summarize


def test_summary_metrics_are_correct():
    rows = [
        {
            "stage": "stage1",
            "scenario": "linear",
            "n": 500,
            "p": 10,
            "learner": "lasso",
            "tabicl_estimators": 0,
            "status": "success",
            "theta": 0.9,
            "standard_error": 0.1,
            "ci_lower": 0.704,
            "ci_upper": 1.096,
            "runtime_seconds": 2.0,
            "l_mse": 0.3,
            "m_mse": 0.2,
        },
        {
            "stage": "stage1",
            "scenario": "linear",
            "n": 500,
            "p": 10,
            "learner": "lasso",
            "tabicl_estimators": 0,
            "status": "success",
            "theta": 1.1,
            "standard_error": 0.1,
            "ci_lower": 0.904,
            "ci_upper": 1.296,
            "runtime_seconds": 4.0,
            "l_mse": 0.4,
            "m_mse": 0.3,
        },
    ]
    row = summarize(rows).iloc[0]
    assert np.isclose(row["bias"], 0.0)
    assert np.isclose(row["rmse"], 0.1)
    assert np.isclose(row["coverage"], 1.0)
    assert np.isclose(row["mean_runtime_seconds"], 3.0)
