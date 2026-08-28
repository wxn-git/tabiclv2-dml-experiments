import numpy as np

from tabdml.stage3b_aggregate import aggregate_dml_records, markdown_table


def test_aggregate_reports_core_inference_and_proxy_metrics():
    records = []
    for replication, theta in enumerate((0.9, 1.1)):
        records.append(
            {
                "status": "success",
                "stage": "stage3b_confirmation",
                "scenario": "tree",
                "n": 80,
                "p": 10,
                "replication": replication,
                "learner_l": "lasso",
                "learner_m": "lasso",
                "learner_l_config_hash": "default",
                "learner_m_config_hash": "default",
                "theta": theta,
                "standard_error": 0.1,
                "ci_lower": theta - 0.2,
                "ci_upper": theta + 0.2,
                "l_mse": 0.2,
                "m_mse": 0.1,
                "lm_error_cross": 0.03,
                "theta_proxy": 0.95,
                "proxy_error": theta - 0.95,
                "runtime_seconds": 1.0,
            }
        )

    summary = aggregate_dml_records(records, theta0=1.0)
    row = summary.iloc[0]

    assert np.isclose(row["bias"], 0.0)
    assert np.isclose(row["rmse"], 0.1)
    assert row["coverage"] == 1.0
    assert row["success_count"] == 2
    assert {
        "mean_m_mse",
        "mean_lm_error_cross",
        "mean_theta_proxy",
        "mean_proxy_error",
    } <= set(summary.columns)


def test_markdown_table_does_not_require_optional_tabulate_dependency():
    text = markdown_table(
        __import__("pandas").DataFrame({"method": ["tab"], "rmse": [0.123456]}),
        ["method", "rmse"],
    )

    assert "| method | rmse |" in text
    assert "| tab | 0.1235 |" in text
