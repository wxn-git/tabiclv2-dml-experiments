import pandas as pd

from tabdml.figures import make_accuracy_cost_figure


def test_accuracy_cost_figure_renders_without_gui(tmp_path):
    summary = pd.DataFrame(
        [
            {
                "learner": "lasso",
                "scenario": "linear",
                "rmse": 0.1,
                "mean_runtime_seconds": 1.0,
            },
            {
                "learner": "tabiclv2",
                "scenario": "linear",
                "rmse": 0.08,
                "mean_runtime_seconds": 5.0,
            },
        ]
    )
    output = make_accuracy_cost_figure(summary, tmp_path)
    assert output.exists()
    assert output.stat().st_size > 0

