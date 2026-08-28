import pandas as pd

from tabdml.config import load_config
from tabdml.stages import enumerate_stage1_tasks, select_stage2


def test_stage1_enumerates_expected_task_count():
    cfg = load_config("configs/stage1.yaml")
    assert len(list(enumerate_stage1_tasks(cfg))) == 48 * 20 * 6


def test_selection_returns_seven_unique_configs_with_baseline():
    rows = []
    for index, scenario in enumerate(["linear", "smooth", "tree", "mixed"]):
        for n, p in [(500, 10), (1000, 50), (2000, 100)]:
            base = 0.08 + 0.01 * index + n / 1_000_000 + p / 100_000
            rows.extend(
                [
                    {"scenario": scenario, "n": n, "p": p, "learner": "lasso", "rmse": base},
                    {
                        "scenario": scenario,
                        "n": n,
                        "p": p,
                        "learner": "tabiclv2",
                        "rmse": base + (index - 1.5) * 0.02 + n / 10_000_000,
                    },
                ]
            )
    selected = select_stage2(pd.DataFrame(rows))
    keys = {(x["scenario"], x["n"], x["p"]) for x in selected}
    assert len(selected) == len(keys) == 7
    assert ("linear", 2000, 50) in keys

