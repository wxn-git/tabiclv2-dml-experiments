from sklearn.base import clone

from tabdml.learners import make_configured_tree_learner
from tabdml.stage3b_screen import (
    ScreeningTaskSpec,
    _make_screening_model,
    run_screening_task,
    select_screening_winner,
)


def test_configured_extra_trees_is_cloneable():
    model = make_configured_tree_learner(
        "extra_trees",
        {
            "n_estimators": 20,
            "max_features": 1.0,
            "min_samples_leaf": 2,
        },
        seed=9,
        fast=True,
    )

    assert clone(model) is not model


def test_selection_uses_validation_d_mse_not_m0_mse():
    records = [
        {
            "candidate": "a",
            "candidate_group": "xgboost_tuned",
            "status": "success",
            "training_target": "d",
            "validation_d_mse": 1.2,
            "validation_m0_mse": 0.01,
            "params": {"max_depth": 2},
        },
        {
            "candidate": "b",
            "candidate_group": "xgboost_tuned",
            "status": "success",
            "training_target": "d",
            "validation_d_mse": 1.1,
            "validation_m0_mse": 0.50,
            "params": {"max_depth": 4},
        },
        {
            "candidate": "oracle_leak",
            "candidate_group": "xgboost_tuned",
            "status": "success",
            "training_target": "m0",
            "validation_d_mse": 0.1,
            "validation_m0_mse": 0.0,
            "params": {"max_depth": 9},
        },
    ]

    selected = select_screening_winner(records, "xgboost_tuned")

    assert selected["candidate"] == "b"
    assert selected["selection_metric"] == "mean_validation_d_mse"


def test_fast_screening_task_records_observable_and_oracle_metrics(tmp_path):
    task = ScreeningTaskSpec(
        stage="stage3b_mscreen_pilot",
        seed_namespace="stage3b_mscreen_pilot",
        scenario="tree",
        n=80,
        p=10,
        replication=0,
        candidate="extra_leaf2",
        candidate_group="extra_trees",
        learner_kind="extra_trees",
        params={
            "n_estimators": 20,
            "max_features": 1.0,
            "min_samples_leaf": 2,
        },
        training_target="d",
        validation_fraction=0.25,
    )

    record = run_screening_task(task, output_root=tmp_path, fast=True)

    assert record["status"] == "success"
    assert record["training_target"] == "d"
    assert record["validation_d_mse"] > 0
    assert record["validation_m0_mse"] >= 0
    assert (tmp_path / f"{task.key}.json").exists()


def test_current_xgboost_baseline_uses_existing_project_factory():
    task = ScreeningTaskSpec(
        stage="stage3b_mscreen_pilot",
        seed_namespace="stage3b_mscreen_pilot",
        scenario="tree",
        n=80,
        p=10,
        replication=0,
        candidate="current_xgboost",
        candidate_group="xgb_baseline",
        learner_kind="xgboost",
        params={},
        training_target="d",
        validation_fraction=0.25,
    )

    model = _make_screening_model(task, seed=7, fast=True)

    assert model.max_depth == 3
    assert model.learning_rate == 0.1
    assert model.subsample == 0.8
