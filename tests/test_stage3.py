import numpy as np

from tabdml.config import TaskSpec, derive_seed
from tabdml.stage3 import Stage3TaskSpec, legacy_learner_seed, run_stage3_task


def _task(learner_l="tabiclv2_1", learner_m="xgboost", seed_namespace="stage3"):
    return Stage3TaskSpec(
        stage="stage3_tree_diagnosis",
        seed_namespace=seed_namespace,
        scenario="tree",
        n=80,
        p=10,
        replication=2,
        learner_l=learner_l,
        learner_m=learner_m,
        tabicl_estimators=1,
    )


def test_stage3_key_contains_ordered_nuisance_learners():
    forward = _task("tabiclv2_1", "xgboost")
    reverse = _task("xgboost", "tabiclv2_1")

    assert "__ltabiclv2_1__mxgboost__" in forward.key
    assert "__lxgboost__mtabiclv2_1__" in reverse.key
    assert forward.key != reverse.key


def test_stage2_seed_namespace_reproduces_legacy_learner_seed():
    task = _task("xgboost", "xgboost", seed_namespace="stage2")
    legacy = TaskSpec(
        "stage2",
        task.scenario,
        task.n,
        task.p,
        task.replication,
        "xgboost",
        0,
    )

    assert legacy_learner_seed(task, "xgboost") == derive_seed(
        legacy.key, "learners"
    )


def test_oracle_stage3_task_writes_exact_zero_nuisance_errors(tmp_path):
    task = _task("oracle", "oracle")

    result = run_stage3_task(
        task,
        folds_count=4,
        theta0=1.0,
        output_root=tmp_path,
        fast=True,
    )

    assert result["status"] == "success"
    assert result["learner_l"] == "oracle"
    assert result["learner_m"] == "oracle"
    assert result["l_mse"] == 0.0
    assert result["m_mse"] == 0.0
    assert result["nuisance_error_product"] == 0.0
    assert np.isfinite(result["theta"])
    assert result["standard_error"] > 0
    assert (tmp_path / f"{task.key}.json").exists()

