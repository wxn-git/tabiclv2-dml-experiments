from tabdml.config import TaskSpec
from tabdml.stage3 import Stage3TaskSpec
from tabdml.storage import ResultStore


def test_store_round_trip_and_resume(tmp_path):
    store = ResultStore(tmp_path)
    task = TaskSpec("stage1", "linear", 500, 10, 0, "lasso", 0)
    store.write({"task_key": task.key, "status": "success", "theta": 1.0})
    assert store.exists(task)
    assert store.read_all()[0]["theta"] == 1.0


def test_store_does_not_treat_failed_task_as_success(tmp_path):
    store = ResultStore(tmp_path)
    task = TaskSpec("stage1", "linear", 500, 10, 0, "lasso", 0)
    store.write({"task_key": task.key, "status": "failed"})
    assert not store.exists(task)


def test_store_accepts_stage3_task_objects(tmp_path):
    store = ResultStore(tmp_path)
    task = Stage3TaskSpec(
        "stage3_tree_diagnosis",
        "stage3_tree_diagnosis",
        "tree",
        80,
        10,
        0,
        "oracle",
        "xgboost",
        1,
    )
    store.write({"task_key": task.key, "status": "success"})

    assert store.exists(task)
