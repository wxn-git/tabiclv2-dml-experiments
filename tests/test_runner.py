from tabdml.config import TaskSpec
from tabdml.runner import classify_failure


def test_cuda_memory_failure_is_classified_as_oom():
    assert classify_failure(RuntimeError("CUDA out of memory")) == "oom"
    assert classify_failure(MemoryError()) == "oom"
    assert classify_failure(ValueError("bad input")) == "failed"


def test_task_identity_is_preserved():
    task = TaskSpec("stage1", "tree", 500, 10, 2, "lasso", 0)
    assert task.key.startswith("stage1__tree")

