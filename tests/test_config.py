from tabdml.config import TaskSpec, derive_seed, load_config


def test_seed_is_stable_and_sensitive():
    assert derive_seed("linear", 500, 10, 0) == derive_seed("linear", 500, 10, 0)
    assert derive_seed("linear", 500, 10, 0) != derive_seed("linear", 500, 10, 1)


def test_task_key_contains_all_identity_fields():
    task = TaskSpec("stage1", "linear", 500, 10, 0, "lasso", 0)
    assert task.key == "stage1__linear__n500__p10__r000__lasso__e0"


def test_stage1_grid_has_48_configurations():
    cfg = load_config("configs/stage1.yaml")
    assert len(cfg.scenarios) * len(cfg.sample_sizes) * len(cfg.dimensions) == 48
    assert cfg.folds == 5
    assert cfg.replications == 20

