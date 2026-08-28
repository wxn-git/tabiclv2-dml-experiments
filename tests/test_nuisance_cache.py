import numpy as np
import pytest

from tabdml.nuisance_cache import NuisanceCache, NuisanceTaskSpec


def _task(target="l"):
    return NuisanceTaskSpec(
        seed_namespace="stage3b",
        scenario="tree",
        n=80,
        p=10,
        replication=0,
        target=target,
        learner="lasso",
        tabicl_estimators=0,
        folds_count=5,
        learner_seed=123,
    )


def test_nuisance_cache_round_trip(tmp_path):
    task = _task()
    cache = NuisanceCache(tmp_path)
    cache.write(
        task,
        prediction=np.arange(80.0),
        fold_seconds=(0.1, 0.2),
        peak_gpu_mb=None,
        fallback_reason=None,
    )

    loaded = cache.read(task, expected_length=80)

    np.testing.assert_array_equal(loaded.prediction, np.arange(80.0))
    assert loaded.fold_seconds == (0.1, 0.2)
    assert loaded.peak_gpu_mb is None
    assert loaded.fallback_reason is None


def test_nuisance_cache_rejects_nonfinite_payload(tmp_path):
    task = _task("m")
    cache = NuisanceCache(tmp_path)
    cache.write(task, np.full(80, np.nan), (), None, None)

    with pytest.raises(ValueError, match="finite"):
        cache.read(task, expected_length=80)


def test_nuisance_cache_rejects_wrong_expected_length(tmp_path):
    task = _task()
    cache = NuisanceCache(tmp_path)
    cache.write(task, np.arange(80.0), (), None, None)

    with pytest.raises(ValueError, match="length"):
        cache.read(task, expected_length=79)
