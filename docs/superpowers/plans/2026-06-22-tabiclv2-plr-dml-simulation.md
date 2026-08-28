# TabICLv2-PLR-DML Simulation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible Monte Carlo experiment that compares TabICLv2 with Lasso, random forest, XGBoost, MLP, and a traditional ensemble as nuisance learners in five-fold cross-fitted PLR-DML.

**Architecture:** A small Python package separates data-generating processes, learner factories, cross-fitted DML estimation, durable task execution, aggregation, and reporting. Every Monte Carlo task is deterministic and atomically persisted; optional heavy learners are loaded lazily so statistical unit tests can run without downloading model weights.

**Tech Stack:** Python 3.12, numpy, pandas, scipy, scikit-learn, xgboost, torch, tabicl, doubleml, matplotlib, seaborn, pytest, PyYAML.

## Global Constraints

- The structural parameter is fixed at `theta0 = 1.0`.
- Scenarios are `linear`, `smooth`, `tree`, and `mixed`.
- Stage-one grid is `n in {500, 1000, 2000, 5000}` and `p in {10, 50, 100}`, with 20 replications and five folds.
- Stage one uses `TabICLv2 n_estimators=1`; selected stage-two configurations use 100 new replications and `n_estimators=8`.
- Every learner uses identical generated data and identical outer folds.
- Preprocessing, tuning, early stopping, and ensemble weighting use outer-training data only.
- The available GPU is an NVIDIA GeForce RTX 5060 Laptop GPU with 8,151 MiB VRAM; OOM must be recorded, not imputed.
- TabICLv2 pretraining cost is excluded from downstream runtime and disclosed in reports.
- The directory is not a Git repository, so commit steps are recorded as not applicable.

---

### Task 1: Project configuration and deterministic experiment identities

**Files:**
- Create: `pyproject.toml`
- Create: `configs/stage1.yaml`
- Create: `configs/stage2.yaml`
- Create: `src/tabdml/__init__.py`
- Create: `src/tabdml/config.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Produces: `ExperimentConfig`, `TaskSpec`, `load_config(path)`, `derive_seed(*parts)`, and `TaskSpec.key`.

- [ ] **Step 1: Write the failing configuration tests**

```python
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
```

- [ ] **Step 2: Run the tests and verify RED**

Run: `python -m pytest tests/test_config.py -v`
Expected: collection fails with `ModuleNotFoundError: No module named 'tabdml'`.

- [ ] **Step 3: Implement configuration types and YAML files**

```python
@dataclass(frozen=True)
class TaskSpec:
    stage: str
    scenario: str
    n: int
    p: int
    replication: int
    learner: str
    tabicl_estimators: int

    @property
    def key(self) -> str:
        return (
            f"{self.stage}__{self.scenario}__n{self.n}__p{self.p}"
            f"__r{self.replication:03d}__{self.learner}__e{self.tabicl_estimators}"
        )


def derive_seed(*parts: object) -> int:
    digest = hashlib.blake2b("|".join(map(str, parts)).encode(), digest_size=8).digest()
    return int.from_bytes(digest, "little") % (2**32 - 1)
```

The YAML files must list the exact grids and learners from Global Constraints.

- [ ] **Step 4: Verify GREEN**

Run: `python -m pytest tests/test_config.py -v`
Expected: 3 tests pass.

### Task 2: Four calibrated PLR data-generating processes

**Files:**
- Create: `src/tabdml/dgp.py`
- Test: `tests/test_dgp.py`

**Interfaces:**
- Produces: `SimulatedData(X, y, d, l0, m0, g0, theta0, categorical_indices)`.
- Produces: `simulate_plr(scenario, n, p, seed, theta0=1.0)`.

- [ ] **Step 1: Write failing tests for shape, reproducibility, types, and structural identity**

```python
import numpy as np
import pytest
from tabdml.dgp import simulate_plr


@pytest.mark.parametrize("scenario", ["linear", "smooth", "tree", "mixed"])
def test_dgp_is_reproducible_and_obeys_plr_identity(scenario):
    a = simulate_plr(scenario, n=300, p=10, seed=7)
    b = simulate_plr(scenario, n=300, p=10, seed=7)
    np.testing.assert_allclose(a.X, b.X)
    np.testing.assert_allclose(a.l0, a.theta0 * a.m0 + a.g0)
    assert a.X.shape == (300, 10)
    assert a.y.shape == a.d.shape == (300,)


def test_mixed_scenario_marks_binary_and_categorical_columns():
    data = simulate_plr("mixed", 300, 10, 11)
    assert len(data.categorical_indices) >= 2
```

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests/test_dgp.py -v`
Expected: import failure for `tabdml.dgp`.

- [ ] **Step 3: Implement calibrated structural functions**

Use deterministic formulas:

```python
linear_m = 0.8*x0 - 0.6*x1 + 0.4*x2
linear_g = 0.7*x0 + 0.5*x1 - 0.4*x3
smooth_m = np.sin(x0) + 0.5*x1**2 - 0.4*np.exp(-x2**2)
smooth_g = 0.8*np.cos(x0) + 0.5*np.abs(x1) + 0.3*x2*x3
tree_m = 0.9*(x0 > 0) - 0.7*(x1 > 0.5) + 0.5*(x2*x3 > 0)
tree_g = 0.8*(x0+x1 > 0) + 0.6*(x2 > 0)*(x3 < 0) - 0.5*(x4 > 0.5)
```

For `mixed`, convert the last columns into binary and three-level categorical values and include cross-type interactions. Center and rescale every raw structural function to unit standard deviation. Generate `v, epsilon ~ N(0,1)`, `d=m0+v`, `y=theta0*d+g0+epsilon`, and `l0=theta0*m0+g0`.

- [ ] **Step 4: Verify GREEN**

Run: `python -m pytest tests/test_dgp.py -v`
Expected: 5 tests pass.

### Task 3: PLR-DML point estimate, robust inference, and oracle validation

**Files:**
- Create: `src/tabdml/dml.py`
- Test: `tests/test_dml.py`

**Interfaces:**
- Produces: `DMLResult(theta, standard_error, ci_lower, ci_upper, score)`.
- Produces: `estimate_plr_dml(y, d, l_hat, m_hat, alpha=0.05)`.

- [ ] **Step 1: Write failing formula tests**

```python
import numpy as np
from tabdml.dgp import simulate_plr
from tabdml.dml import estimate_plr_dml


def test_point_estimate_matches_manual_residual_regression():
    y = np.array([2.0, 4.0, 5.0, 8.0])
    d = np.array([1.0, 2.0, 2.0, 4.0])
    zeros = np.zeros(4)
    result = estimate_plr_dml(y, d, zeros, zeros)
    assert np.isclose(result.theta, d @ y / (d @ d))


def test_oracle_nuisances_recover_theta():
    data = simulate_plr("smooth", 20000, 10, 99)
    result = estimate_plr_dml(data.y, data.d, data.l0, data.m0)
    assert abs(result.theta - 1.0) < 0.04
    assert result.ci_lower < result.theta < result.ci_upper
```

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests/test_dml.py -v`
Expected: import failure for `tabdml.dml`.

- [ ] **Step 3: Implement the orthogonal estimator**

```python
v = np.asarray(d) - np.asarray(m_hat)
u = np.asarray(y) - np.asarray(l_hat)
denominator = np.mean(v * v)
theta = np.mean(v * u) / denominator
score = v * (u - theta * v)
standard_error = np.sqrt(np.mean(score * score) / (len(y) * denominator**2))
critical = scipy.stats.norm.ppf(1 - alpha / 2)
```

Raise `ValueError` for unequal lengths, non-finite values, or denominator below `1e-12`.

- [ ] **Step 4: Verify GREEN**

Run: `python -m pytest tests/test_dml.py -v`
Expected: 2 tests pass.

### Task 4: Leakage-safe nuisance learner factories

**Files:**
- Create: `src/tabdml/learners.py`
- Create: `src/tabdml/ensemble.py`
- Test: `tests/test_learners.py`

**Interfaces:**
- Produces: `make_learner(name, seed, categorical_indices=(), tabicl_estimators=1)`.
- Produces sklearn-compatible regressors for `lasso`, `random_forest`, `xgboost`, `mlp`, `ensemble`, and `tabiclv2`.
- Produces: `ConvexOOFEnsemble`.

- [ ] **Step 1: Write failing interface and leakage tests**

```python
import numpy as np
import pytest
from sklearn.base import clone
from tabdml.learners import make_learner


@pytest.mark.parametrize("name", ["lasso", "random_forest", "xgboost", "mlp"])
def test_traditional_learner_is_cloneable_and_predicts(name):
    model = make_learner(name, seed=1)
    clone(model).fit(np.random.default_rng(1).normal(size=(40, 6)), np.arange(40.0))
    assert clone(model) is not model


def test_preprocessing_is_inside_pipeline():
    model = make_learner("lasso", seed=1)
    assert "preprocess" in model.named_steps
```

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests/test_learners.py -v`
Expected: import failure for `tabdml.learners`.

- [ ] **Step 3: Implement fixed, lightweight model specifications**

Use:

- Lasso: `StandardScaler` + `LassoCV(alphas=logspace(-4,1,20), cv=3)`.
- Random forest: inner `GridSearchCV` over `max_features in {0.5,1.0}` and `min_samples_leaf in {2,10}`, 300 trees.
- XGBoost: inner `GridSearchCV` over depth `{3,6}` and learning rate `{0.03,0.1}`, 500 trees with fixed subsampling.
- MLP: `StandardScaler` + `GridSearchCV` over `(64,)` and `(128,64)`, early stopping enabled.
- Ensemble: four base models, three-fold OOF predictions, non-negative simplex weights solved with `scipy.optimize.minimize`; equal weights on optimization failure.
- TabICLv2: lazy import of `TabICLRegressor`; set `n_estimators` and device, expose categorical indices only through supported fit parameters.

- [ ] **Step 4: Verify GREEN for traditional learners**

Run: `python -m pytest tests/test_learners.py -v`
Expected: 5 tests pass without downloading TabICLv2.

### Task 5: Shared cross-fitting and per-task diagnostics

**Files:**
- Create: `src/tabdml/crossfit.py`
- Test: `tests/test_crossfit.py`

**Interfaces:**
- Produces: `make_folds(n, folds, seed)`.
- Produces: `crossfit_nuisances(data, learner_name, folds, seed, tabicl_estimators)`.
- Produces: `CrossfitResult(l_hat, m_hat, fold_seconds, peak_gpu_mb, fallback_reason)`.

- [ ] **Step 1: Write failing shared-fold and out-of-fold tests**

```python
import numpy as np
from tabdml.crossfit import make_folds, crossfit_nuisances
from tabdml.dgp import simulate_plr


def test_folds_are_deterministic_and_partition_all_rows():
    a = make_folds(101, 5, 42)
    b = make_folds(101, 5, 42)
    assert all(np.array_equal(x[1], y[1]) for x, y in zip(a, b))
    assert sorted(np.concatenate([test for _, test in a]).tolist()) == list(range(101))


def test_crossfit_produces_one_prediction_per_row():
    data = simulate_plr("linear", 120, 10, 3)
    result = crossfit_nuisances(data, "lasso", make_folds(120, 5, 4), 5, 0)
    assert np.isfinite(result.l_hat).all()
    assert np.isfinite(result.m_hat).all()
```

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests/test_crossfit.py -v`
Expected: import failure for `tabdml.crossfit`.

- [ ] **Step 3: Implement fold-local fitting**

For each outer fold, instantiate two fresh learners using fold-specific seeds, fit one to `y[train]` and one to `d[train]`, then predict only `test`. Measure each fold with `time.perf_counter`; synchronize CUDA before timing and read peak memory when torch/CUDA are available.

- [ ] **Step 4: Verify GREEN**

Run: `python -m pytest tests/test_crossfit.py -v`
Expected: 2 tests pass.

### Task 6: Atomic task persistence, resume behavior, and structured failure records

**Files:**
- Create: `src/tabdml/storage.py`
- Create: `src/tabdml/runner.py`
- Test: `tests/test_storage.py`
- Test: `tests/test_runner.py`

**Interfaces:**
- Produces: `ResultStore(root)`, `ResultStore.exists(task)`, `ResultStore.write(record)`, `ResultStore.read_all()`.
- Produces: `run_task(task, retry_failed=False) -> dict`.

- [ ] **Step 1: Write failing persistence tests**

```python
from tabdml.config import TaskSpec
from tabdml.storage import ResultStore


def test_store_round_trip_and_resume(tmp_path):
    store = ResultStore(tmp_path)
    task = TaskSpec("stage1", "linear", 500, 10, 0, "lasso", 0)
    store.write({"task_key": task.key, "status": "success", "theta": 1.0})
    assert store.exists(task)
    assert store.read_all()[0]["theta"] == 1.0
```

Test `run_task` classifies `MemoryError` and CUDA out-of-memory messages as `oom`, all other exceptions as `failed`, and never overwrites a success unless explicitly requested.

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests/test_storage.py tests/test_runner.py -v`
Expected: import failures.

- [ ] **Step 3: Implement one-JSON-per-task atomic storage**

Write UTF-8 JSON to `<task_key>.json.tmp`, flush and close it, then use `os.replace` to create `<task_key>.json`. A successful task record includes DML estimates, nuisance MSEs, error product, runtime, fold times, seeds, and environment identifier.

- [ ] **Step 4: Verify GREEN**

Run: `python -m pytest tests/test_storage.py tests/test_runner.py -v`
Expected: all persistence and failure tests pass.

### Task 7: Stage orchestration and automatic selection of seven stage-two configurations

**Files:**
- Create: `src/tabdml/stages.py`
- Create: `scripts/run_stage1.py`
- Create: `scripts/select_stage2.py`
- Create: `scripts/run_stage2.py`
- Test: `tests/test_stages.py`

**Interfaces:**
- Produces: `enumerate_stage1_tasks(config)`.
- Produces: `select_stage2(summary, baseline=("linear", 2000, 50))`.
- Produces: `configs/stage2_selected.yaml`.

- [ ] **Step 1: Write failing enumeration and selection tests**

```python
from tabdml.config import load_config
from tabdml.stages import enumerate_stage1_tasks, select_stage2


def test_stage1_enumerates_expected_task_count():
    cfg = load_config("configs/stage1.yaml")
    assert len(list(enumerate_stage1_tasks(cfg))) == 48 * 20 * 6


def test_selection_returns_seven_unique_configs_with_baseline(fake_summary):
    selected = select_stage2(fake_summary)
    assert len(selected) == 7
    assert len({(x["scenario"], x["n"], x["p"]) for x in selected}) == 7
    assert {"scenario": "linear", "n": 2000, "p": 50} in selected
```

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests/test_stages.py -v`
Expected: import failure for `tabdml.stages`.

- [ ] **Step 3: Implement ranking and deduplication**

Aggregate only successful stage-one records. For each configuration, compute TabICLv2 RMSE minus the minimum RMSE among traditional learners. Select the three smallest and three largest differences, skip duplicates and the fixed baseline, then append the baseline. Emit stage-two tasks with replications `0..99`, traditional learner settings unchanged, and separate TabICLv2-1 and TabICLv2-8 labels.

- [ ] **Step 4: Verify GREEN**

Run: `python -m pytest tests/test_stages.py -v`
Expected: all stage tests pass.

### Task 8: Aggregation, figures, environment metadata, and Chinese report

**Files:**
- Create: `src/tabdml/aggregate.py`
- Create: `src/tabdml/report.py`
- Create: `scripts/aggregate_results.py`
- Create: `scripts/make_figures.py`
- Create: `scripts/environment_report.py`
- Create: `scripts/make_report.py`
- Create: `README.md`
- Test: `tests/test_aggregate.py`
- Test: `tests/test_report.py`

**Interfaces:**
- Produces: `summarize(records, theta0=1.0) -> pandas.DataFrame`.
- Produces: `results/summary_stage1.csv`, `results/summary_stage2.csv`, figures under `results/figures/`, `results/environment.json`, and `results/report_zh.md`.

- [ ] **Step 1: Write failing metric tests**

```python
import numpy as np
from tabdml.aggregate import summarize


def test_summary_metrics_are_correct():
    rows = [
        {"scenario":"linear","n":500,"p":10,"learner":"lasso","status":"success",
         "theta":0.9,"standard_error":0.1,"ci_lower":0.704,"ci_upper":1.096,"runtime_seconds":2.0},
        {"scenario":"linear","n":500,"p":10,"learner":"lasso","status":"success",
         "theta":1.1,"standard_error":0.1,"ci_lower":0.904,"ci_upper":1.296,"runtime_seconds":4.0},
    ]
    row = summarize(rows).iloc[0]
    assert np.isclose(row["bias"], 0.0)
    assert np.isclose(row["rmse"], 0.1)
    assert np.isclose(row["coverage"], 1.0)
    assert np.isclose(row["mean_runtime_seconds"], 3.0)
```

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests/test_aggregate.py tests/test_report.py -v`
Expected: import failures.

- [ ] **Step 3: Implement summaries and outputs**

Group by stage, scenario, `n`, `p`, learner, and TabICLv2 estimator count. Compute bias, RMSE, empirical SD, mean SE, coverage, interval length, runtime, nuisance MSEs, success count, failure count, OOM count, and Monte Carlo standard errors. Generate heatmaps, error-bar plots, nuisance-versus-treatment-error plots, and accuracy-cost Pareto plots.

The Chinese report must explicitly state:

- stage one is screening with 20 repetitions;
- stage two is data-selected validation;
- TabICLv2 pretraining cost is excluded;
- runtime is hardware-specific;
- “改进” is used only when differences are stable relative to Monte Carlo uncertainty.

- [ ] **Step 4: Verify GREEN**

Run: `python -m pytest tests/test_aggregate.py tests/test_report.py -v`
Expected: all aggregation and report tests pass.

### Task 9: Dependency installation, complete verification, smoke run, and DoubleML cross-check

**Files:**
- Create: `scripts/validate_doubleml.py`
- Create: `tests/test_doubleml_validation.py`
- Modify: `README.md`

**Interfaces:**
- Produces: `results/doubleml_validation.json`.
- Produces a documented smoke command that runs one replication for every learner.

- [ ] **Step 1: Install the locked project dependencies**

Run: `python -m pip install -e ".[test]"`
Expected: numpy, pandas, scipy, scikit-learn, xgboost, torch, tabicl, doubleml, plotting libraries, and pytest install successfully.

- [ ] **Step 2: Run the full fast test suite**

Run: `python -m pytest -m "not gpu and not integration" -v`
Expected: all fast tests pass.

- [ ] **Step 3: Add and verify the DoubleML numerical cross-check**

Construct one fixed linear dataset and fixed five-fold split. Feed identical out-of-fold nuisance predictions to the custom estimator and compare against the corresponding DoubleML PLR score calculation with tolerance `1e-8` for theta and `1e-6` for standard error.

Run: `python -m pytest tests/test_doubleml_validation.py -v`
Expected: PASS.

- [ ] **Step 4: Run traditional-learner smoke experiments**

Run: `python scripts/run_stage1.py --scenarios linear --sample-sizes 500 --dimensions 10 --replications 1 --learners lasso random_forest xgboost mlp ensemble`
Expected: five successful JSON task records.

- [ ] **Step 5: Run the TabICLv2 smoke experiment**

Run: `python scripts/run_stage1.py --scenarios linear --sample-sizes 500 --dimensions 10 --replications 1 --learners tabiclv2`
Expected: checkpoint downloads once, then one successful record; if 8GB VRAM is insufficient, an `oom` record is produced without crashing.

- [ ] **Step 6: Generate smoke summaries and report**

Run: `python scripts/aggregate_results.py`

Run: `python scripts/environment_report.py`

Run: `python scripts/make_report.py`

Expected: CSV summary, environment JSON, and Chinese Markdown report are produced.

- [ ] **Step 7: Run final verification**

Run: `python -m pytest -v`
Expected: all tests pass; GPU/integration tests either pass or carry explicit skip reasons.

Run: `python -m compileall src scripts`
Expected: compilation succeeds with no syntax errors.
