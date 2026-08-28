# Stage 3B Publication Diagnostics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run a cached, resumable Stage 3B pipeline that decomposes the tree-DGP DML bias, screens stronger treatment nuisance learners, and confirms whether lower `m_mse` improves inference.

**Architecture:** Split nuisance fitting from DML composition. A single-target cross-fit primitive writes deterministic OOF prediction caches; diagnostic and confirmation runners combine cached `l` and `m` predictions without retraining. Screening uses an independent seed namespace and observable validation `D` MSE to freeze one XGBoost and one ExtraTrees configuration before confirmation.

**Tech Stack:** Python 3.12, NumPy, pandas, SciPy, scikit-learn 1.9, XGBoost 3.3 CPU, PyTorch 2.11 CUDA, TabICLv2 2.1.1, PyYAML, pytest.

## Global Constraints

- Keep `tree`, `theta0=1.0`, `n=2000`, `p=10`, and five outer folds unchanged.
- Use one TabICLv2 GPU worker and at most eight CPU workers.
- Never overwrite Stage 1, Stage 2, or Stage 3A raw results.
- Batch A uses `stage3_tree_diagnosis`; screening uses `stage3b_mscreen_pilot`; confirmation uses `stage3b_confirmation`.
- Formal candidate selection uses validation `D` MSE only; `m0` is diagnostic-only.
- Every cache/result write is atomic and resumable; failures and fallbacks remain explicit.
- Preserve exact legacy Stage 3A learner seeds for TabICLv2-1 and current XGBoost.
- The workspace has no usable Git repository; replace commit checkpoints with fresh focused and full-suite test evidence.

---

### Task 1: Nuisance Error Decomposition

**Files:**
- Create: `src/tabdml/diagnostics.py`
- Create: `tests/test_diagnostics.py`

**Interfaces:**
- Consumes: `SimulatedData`, OOF `l_hat`, OOF `m_hat`, and `theta0`.
- Produces: `NuisanceDiagnostics` and `compute_nuisance_diagnostics(data, l_hat, m_hat, theta0)`.

- [ ] **Step 1: Write failing formula tests**

```python
def test_diagnostics_compute_signed_cross_term_and_proxy():
    data = simulate_plr("tree", 80, 10, 19)
    dl = np.linspace(-0.2, 0.2, 80)
    dm = np.linspace(0.1, -0.1, 80)
    result = compute_nuisance_diagnostics(data, data.l0 + dl, data.m0 + dm, 1.0)
    assert np.isclose(result.l_mse, np.mean(dl**2))
    assert np.isclose(result.m_mse, np.mean(dm**2))
    assert np.isclose(result.lm_error_cross, np.mean(dl * dm))
    assert np.isclose(result.theta_proxy, (1 + np.mean(dl * dm)) / (1 + np.mean(dm**2)))

def test_oracle_diagnostics_are_zero():
    data = simulate_plr("tree", 80, 10, 20)
    result = compute_nuisance_diagnostics(data, data.l0, data.m0, 1.0)
    assert result.l_mse == result.m_mse == result.lm_error_cross == 0.0
    assert result.theta_proxy == 1.0
```

- [ ] **Step 2: Run the focused test and confirm an import failure**

Run: `.venv\Scripts\python.exe -m pytest tests/test_diagnostics.py -q`

Expected: FAIL because `tabdml.diagnostics` does not exist.

- [ ] **Step 3: Implement the immutable diagnostic result**

```python
@dataclass(frozen=True)
class NuisanceDiagnostics:
    l_mse: float
    m_mse: float
    lm_error_cross: float
    residual_d_variance: float
    bias_numerator_proxy: float
    theta_proxy: float

def compute_nuisance_diagnostics(data, l_hat, m_hat, theta0):
    dl = np.asarray(l_hat, dtype=float) - data.l0
    dm = np.asarray(m_hat, dtype=float) - data.m0
    l_mse = float(np.mean(dl**2))
    m_mse = float(np.mean(dm**2))
    cross = float(np.mean(dl * dm))
    return NuisanceDiagnostics(
        l_mse=l_mse,
        m_mse=m_mse,
        lm_error_cross=cross,
        residual_d_variance=float(np.mean((data.d - np.asarray(m_hat)) ** 2)),
        bias_numerator_proxy=cross - float(theta0) * m_mse,
        theta_proxy=float((float(theta0) + cross) / (1.0 + m_mse)),
    )
```

- [ ] **Step 4: Run the focused test and full suite**

Run: `.venv\Scripts\python.exe -m pytest tests/test_diagnostics.py -q`

Expected: 2 passed.

Run: `.venv\Scripts\python.exe -m pytest -q`

Expected: all existing and new tests pass.

---

### Task 2: Single-Target Cross-Fitting and Atomic Prediction Cache

**Files:**
- Modify: `src/tabdml/crossfit.py`
- Create: `src/tabdml/nuisance_cache.py`
- Modify: `tests/test_crossfit.py`
- Create: `tests/test_nuisance_cache.py`

**Interfaces:**
- Produces: `SingleNuisanceResult`, `crossfit_single_nuisance(...)`, `NuisanceTaskSpec`, and `NuisanceCache`.
- Cache payload: `prediction`, `fold_seconds`, `peak_gpu_mb`, `fallback_reason`, and exact task metadata.

- [ ] **Step 1: Add failing l/m seed-compatibility tests**

```python
def test_single_crossfit_matches_pair_sides_exactly():
    data = simulate_plr("linear", 90, 10, 41)
    folds = make_folds(90, 3, 19)
    pair = crossfit_nuisance_pair(data, "lasso", "lasso", folds, 303, 303, 0, fast=True)
    l = crossfit_single_nuisance(data, "l", "lasso", folds, 303, 0, fast=True)
    m = crossfit_single_nuisance(data, "m", "lasso", folds, 303, 0, fast=True)
    np.testing.assert_array_equal(l.prediction, pair.l_hat)
    np.testing.assert_array_equal(m.prediction, pair.m_hat)
```

- [ ] **Step 2: Add failing cache round-trip and corruption tests**

```python
def test_nuisance_cache_round_trip(tmp_path):
    task = NuisanceTaskSpec("stage3b", "tree", 80, 10, 0, "l", "lasso", 0, 5, 123)
    cache = NuisanceCache(tmp_path)
    cache.write(task, np.arange(80.0), (0.1,), None, None)
    loaded = cache.read(task, expected_length=80)
    np.testing.assert_array_equal(loaded.prediction, np.arange(80.0))

def test_nuisance_cache_rejects_nonfinite_payload(tmp_path):
    task = NuisanceTaskSpec("stage3b", "tree", 80, 10, 0, "m", "lasso", 0, 5, 123)
    cache = NuisanceCache(tmp_path)
    cache.write(task, np.full(80, np.nan), (), None, None)
    with pytest.raises(ValueError, match="finite"):
        cache.read(task, expected_length=80)
```

- [ ] **Step 3: Confirm both focused test files fail**

Run: `.venv\Scripts\python.exe -m pytest tests/test_crossfit.py tests/test_nuisance_cache.py -q`

Expected: FAIL because the new interfaces do not exist.

- [ ] **Step 4: Extract exact single-target fitting logic**

```python
@dataclass(frozen=True)
class SingleNuisanceResult:
    prediction: NDArray[np.float64]
    fold_seconds: tuple[float, ...]
    peak_gpu_mb: float | None
    fallback_reason: str | None

def crossfit_single_nuisance(data, target, learner_name, folds, seed, tabicl_estimators, fast=False):
    if target not in {"l", "m"}:
        raise ValueError("target must be 'l' or 'm'")
    # Use the existing l seed expression for target l and the existing
    # derive_seed(derive_seed(seed, learner_name, fold_index), "m") expression for target m.
```

Refactor `crossfit_nuisance_pair` to call the same private fold-fit helper while preserving output arrays, seeds, fallbacks, and CUDA behavior.

- [ ] **Step 5: Implement deterministic cache keys and atomic `.npz` writes**

```python
@dataclass(frozen=True)
class NuisanceTaskSpec:
    seed_namespace: str
    scenario: str
    n: int
    p: int
    replication: int
    target: str
    learner: str
    tabicl_estimators: int
    folds_count: int
    learner_seed: int

    @property
    def key(self):
        return f"{self.seed_namespace}__{self.scenario}__n{self.n}__p{self.p}__r{self.replication:03d}__{self.target}__{self.learner}__e{self.tabicl_estimators}__k{self.folds_count}__s{self.learner_seed}"
```

Write to `<key>.npz.tmp`, flush and `os.fsync`, then `os.replace`; read validates metadata equality, expected length, one-dimensional shape, and finite predictions.

- [ ] **Step 6: Verify focused tests, full suite, and real legacy compatibility**

Run: `.venv\Scripts\python.exe -m pytest tests/test_crossfit.py tests/test_nuisance_cache.py -q`

Expected: all focused tests pass.

Run: `.venv\Scripts\python.exe -m pytest -q`

Expected: full suite passes.

Run a fixed-seed TabICLv2-1 and XGBoost comparison between `crossfit_nuisance_pair` and two single-target calls; require maximum absolute prediction difference `0.0` before continuing.

---

### Task 3: Configured Tree Learners and Independent Screening

**Files:**
- Modify: `src/tabdml/learners.py`
- Create: `src/tabdml/stage3b_screen.py`
- Create: `tests/test_stage3b_screen.py`
- Create: `configs/stage3b_tree_publication.yaml`
- Create: `scripts/run_stage3b_screen.py`

**Interfaces:**
- Produces: `make_configured_tree_learner(kind, params, seed, fast=False)`.
- Produces: `iter_screening_tasks`, `run_screening_task`, and `select_screening_winners`.
- Writes: `results/stage3b_screening_raw/*.json` and `results/stage3b_screening/selected_models.json`.

- [ ] **Step 1: Add failing cloneability and deterministic selection tests**

```python
def test_configured_extra_trees_is_cloneable():
    model = make_configured_tree_learner("extra_trees", {"n_estimators": 20, "max_features": 1.0, "min_samples_leaf": 2}, 9, fast=True)
    assert clone(model) is not model

def test_selection_uses_validation_d_mse_not_m0_mse():
    records = [
        {"candidate": "a", "status": "success", "validation_d_mse": 1.2, "validation_m0_mse": 0.01},
        {"candidate": "b", "status": "success", "validation_d_mse": 1.1, "validation_m0_mse": 0.50},
    ]
    assert select_screening_winners(records, kind="xgboost")["candidate"] == "b"
```

- [ ] **Step 2: Confirm focused tests fail**

Run: `.venv\Scripts\python.exe -m pytest tests/test_stage3b_screen.py -q`

Expected: FAIL on missing screening interfaces.

- [ ] **Step 3: Implement fixed candidate factories**

```python
def make_configured_tree_learner(kind, params, seed, fast=False):
    if kind == "xgboost":
        return XGBRegressor(objective="reg:squarederror", random_state=seed, n_jobs=1, **params)
    if kind == "extra_trees":
        return ExtraTreesRegressor(random_state=seed, n_jobs=1, **params)
    raise ValueError(f"Unknown configured learner: {kind}")
```

The YAML contains explicit named candidates. XGBoost varies `max_depth`, `learning_rate`, `min_child_weight`, and `reg_lambda`; ExtraTrees varies `max_features` and `min_samples_leaf`. TabICLv2-1, TabICLv2-8, and current XGBoost are fixed baselines.

- [ ] **Step 4: Implement screening task identity and evaluation**

Each task derives data and split seeds from `stage3b_mscreen_pilot`, fits on training `D`, and writes validation `D` MSE, diagnostic validation `m0` MSE, runtime, GPU memory, fallback, and complete params. The selector groups successful records by candidate, minimizes mean validation `D` MSE, and writes frozen winners atomically.

- [ ] **Step 5: Verify tests and CLI smoke**

Run: `.venv\Scripts\python.exe -m pytest tests/test_stage3b_screen.py -q`

Expected: focused tests pass.

Run: `.venv\Scripts\python.exe scripts/run_stage3b_screen.py --replications 1 --fast --cpu-only-candidates`

Expected: all configured CPU candidate smoke records are successful and a pilot-only selection JSON is produced under a smoke directory.

---

### Task 4: Batch A Cached Decomposition Runner

**Files:**
- Create: `src/tabdml/stage3b.py`
- Create: `tests/test_stage3b.py`
- Create: `scripts/run_stage3b_cache.py`
- Create: `scripts/compose_stage3b_batch_a.py`

**Interfaces:**
- Produces: `Stage3BNuisanceSpec`, `build_nuisance_spec`, `fit_cached_nuisance`, and `compose_dml_record`.
- Reads/writes: `results/stage3b_cache_batch_a/*.npz` and `results/stage3b_batch_a_raw/*.json`.

- [ ] **Step 1: Write failing cache-reuse and decomposition tests**

```python
def test_fit_cached_nuisance_reuses_successful_prediction(monkeypatch, tmp_path):
    first = fit_cached_nuisance(task, cache_root=tmp_path, fast=True)
    monkeypatch.setattr("tabdml.stage3b.crossfit_single_nuisance", lambda *a, **k: (_ for _ in ()).throw(AssertionError("refit")))
    second = fit_cached_nuisance(task, cache_root=tmp_path, fast=True)
    np.testing.assert_array_equal(first.prediction, second.prediction)

def test_compose_record_contains_proxy_error():
    record = compose_dml_record(data, task, data.l0, data.m0)
    assert record["lm_error_cross"] == 0.0
    assert record["theta_proxy"] == 1.0
    assert np.isclose(record["proxy_error"], record["theta"] - 1.0)
```

- [ ] **Step 2: Confirm focused tests fail**

Run: `.venv\Scripts\python.exe -m pytest tests/test_stage3b.py -q`

Expected: FAIL because `tabdml.stage3b` does not exist.

- [ ] **Step 3: Implement cache fitting and DML composition**

`fit_cached_nuisance` reconstructs data/folds from the task, reads a valid cache when available, otherwise calls `crossfit_single_nuisance` and writes the cache. `compose_dml_record` calls `estimate_plr_dml` and `compute_nuisance_diagnostics`, then stores the complete point estimate, interval, diagnostics, seeds, learner names, runtime, GPU memory, and fallback.

- [ ] **Step 4: Add Batch A CLIs**

`run_stage3b_cache.py` accepts target/learner filters and deterministic replication shards. `compose_stage3b_batch_a.py` requires all expected caches, composes the nine Stage 3A pairs, compares each theta with the existing Stage 3A JSON, and writes `stage3a_theta_difference`.

- [ ] **Step 5: Verify unit tests and a three-pair smoke**

Run: `.venv\Scripts\python.exe -m pytest tests/test_stage3b.py -q`

Expected: focused tests pass.

Run one replication for Oracle/Oracle, Tab/Oracle, and Oracle/XGBoost. Require three successful records, finite metrics, and exact Oracle diagnostics.

---

### Task 5: Batch C Confirmation and Resource-Aware Supervisor

**Files:**
- Create: `src/tabdml/stage3b_parallel.py`
- Create: `tests/test_stage3b_parallel.py`
- Create: `scripts/run_stage3b_parallel.py`
- Create: `scripts/compose_stage3b_confirmation.py`

**Interfaces:**
- Produces deterministic worker commands for `batch-a`, `screen`, and `confirmation`.
- Uses existing `tabdml.parallel.run_workers` state/log behavior.

- [ ] **Step 1: Add failing worker ownership tests**

```python
def test_confirmation_uses_one_gpu_and_eight_cpu_workers():
    commands = build_stage3b_confirmation_commands("python", Path("."), cpu_workers=8, replications=5)
    assert commands[0].name == "gpu_stage3b_cache"
    assert len(commands) == 9
    assert all("--num-shards" in c.argv for c in commands[1:])
```

- [ ] **Step 2: Confirm the focused test fails**

Run: `.venv\Scripts\python.exe -m pytest tests/test_stage3b_parallel.py -q`

Expected: FAIL on missing command builder.

- [ ] **Step 3: Implement worker command builders**

The GPU command owns TabICLv2 cache tasks. Eight CPU commands shard current XGBoost, frozen tuned-XGBoost, frozen ExtraTrees, and composition-ready CPU caches. No CPU worker initializes CUDA. Composition starts only after all cache workers exit zero.

- [ ] **Step 4: Implement confirmation composition**

Load frozen winners, enumerate `l in {tabiclv2_1, xgboost, oracle}` and `m in {tabiclv2_1, xgboost, xgboost_tuned, extra_trees, oracle}`, require caches, and write 15 pair records per replication under `results/stage3b_confirmation_raw`.

- [ ] **Step 5: Verify tests and five-replication smoke**

Run: `.venv\Scripts\python.exe -m pytest tests/test_stage3b_parallel.py -q`

Expected: focused tests pass.

Run: `.venv\Scripts\python.exe scripts/run_stage3b_parallel.py --batch confirmation --replications 5 --cpu-workers 8`

Expected: 75 unique successful records, 15 pairs with 5 records each, empty stderr logs, no fallback, and Oracle/Oracle exactness.

---

### Task 6: Aggregation, Validation, and Experiment Execution

**Files:**
- Create: `src/tabdml/stage3b_aggregate.py`
- Create: `tests/test_stage3b_aggregate.py`
- Create: `scripts/aggregate_stage3b.py`
- Produce: `results/stage3b_analysis/batch_a_summary.csv`
- Produce: `results/stage3b_analysis/screening_summary.csv`
- Produce: `results/stage3b_analysis/confirmation_summary.csv`
- Produce: `results/stage3b_analysis/analysis_report_zh.md`

**Interfaces:**
- Produces publication-facing grouped metrics and validation checks.

- [ ] **Step 1: Add failing aggregation tests**

```python
def test_aggregate_reports_bias_rmse_coverage_and_proxy_error():
    summary = aggregate_confirmation(fake_records(), theta0=1.0)
    assert {"bias", "rmse", "coverage", "mean_m_mse", "mean_lm_error_cross", "mean_proxy_error"} <= set(summary.columns)
```

- [ ] **Step 2: Confirm the focused test fails**

Run: `.venv\Scripts\python.exe -m pytest tests/test_stage3b_aggregate.py -q`

Expected: FAIL because the aggregate module does not exist.

- [ ] **Step 3: Implement grouped metrics and checks**

Aggregate by batch, scenario, n, p, learner_l, learner_m, and frozen config hash. Compute Bias, RMSE, empirical SD, mean SE, Coverage, interval length, nuisance diagnostics, proxy agreement, runtime, success/failure/OOM counts, and Monte Carlo standard errors.

- [ ] **Step 4: Run all automated verification**

Run: `.venv\Scripts\python.exe -m pytest -q`

Expected: every test passes; only documented sklearn convergence warnings are allowed.

- [ ] **Step 5: Run Batch A**

Run the Batch A parallel cache supervisor for 50 replications, compose all nine pairs, and aggregate. Verify 450 unique successful records and exact compatibility fields for existing Stage 3A pairs.

- [ ] **Step 6: Run Batch B**

Run the 10-replication screening, verify every candidate count, freeze one XGBoost and one ExtraTrees winner, and record that selection used validation `D` MSE.

- [ ] **Step 7: Run Batch C smoke and full confirmation**

Run five confirmation replications. If implementation checks pass, resume to 50 without deleting smoke records. Verify 750 unique records, 15 pairs times 50 replications, all finite values, no silent fallback, and empty stderr logs.

- [ ] **Step 8: Generate and inspect the Chinese analysis report**

Run: `.venv\Scripts\python.exe scripts/aggregate_stage3b.py`

Expected: all three CSV files and `analysis_report_zh.md` exist and contain the hypothesis outcomes regardless of whether candidate learners improve.

