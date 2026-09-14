# Stage 4 Tree DGP Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and validate a resumable two-panel Stage 4 experiment that compares TabICLv2 with fairly tuned XGBoost across 24 predeclared axis-aligned tree DGP configurations and six independently confirmed configurations.

**Architecture:** Add three DGP names to the existing simulator, then keep all Stage 4 experiment logic in new `stage4_*` modules so Stage 1–3B behavior remains unchanged. A YAML file is the single source of truth for cells, XGBoost candidates, methods, seed namespaces, and paths; task runners write atomic JSON/cache artifacts, selectors freeze per-cell/per-target XGBoost models and six confirmation cells, and a final analysis module performs paired tests, Holm correction, exact coverage intervals, and decision-rule evaluation.

**Tech Stack:** Python 3.12, NumPy, pandas, SciPy, scikit-learn 1.9, XGBoost 3.3 CPU, PyTorch 2.11 CUDA, TabICLv2 2.1.1, PyYAML, pytest, existing `tabdml` result/cache/parallel utilities.

## Global Constraints

- Keep `theta0=1.0`, `Var(V)=1`, `Var(epsilon)=1`, and five-fold cross-fitting in the main experiment.
- Use only axis-aligned threshold functions; no `Xj*Xk > c`, `Xj+Xk > c`, pure XOR, or truth-derived engineered learner features.
- Panel A contains `n in {1000, 2000}`, `p in {10, 50}`; Panel B contains `n in {300, 500}`, `p in {50, 100}`; both contain all three structures.
- Tuning uses validation `Y`-MSE for target `l` and validation `D`-MSE for target `m`; `l0/m0` remain diagnostic-only.
- Use 10 tuning replications, 20 screening replications, 5 confirmation smoke replications, and 100 confirmation replications.
- Freeze one `l` XGBoost and one `m` XGBoost configuration per cell before screening.
- Freeze one confirmation cell for each `panel * structure` group, including groups where TabICLv2 loses.
- Run at most one TabICLv2 GPU worker and eight CPU workers; XGBoost remains on CPU.
- Never overwrite Stage 1, Stage 2, Stage 3B, or tree-simple Stage 3B artifacts.
- All writes are atomic, completed tasks are resumable, and failures/fallbacks remain explicit.
- Implement every behavior test-first and commit after each independently passing task.

---

## File Structure

**Modified files**

- `src/tabdml/dgp.py`: add the three Stage 4 structural functions and simulator scenarios.
- `tests/test_dgp.py`: verify exact formulas, reproducibility, and backward compatibility.
- `README.md`: add Stage 4 smoke and resume commands after the pipeline exists.
- `REPRODUCIBILITY.md`: record Stage 4 namespaces, artifact locations, and restart procedure.
- `.gitignore`: ignore Stage 4 raw/cache/log directories while allowing compact published outputs.
- `scripts/environment_report.py`: accept an explicit output path while preserving its existing default.

**New source modules**

- `src/tabdml/stage4_config.py`: validate the YAML and enumerate the 24 benchmark cells.
- `src/tabdml/stage4_structure.py`: calculate split gains and produce the hidden-XOR audit.
- `src/tabdml/stage4_tuning.py`: enumerate/run observable-target XGBoost tuning tasks and freeze per-target winners.
- `src/tabdml/stage4_experiment.py`: map Stage 4 methods to cached nuisance fits and compose DML records.
- `src/tabdml/stage4_selection.py`: freeze six confirmation cells from screening records.
- `src/tabdml/stage4_analysis.py`: aggregate records, run paired inference/Holm correction, and apply publication decision rules.
- `src/tabdml/stage4_parallel.py`: build CPU/GPU worker commands for tuning, screening, and confirmation.
- `src/tabdml/stage4_publish.py`: validate final result counts and copy only compact publication artifacts.

**New configuration and scripts**

- `configs/stage4_tree_benchmark.yaml`: exact panels, structures, candidates, methods, replication counts, and namespaces.
- `scripts/check_stage4_tree_structures.py`: write structure diagnostics.
- `scripts/run_stage4_tuning.py`: run sharded tuning tasks and freeze XGBoost winners.
- `scripts/run_stage4_cache.py`: populate nuisance caches for screening or confirmation.
- `scripts/compose_stage4_dml.py`: compose cached nuisances into DML JSON records.
- `scripts/select_stage4_confirmation.py`: freeze six confirmation cells.
- `scripts/analyze_stage4.py`: write summaries, paired comparisons, figures-input CSVs, and Chinese report.
- `scripts/publish_stage4.py`: validate final counts and copy compact final artifacts into the Git-tracked publication directory.
- `scripts/run_stage4_parallel.py`: launch one GPU worker and up to eight CPU workers, then compose results.

**New tests**

- `tests/test_stage4_config.py`
- `tests/test_stage4_structure.py`
- `tests/test_stage4_tuning.py`
- `tests/test_stage4_experiment.py`
- `tests/test_stage4_selection.py`
- `tests/test_stage4_analysis.py`
- `tests/test_stage4_parallel.py`
- `tests/test_stage4_cli.py`
- `tests/test_stage4_publish.py`
- `tests/test_environment_report.py`

---

### Task 1: Add the three exact axis-aligned tree DGPs

**Files:**
- Modify: `src/tabdml/dgp.py`
- Modify: `tests/test_dgp.py`

**Interfaces:**
- Consumes: `simulate_plr(scenario: str, n: int, p: int, seed: int, theta0: float = 1.0) -> SimulatedData`.
- Produces: scenarios `tree_stumps`, `tree_hierarchical`, and `tree_forest_sum`; private helper `_hierarchical_raw(X, root, left, right, a, b, c) -> NDArray[np.float64]`.

- [ ] **Step 1: Write failing exact-formula and reproducibility tests**

Append tests that reconstruct every raw function from `data.X` and compare the standardized values:

```python
def _scale(values):
    values = np.asarray(values, dtype=float)
    return (values - values.mean()) / values.std()


def _h(X, root, left, right, a, b, c):
    return (
        a * (X[:, root] > 0)
        + b * ((X[:, root] > 0) & (X[:, left] > 0))
        + c * ((X[:, root] <= 0) & (X[:, right] > 0))
    )


@pytest.mark.parametrize(
    "scenario", ["tree_stumps", "tree_hierarchical", "tree_forest_sum"]
)
def test_stage4_tree_dgps_are_reproducible(scenario):
    first = simulate_plr(scenario, n=500, p=10, seed=29)
    second = simulate_plr(scenario, n=500, p=10, seed=29)
    np.testing.assert_array_equal(first.X, second.X)
    np.testing.assert_array_equal(first.m0, second.m0)
    np.testing.assert_array_equal(first.g0, second.g0)


def test_tree_stumps_matches_declared_formula():
    data = simulate_plr("tree_stumps", n=500, p=10, seed=31)
    X = data.X
    raw_m = 0.9 * (X[:, 0] > 0) - 0.7 * (X[:, 1] > 0) + 0.5 * (X[:, 2] > 0)
    raw_g = 0.8 * (X[:, 0] > 0) + 0.6 * (X[:, 3] > 0) - 0.5 * (X[:, 4] > 0)
    np.testing.assert_allclose(data.m0, _scale(raw_m))
    np.testing.assert_allclose(data.g0, _scale(raw_g))


def test_tree_hierarchical_matches_declared_formula():
    data = simulate_plr("tree_hierarchical", n=500, p=10, seed=37)
    np.testing.assert_allclose(data.m0, _scale(_h(data.X, 0, 1, 2, 0.8, 0.6, -0.4)))
    np.testing.assert_allclose(data.g0, _scale(_h(data.X, 0, 3, 4, 0.7, 0.5, -0.4)))


def test_tree_forest_sum_matches_declared_formula():
    data = simulate_plr("tree_forest_sum", n=500, p=10, seed=41)
    raw_m = _h(data.X, 0, 1, 2, 0.55, 0.40, -0.30) + _h(
        data.X, 3, 4, 5, 0.45, -0.35, 0.30
    )
    raw_g = _h(data.X, 0, 6, 7, 0.50, 0.35, -0.25) + _h(
        data.X, 3, 8, 9, 0.40, -0.30, 0.25
    )
    np.testing.assert_allclose(data.m0, _scale(raw_m))
    np.testing.assert_allclose(data.g0, _scale(raw_g))


def test_tree_forest_sum_requires_ten_columns():
    with pytest.raises(ValueError, match="requires p >= 10"):
        simulate_plr("tree_forest_sum", n=500, p=9, seed=43)
```

- [ ] **Step 2: Run the new tests and verify unknown-scenario failure**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_dgp.py -q
```

Expected: the three new scenario tests fail with `ValueError: Unknown scenario` while existing tests pass.

- [ ] **Step 3: Add the minimal structural helper and scenario branches**

Add to `src/tabdml/dgp.py`:

```python
def _hierarchical_raw(
    X: NDArray[np.float64],
    root: int,
    left: int,
    right: int,
    a: float,
    b: float,
    c: float,
) -> NDArray[np.float64]:
    root_positive = X[:, root] > 0
    return np.asarray(
        a * root_positive
        + b * (root_positive & (X[:, left] > 0))
        + c * ((~root_positive) & (X[:, right] > 0)),
        dtype=float,
    )
```

Add branches before `mixed`:

```python
elif scenario == "tree_stumps":
    raw_m = 0.9 * (x0 > 0) - 0.7 * (x1 > 0) + 0.5 * (x2 > 0)
    raw_g = 0.8 * (x0 > 0) + 0.6 * (x3 > 0) - 0.5 * (x4 > 0)
elif scenario == "tree_hierarchical":
    raw_m = _hierarchical_raw(X, 0, 1, 2, 0.8, 0.6, -0.4)
    raw_g = _hierarchical_raw(X, 0, 3, 4, 0.7, 0.5, -0.4)
elif scenario == "tree_forest_sum":
    if p < 10:
        raise ValueError("tree_forest_sum requires p >= 10.")
    raw_m = _hierarchical_raw(X, 0, 1, 2, 0.55, 0.40, -0.30) + _hierarchical_raw(
        X, 3, 4, 5, 0.45, -0.35, 0.30
    )
    raw_g = _hierarchical_raw(X, 0, 6, 7, 0.50, 0.35, -0.25) + _hierarchical_raw(
        X, 3, 8, 9, 0.40, -0.30, 0.25
    )
```

- [ ] **Step 4: Run DGP and full regression tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_dgp.py -q
.\.venv\Scripts\python.exe -m pytest -q
```

Expected: all tests pass; the existing `tree_simple` outputs remain unchanged for a fixed seed.

- [ ] **Step 5: Commit**

```powershell
git add src/tabdml/dgp.py tests/test_dgp.py
git commit -m "Add Stage 4 axis-aligned tree DGPs"
```

---

### Task 2: Add the hidden-XOR and split-gain structure audit

**Files:**
- Create: `src/tabdml/stage4_structure.py`
- Create: `scripts/check_stage4_tree_structures.py`
- Create: `tests/test_stage4_structure.py`

**Interfaces:**
- Consumes: `simulate_plr` and scenario/root mappings.
- Produces: `split_gain(values, feature, threshold=0.0) -> float`, `audit_tree_structures(n=200_000, seed=20260903) -> list[dict]`, and `write_structure_audit(records, output_dir) -> None`.

- [ ] **Step 1: Write failing split-gain and audit tests**

```python
import numpy as np

from tabdml.stage4_structure import audit_tree_structures, split_gain


def test_split_gain_detects_signal_and_rejects_xor_root():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(200_000, 2))
    stump = (X[:, 0] > 0).astype(float)
    xor = (X[:, 0] * X[:, 1] > 0).astype(float)
    assert split_gain(stump, X[:, 0]) > 0.24
    assert split_gain(xor, X[:, 0]) < 1e-4
    assert split_gain(xor, X[:, 1]) < 1e-4


def test_all_declared_stage4_roots_have_positive_gain():
    rows = audit_tree_structures(n=200_000, seed=20260903)
    assert len(rows) == 12
    assert {row["scenario"] for row in rows} == {
        "tree_stumps",
        "tree_hierarchical",
        "tree_forest_sum",
    }
    assert all(row["split_gain"] > 1e-3 for row in rows)
    assert all(0.45 < row["left_probability"] < 0.55 for row in rows)
```

The expected 12 rows are S1 (`m`: roots 0,1,2; `g`: roots 0,3,4), S2 (`m/g`: root 0), and S3 (`m/g`: roots 0,3).

- [ ] **Step 2: Run the tests and verify import failure**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_stage4_structure.py -q
```

Expected: FAIL because `tabdml.stage4_structure` does not exist.

- [ ] **Step 3: Implement deterministic split-gain diagnostics**

Use weighted child variance reduction:

```python
def split_gain(values, feature, threshold=0.0):
    values = np.asarray(values, dtype=float)
    left = np.asarray(feature) <= threshold
    right = ~left
    if not left.any() or not right.any():
        raise ValueError("A split must have observations on both sides.")
    parent = float(np.var(values))
    child = float(left.mean() * np.var(values[left]) + right.mean() * np.var(values[right]))
    return parent - child
```

Define an explicit mapping of `(scenario, target, root_variable)` and obtain `m0/g0` from `simulate_plr`. Each audit row must contain `scenario`, `target`, `root_variable`, `threshold`, `split_gain`, `left_probability`, `left_mean`, and `right_mean`.

- [ ] **Step 4: Add atomic JSON/CSV output and CLI**

The script accepts:

```powershell
.\.venv\Scripts\python.exe scripts\check_stage4_tree_structures.py --n 200000 --seed 20260903 --output-dir results\stage4_tree_structure_checks
```

It must write `structure_checks.json` through a temporary file plus `os.replace`, write `structure_checks.csv`, print every gain, and exit nonzero if a gain is at most `1e-3`.

- [ ] **Step 5: Verify audit and commit**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_stage4_structure.py -q
.\.venv\Scripts\python.exe scripts\check_stage4_tree_structures.py --n 200000 --seed 20260903 --output-dir tmp\stage4_structure_check
```

Expected: tests pass; CLI writes 12 passing rows and returns exit code 0.

```powershell
git add src/tabdml/stage4_structure.py scripts/check_stage4_tree_structures.py tests/test_stage4_structure.py
git commit -m "Add Stage 4 tree structure audit"
```

---

### Task 3: Add the immutable Stage 4 configuration and 24-cell enumeration

**Files:**
- Create: `configs/stage4_tree_benchmark.yaml`
- Create: `src/tabdml/stage4_config.py`
- Create: `tests/test_stage4_config.py`

**Interfaces:**
- Produces: `TreeBenchmarkCell(panel: str, scenario: str, n: int, p: int)`, `load_stage4_config(path) -> dict`, `iter_tree_cells(config) -> tuple[TreeBenchmarkCell, ...]`, and `cell_key(cell) -> str` through the dataclass `key` property.

- [ ] **Step 1: Write failing configuration tests**

```python
from pathlib import Path

import pytest

from tabdml.stage4_config import iter_tree_cells, load_stage4_config


CONFIG = Path("configs/stage4_tree_benchmark.yaml")


def test_stage4_config_enumerates_exactly_two_twelve_cell_panels():
    cells = iter_tree_cells(load_stage4_config(CONFIG))
    assert len(cells) == 24
    assert len({cell.key for cell in cells}) == 24
    assert sum(cell.panel == "standard" for cell in cells) == 12
    assert sum(cell.panel == "small_n_high_p" for cell in cells) == 12
    assert {cell.scenario for cell in cells} == {
        "tree_stumps",
        "tree_hierarchical",
        "tree_forest_sum",
    }


def test_stage4_config_keeps_panel_ranges_disjoint():
    cells = iter_tree_cells(load_stage4_config(CONFIG))
    assert {(cell.n, cell.p) for cell in cells if cell.panel == "standard"} == {
        (1000, 10), (1000, 50), (2000, 10), (2000, 50)
    }
    assert {(cell.n, cell.p) for cell in cells if cell.panel == "small_n_high_p"} == {
        (300, 50), (300, 100), (500, 50), (500, 100)
    }


def test_invalid_duplicate_or_low_dimension_cell_is_rejected(tmp_path):
    config = load_stage4_config(CONFIG)
    config["panels"]["standard"]["dimensions"] = [9]
    with pytest.raises(ValueError, match="p >= 10"):
        iter_tree_cells(config)
```

- [ ] **Step 2: Run tests and verify module/config absence**

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_stage4_config.py -q
```

Expected: FAIL because the module and YAML do not exist.

- [ ] **Step 3: Write the exact YAML**

The YAML must contain:

```yaml
theta0: 1.0
folds: 5
structures: [tree_stumps, tree_hierarchical, tree_forest_sum]
panels:
  standard: {sample_sizes: [1000, 2000], dimensions: [10, 50]}
  small_n_high_p: {sample_sizes: [300, 500], dimensions: [50, 100]}
tuning:
  stage: stage4_tree_tuning
  seed_namespace: stage4_tree_tuning
  replications: 10
  validation_fraction: 0.25
  targets: [l, m]
  xgboost_candidates:
    - {name: xgb_d1_lr003, params: {n_estimators: 800, max_depth: 1, learning_rate: 0.03, min_child_weight: 1, reg_lambda: 1.0, subsample: 0.9, colsample_bytree: 1.0, tree_method: hist}}
    - {name: xgb_d2_lr003, params: {n_estimators: 800, max_depth: 2, learning_rate: 0.03, min_child_weight: 1, reg_lambda: 1.0, subsample: 0.9, colsample_bytree: 1.0, tree_method: hist}}
    - {name: xgb_d2_lr005, params: {n_estimators: 600, max_depth: 2, learning_rate: 0.05, min_child_weight: 5, reg_lambda: 1.0, subsample: 0.9, colsample_bytree: 1.0, tree_method: hist}}
    - {name: xgb_d3_lr003, params: {n_estimators: 800, max_depth: 3, learning_rate: 0.03, min_child_weight: 5, reg_lambda: 1.0, subsample: 0.9, colsample_bytree: 1.0, tree_method: hist}}
    - {name: xgb_d3_lr005, params: {n_estimators: 600, max_depth: 3, learning_rate: 0.05, min_child_weight: 10, reg_lambda: 2.0, subsample: 0.9, colsample_bytree: 1.0, tree_method: hist}}
    - {name: xgb_d4_lr003, params: {n_estimators: 800, max_depth: 4, learning_rate: 0.03, min_child_weight: 5, reg_lambda: 2.0, subsample: 0.9, colsample_bytree: 1.0, tree_method: hist}}
screening:
  stage: stage4_tree_screening
  seed_namespace: stage4_tree_screening
  replications: 20
  methods: [tabiclv2_1, tabiclv2_8, xgboost, xgboost_tuned, extra_trees, oracle]
confirmation:
  stage: stage4_tree_confirmation
  seed_namespace: stage4_tree_confirmation
  smoke_replications: 5
  replications: 100
  methods: [tabiclv2_1, tabiclv2_8, xgboost, xgboost_tuned, extra_trees, oracle]
extra_trees:
  params: {n_estimators: 600, max_features: 1.0, min_samples_leaf: 2}
```

- [ ] **Step 4: Implement strict loading and enumeration**

`load_stage4_config` must reject missing top-level sections, duplicate structures/candidate names, unknown scenarios, `p < 10`, nonpositive replications, folds below 2, and invalid validation fractions. `TreeBenchmarkCell.key` must return:

```python
return f"{self.panel}__{self.scenario}__n{self.n}__p{self.p}"
```

- [ ] **Step 5: Run tests and commit**

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_stage4_config.py -q
git add configs/stage4_tree_benchmark.yaml src/tabdml/stage4_config.py tests/test_stage4_config.py
git commit -m "Add Stage 4 benchmark configuration"
```

Expected: all configuration tests pass and exactly 24 unique keys are printed by an optional local inspection.

---

### Task 4: Implement observable-target XGBoost tuning and per-cell freezing

**Files:**
- Create: `src/tabdml/stage4_tuning.py`
- Create: `scripts/run_stage4_tuning.py`
- Create: `tests/test_stage4_tuning.py`
- Create: `tests/test_stage4_cli.py`

**Interfaces:**
- Consumes: `TreeBenchmarkCell`, YAML candidates, `simulate_plr`, `make_configured_tree_learner`, `ResultStore`.
- Produces: `Stage4TuningTask`, `iter_tuning_tasks`, `run_tuning_task`, `select_tuned_xgboost`, and `write_tuned_xgboost`.

- [ ] **Step 1: Write failing task-key, target-isolation, and no-truth-selection tests**

```python
from tabdml.stage4_config import TreeBenchmarkCell


CELL = TreeBenchmarkCell("standard", "tree_stumps", 1000, 10)
CELL_KEY = CELL.key


def _record(target, candidate, observed, diagnostic):
    return {
        "status": "success",
        "stage": "stage4_tree_tuning",
        "panel": CELL.panel,
        "scenario": CELL.scenario,
        "n": CELL.n,
        "p": CELL.p,
        "replication": 0,
        "target": target,
        "candidate": candidate,
        "params": {"max_depth": 1 if candidate == "a" else 2},
        "config_hash": f"{candidate}-hash",
        "validation_observed_mse": observed,
        "validation_truth_mse_diagnostic": diagnostic,
    }


def test_tuning_enumerates_cell_target_candidate_replication_product(config):
    tasks = tuple(iter_tuning_tasks(config, replications=2))
    assert len(tasks) == 24 * 2 * 6 * 2
    assert len({task.key for task in tasks}) == len(tasks)
    assert {task.target for task in tasks} == {"l", "m"}


def test_l_and_m_winners_use_observable_losses_independently():
    records = [
        _record("l", "a", observed=1.0, diagnostic=9.0),
        _record("l", "b", observed=2.0, diagnostic=0.0),
        _record("m", "a", observed=3.0, diagnostic=0.0),
        _record("m", "b", observed=1.5, diagnostic=9.0),
    ]
    selected = select_tuned_xgboost(records, expected_replications=1)
    assert selected["cells"][CELL_KEY]["l"]["candidate"] == "a"
    assert selected["cells"][CELL_KEY]["m"]["candidate"] == "b"
    assert selected["selection_metric_l"] == "mean_validation_y_mse"
    assert selected["selection_metric_m"] == "mean_validation_d_mse"
```

The `_record` test helper must provide `panel`, `scenario`, `n`, `p`, `replication`, `candidate`, `params`, `config_hash`, `status="success"`, `target`, `validation_observed_mse`, and `validation_truth_mse_diagnostic`.

- [ ] **Step 2: Run tests and verify import failure**

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_stage4_tuning.py -q
```

Expected: FAIL because `stage4_tuning` does not exist.

- [ ] **Step 3: Implement task enumeration and one tuning fit**

`Stage4TuningTask.key` must include stage, panel, scenario, `n`, `p`, replication, target, candidate, and the `_params_hash` from `stage3b_screen`.

Use these target mappings:

```python
response = data.y if task.target == "l" else data.d
truth = data.l0 if task.target == "l" else data.m0
metric_name = "validation_y_mse" if task.target == "l" else "validation_d_mse"
```

Fit only on the 75% training indices. Record both:

```python
"validation_observed_mse": float(np.mean((prediction - response[validation]) ** 2)),
"validation_truth_mse_diagnostic": float(np.mean((prediction - truth[validation]) ** 2)),
"selection_metric": metric_name,
```

The selector must group by `(cell_key, target, candidate)`, require exactly the expected replication set for every candidate, rank only `validation_observed_mse`, and write separate `l` and `m` winners.

- [ ] **Step 4: Add resumable sharded CLI**

The CLI accepts `--config`, `--output-root`, `--selected-output`, `--replications`, `--num-shards`, `--shard-index`, `--retry-failed`, `--fast`, and `--select`. A normal task writes through `ResultStore`; `--select` refuses to write if any cell/target/candidate count is incomplete.

- [ ] **Step 5: Verify a one-cell fast run and commit**

Add a CLI test invoking `main()` with a temporary one-cell config, one candidate, one replication, and `--fast`. Then run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_stage4_tuning.py tests\test_stage4_cli.py -q
```

Expected: tests pass; the temporary result contains observable and diagnostic losses but selection names only the observable loss.

```powershell
git add src/tabdml/stage4_tuning.py scripts/run_stage4_tuning.py tests/test_stage4_tuning.py tests/test_stage4_cli.py
git commit -m "Add Stage 4 per-target XGBoost tuning"
```

---

### Task 5: Implement Stage 4 cached nuisance fitting and DML composition

**Files:**
- Create: `src/tabdml/stage4_experiment.py`
- Create: `scripts/run_stage4_cache.py`
- Create: `scripts/compose_stage4_dml.py`
- Create: `tests/test_stage4_experiment.py`
- Modify: `tests/test_stage4_cli.py`

**Interfaces:**
- Consumes: frozen tuning JSON, `NuisanceCache`, `Stage3BPairSpec`, `fit_cached_nuisance`, `compose_dml_record`.
- Produces: `Stage4PairSpec`, `iter_stage4_pairs`, `build_stage4_nuisance_spec`, `resolve_method`, `fit_stage4_nuisance`, and `compose_stage4_record`.

- [ ] **Step 1: Write failing panel-key and per-target-config tests**

```python
import numpy as np
import pytest

from tabdml.stage4_experiment import (
    Stage4PairSpec,
    fit_stage4_nuisance,
    resolve_method,
)


def make_pair(panel="standard", learner_l="xgboost", learner_m="xgboost"):
    return Stage4PairSpec(
        stage="stage4_tree_screening",
        seed_namespace="stage4_tree_screening",
        panel=panel,
        scenario="tree_stumps",
        n=80,
        p=10,
        replication=0,
        learner_l=learner_l,
        learner_m=learner_m,
        folds_count=2,
        theta0=1.0,
    )


@pytest.fixture
def frozen():
    return {
        "cells": {
            "standard__tree_stumps__n80__p10": {
                "l": {
                    "learner_kind": "xgboost",
                    "params": {"max_depth": 1, "n_estimators": 20},
                    "config_hash": "l-hash",
                },
                "m": {
                    "learner_kind": "xgboost",
                    "params": {"max_depth": 2, "n_estimators": 20},
                    "config_hash": "m-hash",
                },
            }
        }
    }


def fail_if_called(*args, **kwargs):
    raise AssertionError("a complete nuisance cache entry must be reused")


def test_stage4_pair_key_contains_panel_and_ordered_methods():
    pair = make_pair(panel="small_n_high_p", learner_l="tabiclv2_1", learner_m="xgboost_tuned")
    assert "small_n_high_p" in pair.key
    assert "__ltabiclv2_1__mxgboost_tuned__" in pair.key


def test_tuned_xgboost_resolves_separate_l_and_m_hashes(frozen):
    pair = make_pair(learner_l="xgboost_tuned", learner_m="xgboost_tuned")
    l = resolve_method(pair, "l", frozen, extra_trees_params={})
    m = resolve_method(pair, "m", frozen, extra_trees_params={})
    assert l.config_hash == "l-hash"
    assert m.config_hash == "m-hash"
    assert l.params["max_depth"] == 1
    assert m.params["max_depth"] == 2


def test_cached_stage4_nuisance_is_reused(monkeypatch, tmp_path, frozen):
    pair = make_pair(learner_l="xgboost", learner_m="oracle")
    first = fit_stage4_nuisance(pair, "l", frozen, {}, tmp_path, fast=True)
    monkeypatch.setattr("tabdml.stage3b.crossfit_single_nuisance", fail_if_called)
    second = fit_stage4_nuisance(pair, "l", frozen, {}, tmp_path, fast=True)
    np.testing.assert_array_equal(first.prediction, second.prediction)
```

- [ ] **Step 2: Run tests and verify import failure**

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_stage4_experiment.py -q
```

Expected: FAIL because `stage4_experiment` does not exist.

- [ ] **Step 3: Implement Stage 4 wrappers without changing Stage 3B**

`Stage4PairSpec` contains `panel` plus all fields in `Stage3BPairSpec`. Its `effective_seed_namespace` must be:

```python
f"{self.seed_namespace}__{self.panel}"
```

Build an internal `Stage3BPairSpec` using that effective namespace so the existing deterministic cache and DML math remain unchanged. `compose_stage4_record` calls `compose_dml_record`, then overwrites `task_key` with the Stage 4 key and adds `panel`.

Define `ResolvedMethod` with `learner`, `learner_kind`, `params`, and `config_hash`. Map methods exactly:

```python
tabiclv2_1 -> learner="tabiclv2_1", config_hash="default"
tabiclv2_8 -> learner="tabiclv2_8", config_hash="default"
xgboost -> learner="xgboost", config_hash="default"
xgboost_tuned -> learner="xgboost_tuned", per-cell/per-target frozen params/hash
extra_trees -> learner="extra_trees", YAML params and stable hash
oracle -> learner="oracle", config_hash="default"
```

- [ ] **Step 4: Add cache and composition CLIs**

`run_stage4_cache.py` accepts `--phase screening|confirmation`, `--device-group gpu|cpu`, selected tuning JSON, selected confirmation JSON when required, shard options, and `--fast`. GPU requests contain only TabICLv2 methods; CPU requests contain only XGBoost, ExtraTrees, and Oracle.

`compose_stage4_dml.py` enumerates same-method pairs plus these oracle diagnostics for every cell:

```text
oracle/xgboost_tuned
xgboost_tuned/oracle
oracle/tabiclv2_1
tabiclv2_1/oracle
```

It refuses composition when either cache entry is missing or corrupt.

- [ ] **Step 5: Run fast cache/composition tests and commit**

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_stage4_experiment.py tests\test_stage4_cli.py -q
.\.venv\Scripts\python.exe -m pytest tests\test_stage3b.py tests\test_nuisance_cache.py -q
```

Expected: new tests pass and Stage 3B cache behavior remains unchanged.

```powershell
git add src/tabdml/stage4_experiment.py scripts/run_stage4_cache.py scripts/compose_stage4_dml.py tests/test_stage4_experiment.py tests/test_stage4_cli.py
git commit -m "Add Stage 4 cached DML experiment pipeline"
```

---

### Task 6: Freeze six confirmation cells using the predeclared rule

**Files:**
- Create: `src/tabdml/stage4_selection.py`
- Create: `scripts/select_stage4_confirmation.py`
- Create: `tests/test_stage4_selection.py`

**Interfaces:**
- Consumes: 24-cell screening DML JSON records.
- Produces: `paired_squared_error_advantage`, `select_confirmation_cells`, and atomic `write_confirmation_cells`.

- [ ] **Step 1: Write failing selection tests, including all-Tab-loss case**

```python
import pytest

from tabdml.stage4_selection import select_confirmation_cells


PANELS = {
    "standard": ((1000, 10), (1000, 50), (2000, 10), (2000, 50)),
    "small_n_high_p": ((300, 50), (300, 100), (500, 50), (500, 100)),
}
SCENARIOS = ("tree_stumps", "tree_hierarchical", "tree_forest_sum")


def _screen_row(panel, scenario, n, p, replication, method, theta):
    return {
        "status": "success",
        "panel": panel,
        "scenario": scenario,
        "n": n,
        "p": p,
        "replication": replication,
        "learner_l": method,
        "learner_m": method,
        "theta": theta,
    }


def records_for_all_24_cells(tab_multiplier=0.8):
    rows = []
    for panel, sizes in PANELS.items():
        for scenario in SCENARIOS:
            for cell_index, (n, p) in enumerate(sizes):
                for replication in range(2):
                    xgb_error = 0.02 + 0.002 * cell_index + 0.001 * replication
                    rows.append(_screen_row(panel, scenario, n, p, replication, "xgboost_tuned", 1 + xgb_error))
                    rows.append(_screen_row(panel, scenario, n, p, replication, "tabiclv2_1", 1 + tab_multiplier * xgb_error))
    return rows


def records_where_tab_squared_error_is_always_larger():
    return records_for_all_24_cells(tab_multiplier=1.2)


def incomplete_records():
    rows = records_for_all_24_cells()
    return rows[:-1]


def test_selector_returns_one_cell_per_panel_and_structure():
    selected = select_confirmation_cells(records_for_all_24_cells(), theta0=1.0, expected_replications=2)
    assert len(selected["cells"]) == 6
    assert {(row["panel"], row["scenario"]) for row in selected["cells"]} == {
        (panel, scenario)
        for panel in ("standard", "small_n_high_p")
        for scenario in ("tree_stumps", "tree_hierarchical", "tree_forest_sum")
    }


def test_selector_keeps_least_negative_cell_when_tab_loses_everywhere():
    records = records_where_tab_squared_error_is_always_larger()
    selected = select_confirmation_cells(records, theta0=1.0, expected_replications=2)
    row = selected["cells"][0]
    assert row["mean_paired_squared_error_difference"] > 0
    assert row["selection_rule"] == "minimum_mean_tab_minus_xgb_squared_error"


def test_selector_rejects_missing_or_unpaired_replications():
    with pytest.raises(ValueError, match="complete paired replications"):
        select_confirmation_cells(incomplete_records(), theta0=1.0, expected_replications=2)
```

- [ ] **Step 2: Run tests and verify import failure**

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_stage4_selection.py -q
```

Expected: FAIL because `stage4_selection` does not exist.

- [ ] **Step 3: Implement paired selection with deterministic ties**

For each cell, inner-join successful `tabiclv2_1/tabiclv2_1` and `xgboost_tuned/xgboost_tuned` rows on replication. Compute:

```python
delta = (theta_tab - theta0) ** 2 - (theta_xgb - theta0) ** 2
score = float(delta.mean())
```

Within each `(panel, scenario)`, select by `(score, n, p)` ascending. Store all 24 scores in `screening_ranking` and the six frozen rows in `cells`.

- [ ] **Step 4: Add CLI completeness checks and atomic output**

The CLI accepts screening root, config, output path, and expected replications. It rejects duplicate task keys, missing methods, asymmetric replication sets, non-success records, and fewer/more than 24 cells.

- [ ] **Step 5: Run tests and commit**

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_stage4_selection.py -q
git add src/tabdml/stage4_selection.py scripts/select_stage4_confirmation.py tests/test_stage4_selection.py
git commit -m "Add Stage 4 confirmation cell freezing"
```

---

### Task 7: Add paired inference, Holm correction, coverage intervals, and report

**Files:**
- Create: `src/tabdml/stage4_analysis.py`
- Create: `scripts/analyze_stage4.py`
- Create: `tests/test_stage4_analysis.py`

**Interfaces:**
- Consumes: Stage 4 records and six-cell selection JSON.
- Produces: `aggregate_stage4`, `holm_adjust`, `exact_coverage_interval`, `paired_primary_comparisons`, `apply_superiority_rule`, `write_stage4_report`, and `write_stage4_figures`.

- [ ] **Step 1: Write failing statistical unit tests**

```python
import numpy as np

from tabdml.stage4_analysis import (
    apply_superiority_rule,
    exact_coverage_interval,
    holm_adjust,
    paired_primary_comparisons,
)


def _comparison(**overrides):
    values = {
        "rmse_improvement_pct": 12.0,
        "holm_p_value": 0.01,
        "tab_coverage": 0.94,
        "xgb_coverage": 0.95,
        "coverage_difference": -0.01,
        "symmetric_success": True,
    }
    values.update(overrides)
    return values


def _primary_row(method, replication, theta):
    return {
        "status": "success",
        "panel": "standard",
        "scenario": "tree_stumps",
        "n": 1000,
        "p": 10,
        "replication": replication,
        "learner_l": method,
        "learner_m": method,
        "theta": theta,
        "standard_error": 0.03,
        "ci_lower": theta - 0.06,
        "ci_upper": theta + 0.06,
    }


def shuffled_primary_records():
    tab = [
        _primary_row("tabiclv2_1", replication, 1.0 + 0.005 * (-1) ** replication)
        for replication in range(100)
    ]
    xgb = [
        _primary_row("xgboost_tuned", replication, 1.0 + 0.02 * (-1) ** replication)
        for replication in range(100)
    ]
    return list(reversed(xgb)) + tab


def test_holm_adjust_is_monotone_in_sorted_order():
    adjusted = holm_adjust([0.01, 0.04, 0.03])
    np.testing.assert_allclose(adjusted, [0.03, 0.06, 0.06])


def test_exact_coverage_interval_contains_observed_fraction():
    lower, upper = exact_coverage_interval(94, 100, alpha=0.05)
    assert lower < 0.94 < upper
    assert 0.87 < lower < 0.90
    assert 0.97 < upper < 1.0


def test_superiority_requires_all_five_conditions():
    passing = _comparison()
    assert apply_superiority_rule(passing)["superior"] is True
    assert apply_superiority_rule({**passing, "tab_coverage": 0.89})["superior"] is False
    assert apply_superiority_rule({**passing, "rmse_improvement_pct": 9.9})["superior"] is False


def test_paired_comparison_joins_on_replication_not_row_order():
    result = paired_primary_comparisons(shuffled_primary_records(), theta0=1.0)
    assert result.iloc[0]["paired_count"] == 100
    assert result.iloc[0]["mean_squared_error_difference"] < 0
```

- [ ] **Step 2: Run tests and verify import failure**

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_stage4_analysis.py -q
```

Expected: FAIL because `stage4_analysis` does not exist.

- [ ] **Step 3: Implement aggregation and statistical inference**

Use `scipy.stats.ttest_1samp(delta, 0.0)` for the two-sided paired test and `scipy.stats.beta.ppf` for Clopper–Pearson bounds, with boundary values 0 and 1 handled explicitly. Implement Holm adjustment by sorting p-values, multiplying by remaining hypotheses, taking a cumulative maximum, clipping at 1, and restoring original order.

The primary comparison frame must contain:

```text
panel, scenario, n, p, paired_count,
tab_rmse, xgb_rmse, rmse_improvement_pct,
mean_squared_error_difference, difference_ci_lower, difference_ci_upper,
paired_p_value, holm_p_value, tab_abs_error_win_rate,
tab_bias, xgb_bias, tab_coverage, xgb_coverage,
tab_coverage_ci_lower, tab_coverage_ci_upper,
xgb_coverage_ci_lower, xgb_coverage_ci_upper,
coverage_difference, symmetric_success, superior, failed_conditions
```

- [ ] **Step 4: Implement deterministic CSV/Markdown report output**

Write:

```text
screening_summary.csv
screening_cell_ranking.csv
confirmation_summary.csv
primary_paired_comparisons.csv
coverage_diagnostics.csv
nuisance_diagnostics.csv
analysis_report_zh.md
figures/dml_rmse_by_panel.png
figures/nuisance_mse_by_panel.png
figures/coverage_by_panel.png
```

The Chinese report must state the fixed rules, list all six primary comparisons, distinguish nuisance-only improvements, summarize panel-level claims exactly as specified, and explicitly report zero qualifying cells when applicable. Use matplotlib with a noninteractive backend; every figure must label panel, structure, method, and metric, use the same method colors across figures, and be saved at 160 DPI or higher.

- [ ] **Step 5: Run tests and commit**

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_stage4_analysis.py -q
git add src/tabdml/stage4_analysis.py scripts/analyze_stage4.py tests/test_stage4_analysis.py
git commit -m "Add Stage 4 confirmatory statistical analysis"
```

---

### Task 8: Add one-GPU/eight-CPU orchestration and progress state

**Files:**
- Create: `src/tabdml/stage4_parallel.py`
- Create: `scripts/run_stage4_parallel.py`
- Create: `tests/test_stage4_parallel.py`
- Modify: `tests/test_stage4_cli.py`

**Interfaces:**
- Consumes: existing `WorkerCommand` and `run_workers`.
- Produces: `build_stage4_tuning_commands`, `build_stage4_cache_commands`, `run_stage4_phase`, and CLI phases `tuning`, `screening`, `confirmation`.

- [ ] **Step 1: Write failing worker-isolation tests**

```python
def test_tuning_uses_only_eight_cpu_workers():
    commands = build_stage4_tuning_commands("python", ROOT, CONFIG, RAW, cpu_workers=8, replications=1)
    assert len(commands) == 8
    assert all(command.name.startswith("cpu_stage4_tuning_") for command in commands)


def test_screening_cache_uses_one_gpu_and_eight_cpu_workers():
    commands = build_stage4_cache_commands(
        "python", ROOT, CONFIG, TUNED, CACHE, phase="screening", cpu_workers=8, replications=1
    )
    assert len(commands) == 9
    assert commands[0].name == "gpu_stage4_screening"
    assert "--device-group" in commands[0].argv
    assert commands[0].argv[commands[0].argv.index("--device-group") + 1] == "gpu"
    assert all("tabiclv2" not in " ".join(command.argv) for command in commands[1:])


def test_confirmation_requires_frozen_cells_path():
    with pytest.raises(ValueError, match="selected confirmation cells"):
        build_stage4_cache_commands(
            "python", ROOT, CONFIG, TUNED, CACHE, phase="confirmation", cpu_workers=8, replications=5, selected_cells=None
        )
```

- [ ] **Step 2: Run tests and verify import failure**

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_stage4_parallel.py -q
```

Expected: FAIL because `stage4_parallel` does not exist.

- [ ] **Step 3: Implement deterministic commands and phase chaining**

Tuning produces eight CPU shard commands. Screening/confirmation cache produces one unsharded GPU command restricted to TabICLv2 plus eight CPU shard commands restricted to traditional/Oracle methods. After all cache workers return zero, the parent runs `compose_stage4_dml.py`. A nonzero worker prevents composition and is returned to the shell.

Shard complete task keys rather than replication numbers so 24 cells and multiple candidates balance across workers. Use the same stable assignment in tuning and cache CLIs:

```python
def task_belongs_to_shard(task_key: str, num_shards: int, shard_index: int) -> bool:
    validate_shard(num_shards, shard_index)
    digest = hashlib.blake2b(task_key.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little") % num_shards == shard_index
```

Tests must prove that every enumerated CPU task belongs to exactly one shard and that the union of eight shards equals the unsharded task set.

Use separate log directories:

```text
results/logs/stage4_tree/tuning
results/logs/stage4_tree/screening
results/logs/stage4_tree/confirmation
```

The existing `run_workers` state JSON records command, PID, status, exit code, start, and finish. Add a parent `progress.json` containing `phase`, `planned_tasks`, `successful_tasks`, `failed_tasks`, `started_at`, and `updated_at`; update it atomically before and after each child stage.

- [ ] **Step 4: Add CLI and dry-run command inspection**

The CLI requires `--phase`, supports `--replications`, `--cpu-workers`, `--config`, `--tuned-models`, `--selected-cells`, `--cache-root`, `--output-root`, `--log-dir`, `--fast`, and `--dry-run`. `--dry-run` prints commands without starting processes.

- [ ] **Step 5: Run tests and commit**

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_stage4_parallel.py tests\test_stage4_cli.py -q
git add src/tabdml/stage4_parallel.py scripts/run_stage4_parallel.py tests/test_stage4_parallel.py tests/test_stage4_cli.py
git commit -m "Add Stage 4 resumable parallel orchestration"
```

---

### Task 9: Verify the complete smoke pipeline and document restart commands

**Files:**
- Create: `src/tabdml/stage4_publish.py`
- Create: `scripts/publish_stage4.py`
- Create: `tests/test_stage4_publish.py`
- Create: `tests/test_environment_report.py`
- Modify: `scripts/environment_report.py`
- Modify: `README.md`
- Modify: `REPRODUCIBILITY.md`
- Modify: `.gitignore`
- Test: entire `tests/` suite

**Interfaces:**
- Consumes: all Stage 4 CLIs.
- Produces: `validate_stage4_publication`, `publish_stage4`, a documented restartable smoke workflow, and clean Git boundaries.

- [ ] **Step 1: Write failing publication validation tests**

The publisher must require `48` frozen tuning entries, `24` screening cells, `6` confirmation cells, `100` successful primary replications per confirmation cell/method, all seven tabular/report artifacts, and all three figures. It must validate everything before creating or replacing the destination.

```python
import json
import sys
from pathlib import Path

import pytest

from scripts.environment_report import main as environment_main
from tabdml.stage4_publish import publish_stage4, validate_stage4_publication


def test_publisher_rejects_incomplete_results(tmp_path):
    results_root = tmp_path / "results"
    results_root.mkdir()
    with pytest.raises(ValueError, match="Stage 4 publication is incomplete"):
        validate_stage4_publication(results_root, expected_replications=100)


def test_failed_validation_does_not_create_destination(tmp_path):
    results_root = tmp_path / "results"
    destination = tmp_path / "published"
    results_root.mkdir()
    with pytest.raises(ValueError, match="Stage 4 publication is incomplete"):
        publish_stage4(results_root, destination, expected_replications=100)
    assert not destination.exists()


def test_environment_cli_accepts_explicit_output(monkeypatch, tmp_path):
    output = tmp_path / "environment.json"
    monkeypatch.setattr(sys, "argv", ["environment_report.py", "--output", str(output)])
    environment_main()
    assert json.loads(output.read_text(encoding="utf-8"))["python"]
```

- [ ] **Step 2: Run tests and verify import failure**

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_stage4_publish.py -q
```

Expected: FAIL because `stage4_publish` does not exist.

- [ ] **Step 3: Implement validation-first compact publication**

`validate_stage4_publication(results_root, expected_replications)` must validate these paths and return a manifest dictionary only after every check passes:

```python
required = {
    "structure_json": root / "stage4_tree_structure_checks" / "structure_checks.json",
    "structure_csv": root / "stage4_tree_structure_checks" / "structure_checks.csv",
    "tuned_models": root / "stage4_tree_tuning" / "selected_xgboost.json",
    "screening_summary": root / "stage4_tree_screening" / "screening_summary.csv",
    "selected_cells": root / "stage4_tree_screening" / "selected_confirmation_cells.json",
    "confirmation_summary": root / "stage4_tree_confirmation" / "confirmation_summary.csv",
    "primary_comparisons": root / "stage4_tree_confirmation" / "primary_paired_comparisons.csv",
    "coverage": root / "stage4_tree_confirmation" / "coverage_diagnostics.csv",
    "nuisance": root / "stage4_tree_confirmation" / "nuisance_diagnostics.csv",
    "report": root / "stage4_tree_confirmation" / "analysis_report_zh.md",
    "environment": root / "stage4_tree_confirmation" / "environment.json",
    "rmse_figure": root / "stage4_tree_confirmation" / "figures" / "dml_rmse_by_panel.png",
    "nuisance_figure": root / "stage4_tree_confirmation" / "figures" / "nuisance_mse_by_panel.png",
    "coverage_figure": root / "stage4_tree_confirmation" / "figures" / "coverage_by_panel.png",
}
```

It must also read `results/stage4_tree_confirmation_raw/*.json`, group successful same-method records by `(panel, scenario, n, p, learner_l, learner_m)`, and require `expected_replications` distinct replications for all six methods in all six selected cells. `publish_stage4` copies validated files into a temporary sibling directory, writes `manifest.json` containing file SHA-256 hashes and counts, then atomically replaces a nonexistent or explicitly replaceable destination. The CLI accepts `--results-root`, `--destination`, `--expected-replications`, and `--replace`.

Update `scripts/environment_report.py` with `argparse` so `--output` defaults to its existing `results/environment.json`. Keep `collect_environment()` unchanged and write parent directories before the JSON.

- [ ] **Step 4: Run targeted publication tests**

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_stage4_publish.py tests\test_environment_report.py -q
```

Expected: tests pass and failed validation leaves no destination directory.

- [ ] **Step 5: Add ignored raw/cache/log paths and keep published summaries trackable**

Append exact patterns:

```gitignore
results/stage4_tree_tuning_raw/
results/stage4_tree_screening_raw/
results/stage4_tree_confirmation_raw/
results/stage4_tree_cache/
results/logs/stage4_tree/
```

Do not ignore `results/stage4_tree_structure_checks/`, `results/stage4_tree_tuning/`, `results/stage4_tree_screening/`, `results/stage4_tree_confirmation/`, or `results/published/stage4_tree_benchmark/`.

- [ ] **Step 6: Document exact smoke and resume commands**

Add this ordered workflow to `REPRODUCIBILITY.md` and summarize it in `README.md`:

```powershell
.\.venv\Scripts\python.exe scripts\check_stage4_tree_structures.py --n 200000 --seed 20260903 --output-dir results\stage4_tree_structure_checks
.\.venv\Scripts\python.exe scripts\run_stage4_parallel.py --phase tuning --replications 1 --cpu-workers 8 --fast
.\.venv\Scripts\python.exe scripts\run_stage4_tuning.py --config configs\stage4_tree_benchmark.yaml --output-root results\stage4_tree_tuning_raw --selected-output results\stage4_tree_tuning\selected_xgboost.json --replications 1 --select --fast
.\.venv\Scripts\python.exe scripts\run_stage4_parallel.py --phase screening --replications 1 --cpu-workers 8 --fast
.\.venv\Scripts\python.exe scripts\select_stage4_confirmation.py --screening-root results\stage4_tree_screening_raw --expected-replications 1 --output results\stage4_tree_screening\selected_confirmation_cells.json
.\.venv\Scripts\python.exe scripts\run_stage4_parallel.py --phase confirmation --replications 1 --cpu-workers 8 --fast --selected-cells results\stage4_tree_screening\selected_confirmation_cells.json
.\.venv\Scripts\python.exe scripts\analyze_stage4.py --screening-root results\stage4_tree_screening_raw --confirmation-root results\stage4_tree_confirmation_raw --selected-cells results\stage4_tree_screening\selected_confirmation_cells.json --output-dir results\stage4_tree_confirmation
.\.venv\Scripts\python.exe scripts\environment_report.py --output results\stage4_tree_confirmation\environment.json
```

Document that rerunning the same commands resumes from atomic successful JSON/cache files and that failed tasks require the explicit retry flag where supported.

Document this command separately as a full-run-only publication gate; do not run it on smoke outputs:

```powershell
.\.venv\Scripts\python.exe scripts\publish_stage4.py --results-root results --expected-replications 100 --destination results\published\stage4_tree_benchmark
```

- [ ] **Step 7: Run all unit tests**

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Expected: all existing and new tests pass.

- [ ] **Step 8: Run the complete one-replication smoke pipeline**

Run the seven commands from Step 2. Expected:

- structure audit returns 12 passing rows;
- tuning selects 48 winners entries (`24 cells * 2 targets`);
- screening has all 24 cells and all configured methods;
- selection freezes exactly six cells;
- confirmation contains all six cells and configured methods/diagnostics;
- analysis writes all seven tabular/report artifacts and three figures;
- environment collection writes the current Python, package, CUDA, GPU, and driver record;
- no stderr log contains an unexplained traceback, OOM, or fallback;
- `progress.json` reports completion.

- [ ] **Step 9: Inspect Git boundaries and commit**

```powershell
git status --short
git check-ignore results/stage4_tree_tuning_raw/example.json
git check-ignore results/stage4_tree_cache/example.npz
git diff --check
git add .gitignore README.md REPRODUCIBILITY.md src/tabdml/stage4_publish.py scripts/publish_stage4.py scripts/environment_report.py tests/test_stage4_publish.py tests/test_environment_report.py
git commit -m "Document Stage 4 tree benchmark workflow"
```

Expected: raw/cache/log paths are ignored, compact analysis/structure outputs remain visible, and no historical result file is modified.

---

### Task 10: Review implementation before starting the expensive experiment

**Files:**
- Review: all files changed in Tasks 1–9
- Test: entire `tests/` suite and one-replication smoke artifacts

**Interfaces:**
- Consumes: complete Stage 4 implementation.
- Produces: a reviewed commit set ready for 10-rep tuning, 20-rep screening, and 100-rep confirmation.

- [ ] **Step 1: Run final verification from a clean shell**

```powershell
.\.venv\Scripts\python.exe -m pytest -q
git diff --check e2a9799..HEAD
git status --short
```

Expected: all tests pass, no whitespace errors, and the worktree contains only expected compact smoke outputs if they are intentionally retained.

- [ ] **Step 2: Request code review**

Review specifically for:

```text
DGP formulas and absence of hidden XOR
observable-only tuning selection
per-cell/per-target model hashes
paired replication integrity
one-GPU isolation
resume safety and atomic writes
Holm and exact coverage calculations
six-cell selection even when TabICLv2 loses
historical artifact isolation
```

- [ ] **Step 3: Apply only verified review fixes and rerun targeted plus full tests**

For each accepted finding, first add or update a failing regression test, verify failure, make the smallest source change, run the targeted test, then run:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Expected: all tests pass after review fixes.

- [ ] **Step 4: Commit review fixes when present**

```powershell
git add src scripts tests configs README.md REPRODUCIBILITY.md .gitignore
git commit -m "Harden Stage 4 tree benchmark pipeline"
```

If review finds no actionable issue, do not create an empty commit.

- [ ] **Step 5: Stop at the experiment launch gate**

Report smoke counts, test counts, selected paths, estimated runtime from smoke timings, and the exact full commands. Do not launch the 10/20/100-replication experiment until the user explicitly approves the expensive run.
