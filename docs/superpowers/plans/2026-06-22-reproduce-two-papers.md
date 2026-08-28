# Two-Paper Experiment Reproduction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build runnable, documented reproductions for the accessible experiments in TabICLv2 (Qu et al., 2026) and Double/Debiased Machine Learning (Chernozhukov et al., 2018), with machine-readable results and explicit fidelity limits.

**Architecture:** Use one isolated Python environment and two independent experiment packages. The TabICLv2 package evaluates the released checkpoints on deterministic classification, regression, and scaling tasks; the DML package reproduces the empirical designs with orthogonal scores, cross-fitting, and the learners listed in the paper. Each runner writes CSV/JSON outputs, and a top-level report compares reproduced values with the paper.

**Tech Stack:** Python 3.12, PyTorch, tabicl, scikit-learn, DoubleML, pandas, numpy, matplotlib, pytest.

## Global Constraints

- Hardware available: NVIDIA GeForce RTX 5060 Laptop GPU with 8,151 MiB VRAM.
- TabICLv2 paper benchmark used an NVIDIA H100; runtime values are not directly comparable across hardware.
- TabICLv2 v2 pretraining code is not publicly released as of June 22, 2026; reproduce released-checkpoint inference experiments, not the unavailable full pretraining/ablation pipeline.
- Use fixed random seeds and record package, CUDA, GPU, and CPU metadata.
- Never present a reduced benchmark as an exact reproduction of the full 51-dataset TabArena or 300-dataset TALENT benchmark.

---

### Task 1: Reproducible environment and provenance

**Files:**
- Create: `requirements.txt`
- Create: `scripts/environment_report.py`
- Create: `tests/test_environment_report.py`

**Interfaces:**
- Produces: `collect_environment() -> dict[str, object]`
- Produces: `results/environment.json`

- [ ] **Step 1: Write the failing test**

```python
from scripts.environment_report import collect_environment


def test_environment_report_has_reproduction_fields():
    report = collect_environment()
    assert {"python", "platform", "packages", "cuda", "gpu"} <= report.keys()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_environment_report.py -v`
Expected: FAIL because `scripts.environment_report` does not exist.

- [ ] **Step 3: Implement environment reporting**

Create a collector using `platform`, `importlib.metadata`, and optional `torch` APIs. Serialize it as UTF-8 JSON under `results/environment.json`.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_environment_report.py -v`
Expected: PASS.

### Task 2: TabICLv2 deterministic classification and regression reproduction

**Files:**
- Create: `tabiclv2_experiments/run_core.py`
- Create: `tabiclv2_experiments/__init__.py`
- Create: `tests/test_tabiclv2_core.py`

**Interfaces:**
- Produces: `run_classification(seed: int, n_estimators: int) -> dict[str, float]`
- Produces: `run_regression(seed: int, n_estimators: int) -> dict[str, float]`
- Produces: `results/tabiclv2_core.csv`

- [ ] **Step 1: Write tests for result schemas and metric bounds**

Use tiny synthetic inputs and dependency injection for estimators so tests do not download model weights.

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_tabiclv2_core.py -v`
Expected: FAIL because the module does not exist.

- [ ] **Step 3: Implement the experiment runner**

Use fixed train/test splits. Evaluate `TabICLClassifier(n_estimators=8)` with ROC AUC/log loss and `TabICLRegressor(n_estimators=8)` with RMSE, matching the paper's metric families.

- [ ] **Step 4: Run unit tests**

Run: `python -m pytest tests/test_tabiclv2_core.py -v`
Expected: PASS.

- [ ] **Step 5: Run released-checkpoint experiments**

Run: `python -m tabiclv2_experiments.run_core`
Expected: checkpoint downloads once, both experiments finish, and `results/tabiclv2_core.csv` is created.

### Task 3: TabICLv2 local scaling and runtime curve

**Files:**
- Create: `tabiclv2_experiments/run_scaling.py`
- Create: `tests/test_tabiclv2_scaling.py`

**Interfaces:**
- Consumes: released TabICLv2 classifier checkpoint.
- Produces: `benchmark_sizes() -> list[int]`
- Produces: `results/tabiclv2_scaling.csv`

- [ ] **Step 1: Test that sizes are monotonic and safe for 8 GB VRAM**

Use sizes `[300, 1000, 3000, 10000]`, 50 features, 500 test samples, and one warm-up run.

- [ ] **Step 2: Run test and confirm failure**

Run: `python -m pytest tests/test_tabiclv2_scaling.py -v`
Expected: FAIL because the module does not exist.

- [ ] **Step 3: Implement timed fit-plus-predict measurements**

Record wall-clock seconds, peak CUDA memory, sample count, feature count, estimator count, and hardware. Synchronize CUDA before and after timing.

- [ ] **Step 4: Verify and run**

Run: `python -m pytest tests/test_tabiclv2_scaling.py -v`
Expected: PASS.

Run: `python -m tabiclv2_experiments.run_scaling`
Expected: `results/tabiclv2_scaling.csv` exists; oversized cases are recorded as skipped/OOM rather than crashing the run.

### Task 4: DML Pennsylvania bonus and 401(k) empirical experiments

**Files:**
- Create: `dml_experiments/run_bonus.py`
- Create: `dml_experiments/run_401k.py`
- Create: `dml_experiments/common.py`
- Create: `dml_experiments/__init__.py`
- Create: `tests/test_dml_common.py`

**Interfaces:**
- Produces: learner factories for lasso, random forest, gradient boosting, neural network, and stacking/ensemble.
- Produces: estimates for 2-fold and 5-fold cross-fitting with repeated sample splits.
- Produces: `results/dml_bonus.csv`
- Produces: `results/dml_401k.csv`

- [ ] **Step 1: Test learner construction and output schema**

Assert every learner supports `fit`/`predict`; classifiers also support `predict_proba`; result rows include estimate, standard error, folds, repetition, score, and learner.

- [ ] **Step 2: Run tests and confirm failure**

Run: `python -m pytest tests/test_dml_common.py -v`
Expected: FAIL because modules do not exist.

- [ ] **Step 3: Implement the paper-aligned learners and preprocessing**

Use the official DoubleML datasets, deterministic preprocessing, DML orthogonal scores, 2-fold and 5-fold cross-fitting, and repeated sample splitting. Keep default and median-over-splits results separate.

- [ ] **Step 4: Run unit tests**

Run: `python -m pytest tests/test_dml_common.py -v`
Expected: PASS.

- [ ] **Step 5: Run empirical reproductions**

Run: `python -m dml_experiments.run_bonus`
Expected: treatment effects are negative and close to the paper's approximately -0.08 ATE scale.

Run: `python -m dml_experiments.run_401k`
Expected: eligibility ATE and participation LATE are positive and on the paper's several-thousand-dollar scale.

### Task 5: DML institutions and economic growth experiment

**Files:**
- Create: `dml_experiments/run_ajr.py`
- Create: `tests/test_dml_ajr.py`

**Interfaces:**
- Produces: `results/dml_ajr.csv` when the AJR data are available.
- Produces: a structured `unavailable` record with the exact missing source when data cannot legally or reliably be obtained.

- [ ] **Step 1: Test data validation**

Require outcome, endogenous treatment, instrument, and control columns before estimation.

- [ ] **Step 2: Verify the test fails**

Run: `python -m pytest tests/test_dml_ajr.py -v`
Expected: FAIL because the module does not exist.

- [ ] **Step 3: Implement PLIV estimation and explicit unavailable handling**

Match the paper's institutions/output design and run the same learner/fold grid as the other DML experiments. Never substitute synthetic data while labeling it as the empirical AJR result.

- [ ] **Step 4: Verify**

Run: `python -m pytest tests/test_dml_ajr.py -v`
Expected: PASS whether real estimates or a structured unavailable record is produced.

### Task 6: Comparison report and end-to-end verification

**Files:**
- Create: `make_report.py`
- Create: `REPRODUCTION.md`
- Create: `tests/test_report.py`

**Interfaces:**
- Consumes: all files under `results/`.
- Produces: `results/comparison.md`

- [ ] **Step 1: Test report generation from fixture result rows**

Require sections for both papers, hardware caveats, reproduced metrics, paper targets, absolute/relative differences, and unavailable experiments.

- [ ] **Step 2: Verify the test fails**

Run: `python -m pytest tests/test_report.py -v`
Expected: FAIL because report generation does not exist.

- [ ] **Step 3: Implement report generation and usage documentation**

Document exact commands, expected runtime, checkpoint/data downloads, and the distinction between exact, approximate, and reduced-scope reproduction.

- [ ] **Step 4: Run all tests and experiments**

Run: `python -m pytest -v`
Expected: PASS.

Run: `python make_report.py`
Expected: `results/comparison.md` is generated with no missing-file traceback.

- [ ] **Step 5: Final reproducibility audit**

Run every documented command from a clean environment, confirm CSV/JSON files are readable, and compare the report against Sections 6-7 of TabICLv2 and Section 6 of the DML paper.
