# Stage 3 Tree Diagnosis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run a resumable Stage 3A experiment that cross-fits different learners for `l(X)` and `m(X)`, supports one-sided Oracle nuisances, preserves Stage 1/2 behavior, and stops after a verified 45-task smoke batch.

**Architecture:** Preserve the legacy `TaskSpec`, runner, and raw results. Add a paired cross-fit primitive under the existing wrapper, a Stage 3-specific task/runner/CLI, and a resource-aware supervisor with one GPU process plus CPU shards. Derive each nuisance learner's seed from its matching legacy task key so homogeneous pairs reproduce Stage 2 and hybrid pairs remain paired with their baselines.

**Tech Stack:** Python 3.12, NumPy, SciPy, scikit-learn, XGBoost CPU, PyTorch CUDA, TabICLv2, PyYAML, pytest.

## Global Constraints

- TabICLv2 runs on one GPU worker; XGBoost remains CPU-only with `n_jobs=1`.
- Stage 3A uses tree `n=2000`, `p=10`, 5 folds, 50 replications, `theta0=1.0`, and one Tab estimator.
- The first execution stops after replications 0-4: 45 total tasks.
- Stage 1/2 code paths and result files remain backward compatible and unchanged in meaning.
- All new production behavior is introduced test-first.
- Git metadata is unavailable in this workspace, so commit steps are recorded but cannot be executed here.

---

### Task 1: Paired nuisance cross-fitting

**Files:**
- Modify: `src/tabdml/crossfit.py`
- Test: `tests/test_crossfit.py`

**Interfaces:**
- Produces: `crossfit_nuisance_pair(data, learner_l_name, learner_m_name, folds, seed_l, seed_m, tabicl_estimators, fast=False) -> CrossfitResult`
- Preserves: `crossfit_nuisances(data, learner_name, folds, seed, tabicl_estimators, fast=False)`

- [ ] Add a failing test showing `oracle/oracle` returns exact `data.l0` and `data.m0`.
- [ ] Run `python -m pytest tests/test_crossfit.py -v` and confirm failure because `crossfit_nuisance_pair` is missing.
- [ ] Implement the minimal paired function; skip fitting on an Oracle side and initialize CUDA only if either side starts with `tabiclv2`.
- [ ] Add a failing compatibility test comparing the old wrapper with the paired function for one homogeneous learner and fixed seeds.
- [ ] Make the wrapper delegate while preserving the legacy l-model and m-model seed derivation exactly.
- [ ] Run `python -m pytest tests/test_crossfit.py -v` and confirm all tests pass.

### Task 2: Stage 3 task identity and runner

**Files:**
- Create: `src/tabdml/stage3.py`
- Modify: `src/tabdml/storage.py`
- Test: `tests/test_stage3.py`
- Test: `tests/test_storage.py`

**Interfaces:**
- Produces: immutable `Stage3TaskSpec`
- Produces: `run_stage3_task(task, folds_count, theta0, output_root, retry_failed=False, fast=False) -> dict`
- Produces: `legacy_learner_seed(task, learner_name) -> int`

- [ ] Add failing tests for a key containing both learner names and for distinct keys when the pair order is reversed.
- [ ] Add a failing test that `seed_namespace="stage2"` reproduces legacy data, fold, and homogeneous learner seeds.
- [ ] Add a failing Oracle runner test asserting exact Oracle predictions yield valid DML output and zero nuisance MSE.
- [ ] Generalize `ResultStore` to accept any object exposing `.key`, while preserving string and `TaskSpec` behavior.
- [ ] Implement `Stage3TaskSpec`, seed helpers, and `run_stage3_task` with separate result fields `learner_l` and `learner_m`.
- [ ] Run `python -m pytest tests/test_stage3.py tests/test_storage.py -v` and confirm all tests pass.

### Task 3: Config and Stage 3 CLI

**Files:**
- Create: `configs/stage3_tree_diagnosis.yaml`
- Create: `scripts/run_stage3.py`
- Test: `tests/test_stage3_cli.py`

**Interfaces:**
- CLI arguments: `--config`, `--replications`, `--pair-names`, `--output-root`, `--retry-failed`, `--fast`, `--num-shards`, `--shard-index`

- [ ] Add a failing CLI test that parses all nine configured pairs and produces 45 unique tasks for five replications.
- [ ] Implement the YAML configuration with stable pair names and `seed_namespace: stage3_tree_diagnosis`.
- [ ] Implement task iteration and shard filtering by replication, using `run_stage3_task` for execution.
- [ ] Run `python -m pytest tests/test_stage3_cli.py -v` and confirm all tests pass.

### Task 4: Resource-aware supervisor

**Files:**
- Modify: `src/tabdml/parallel.py`
- Create: `scripts/run_stage3_parallel.py`
- Test: `tests/test_parallel.py`

**Interfaces:**
- Produces: `build_stage3_worker_commands(python_executable, stage3_script, output_root, config, cpu_workers, replications) -> tuple[WorkerCommand, ...]`

- [ ] Add a failing test asserting exactly one GPU command owns the five Tab pairs and CPU commands own only four non-Tab pairs.
- [ ] Add a failing test asserting every CPU command includes `--num-shards 8` and a unique `--shard-index`.
- [ ] Implement the command builder and supervisor script using existing `run_workers` state/log handling.
- [ ] Run `python -m pytest tests/test_parallel.py -v` and confirm all tests pass.

### Task 5: Environment and full verification

**Files:**
- Modify only if needed: `results/environment_stage3.json`

- [ ] Restore exact recorded package versions when available: NumPy 2.3.5, pandas 3.0.1, SciPy 1.18.0, scikit-learn 1.9.0, XGBoost 3.3.0, Torch 2.11.0+cu128, TabICL 2.1.1, DoubleML 0.11.3.
- [ ] Run an import/CUDA check and confirm the RTX 5060 Laptop GPU is visible.
- [ ] Run the complete non-GPU suite with `python -m pytest -m "not gpu" -v`.
- [ ] Run the smallest existing TabICLv2 GPU learner test.
- [ ] Run a fixed-seed legacy-vs-paired compatibility check for Tab/Tab and XGB/XGB.
- [ ] Write `results/environment_stage3.json` with exact versions and hardware.

### Task 6: Run and validate the 45-task smoke batch

**Files:**
- Produce: `results/stage3_tree_diagnosis_raw/*.json`
- Produce: `results/logs/stage3_tree_diagnosis_smoke/*`

- [ ] Run `scripts/run_stage3_parallel.py --replications 5 --cpu-workers 8`.
- [ ] Wait for all workers to finish; do not launch the remaining 405 tasks in the same command.
- [ ] Verify exactly 45 unique records and nine pairs with five successes each.
- [ ] Verify zero failures/OOM, finite estimates, positive standard errors, ordered confidence intervals, and zero Oracle nuisance MSE on the corresponding side.
- [ ] Compare Oracle/Oracle with direct DML for every replication.
- [ ] Summarize smoke Bias, RMSE, coverage, `l_mse`, `m_mse`, nuisance product, and runtime by pair.
- [ ] Stop for a result review before changing `--replications` from 5 to 50.

