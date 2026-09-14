# Stage 3B `tree_simple` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an isolated, axis-aligned `tree_simple` DGP and rerun all three Stage 3B batches without changing or overwriting the original `tree` experiment.

**Architecture:** Extend the existing DGP dispatcher with one new scenario. Parameterize the Stage 3B parallel launchers so stage, seed namespace, scenario, and config flow from a new YAML file into cache, screening, composition, and aggregation processes while preserving all old defaults.

**Tech Stack:** Python 3.10, NumPy, scikit-learn, XGBoost, TabICLv2/PyTorch, pytest, YAML, pandas, PowerShell.

## Global Constraints

- Keep the existing `tree` DGP and all existing Stage 3B outputs unchanged.
- Use `n=2000`, `p=10`, `theta0=1`, and 5-fold cross-fitting.
- Use 50 Batch A replications, 10 Batch B replications, and 50 Batch C replications.
- Use independent `tree_simple` stage names, seed namespaces, cache roots, result roots, log roots, selected-model output, and analysis output.
- Run TabICLv2 on one GPU worker and traditional learners on 8 CPU workers.
- Select Batch B winners only by mean validation `D`-MSE.

---

### Task 1: Add the `tree_simple` DGP with TDD

**Files:**
- Modify: `tests/test_dgp.py`
- Modify: `src/tabdml/dgp.py`

**Interfaces:**
- Consumes: `simulate_plr(scenario: str, n: int, p: int, seed: int, theta0: float = 1.0)`.
- Produces: support for `scenario="tree_simple"` with the formulas in the design spec.

- [ ] Add a test that reconstructs the expected standardized `raw_m` and `raw_g` from `data.X`, asserts equality with `data.m0` and `data.g0`, and asserts `data.l0 == data.theta0 * data.m0 + data.g0`.
- [ ] Run `.\.venv\Scripts\python.exe -m pytest -q tests/test_dgp.py` and verify failure with `Unknown scenario: tree_simple`.
- [ ] Add an `elif scenario == "tree_simple"` branch using only `x0 > 0`, `x1 > 0`, `x2 > 0`, `x3 > 0`, and `x4 > 0` indicators.
- [ ] Re-run the target test and verify it passes.
- [ ] Run the full test suite and verify no regressions.

### Task 2: Parameterize Stage 3B launchers without changing old defaults

**Files:**
- Modify: `tests/test_stage3b_parallel.py`
- Modify: `src/tabdml/stage3b_parallel.py`
- Modify: `scripts/run_stage3b_batch_a_parallel.py`
- Modify: `scripts/compose_stage3b_batch_a.py`
- Modify: `scripts/run_stage3b_screen_parallel.py`
- Modify: `scripts/run_stage3b_parallel.py`

**Interfaces:**
- `build_stage3b_batch_a_commands(..., stage="stage3b_batch_a", seed_namespace="stage3_tree_diagnosis", scenario="tree")`.
- `build_stage3b_screening_commands(..., config_path="configs/stage3b_tree_publication.yaml")`.
- `build_stage3b_confirmation_commands(..., stage="stage3b_confirmation", seed_namespace="stage3b_confirmation", scenario="tree", n=2000, p=10, folds=5, theta0=1.0)`.

- [ ] Add command-builder tests asserting custom tree-simple values appear in every worker command and old calls retain old defaults.
- [ ] Run `tests/test_stage3b_parallel.py` and verify the new assertions fail because parameters are not yet accepted or propagated.
- [ ] Add optional parameters to the three command builders and append the corresponding CLI arguments to worker commands.
- [ ] Add matching CLI options to Batch A and config propagation to screening and confirmation launchers/composers.
- [ ] Ensure Batch A legacy Stage 3A comparison executes only for the original `tree` namespace and does not require an old result for `tree_simple`.
- [ ] Re-run the target tests and the full suite.

### Task 3: Add isolated configuration and aggregation

**Files:**
- Create: `configs/stage3b_tree_simple.yaml`
- Modify: `scripts/aggregate_stage3b.py`
- Test: `tests/test_stage3b_cli.py`

**Interfaces:**
- New config uses `selected_configuration.scenario: tree_simple` and independent screening/confirmation stage and seed namespace values.
- Aggregator accepts input roots and output root as CLI parameters while retaining existing defaults.

- [ ] Add a failing CLI/config test that loads the new YAML, checks the exact scenario/stage/namespaces/counts, and checks custom aggregation arguments parse.
- [ ] Run the target test and verify failure because the config and aggregation arguments do not yet exist.
- [ ] Copy the candidate grid from `stage3b_tree_publication.yaml`, changing only scenario and stage/seed namespaces.
- [ ] Add argparse options for Batch A, screening, confirmation, original-comparison, and output roots to `aggregate_stage3b.py`.
- [ ] Re-run target and full tests.

### Task 4: Smoke-test the complete data flow

**Files:**
- Outputs only under `results/tree_simple_stage3b_smoke_*`.

**Interfaces:**
- Consumes the new config and parameterized launchers.
- Produces isolated smoke caches, JSON results, selected models, logs, and summaries.

- [ ] Run Batch A with 1 replication, 1 CPU worker, isolated smoke roots, and `--fast` where supported by direct worker smoke commands.
- [ ] Run screening with 1 replication and a minimal GPU/CPU candidate subset, then write selected-model JSON from eligible tuned-XGBoost and ExtraTrees candidates.
- [ ] Run confirmation with 1 replication and isolated roots.
- [ ] Aggregate smoke results and verify every JSON status is `success`, cache metadata says `tree_simple`, and no original Stage 3B directory timestamp or file count changes.

### Task 5: Run the formal experiment sequentially

**Files:**
- Create outputs under `results/stage3b_tree_simple_*` and `results/logs/stage3b_tree_simple_*`.

**Interfaces:**
- Batch A produces 450 records.
- Batch B produces 170 records plus frozen `selected_models.json`.
- Batch C produces 750 records.

- [ ] Launch Batch A with 50 replications and 8 CPU workers; wait for all workers and composition to finish.
- [ ] Verify 450 success records before starting Batch B.
- [ ] Launch Batch B with 10 replications and 8 CPU workers; wait for selection output.
- [ ] Verify 170 success records and inspect the frozen selection metric before starting Batch C.
- [ ] Launch Batch C with 50 replications and 8 CPU workers; wait for all workers and composition to finish.
- [ ] Verify 750 success records.

### Task 6: Aggregate and compare against the original tree experiment

**Files:**
- Create: `results/stage3b_tree_simple_analysis/batch_a_summary.csv`
- Create: `results/stage3b_tree_simple_analysis/screening_summary.csv`
- Create: `results/stage3b_tree_simple_analysis/confirmation_summary.csv`
- Create: `results/stage3b_tree_simple_analysis/analysis_report_zh.md`
- Create: `results/stage3b_tree_simple_analysis/tree_comparison.csv`

**Interfaces:**
- Consumes original published Stage 3B summaries and new `tree_simple` summaries.
- Produces a scenario-level comparison keyed by learner pair.

- [ ] Run the parameterized aggregator on formal tree-simple roots.
- [ ] Check all expected learner pairs and candidate groups are present.
- [ ] Calculate differences in Bias, RMSE, Coverage, `l`-MSE, and `m`-MSE relative to original tree.
- [ ] Write a Chinese report that separates observed evidence from interpretation and states Monte Carlo uncertainty.
- [ ] Run the full test suite and `git diff --check` before reporting completion.
