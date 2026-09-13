# Stage 5 Sensitivity Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build, validate, and preflight a resumable Stage 5 PLR-DML benchmark that varies feature dimension at fixed sample size and sample size at fixed feature dimension, then produces publication-ready treatment-effect sensitivity figures.

**Architecture:** Stage 5 gets its own validated configuration, seed namespaces, tuning artifact, nuisance cache, orchestration scripts, result store, and analysis outputs. It reuses the established DGP, cross-fitting, nuisance-cache, DML-estimation, learner, storage, sharding, and worker primitives, while keeping Stage 2 and Stage 4 code and results unchanged. One GPU worker handles TabICLv2; standard CPU methods and the full ensemble run in separate CPU pools before deterministic cache composition.

**Tech Stack:** Python 3.11+, NumPy, pandas, scikit-learn, XGBoost, PyTorch/TabICLv2, PyYAML, matplotlib, pytest.

## Global Constraints

- Use exactly six existing DGPs: `linear`, `smooth`, `tree`, `tree_stumps`, `tree_hierarchical`, and `tree_forest_sum`; do not modify their formulas.
- Use `theta0=1`, five-fold cross-fitting, standard-normal covariates, unit-scaled structural functions, and unit-variance treatment and outcome noise through `simulate_plr`.
- Dimension sweep: fixed `n=1000`, `p=[10,50,100]`; sample-size sweep: fixed `p=50`, `n=[500,1000,2000]`.
- Generate the shared center cell `(n=1000,p=50)` once, giving exactly five cells per DGP and 30 cells overall.
- Use exactly six strategies: `tabiclv2_1`, `tabiclv2_8`, `xgboost_tuned`, `extra_trees`, `lasso`, and the existing full `ensemble`.
- Tune XGBoost independently for `l` and `m` at `(n=1000,p=50)` for each DGP with 10 replications and the exact Stage 4 six-candidate grid; select by mean observed validation MSE with deterministic tie-breaking.
- Keep tuning, smoke, preflight, and formal seed namespaces distinct; pair data and fold seeds across methods within every DGP-cell-replication.
- Smoke has 1 replication (180 records), preflight has 5 (900 records), and formal has 100 (18,000 records).
- Run one unsharded GPU worker for both TabICLv2 methods; run XGBoost-tuned, ExtraTrees, and Lasso on 5–8 CPU shards; run the full ensemble separately on 2–4 CPU shards.
- Do not use GPU XGBoost and do not replace any benchmark learner with an approximation.
- Write incrementally, skip successful existing work on resume, and reject duplicate keys or incompatible provenance.
- Do not start the formal phase until all tests, smoke, and preflight gates pass and the user explicitly approves the estimated formal cost.

---

### Task 1: Frozen Stage 5 configuration and 30-cell grid

**Files:**
- Create: `configs/stage5_sensitivity.yaml`
- Create: `src/tabdml/stage5_config.py`
- Create: `tests/test_stage5_config.py`

**Interfaces:**
- Consumes: `yaml.safe_load(path.read_text(encoding="utf-8"))` and the existing DGP names accepted by `simulate_plr`.
- Produces: `SensitivityCell`, `load_stage5_config(path)`, `iter_sensitivity_cells(config)`, `resolve_stage5_profile(config, profile, replications=None)`, and `stage5_config_fingerprint(config)`.

- [ ] **Step 1: Write failing grid, validation, profile, and fingerprint tests**

```python
def test_stage5_grid_has_six_scenarios_and_deduplicated_center():
    config = load_stage5_config(Path("configs/stage5_sensitivity.yaml"))
    cells = list(iter_sensitivity_cells(config))
    assert len(cells) == 30
    assert len({cell.key for cell in cells}) == 30
    assert sum((cell.n, cell.p) == (1000, 50) for cell in cells) == 6
    assert {(cell.n, cell.p) for cell in cells if cell.scenario == "linear"} == {
        (1000, 10), (1000, 50), (1000, 100), (500, 50), (2000, 50)
    }

def test_stage5_profiles_are_isolated():
    config = load_stage5_config(Path("configs/stage5_sensitivity.yaml"))
    profiles = [resolve_stage5_profile(config, name) for name in ("smoke", "preflight", "formal")]
    assert [profile.replications for profile in profiles] == [1, 5, 100]
    assert len({profile.seed_namespace for profile in profiles}) == 3
```

- [ ] **Step 2: Run tests and confirm they fail because Stage 5 does not exist**

Run: `python -m pytest tests/test_stage5_config.py -q`

Expected: collection fails with `ModuleNotFoundError: No module named 'tabdml.stage5_config'`.

- [ ] **Step 3: Add the exact YAML contract and strict parser**

```yaml
theta0: 1.0
folds: 5
scenarios: [linear, smooth, tree, tree_stumps, tree_hierarchical, tree_forest_sum]
sweeps:
  dimension: {fixed_n: 1000, values: [10, 50, 100]}
  sample_size: {fixed_p: 50, values: [500, 1000, 2000]}
profiles:
  smoke: {stage: stage5_smoke, seed_namespace: stage5_smoke_v1, replications: 1, full_settings: false}
  preflight: {stage: stage5_preflight, seed_namespace: stage5_preflight_v1, replications: 5, full_settings: true}
  formal: {stage: stage5_formal, seed_namespace: stage5_formal_v1, replications: 100, full_settings: true}
tuning:
  stage: stage5_tuning
  seed_namespace: stage5_tuning_v1
  n: 1000
  p: 50
  replications: 10
  validation_fraction: 0.25
  targets: [l, m]
methods: [tabiclv2_1, tabiclv2_8, xgboost_tuned, extra_trees, lasso, ensemble]
```

Implement immutable dataclasses, exact-list checks, native-positive-integer checks, duplicate detection, distinct seed-namespace checks, exact workload checks, and a canonical SHA-256 fingerprint. Copy the six Stage 4 XGBoost candidates and ExtraTrees parameters into the YAML without changing their numeric values.

- [ ] **Step 4: Run configuration tests**

Run: `python -m pytest tests/test_stage5_config.py -q`

Expected: all tests pass and report exactly 30 unique cells.

- [ ] **Step 5: Commit the configuration boundary**

```powershell
git add configs/stage5_sensitivity.yaml src/tabdml/stage5_config.py tests/test_stage5_config.py
git commit -m "Add Stage 5 sensitivity configuration"
```

### Task 2: DGP-level, target-specific XGBoost tuning and freezing

**Files:**
- Create: `src/tabdml/stage5_tuning.py`
- Create: `scripts/run_stage5_tuning.py`
- Create: `scripts/select_stage5_tuning.py`
- Create: `tests/test_stage5_tuning.py`
- Create: `tests/test_stage5_cli.py`

**Interfaces:**
- Consumes: `derive_seed`, `simulate_plr`, `make_configured_tree_learner`, Stage 5 tuning config, and `ResultStore`.
- Produces: `Stage5TuningTask`, `iter_stage5_tuning_tasks(config, replications, execution_profile)`, `run_stage5_tuning_task(task, config)`, `select_stage5_tuning(records, config, execution_profile)`, and a frozen JSON keyed by `scenario -> target`.

- [ ] **Step 1: Write failing task-count, seed, metric, and tie-break tests**

```python
def test_full_tuning_has_720_tasks_and_distinct_target_seeds():
    config = load_stage5_config(Path("configs/stage5_sensitivity.yaml"))
    tasks = list(iter_stage5_tuning_tasks(config, replications=10, execution_profile="full"))
    assert len(tasks) == 6 * 2 * 6 * 10
    assert {(task.n, task.p) for task in tasks} == {(1000, 50)}
    assert len({task.key for task in tasks}) == len(tasks)

def test_selection_uses_observed_mse_then_candidate_order():
    frozen = select_stage5_tuning(make_tied_records(), config, "full")
    assert frozen["scenarios"]["linear"]["l"]["candidate"] == "xgb_d1_lr003"
    assert frozen["selection_metric_l"] == "mean_validation_y_mse"
    assert frozen["selection_metric_m"] == "mean_validation_d_mse"
```

- [ ] **Step 2: Run tests and confirm missing-module failures**

Run: `python -m pytest tests/test_stage5_tuning.py tests/test_stage5_cli.py -q`

Expected: failure because tuning interfaces and scripts do not exist.

- [ ] **Step 3: Implement tuning records and deterministic selection**

Each task must simulate only its DGP center cell, split with a derived validation seed, fit one candidate to one target, and store observed and truth-diagnostic MSE separately. The selector must require all expected candidate-replication keys, sort by `(mean_validation_observed_mse, candidate_order)`, and write the selected nominal/effective parameters, parameter hashes, config fingerprint, tuning fingerprint, profile, and replication count.

- [ ] **Step 4: Add resumable sharded tuning and selection CLIs**

```powershell
python scripts/run_stage5_tuning.py --config configs/stage5_sensitivity.yaml --output results/stage5/tuning/records.jsonl --execution-profile fast --num-shards 1 --shard-index 0
python scripts/select_stage5_tuning.py --config configs/stage5_sensitivity.yaml --input results/stage5/tuning/records.jsonl --output results/stage5/tuning/frozen-fast.json --execution-profile fast
```

The runner must append atomically through `ResultStore`, skip only valid successful matching keys, return nonzero on any failure, and never inspect treatment-effect results.

- [ ] **Step 5: Run tuning tests**

Run: `python -m pytest tests/test_stage5_tuning.py tests/test_stage5_cli.py -q`

Expected: all tests pass, including incomplete-input and provenance-rejection cases.

- [ ] **Step 6: Commit the tuning pipeline**

```powershell
git add src/tabdml/stage5_tuning.py scripts/run_stage5_tuning.py scripts/select_stage5_tuning.py tests/test_stage5_tuning.py tests/test_stage5_cli.py
git commit -m "Add frozen Stage 5 XGBoost tuning"
```

### Task 3: Paired DML task generation, nuisance caching, and composition

**Files:**
- Create: `src/tabdml/stage5_experiment.py`
- Create: `scripts/run_stage5_cache.py`
- Create: `scripts/compose_stage5_dml.py`
- Create: `tests/test_stage5_experiment.py`
- Modify: `tests/test_stage5_cli.py`

**Interfaces:**
- Consumes: `SensitivityCell`, `Stage5Profile`, frozen tuning JSON, `NuisanceTaskSpec`, `NuisanceCache`, `crossfit_single_nuisance`, `estimate_plr_dml`, and existing learners.
- Produces: `Stage5PairSpec`, `ResolvedStage5Method`, `iter_stage5_pairs`, `resolve_stage5_method`, `build_stage5_nuisance_spec`, `fit_stage5_nuisance`, `compose_stage5_record`, cache CLI, and composition CLI.

- [ ] **Step 1: Write failing workload, pairing, routing, and composition tests**

```python
def test_smoke_generates_180_paired_results():
    pairs = list(iter_stage5_pairs(config, frozen_fast, profile="smoke"))
    assert len(pairs) == 30 * 6
    grouped = groupby_data_identity(pairs)
    assert all(len({pair.data_seed for pair in group}) == 1 for group in grouped.values())
    assert all(len({pair.fold_seed for pair in group}) == 1 for group in grouped.values())

def test_method_routing_is_hybrid():
    assert methods_for_device("gpu") == ("tabiclv2_1", "tabiclv2_8")
    assert methods_for_device("cpu") == ("xgboost_tuned", "extra_trees", "lasso")
    assert methods_for_device("ensemble") == ("ensemble",)
```

Also assert that the center cell appears once per DGP-method-replication, `l` and `m` tuned XGBoost parameters can differ, cache keys include target/config/profile provenance, and a composed oracle-shaped fixture stores squared error, coverage, nuisance MSEs, cross term, timings, fallback, and hashes.

- [ ] **Step 2: Run tests and confirm missing-interface failures**

Run: `python -m pytest tests/test_stage5_experiment.py tests/test_stage5_cli.py -q`

Expected: failure because Stage 5 experiment interfaces do not exist.

- [ ] **Step 3: Implement paired task generation and learner resolution**

Use one data seed and one fold seed derived from `(profile.seed_namespace, scenario, n, p, replication)` before adding the method. Resolve `xgboost_tuned` independently for target `l` and `m`; resolve ExtraTrees from fixed YAML parameters; resolve the remaining learners through `make_learner`. In smoke only, cap tree counts and ensemble base-model iterations while retaining the same learner identities; preflight and formal must use full settings.

- [ ] **Step 4: Implement cache fitting and complete DML composition**

Cache out-of-fold predictions, truth vectors, fit/total timing, device request and observation, peak allocated GPU memory when CUDA is used, fallback state, fold metadata, data/fold seeds, and config hashes. Compose only when both nuisance targets are valid and compatible; calculate `theta_hat`, standard error, 95% interval, coverage, squared error, `l_mse`, `m_mse`, nuisance error product, cross term, and status with finite-value checks.

- [ ] **Step 5: Add sharded cache and deterministic composition CLIs**

```powershell
python scripts/run_stage5_cache.py --config configs/stage5_sensitivity.yaml --profile smoke --frozen-tuning results/stage5/tuning/frozen-fast.json --cache-root results/stage5/smoke/cache --device-group gpu
python scripts/run_stage5_cache.py --config configs/stage5_sensitivity.yaml --profile smoke --frozen-tuning results/stage5/tuning/frozen-fast.json --cache-root results/stage5/smoke/cache --device-group cpu --num-shards 5 --shard-index 0
python scripts/run_stage5_cache.py --config configs/stage5_sensitivity.yaml --profile smoke --frozen-tuning results/stage5/tuning/frozen-fast.json --cache-root results/stage5/smoke/cache --device-group ensemble --num-shards 2 --shard-index 0
python scripts/compose_stage5_dml.py --config configs/stage5_sensitivity.yaml --profile smoke --frozen-tuning results/stage5/tuning/frozen-fast.json --cache-root results/stage5/smoke/cache --output results/stage5/smoke/raw.jsonl
```

- [ ] **Step 6: Run experiment and CLI tests**

Run: `python -m pytest tests/test_stage5_experiment.py tests/test_stage5_cli.py -q`

Expected: all tests pass with exact 180/900/18,000 workload assertions.

- [ ] **Step 7: Commit the experiment core**

```powershell
git add src/tabdml/stage5_experiment.py scripts/run_stage5_cache.py scripts/compose_stage5_dml.py tests/test_stage5_experiment.py tests/test_stage5_cli.py
git commit -m "Add Stage 5 paired DML cache pipeline"
```

### Task 4: Parallel controller, progress, resume, and launch gates

**Files:**
- Create: `src/tabdml/stage5_parallel.py`
- Create: `scripts/run_stage5_parallel.py`
- Create: `tests/test_stage5_parallel.py`

**Interfaces:**
- Consumes: `WorkerCommand`, `run_workers`, the Stage 5 tuning/cache/composition CLIs, `iter_stage5_pairs`, and the cache/result validators.
- Produces: `build_stage5_cache_commands`, `run_stage5_parallel`, `write_stage5_progress`, `validate_stage5_gate`, and a single resumable controller CLI.

- [ ] **Step 1: Write failing worker-layout and gate tests**

```python
def test_parallel_layout_has_one_gpu_standard_cpu_and_ensemble_batches():
    commands = build_stage5_cache_commands(args, project_root)
    assert sum("--device-group gpu" in command_text(c) for c in commands.concurrent) == 1
    assert sum("--device-group cpu" in command_text(c) for c in commands.concurrent) == 5
    assert sum("--device-group ensemble" in command_text(c) for c in commands.ensemble) == 2

def test_formal_gate_rejects_unapproved_or_incomplete_preflight():
    with pytest.raises(ValueError, match="explicit formal approval"):
        validate_stage5_gate(profile="formal", preflight_summary=complete_summary, formal_approved=False)
```

Also test absolute interpreter resolution, paths with spaces, nonzero worker propagation, resume after partial cache completion, unique output keys, GPU worker non-sharding, and rejection of OOM/fallback/non-finite/provenance mismatches.

- [ ] **Step 2: Run tests and confirm missing-controller failures**

Run: `python -m pytest tests/test_stage5_parallel.py -q`

Expected: failure because `tabdml.stage5_parallel` does not exist.

- [ ] **Step 3: Implement phase-aware orchestration**

Launch the GPU worker and standard CPU shards concurrently, wait for all to succeed, launch ensemble shards separately, validate cache completeness, compose once, and atomically write `progress.json`. Include expected/completed/failed/missing counts per method, elapsed time, latest update, command return codes, and resumable phase state.

- [ ] **Step 4: Enforce formal launch evidence**

The formal CLI must require `--formal-approved` plus a compatible preflight summary containing exactly 900 successful results and zero failures, OOMs, fallbacks, missing keys, duplicate keys, non-finite estimates, and provenance mismatches. Smoke and preflight must reject that switch. This is a technical backstop; the flag is supplied only after the user explicitly approves.

- [ ] **Step 5: Run controller tests**

Run: `python -m pytest tests/test_stage5_parallel.py -q`

Expected: all tests pass on Windows-style and POSIX-style fixture paths.

- [ ] **Step 6: Commit orchestration**

```powershell
git add src/tabdml/stage5_parallel.py scripts/run_stage5_parallel.py tests/test_stage5_parallel.py
git commit -m "Add resumable Stage 5 orchestration"
```

### Task 5: Statistical summaries, bootstrap intervals, and publication figures

**Files:**
- Create: `src/tabdml/stage5_analysis.py`
- Create: `scripts/analyze_stage5.py`
- Create: `tests/test_stage5_analysis.py`
- Modify: `src/tabdml/figures.py`
- Modify: `tests/test_figures.py`

**Interfaces:**
- Consumes: valid Stage 5 JSONL records and matplotlib.
- Produces: `summarize_stage5`, `bootstrap_mse_interval`, `build_stage5_plot_data`, `estimate_formal_runtime`, `make_stage5_sensitivity_figures`, summary CSV/JSON, exact plot-data CSVs, two primary PNG/PDF figures, and four supplementary metric figures.

- [ ] **Step 1: Write failing aggregation, bootstrap, center-cell, and plotting tests**

```python
def test_summary_reports_required_causal_and_nuisance_metrics():
    summary = summarize_stage5(make_complete_records(replications=5))
    required = {"bias", "rmse", "treatment_mse", "empirical_sd", "mean_se", "coverage", "interval_width", "l_mse", "m_mse", "mean_runtime"}
    assert required.issubset(summary.columns)

def test_center_cell_is_identical_in_both_plot_datasets():
    fixed_n, fixed_p = build_stage5_plot_data(summary)
    left = fixed_n.query("n == 1000 and p == 50").sort_values(["scenario", "method"])
    right = fixed_p.query("n == 1000 and p == 50").sort_values(["scenario", "method"])
    pd.testing.assert_frame_equal(left.reset_index(drop=True), right.reset_index(drop=True))
```

Test the bootstrap against a deterministic hand fixture, require paired replication sets, verify log axis with original-unit tick labels, six panel titles, identical method style mapping, and exported CSV equality.

- [ ] **Step 2: Run tests and confirm missing-analysis failures**

Run: `python -m pytest tests/test_stage5_analysis.py tests/test_figures.py -q`

Expected: failure because Stage 5 analysis functions do not exist.

- [ ] **Step 3: Implement summaries and deterministic bootstrap**

Group by `(scenario,n,p,method)`. Compute bias, MSE, RMSE, empirical SD, mean SE, coverage, mean interval width, nuisance MSEs, runtime, and all status counts. Bootstrap the replication-level squared errors with 10,000 deterministic resamples and percentile limits at 2.5% and 97.5%; never bootstrap already-aggregated means.

- [ ] **Step 4: Implement the two primary 2-by-3 figures and supplements**

The fixed-`n` figure uses `p=[10,50,100]`; the fixed-`p` figure uses `n=[500,1000,2000]`. Use log-scaled treatment-effect MSE, original-unit tick labels, method-consistent colors/markers, mean lines, translucent 95% ribbons, one shared legend, and the same ordered six DGP panels. Export PNG at 300 DPI, vector PDF, and exact plotting CSV. Reuse the panel engine for Bias, Coverage, `l` MSE, and `m` MSE supplements.

- [ ] **Step 5: Add runtime projection and analysis CLI**

Estimate formal elapsed time from preflight per-task timings separately for the serial GPU lane, parallel standard-CPU lane, and parallel ensemble lane, then report the maximum concurrent lane plus composition/analysis overhead. Label the estimate as hardware- and implementation-dependent.

- [ ] **Step 6: Run analysis and figure tests**

Run: `python -m pytest tests/test_stage5_analysis.py tests/test_figures.py -q`

Expected: all tests pass and temporary PNG/PDF/CSV artifacts are nonempty.

- [ ] **Step 7: Commit analysis and plotting**

```powershell
git add src/tabdml/stage5_analysis.py src/tabdml/figures.py scripts/analyze_stage5.py tests/test_stage5_analysis.py tests/test_figures.py
git commit -m "Add Stage 5 sensitivity analysis figures"
```

### Task 6: Verification, smoke run, full tuning, and preflight run

**Files:**
- Modify: `README.md`
- Create runtime outputs under ignored directory: `results/stage5/`

**Interfaces:**
- Consumes: all Stage 5 commands from Tasks 1–5.
- Produces: verified code, smoke artifacts, frozen full tuning, preflight artifacts, preflight analysis, and a formal runtime estimate for user approval.

- [ ] **Step 1: Run focused Stage 5 tests**

Run: `python -m pytest tests/test_stage5_config.py tests/test_stage5_tuning.py tests/test_stage5_experiment.py tests/test_stage5_parallel.py tests/test_stage5_analysis.py tests/test_stage5_cli.py tests/test_figures.py -q`

Expected: all focused tests pass.

- [ ] **Step 2: Run the full regression suite**

Run: `python -m pytest -q`

Expected: all repository tests pass; existing documented MLP convergence warnings are acceptable, but failures are not.

- [ ] **Step 3: Run fast tuning and the 180-result smoke phase**

```powershell
python scripts/run_stage5_tuning.py --config configs/stage5_sensitivity.yaml --output results/stage5/tuning/fast-records.jsonl --execution-profile fast --num-shards 1 --shard-index 0
python scripts/select_stage5_tuning.py --config configs/stage5_sensitivity.yaml --input results/stage5/tuning/fast-records.jsonl --output results/stage5/tuning/frozen-fast.json --execution-profile fast
python scripts/run_stage5_parallel.py --config configs/stage5_sensitivity.yaml --profile smoke --frozen-tuning results/stage5/tuning/frozen-fast.json --output-root results/stage5/smoke --cpu-workers 5 --ensemble-workers 2
python scripts/analyze_stage5.py --config configs/stage5_sensitivity.yaml --profile smoke --input results/stage5/smoke/raw.jsonl --output-root results/stage5/smoke/analysis
```

Expected: 180 unique successes, zero failures/OOM/fallback/non-finite/missing/duplicate/provenance errors, and one GPU process handling both TabICLv2 methods.

- [ ] **Step 4: Run full 10-replication tuning and freeze 12 winners**

```powershell
python scripts/run_stage5_tuning.py --config configs/stage5_sensitivity.yaml --output results/stage5/tuning/full-records.jsonl --execution-profile full --num-shards 5 --shard-index 0
python scripts/run_stage5_tuning.py --config configs/stage5_sensitivity.yaml --output results/stage5/tuning/full-records.jsonl --execution-profile full --num-shards 5 --shard-index 1
python scripts/run_stage5_tuning.py --config configs/stage5_sensitivity.yaml --output results/stage5/tuning/full-records.jsonl --execution-profile full --num-shards 5 --shard-index 2
python scripts/run_stage5_tuning.py --config configs/stage5_sensitivity.yaml --output results/stage5/tuning/full-records.jsonl --execution-profile full --num-shards 5 --shard-index 3
python scripts/run_stage5_tuning.py --config configs/stage5_sensitivity.yaml --output results/stage5/tuning/full-records.jsonl --execution-profile full --num-shards 5 --shard-index 4
python scripts/select_stage5_tuning.py --config configs/stage5_sensitivity.yaml --input results/stage5/tuning/full-records.jsonl --output results/stage5/tuning/frozen-full.json --execution-profile full
```

The five runners must be started as independent processes by the controller in actual execution; the commands above document the exact shards. Expected: 720 unique successful tuning records and 12 frozen winners, one for each DGP-target pair.

- [ ] **Step 5: Run the 900-result full-settings preflight**

```powershell
python scripts/run_stage5_parallel.py --config configs/stage5_sensitivity.yaml --profile preflight --frozen-tuning results/stage5/tuning/frozen-full.json --output-root results/stage5/preflight --cpu-workers 5 --ensemble-workers 2
python scripts/analyze_stage5.py --config configs/stage5_sensitivity.yaml --profile preflight --input results/stage5/preflight/raw.jsonl --output-root results/stage5/preflight/analysis
```

Expected: 900 unique successes, zero gate violations, exact paired seeds, deduplicated centers, and a formal runtime projection.

- [ ] **Step 6: Document reproducible commands and artifact meanings**

Add a Stage 5 README section explaining the two sweeps, tuning freeze, hybrid hardware routing, resume commands, progress inspection, result directories, gate checks, interpretation limits, and why formal execution requires a separate approval.

- [ ] **Step 7: Commit documentation and stop before formal execution**

```powershell
git add README.md
git commit -m "Document Stage 5 benchmark workflow"
```

Report smoke and preflight gate evidence, measured elapsed time, projected formal elapsed time, GPU/CPU utilization observations, and estimated cost. Do not run `--profile formal` in this task.

