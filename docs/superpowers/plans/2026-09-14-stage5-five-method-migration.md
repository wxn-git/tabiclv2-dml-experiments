# Stage 5 Five-Method Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert Stage 5 permanently to five methods, reuse the completed non-ensemble preflight predictions through an audited migration, produce 750 validated preflight DML results and figures, and stop before the 15,000-result formal run.

**Architecture:** A new five-method config becomes the only active Stage 5 protocol. A dedicated migration module validates old six-method cache entries against both protocols and atomically rewrites them under the new provenance, without recomputing predictions or touching the source cache. Existing orchestration, composition, analysis, plotting, and formal-gate code then operate on the exact five-method universe.

**Tech Stack:** Python 3.12, pytest, NumPy, pandas, scikit-learn, XGBoost, PyTorch/TabICLv2, matplotlib, YAML, JSON/NPZ, Git.

## Global Constraints

- Stage 1--4 code, configurations, and results remain unchanged.
- Stage 5 methods are exactly `tabiclv2_1`, `tabiclv2_8`, `xgboost_tuned`, `extra_trees`, and `lasso`.
- Exact DML counts are smoke 150, preflight 750, and formal 15,000; nuisance-cache counts are 300, 1,500, and 30,000.
- The center `(n,p)=(1000,50)` is counted once in each DGP.
- The old source cache is read-only; migrated output uses `results/stage5_five/`.
- Formal execution requires a clean 750-result preflight plus explicit user approval and is not authorized by this plan.
- Preserve the user's untracked `artifacts/` directory.

---

### Task 1: Five-method configuration and exact protocol counts

**Files:**
- Create: `configs/stage5_sensitivity_five.yaml`
- Modify: `src/tabdml/stage5_config.py`
- Modify: `src/tabdml/stage5_parallel.py`
- Modify: `tests/test_stage5_config.py`
- Modify: `tests/test_stage5_parallel.py`

**Interfaces:**
- Consumes: existing Stage 5 profile, grid, tuning, fingerprint, pair, cache, and gate interfaces.
- Produces: validated five-method config and exact count constants derived from `len(config["methods"])` rather than the retired six-method assumptions.

- [ ] **Step 1: Write failing exact-universe tests**

Add tests that load `stage5_sensitivity_five.yaml`, assert the exact ordered five methods, and assert pair counts 150/750/15,000. Add gate tests that accept exactly 750 clean preflight records and reject 900, 750-as-float, stale fingerprints, ensemble output, and any nonzero violation.

- [ ] **Step 2: Run the tests and confirm RED**

Run: `\.venv\Scripts\python.exe -m pytest tests/test_stage5_config.py tests/test_stage5_parallel.py -q`

Expected: failures because the five-method config and new exact counts do not exist.

- [ ] **Step 3: Implement the five-method protocol**

Create the config by copying all DGP, sweep, tuning, profile, and Extra Trees fields unchanged and setting:

```yaml
methods: [tabiclv2_1, tabiclv2_8, xgboost_tuned, extra_trees, lasso]
```

Change config validation to accept only the legacy six-method config or the new exact five-method config, while exposing the loaded ordered method tuple. In the five-method controller, remove the ensemble batch and derive expected results as:

```python
expected = 30 * profile.replications * len(config["methods"])
```

For the five-method protocol require 150/750/15,000 and validate formal preflight summaries against 750 successes. Keep legacy behavior available only for reading and migrating old artifacts.

- [ ] **Step 4: Run focused and Stage 4 regression tests**

Run: `\.venv\Scripts\python.exe -m pytest tests/test_stage5_config.py tests/test_stage5_parallel.py tests/test_stage4_parallel.py -q`

Expected: all pass.

- [ ] **Step 5: Commit**

```powershell
git add configs/stage5_sensitivity_five.yaml src/tabdml/stage5_config.py src/tabdml/stage5_parallel.py tests/test_stage5_config.py tests/test_stage5_parallel.py
git commit -m "Adopt five-method Stage 5 protocol"
```

### Task 2: Audited migration of 1,500 completed nuisance caches

**Files:**
- Create: `src/tabdml/stage5_migration.py`
- Create: `scripts/migrate_stage5_cache.py`
- Create: `tests/test_stage5_migration.py`

**Interfaces:**
- Consumes: `iter_stage5_pairs`, `resolve_stage5_method`, `build_stage5_nuisance_spec`, `read_stage5_nuisance`, `NuisanceCache`, the old/new configs, and the validated old frozen tuning artifact.
- Produces: `rebind_stage5_tuning(...) -> dict`, `migrate_stage5_cache(...) -> dict`, a new five-method frozen tuning artifact, 1,500 atomic destination NPZ/sidecars, and `migration_manifest.json`.

- [ ] **Step 1: Write migration safety tests**

Use tiny synthetic old/new task universes. Assert that tuning rebinding preserves all 12 selected candidates, rankings, losses, and effective parameters while atomically replacing only config-derived fingerprints. Assert that migration copies only the five retained methods; prediction bytes decode to equal finite arrays; fold timings, device metadata, seeds, learner kind, and effective parameters match; source files remain unchanged; destination task keys/fingerprints are new; and the manifest contains exact overall/per-method counts and SHA-256 array hashes. Test rejection of changed DGP fields, folds, profile namespace, tuning winners, learner parameters, seeds, corrupt caches, missing sidecars, fallback, ensemble destination entries, pre-existing conflicting destination files, and partial migration without `--resume`.

- [ ] **Step 2: Run the test and confirm RED**

Run: `\.venv\Scripts\python.exe -m pytest tests/test_stage5_migration.py -q`

Expected: import failure because the migration module does not exist.

- [ ] **Step 3: Implement fail-closed migration**

Validate the old tuning artifact, copy its rankings and 12 winners, replace its config fingerprint and tuning-run fingerprint with values derived from the new config, then validate and atomically write the new artifact. Build old/new task maps keyed by `(scenario,n,p,replication,method,target)`. Before writing, compare every computation-affecting config field after removing only `methods`; require equal target-specific effective XGBoost winners. For each retained task, read the old validated result, construct the new task/result metadata, verify array and timing invariants, and write through temporary files plus `fsync`/`os.replace`. Hash canonical prediction bytes before and after. Write the manifest only after the exact destination universe validates.

- [ ] **Step 4: Run migration and regression tests**

Run: `\.venv\Scripts\python.exe -m pytest tests/test_stage5_migration.py tests/test_stage5_experiment.py tests/test_nuisance_cache.py -q`

Expected: all pass.

- [ ] **Step 5: Commit**

```powershell
git add src/tabdml/stage5_migration.py scripts/migrate_stage5_cache.py tests/test_stage5_migration.py
git commit -m "Add audited Stage 5 cache migration"
```

### Task 3: Five-method composition, analysis, figures, and final verification

**Files:**
- Modify: `src/tabdml/stage5_analysis.py`
- Modify: `src/tabdml/figures.py`
- Modify: `scripts/analyze_stage5.py`
- Modify: `README.md`
- Modify: `tests/test_stage5_analysis.py`
- Modify: `tests/test_figures.py`

**Interfaces:**
- Consumes: migrated five-method cache, frozen full tuning, Stage 5 composer/controller, summary and plotting functions.
- Produces: exactly 750 composed results, clean preflight gate summary, runtime projection, two primary and four supplementary five-method figures, and documented reproducible commands.

- [ ] **Step 1: Write failing five-method output tests**

Assert summaries contain exactly 150 groups, legends contain exactly five methods, plot CSVs contain no `ensemble`, the center row is identical in both sweeps, and preflight analysis emits exactly 750 successes with paired-seed and center-dedup checks true. Assert six-method or unexpected ensemble inputs fail closed.

- [ ] **Step 2: Run tests and confirm RED**

Run: `\.venv\Scripts\python.exe -m pytest tests/test_stage5_analysis.py tests/test_figures.py -q`

Expected: failures because plotting and analysis still require six methods.

- [ ] **Step 3: Implement five-method outputs**

Derive method order from the validated config, omit ensemble styles and lanes, compute the formal runtime projection as `max(serial_gpu_lane, parallel_cpu_lane) + composition/analysis overhead`, and produce a gate summary with exactly 750 expected/successful records and zero violations. Update README with migration, resume, progress, analysis, and formal-gate commands.

- [ ] **Step 4: Execute the audited migration**

Run:

```powershell
\.venv\Scripts\python.exe scripts/migrate_stage5_cache.py --source-config configs/stage5_sensitivity.yaml --destination-config configs/stage5_sensitivity_five.yaml --profile preflight --source-tuning results/stage5/tuning/frozen-full.json --destination-tuning results/stage5_five/tuning/frozen-full.json --source-cache results/stage5/preflight/cache --destination-cache results/stage5_five/preflight/cache --manifest results/stage5_five/preflight/migration_manifest.json
```

Expected: exactly 1,500 validated migrated caches, five methods each with 300 caches, zero ensemble artifacts, and no source mutation.

- [ ] **Step 5: Resume the controller to compose 750 results**

Run:

```powershell
\.venv\Scripts\python.exe scripts/run_stage5_parallel.py --config configs/stage5_sensitivity_five.yaml --profile preflight --frozen-tuning results/stage5_five/tuning/frozen-full.json --cache-root results/stage5_five/preflight/cache --output-root results/stage5_five/preflight/raw --log-dir results/stage5_five/preflight/log --cpu-workers 5
```

Expected: existing 1,500 caches validate and skip, composition writes exactly 750 successes, and all violation counts are zero.

- [ ] **Step 6: Analyze and render figures**

Run:

```powershell
\.venv\Scripts\python.exe scripts/analyze_stage5.py --config configs/stage5_sensitivity_five.yaml --profile preflight --input results/stage5_five/preflight/raw --output-root results/stage5_five/preflight/analysis --bootstrap-resamples 10000 --cpu-workers 5
```

Expected: exact summary/plot CSVs, two primary PNG/PDF figures, four supplementary PNG/PDF figures, a clean 750-record gate summary, and a hardware-dependent 15,000-result runtime estimate.

- [ ] **Step 7: Run focused and full verification**

Run: `\.venv\Scripts\python.exe -m pytest tests/test_stage5_config.py tests/test_stage5_tuning.py tests/test_stage5_experiment.py tests/test_stage5_parallel.py tests/test_stage5_migration.py tests/test_stage5_analysis.py tests/test_stage5_cli.py tests/test_figures.py -q`

Then run: `\.venv\Scripts\python.exe -m pytest -q`

Expected: all tests pass; existing documented sklearn convergence warnings are acceptable.

- [ ] **Step 8: Commit documentation and stop**

```powershell
git add src/tabdml/stage5_analysis.py src/tabdml/figures.py scripts/analyze_stage5.py README.md tests/test_stage5_analysis.py tests/test_figures.py
git commit -m "Complete five-method Stage 5 preflight"
```

Report exact gate counts, elapsed time, formal runtime estimate, and output paths. Do not launch `--profile formal`; request explicit user approval first.
