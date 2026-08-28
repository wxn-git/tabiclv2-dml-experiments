# Stage 1 CPU/GPU Parallel Execution Design

## Objective

Accelerate the existing Stage 1 Monte Carlo experiment without changing its statistical design, model hyperparameters, random seeds, folds, task identities, or already completed results.

## Constraints

- Preserve every successful JSON record already stored under `results/raw`.
- Keep the six learner definitions unchanged so early and later records remain directly comparable.
- Keep five-fold cross-fitting, the two nuisance targets, all DGPs, and all deterministic seeds unchanged.
- Never let two workers own the same task key.
- Run at most one TabICLv2 worker so GPU jobs cannot compete for VRAM.
- CPU-only workers must not initialize a CUDA context.
- Continue using one atomic JSON file per task and the existing success-record resume check.
- The workspace is not a Git repository, so commit steps are not applicable.

## Alternatives Considered

### 1. Deterministic process-level sharding (selected)

Run one GPU worker for TabICLv2 and eight CPU workers for the five traditional learners. Assign each traditional task to exactly one CPU worker using a stable hash of its task key. This preserves the existing numerical experiment and reuses all completed records.

### 2. Increase `n_jobs` inside nested estimators

This is a smaller code change, but the ensemble already contains several nested cross-validation layers. Parallelizing each layer risks thread oversubscription, unstable memory use, and worse performance. It also gives less control over CPU/GPU separation.

### 3. Simplify the ensemble or move XGBoost to CUDA

This may reduce runtime further, but it changes the benchmark definition. Mixing those records with existing records would invalidate direct comparison, so it is excluded from this acceleration pass.

## Architecture

`scripts/run_stage1.py` gains deterministic shard arguments. After applying scenario, sample-size, dimension, replication, and learner filters, a task belongs to worker `i` exactly when a stable hash of its complete task key modulo the worker count equals `i`.

`scripts/run_stage1_parallel.py` acts as a persistent supervisor:

- spawn one GPU child for learner `tabiclv2`;
- spawn eight CPU children for learners `lasso`, `random_forest`, `xgboost`, `mlp`, and `ensemble`;
- give CPU children shard indices `0..7` with shard count `8`;
- redirect each child to separate UTF-8 stdout and stderr logs;
- record child PIDs and exit codes in a machine-readable state file;
- wait for all children so background process lifetime is tied to the supervisor.

The existing single-process run is stopped only after tests and a smoke run verify the new scheduler. The new workers use the same `results/raw` directory, so successful tasks return `skipped` and unfinished tasks continue normally.

## CUDA Isolation

`crossfit_nuisances` currently initializes PyTorch/CUDA for every learner merely to measure peak GPU memory. With multiple CPU workers this would create unnecessary CUDA contexts and consume VRAM. CUDA helpers will therefore be initialized only for learner names beginning with `tabiclv2`. Traditional records will report `peak_gpu_mb=None`; this is more accurate than reporting PyTorch bookkeeping memory for a CPU-only model and does not affect estimates.

## Data Safety and Failure Handling

- Worker ownership is disjoint by learner group and stable shard.
- `ResultStore.exists` skips successful records.
- Each result remains written through a temporary file followed by `os.replace`.
- A worker crash does not corrupt completed JSON files; restarting the supervisor resumes missing tasks.
- The supervisor reports nonzero child exit codes but does not delete or rewrite successful records.
- The GPU worker count is fixed at one. CPU worker count is configurable, with eight as the initial value for the available 32 logical processors.

## Verification

Automated tests will prove:

- every filtered task maps to exactly one shard;
- different shard indices are disjoint and their union is complete;
- invalid shard arguments are rejected;
- CPU learners do not call CUDA helpers;
- TabICLv2 still calls CUDA helpers;
- the supervisor constructs one GPU command and the requested number of disjoint CPU commands.

A smoke run will use a separate output directory and reduced workload. After restarting the real experiment, verification requires:

- the old process is absent;
- one supervisor, one GPU worker, and eight CPU workers are alive;
- existing successful JSON count never decreases;
- result count increases from multiple worker logs;
- GPU activity appears during the dedicated TabICLv2 stream;
- no duplicate task ownership or JSON parsing failures occur.

## Expected Effect

The change removes the one-lane task queue. Traditional learners remain individually single-threaded, but up to eight independent traditional tasks can run at once, while TabICLv2 continuously receives its own GPU queue. The exact speed-up will be measured after launch; no fixed multiplier is assumed in advance.
