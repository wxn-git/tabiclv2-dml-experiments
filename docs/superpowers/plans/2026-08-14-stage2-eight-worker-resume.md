# Stage 2 Eight-Worker Resume Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Resume the 16 missing Stage 2 ensemble replications with eight CPU workers, two missing replications per worker.

**Architecture:** Add deterministic modulo-based replication sharding to the Stage 2 CLI, then add a dedicated supervisor that launches eight workers only for `ensemble_05.yaml`. A CMD launcher supplies the `src` import path, single-thread limits, and persistent logs for Task Scheduler.

**Tech Stack:** Python 3.12, argparse, pytest, Windows CMD, Windows Task Scheduler.

## Global Constraints

- Keep all existing result JSON files unchanged and rely on atomic skip/resume behavior.
- Do not change seeds, folds, model hyperparameters, data generation, or estimands.
- Assign replication `r` to shard `r % 8`, giving exactly two of `r084-r099` to each worker.
- Set OMP, MKL, OpenBLAS, and NumExpr thread counts to one per process.

---

### Task 1: Replication Sharding

**Files:**
- Modify: `src/tabdml/sharding.py`
- Modify: `tests/test_sharding.py`

**Interfaces:**
- Produces: `replication_belongs_to_shard(replication: int, num_shards: int, shard_index: int) -> bool`

- [x] Add a failing test asserting disjoint, complete ownership and two assignments per shard for replications 84 through 99.
- [x] Run `python -m pytest tests/test_sharding.py -v` and verify failure because the function is missing.
- [x] Implement the function by validating shard arguments and returning `replication % num_shards == shard_index`.
- [x] Re-run the test and verify it passes.

### Task 2: Stage 2 CLI Shard Filtering

**Files:**
- Modify: `scripts/run_stage2.py`
- Create: `tests/test_stage2_cli.py`

**Interfaces:**
- Consumes: `replication_belongs_to_shard(...)`
- Produces: CLI options `--num-shards` and `--shard-index`

- [x] Add a failing CLI test that invokes a temporary two-replication fast configuration with two shards and verifies each process creates only its owned replication.
- [x] Run `python -m pytest tests/test_stage2_cli.py -v` and verify argparse rejects the new options.
- [x] Add the two arguments, validate them, and filter replications before constructing `TaskSpec`.
- [x] Re-run the CLI test and verify both shards are disjoint and complete.

### Task 3: Eight-Worker Resume Supervisor

**Files:**
- Modify: `src/tabdml/parallel.py`
- Modify: `tests/test_parallel.py`
- Create: `scripts/run_stage2_resume_parallel.py`
- Create: `scripts/run_stage2_resume_parallel.cmd`

**Interfaces:**
- Produces: `build_stage2_resume_worker_commands(..., cpu_workers: int) -> tuple[WorkerCommand, ...]`
- Produces: eight named workers `cpu_ensemble_resume_00` through `cpu_ensemble_resume_07`

- [x] Add a failing unit test asserting eight worker commands, the ensemble learner, and unique shard arguments.
- [x] Run `python -m pytest tests/test_parallel.py -v` and verify the builder is missing.
- [x] Implement the builder with validation that worker count is at least one.
- [x] Add the supervisor using `run_workers` and a CMD launcher that sets `PYTHONPATH`, persistent logs, and all four single-thread environment variables.
- [x] Run sharding, CLI, and parallel tests together and verify they pass.

### Task 4: Scheduled Execution and Live Verification

**Files:**
- Runtime output: `results/logs/stage2_resume_8/`

**Interfaces:**
- Consumes: `scripts/run_stage2_resume_parallel.cmd`
- Produces: scheduled task `TabDML_Stage2_Resume8`

- [x] Register a Task Scheduler action whose working directory is the project root and whose launcher is the resume CMD file.
- [x] Start the task and query state without holding or timing out its console.
- [x] Verify eight live worker PIDs, `completed=4884`, `missing=16`, and per-worker logs that skip completed results.
- [x] Recheck after a short interval that worker CPU times increase and no stderr log contains a new traceback.
