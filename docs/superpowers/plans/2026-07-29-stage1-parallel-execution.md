# Stage 1 CPU/GPU Parallel Execution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the single Stage 1 queue with one dedicated TabICLv2 GPU worker and eight deterministic, disjoint traditional-learner CPU workers while preserving all existing results and numerical specifications.

**Architecture:** A pure sharding helper assigns task keys to CPU workers by stable hash. The existing Stage 1 CLI applies that shard filter, CUDA accounting is activated only for TabICLv2, and a supervisor process launches and monitors one GPU child plus configurable CPU children with separate logs and a JSON state file.

**Tech Stack:** Python 3.12, pathlib, subprocess, hashlib-derived experiment seeds, pytest, Windows CMD launcher.

## Global Constraints

- Preserve successful records already stored under `results/raw`.
- Do not change learners, hyperparameters, DGPs, folds, seeds, or task keys.
- Exactly one worker owns each traditional task and only one GPU worker runs TabICLv2.
- CPU-only workers must not initialize CUDA.
- Continue using atomic one-JSON-per-task storage and success-record resume behavior.
- The workspace is not a Git repository, so commit steps are not applicable.

---

### Task 1: Deterministic task sharding and CLI filtering

**Files:**
- Create: `src/tabdml/sharding.py`
- Modify: `scripts/run_stage1.py`
- Create: `tests/test_sharding.py`

**Interfaces:**
- Consumes: `tabdml.config.derive_seed`, `TaskSpec.key`.
- Produces: `validate_shard(num_shards: int, shard_index: int) -> None` and `belongs_to_shard(task_key: str, num_shards: int, shard_index: int) -> bool`.

- [ ] **Step 1: Write failing tests for complete, disjoint, deterministic sharding and invalid arguments**

```python
import pytest
from tabdml.sharding import belongs_to_shard, validate_shard


def test_shards_are_disjoint_and_complete():
    keys = [f"task-{index}" for index in range(200)]
    ownership = {
        key: [shard for shard in range(8) if belongs_to_shard(key, 8, shard)]
        for key in keys
    }
    assert all(len(shards) == 1 for shards in ownership.values())
    assert ownership == {
        key: [shard for shard in range(8) if belongs_to_shard(key, 8, shard)]
        for key in keys
    }


@pytest.mark.parametrize("count,index", [(0, 0), (2, -1), (2, 2)])
def test_invalid_shard_arguments_are_rejected(count, index):
    with pytest.raises(ValueError):
        validate_shard(count, index)
```

- [ ] **Step 2: Run the test and verify RED**

Run: `python -m pytest tests/test_sharding.py -v`

Expected: collection fails because `tabdml.sharding` does not exist.

- [ ] **Step 3: Implement the stable shard helper**

```python
from .config import derive_seed


def validate_shard(num_shards: int, shard_index: int) -> None:
    if num_shards < 1:
        raise ValueError("num_shards must be at least 1")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError("shard_index must be in [0, num_shards)")


def belongs_to_shard(task_key: str, num_shards: int, shard_index: int) -> bool:
    validate_shard(num_shards, shard_index)
    return derive_seed("stage1-shard-v1", task_key) % num_shards == shard_index
```

- [ ] **Step 4: Add `--num-shards` and `--shard-index` to `run_stage1.py` and apply the filter after existing task filters**

```python
parser.add_argument("--num-shards", type=int, default=1)
parser.add_argument("--shard-index", type=int, default=0)

validate_shard(args.num_shards, args.shard_index)

if not belongs_to_shard(task.key, args.num_shards, args.shard_index):
    continue
```

- [ ] **Step 5: Verify GREEN**

Run: `python -m pytest tests/test_sharding.py tests/test_stages.py -v`

Expected: all tests pass.

### Task 2: Prevent CPU learners from initializing CUDA

**Files:**
- Modify: `src/tabdml/crossfit.py`
- Modify: `tests/test_crossfit.py`

**Interfaces:**
- Consumes: `learner_name` passed to `crossfit_nuisances`.
- Produces: CUDA timing and peak-memory accounting only for learner names starting with `tabiclv2`.

- [ ] **Step 1: Write a failing CPU isolation test**

```python
def test_cpu_crossfit_does_not_initialize_cuda(monkeypatch):
    data = simulate_plr("linear", 60, 10, 3)

    def fail_if_called():
        raise AssertionError("CPU learner initialized CUDA")

    monkeypatch.setattr("tabdml.crossfit._cuda_helpers", fail_if_called)
    result = crossfit_nuisances(
        data, "lasso", make_folds(60, 3, 4), seed=5, tabicl_estimators=0, fast=True
    )
    assert result.peak_gpu_mb is None
```

- [ ] **Step 2: Run the test and verify RED**

Run: `python -m pytest tests/test_crossfit.py::test_cpu_crossfit_does_not_initialize_cuda -v`

Expected: FAIL with `CPU learner initialized CUDA`.

- [ ] **Step 3: Restrict CUDA initialization to TabICLv2**

```python
torch = _cuda_helpers() if learner_name.startswith("tabiclv2") else None
```

- [ ] **Step 4: Verify GREEN**

Run: `python -m pytest tests/test_crossfit.py -v`

Expected: all cross-fitting tests pass.

### Task 3: Parallel worker command construction and supervisor

**Files:**
- Create: `src/tabdml/parallel.py`
- Create: `scripts/run_stage1_parallel.py`
- Create: `scripts/run_stage1_parallel.cmd`
- Create: `tests/test_parallel.py`

**Interfaces:**
- Produces: immutable `WorkerCommand(name: str, argv: tuple[str, ...])`.
- Produces: `build_worker_commands(python_executable, stage1_script, output_root, cpu_workers, extra_args=()) -> tuple[WorkerCommand, ...]`.
- The first command is `gpu_tabiclv2`; remaining commands are `cpu_00` through `cpu_{N-1}`.

- [ ] **Step 1: Write failing command-construction tests**

```python
import pytest
from tabdml.parallel import build_worker_commands


def test_parallel_commands_have_one_gpu_worker_and_disjoint_cpu_shards():
    commands = build_worker_commands("python", "run_stage1.py", "results/raw", 3)
    assert [command.name for command in commands] == ["gpu_tabiclv2", "cpu_00", "cpu_01", "cpu_02"]
    assert commands[0].argv[-1] == "tabiclv2"
    for index, command in enumerate(commands[1:]):
        assert command.argv[-4:] == ("--num-shards", "3", "--shard-index", str(index))


def test_parallel_commands_reject_zero_cpu_workers():
    with pytest.raises(ValueError):
        build_worker_commands("python", "run_stage1.py", "results/raw", 0)
```

- [ ] **Step 2: Run the test and verify RED**

Run: `python -m pytest tests/test_parallel.py -v`

Expected: collection fails because `tabdml.parallel` does not exist.

- [ ] **Step 3: Implement `WorkerCommand` and `build_worker_commands`**

```python
from dataclasses import dataclass

TRADITIONAL_LEARNERS = ("lasso", "random_forest", "xgboost", "mlp", "ensemble")


@dataclass(frozen=True)
class WorkerCommand:
    name: str
    argv: tuple[str, ...]


def build_worker_commands(python_executable, stage1_script, output_root, cpu_workers, extra_args=()):
    if cpu_workers < 1:
        raise ValueError("cpu_workers must be at least 1")
    common = (str(python_executable), str(stage1_script), "--output-root", str(output_root), *extra_args)
    commands = [WorkerCommand("gpu_tabiclv2", (*common, "--learners", "tabiclv2"))]
    for index in range(cpu_workers):
        commands.append(
            WorkerCommand(
                f"cpu_{index:02d}",
                (*common, "--learners", *TRADITIONAL_LEARNERS,
                 "--num-shards", str(cpu_workers), "--shard-index", str(index)),
            )
        )
    return tuple(commands)
```

- [ ] **Step 4: Verify command construction GREEN**

Run: `python -m pytest tests/test_parallel.py -v`

Expected: both tests pass.

- [ ] **Step 5: Implement the supervisor and Windows launcher**

The supervisor parses `--cpu-workers`, `--output-root`, `--log-dir`, and the existing Stage 1 filters. It calls `build_worker_commands`, opens `<worker>.stdout.log` and `<worker>.stderr.log`, starts every child with `subprocess.Popen(..., cwd=project_root)`, writes `state.json` after starts and exits, waits for all children, and terminates remaining children on `KeyboardInterrupt`.

The CMD launcher sets `PYTHONPATH`, `MPLCONFIGDIR`, `PYTHONUTF8`, and `HF_HUB_DISABLE_PROGRESS_BARS`, then invokes the bundled Python runtime with `scripts/run_stage1_parallel.py --cpu-workers 8 --output-root results/raw --log-dir results/logs/stage1_parallel`.

- [ ] **Step 6: Run the full fast test suite**

Run: `python -m pytest -m "not gpu and not integration" -v`

Expected: all tests pass.

### Task 4: Smoke verification and live migration

**Files:**
- Write smoke outputs under: `results/parallel_smoke_raw/`
- Write smoke logs under: `results/logs/stage1_parallel_smoke/`
- Write live logs under: `results/logs/stage1_parallel/`

**Interfaces:**
- Consumes: tested supervisor and existing `results/raw` success records.
- Produces: a persistent live supervisor with one GPU worker and eight CPU workers.

- [ ] **Step 1: Run a reduced parallel smoke experiment**

Run: `python scripts/run_stage1_parallel.py --cpu-workers 2 --output-root results/parallel_smoke_raw --log-dir results/logs/stage1_parallel_smoke --scenarios linear --sample-sizes 500 --dimensions 10 --replications 1 --fast`

Expected: supervisor exits zero; six unique successful JSON files exist; worker logs and state JSON are valid.

- [ ] **Step 2: Record the current successful result count and exact old PIDs**

Run: PowerShell checks for launcher PID `8088`, Python PID `9016`, and parses every `results/raw/*.json` record.

Expected: existing JSON records remain readable and the count is recorded before migration.

- [ ] **Step 3: Stop only the verified old launcher and child process**

Run: `Stop-Process -Id 9016,8088`

Expected: both exact old PIDs are absent; existing JSON count does not decrease.

- [ ] **Step 4: Start the tested parallel CMD launcher in a hidden background process**

Run: `Start-Process scripts/run_stage1_parallel.cmd -WorkingDirectory <workspace> -WindowStyle Hidden -PassThru`

Expected: a persistent CMD/supervisor process starts and `results/logs/stage1_parallel/state.json` lists nine child workers.

- [ ] **Step 5: Verify sustained parallel progress**

Poll process state, worker logs, JSON count, CPU use, and `nvidia-smi` over at least two result-count increases.

Expected: count grows without parse failures; multiple CPU logs advance; GPU log advances; stderr contains no fatal traceback; GPU activity is attributable to the dedicated worker.

- [ ] **Step 6: Run final verification**

Run: `python -m pytest -v`

Run: `python -m compileall src scripts`

Expected: tests and compilation pass, and the live experiment remains running.
