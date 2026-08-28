from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from os import PathLike
from pathlib import Path
from typing import Sequence


TRADITIONAL_LEARNERS = (
    "lasso",
    "random_forest",
    "xgboost",
    "mlp",
    "ensemble",
)

STAGE3_GPU_PAIRS = (
    "tab_tab",
    "tab_xgb",
    "xgb_tab",
    "oracle_tab",
    "tab_oracle",
)

STAGE3_CPU_PAIRS = (
    "xgb_xgb",
    "oracle_xgb",
    "xgb_oracle",
    "oracle_oracle",
)


@dataclass(frozen=True)
class WorkerCommand:
    name: str
    argv: tuple[str, ...]


def build_worker_commands(
    python_executable: str | PathLike[str],
    stage1_script: str | PathLike[str],
    output_root: str | PathLike[str],
    cpu_workers: int,
    extra_args: Sequence[str] = (),
) -> tuple[WorkerCommand, ...]:
    if cpu_workers < 1:
        raise ValueError("cpu_workers must be at least 1")

    common = (
        str(python_executable),
        str(stage1_script),
        "--output-root",
        str(output_root),
        *map(str, extra_args),
    )
    commands = [
        WorkerCommand("gpu_tabiclv2", (*common, "--learners", "tabiclv2"))
    ]
    for index in range(cpu_workers):
        commands.append(
            WorkerCommand(
                f"cpu_{index:02d}",
                (
                    *common,
                    "--learners",
                    *TRADITIONAL_LEARNERS,
                    "--num-shards",
                    str(cpu_workers),
                    "--shard-index",
                    str(index),
                ),
            )
        )
    return tuple(commands)


def build_stage2_worker_commands(
    python_executable: str | PathLike[str],
    stage2_script: str | PathLike[str],
    output_root: str | PathLike[str],
    selected_config: str | PathLike[str],
    partition_config_dir: str | PathLike[str],
) -> tuple[WorkerCommand, ...]:
    partition_root = str(partition_config_dir).rstrip("/\\")

    def command(
        name: str,
        config: str | PathLike[str],
        *learners: str,
    ) -> WorkerCommand:
        return WorkerCommand(
            name,
            (
                str(python_executable),
                str(stage2_script),
                "--config",
                str(config),
                "--learners",
                *learners,
                "--output-root",
                str(output_root),
            ),
        )

    commands = [
        command("gpu_tabiclv2", selected_config, "tabiclv2_1", "tabiclv2_8"),
        command("cpu_aux", selected_config, "lasso", "mlp"),
        command(
            "cpu_rf_a",
            f"{partition_root}/group_a.yaml",
            "random_forest",
        ),
        command(
            "cpu_rf_b",
            f"{partition_root}/group_b.yaml",
            "random_forest",
        ),
        command("cpu_xgb_a", f"{partition_root}/group_a.yaml", "xgboost"),
        command("cpu_xgb_b", f"{partition_root}/group_b.yaml", "xgboost"),
    ]
    for index in range(7):
        commands.append(
            command(
                f"cpu_ensemble_{index:02d}",
                f"{partition_root}/ensemble_{index:02d}.yaml",
                "ensemble",
            )
        )
    return tuple(commands)


def build_stage2_resume_worker_commands(
    python_executable: str | PathLike[str],
    stage2_script: str | PathLike[str],
    output_root: str | PathLike[str],
    config: str | PathLike[str],
    cpu_workers: int,
) -> tuple[WorkerCommand, ...]:
    if cpu_workers < 1:
        raise ValueError("cpu_workers must be at least 1")

    common = (
        str(python_executable),
        str(stage2_script),
        "--config",
        str(config),
        "--learners",
        "ensemble",
        "--output-root",
        str(output_root),
        "--num-shards",
        str(cpu_workers),
    )
    return tuple(
        WorkerCommand(
            f"cpu_ensemble_resume_{index:02d}",
            (*common, "--shard-index", str(index)),
        )
        for index in range(cpu_workers)
    )


def build_stage3_worker_commands(
    python_executable: str | PathLike[str],
    stage3_script: str | PathLike[str],
    output_root: str | PathLike[str],
    config: str | PathLike[str],
    cpu_workers: int,
    replications: int,
) -> tuple[WorkerCommand, ...]:
    if cpu_workers < 1:
        raise ValueError("cpu_workers must be at least 1")
    if replications < 1:
        raise ValueError("replications must be at least 1")

    common = (
        str(python_executable),
        str(stage3_script),
        "--config",
        str(config),
        "--output-root",
        str(output_root),
        "--replications",
        str(replications),
    )
    commands = [
        WorkerCommand(
            "gpu_stage3_pairs",
            (*common, "--pair-names", *STAGE3_GPU_PAIRS),
        )
    ]
    for index in range(cpu_workers):
        commands.append(
            WorkerCommand(
                f"cpu_stage3_{index:02d}",
                (
                    *common,
                    "--pair-names",
                    *STAGE3_CPU_PAIRS,
                    "--num-shards",
                    str(cpu_workers),
                    "--shard-index",
                    str(index),
                ),
            )
        )
    return tuple(commands)


def _write_state(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(".json.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def run_workers(
    commands: Sequence[WorkerCommand],
    cwd: str | PathLike[str],
    log_dir: str | PathLike[str],
) -> dict[str, int]:
    working_directory = Path(cwd)
    logs = Path(log_dir)
    logs.mkdir(parents=True, exist_ok=True)
    state_path = logs / "state.json"
    state = {
        "status": "starting",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "workers": [],
    }
    processes: list[tuple[subprocess.Popen, object, object, dict]] = []

    try:
        for command in commands:
            stdout_handle = (logs / f"{command.name}.stdout.log").open(
                "a", encoding="utf-8"
            )
            stderr_handle = (logs / f"{command.name}.stderr.log").open(
                "a", encoding="utf-8"
            )
            try:
                process = subprocess.Popen(
                    command.argv,
                    cwd=working_directory,
                    stdout=stdout_handle,
                    stderr=stderr_handle,
                )
            except Exception:
                stdout_handle.close()
                stderr_handle.close()
                raise
            worker_state = {
                "name": command.name,
                "pid": process.pid,
                "argv": list(command.argv),
                "status": "running",
                "exit_code": None,
            }
            state["workers"].append(worker_state)
            processes.append((process, stdout_handle, stderr_handle, worker_state))

        state["status"] = "running"
        _write_state(state_path, state)

        exit_codes: dict[str, int] = {}
        for process, _, _, worker_state in processes:
            exit_code = process.wait()
            worker_state["status"] = "completed" if exit_code == 0 else "failed"
            worker_state["exit_code"] = exit_code
            exit_codes[str(worker_state["name"])] = exit_code
            _write_state(state_path, state)

        state["status"] = "completed" if all(code == 0 for code in exit_codes.values()) else "failed"
        state["finished_at"] = datetime.now(timezone.utc).isoformat()
        _write_state(state_path, state)
        return exit_codes
    except BaseException:
        for process, _, _, worker_state in processes:
            if process.poll() is None:
                process.terminate()
            exit_code = process.wait()
            worker_state["status"] = "terminated"
            worker_state["exit_code"] = exit_code
        state["status"] = "interrupted"
        state["finished_at"] = datetime.now(timezone.utc).isoformat()
        _write_state(state_path, state)
        raise
    finally:
        for _, stdout_handle, stderr_handle, _ in processes:
            stdout_handle.close()
            stderr_handle.close()
