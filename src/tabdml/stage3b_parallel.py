from __future__ import annotations

from os import PathLike
from pathlib import Path

from .parallel import WorkerCommand


SCREEN_GPU_CANDIDATES = (
    "tabiclv2_1",
    "tabiclv2_8",
    "tabiclv2_1_m0_diagnostic",
)

SCREEN_CPU_CANDIDATES = (
    "current_xgboost",
    "xgb_d2_lr003_leaf1",
    "xgb_d2_lr005_leaf5",
    "xgb_d3_lr003_leaf5",
    "xgb_d3_lr005_leaf10",
    "xgb_d4_lr003_leaf5",
    "xgb_d5_lr003_leaf10",
    "extra_f05_leaf1",
    "extra_f05_leaf2",
    "extra_f05_leaf5",
    "extra_f10_leaf1",
    "extra_f10_leaf2",
    "extra_f10_leaf5",
    "current_xgboost_m0_diagnostic",
)


def _validate_workers(cpu_workers: int, replications: int) -> None:
    if cpu_workers < 1:
        raise ValueError("cpu_workers must be at least 1")
    if replications < 1:
        raise ValueError("replications must be at least 1")


def build_stage3b_batch_a_commands(
    python_executable: str | PathLike[str],
    project_root: str | PathLike[str],
    cache_root: str | PathLike[str],
    cpu_workers: int,
    replications: int,
) -> tuple[WorkerCommand, ...]:
    _validate_workers(cpu_workers, replications)
    script = Path(project_root) / "scripts" / "run_stage3b_cache.py"
    common = (
        str(python_executable),
        str(script),
        "--cache-root",
        str(cache_root),
        "--replications",
        str(replications),
    )
    commands = [
        WorkerCommand(
            "gpu_stage3b_batch_a",
            (*common, "--learners", "tabiclv2_1", "--targets", "l", "m"),
        )
    ]
    for index in range(cpu_workers):
        commands.append(
            WorkerCommand(
                f"cpu_stage3b_batch_a_{index:02d}",
                (
                    *common,
                    "--learners",
                    "xgboost",
                    "oracle",
                    "--targets",
                    "l",
                    "m",
                    "--num-shards",
                    str(cpu_workers),
                    "--shard-index",
                    str(index),
                ),
            )
        )
    return tuple(commands)


def build_stage3b_screening_commands(
    python_executable: str | PathLike[str],
    project_root: str | PathLike[str],
    output_root: str | PathLike[str],
    cpu_workers: int,
    replications: int,
) -> tuple[WorkerCommand, ...]:
    _validate_workers(cpu_workers, replications)
    script = Path(project_root) / "scripts" / "run_stage3b_screen.py"
    common = (
        str(python_executable),
        str(script),
        "--output-root",
        str(output_root),
        "--replications",
        str(replications),
    )
    commands = [
        WorkerCommand(
            "gpu_stage3b_screen",
            (*common, "--candidates", *SCREEN_GPU_CANDIDATES),
        )
    ]
    for index in range(cpu_workers):
        commands.append(
            WorkerCommand(
                f"cpu_stage3b_screen_{index:02d}",
                (
                    *common,
                    "--candidates",
                    *SCREEN_CPU_CANDIDATES,
                    "--num-shards",
                    str(cpu_workers),
                    "--shard-index",
                    str(index),
                ),
            )
        )
    return tuple(commands)


def build_stage3b_confirmation_commands(
    python_executable: str | PathLike[str],
    project_root: str | PathLike[str],
    cache_root: str | PathLike[str],
    selected_models: str | PathLike[str],
    cpu_workers: int,
    replications: int,
) -> tuple[WorkerCommand, ...]:
    _validate_workers(cpu_workers, replications)
    root = Path(project_root)
    script = root / "scripts" / "run_stage3b_cache.py"
    common = (
        str(python_executable),
        str(script),
        "--stage",
        "stage3b_confirmation",
        "--seed-namespace",
        "stage3b_confirmation",
        "--cache-root",
        str(cache_root),
        "--replications",
        str(replications),
        "--selected-models",
        str(selected_models),
    )
    commands = [
        WorkerCommand(
            "gpu_stage3b_cache",
            (
                *common,
                "--learner-targets",
                "l:tabiclv2_1",
                "m:tabiclv2_1",
            ),
        )
    ]
    cpu_targets = (
        "l:xgboost",
        "m:xgboost",
        "m:xgboost_tuned",
        "m:extra_trees",
        "l:oracle",
        "m:oracle",
    )
    for index in range(cpu_workers):
        commands.append(
            WorkerCommand(
                f"cpu_stage3b_cache_{index:02d}",
                (
                    *common,
                    "--learner-targets",
                    *cpu_targets,
                    "--num-shards",
                    str(cpu_workers),
                    "--shard-index",
                    str(index),
                ),
            )
        )
    return tuple(commands)
