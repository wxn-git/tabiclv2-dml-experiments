import json
import sys

import pytest

from tabdml.parallel import (
    WorkerCommand,
    build_stage3_worker_commands,
    build_stage2_resume_worker_commands,
    build_stage2_worker_commands,
    build_worker_commands,
    run_workers,
)


def test_parallel_commands_have_one_gpu_worker_and_disjoint_cpu_shards():
    commands = build_worker_commands("python", "run_stage1.py", "results/raw", 3)

    assert [command.name for command in commands] == [
        "gpu_tabiclv2",
        "cpu_00",
        "cpu_01",
        "cpu_02",
    ]
    assert commands[0].argv[-1] == "tabiclv2"
    for index, command in enumerate(commands[1:]):
        assert command.argv[-4:] == (
            "--num-shards",
            "3",
            "--shard-index",
            str(index),
        )


def test_parallel_commands_reject_zero_cpu_workers():
    with pytest.raises(ValueError):
        build_worker_commands("python", "run_stage1.py", "results/raw", 0)


def test_stage2_parallel_commands_partition_cpu_and_gpu_work():
    commands = build_stage2_worker_commands(
        "python",
        "run_stage2.py",
        "results/raw",
        "configs/stage2_selected.yaml",
        "configs/stage2_parallel",
    )

    assert len(commands) == 13
    assert [command.name for command in commands[:6]] == [
        "gpu_tabiclv2",
        "cpu_aux",
        "cpu_rf_a",
        "cpu_rf_b",
        "cpu_xgb_a",
        "cpu_xgb_b",
    ]
    assert commands[0].argv[-4:] == (
        "tabiclv2_1",
        "tabiclv2_8",
        "--output-root",
        "results/raw",
    )
    assert commands[1].argv[-4:] == (
        "lasso",
        "mlp",
        "--output-root",
        "results/raw",
    )
    assert [command.name for command in commands[6:]] == [
        "cpu_ensemble_00",
        "cpu_ensemble_01",
        "cpu_ensemble_02",
        "cpu_ensemble_03",
        "cpu_ensemble_04",
        "cpu_ensemble_05",
        "cpu_ensemble_06",
    ]
    assert "configs/stage2_parallel/ensemble_06.yaml" in commands[-1].argv


def test_stage2_resume_commands_create_eight_unique_ensemble_shards():
    commands = build_stage2_resume_worker_commands(
        "python",
        "run_stage2.py",
        "results/raw",
        "configs/stage2_parallel/ensemble_05.yaml",
        8,
    )

    assert [command.name for command in commands] == [
        f"cpu_ensemble_resume_{index:02d}" for index in range(8)
    ]
    for index, command in enumerate(commands):
        assert command.argv[4:6] == ("--learners", "ensemble")
        assert command.argv[-4:] == (
            "--num-shards",
            "8",
            "--shard-index",
            str(index),
        )


def test_stage2_resume_commands_reject_zero_workers():
    with pytest.raises(ValueError):
        build_stage2_resume_worker_commands(
            "python",
            "run_stage2.py",
            "results/raw",
            "ensemble_05.yaml",
            0,
        )


def test_stage3_commands_keep_tab_pairs_on_one_gpu_worker():
    commands = build_stage3_worker_commands(
        "python",
        "run_stage3.py",
        "results/stage3",
        "configs/stage3.yaml",
        cpu_workers=8,
        replications=5,
    )

    assert len(commands) == 9
    gpu = commands[0]
    assert gpu.name == "gpu_stage3_pairs"
    pair_index = gpu.argv.index("--pair-names")
    assert set(gpu.argv[pair_index + 1 :]) == {
        "tab_tab",
        "tab_xgb",
        "xgb_tab",
        "oracle_tab",
        "tab_oracle",
    }

    for index, command in enumerate(commands[1:]):
        pair_index = command.argv.index("--pair-names")
        shard_index = command.argv.index("--num-shards")
        assert set(command.argv[pair_index + 1 : shard_index]) == {
            "xgb_xgb",
            "oracle_xgb",
            "xgb_oracle",
            "oracle_oracle",
        }
        assert command.argv[-4:] == (
            "--num-shards",
            "8",
            "--shard-index",
            str(index),
        )


def test_run_workers_captures_logs_and_final_state(tmp_path):
    commands = (
        WorkerCommand("first", (sys.executable, "-c", "print('first-ok')")),
        WorkerCommand("second", (sys.executable, "-c", "print('second-ok')")),
    )

    exit_codes = run_workers(commands, cwd=tmp_path, log_dir=tmp_path / "logs")

    assert exit_codes == {"first": 0, "second": 0}
    assert (tmp_path / "logs" / "first.stdout.log").read_text().strip() == "first-ok"
    assert (tmp_path / "logs" / "second.stdout.log").read_text().strip() == "second-ok"
    state = json.loads((tmp_path / "logs" / "state.json").read_text())
    assert state["status"] == "completed"
    assert {worker["exit_code"] for worker in state["workers"]} == {0}
