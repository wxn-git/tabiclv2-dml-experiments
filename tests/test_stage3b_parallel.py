from pathlib import Path

from tabdml.stage3b_parallel import (
    build_stage3b_batch_a_commands,
    build_stage3b_confirmation_commands,
    build_stage3b_screening_commands,
)


def test_confirmation_uses_one_gpu_and_eight_cpu_workers():
    commands = build_stage3b_confirmation_commands(
        python_executable="python",
        project_root=Path("project"),
        cache_root=Path("cache"),
        selected_models=Path("selected.json"),
        cpu_workers=8,
        replications=5,
    )

    assert commands[0].name == "gpu_stage3b_cache"
    assert len(commands) == 9
    assert "l:tabiclv2_1" in commands[0].argv
    assert "m:tabiclv2_1" in commands[0].argv
    for command in commands[1:]:
        assert "--num-shards" in command.argv
        assert "m:xgboost_tuned" in command.argv
        assert "m:extra_trees" in command.argv


def test_confirmation_rejects_invalid_worker_count():
    try:
        build_stage3b_confirmation_commands(
            "python", Path("project"), Path("cache"), Path("selected.json"), 0, 5
        )
    except ValueError as error:
        assert "cpu_workers" in str(error)
    else:
        raise AssertionError("zero workers should fail")


def test_batch_a_and_screening_keep_tab_on_one_gpu_worker():
    batch_a = build_stage3b_batch_a_commands(
        "python", Path("project"), Path("cache"), 8, 5
    )
    screening = build_stage3b_screening_commands(
        "python", Path("project"), Path("raw"), 8, 10
    )

    assert len(batch_a) == 9
    assert len(screening) == 9
    assert "tabiclv2_1" in batch_a[0].argv
    assert "tabiclv2_8" in screening[0].argv
    assert all("tabiclv2_1" not in command.argv for command in screening[1:])
