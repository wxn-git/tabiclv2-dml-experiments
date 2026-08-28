from __future__ import annotations

from .config import derive_seed


def validate_shard(num_shards: int, shard_index: int) -> None:
    if num_shards < 1:
        raise ValueError("num_shards must be at least 1")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError("shard_index must be in [0, num_shards)")


def belongs_to_shard(task_key: str, num_shards: int, shard_index: int) -> bool:
    validate_shard(num_shards, shard_index)
    return derive_seed("stage1-shard-v1", task_key) % num_shards == shard_index


def replication_belongs_to_shard(
    replication: int,
    num_shards: int,
    shard_index: int,
) -> bool:
    validate_shard(num_shards, shard_index)
    return replication % num_shards == shard_index
