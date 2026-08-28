import pytest

from tabdml.sharding import (
    belongs_to_shard,
    replication_belongs_to_shard,
    validate_shard,
)


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


def test_replications_84_to_99_are_evenly_partitioned_across_eight_shards():
    replications = range(84, 100)
    ownership = {
        replication: [
            shard
            for shard in range(8)
            if replication_belongs_to_shard(replication, 8, shard)
        ]
        for replication in replications
    }

    assert all(len(shards) == 1 for shards in ownership.values())
    assert {
        shard: sum(shards == [shard] for shards in ownership.values())
        for shard in range(8)
    } == {shard: 2 for shard in range(8)}
