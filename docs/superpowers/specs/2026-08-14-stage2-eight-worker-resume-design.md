# Stage 2 Eight-Worker Resume Design

## Goal

Finish the 16 missing `ensemble` replications for the `tree, n=5000, p=50`
configuration by running eight CPU workers, with two missing replications assigned
to each worker. Existing successful JSON results must never be recomputed.

## Design

Stage 2 will gain deterministic replication sharding. With eight shards,
replication `r` belongs to shard `r % 8`; therefore `r084` through `r099` are
distributed exactly two per worker. Every worker scans the same 100-replication
configuration, ignores replications outside its shard, and relies on the existing
atomic result store to skip successful files already on disk.

A dedicated resume supervisor will launch only eight `ensemble` workers against
`configs/stage2_parallel/ensemble_05.yaml`. It will not start completed learners,
other configurations, or GPU work. Logs and process state will be written to a
separate resume log directory so old diagnostics remain intact.

The launcher will set `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`,
`OPENBLAS_NUM_THREADS=1`, and `NUMEXPR_NUM_THREADS=1`. This prevents each Python
process from creating its own large thread pool and oversubscribing the 16
physical CPU cores.

## Recovery and Safety

- Each result remains an independent JSON file written atomically.
- Restarting the same supervisor skips every successful result and continues only
  missing work.
- A failed worker does not invalidate results completed by the other workers.
- No experiment seeds, model hyperparameters, folds, or estimands are changed.

## Verification

Automated tests will verify that eight shards are disjoint and complete, and that
`r084` through `r099` are assigned exactly two per worker. Before handoff, the
scheduled task must be `Running`, eight worker processes must be alive, and logs
must show existing replications being skipped before missing work begins.
