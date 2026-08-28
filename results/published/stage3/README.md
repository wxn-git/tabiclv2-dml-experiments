# Stage 3: tree-DGP diagnosis

Stage 3 was designed to locate the source of severe confidence-interval undercoverage in the tree-structured DGP. It separately replaced the outcome nuisance function `l(X) = E[Y | X]` and treatment nuisance function `m(X) = E[D | X]` with oracle, XGBoost, and TabICLv2 learners.

The initial diagnosis showed that treatment-effect estimates remained close to the truth when `m(X)` was oracle, even when `l(X)` was learned. Estimates deteriorated when `m(X)` was learned, including cases where `l(X)` was oracle. This identified treatment-nuisance estimation as the main failure channel to investigate.

Stage 3 is a diagnostic bridge rather than the final publication result. Stage 3B reproduced the diagnosis, screened alternative treatment learners, and confirmed it with 50 Monte Carlo replications. The compact Stage 3B evidence is stored in [`../stage3b/`](../stage3b/). The 450 Stage 3 raw JSON files are included in the `v0.1-stage3b` GitHub Release described by [`ARCHIVE_MANIFEST.md`](../../../ARCHIVE_MANIFEST.md).
