# Curated experiment results

This directory contains compact, publication-ready copies of completed experiment outputs. The source files under `results/` remain unchanged; only selected summaries, reports, figures, and environment records are copied here for Git versioning.

## Layout

- `stage1/`: screening summaries used to select configurations for formal evaluation.
- `stage2/`: formal repeated-experiment summaries, paired comparisons, diagnostics, report, and figure.
- `stage3/`: documentation of the tree-DGP nuisance-function diagnosis and its link to Stage 3B.
- `stage3b/`: publication diagnostics, learner screening, oracle confirmation, and selected-model metadata.
- `figures/`: cross-stage figures suitable for reports and repository previews.
- `environment/`: software, hardware, and DoubleML validation records.

The complete raw JSON outputs and required nuisance caches are stored separately in the `v0.1-stage3b` GitHub Release. See [`ARCHIVE_MANIFEST.md`](../../ARCHIVE_MANIFEST.md) for contents and integrity checks.
