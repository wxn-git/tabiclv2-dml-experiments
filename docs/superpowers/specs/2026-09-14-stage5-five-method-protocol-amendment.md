# Stage 5 Five-Method Protocol Amendment

## Decision

Stage 5 permanently removes `ensemble`. The confirmatory comparison contains exactly:

1. `tabiclv2_1`
2. `tabiclv2_8`
3. `xgboost_tuned`
4. `extra_trees`
5. `lasso`

Stage 1--4 configurations, code, and results remain unchanged. MLP is not added in this amendment; it may be studied later as a separately designed supplementary baseline.

## DGP and random-seed relationship to Stage 2

The `linear`, `smooth`, and original `tree` panels continue to use the exact
same `simulate_plr` formulas as Stage 2. The three additional tree panels have
no Stage 2 counterpart. Stage 5 intentionally uses its own profile-specific
seed namespace, so overlapping Stage 2 and Stage 5 cells are independent draws
from the same DGP rather than the same realized sample. Within every Stage 5
cell and replication, all methods share identical data and fold seeds.

Seeds must not be changed after inspecting preflight curves. The five-replicate
preflight is a correctness and runtime gate only; non-monotone finite-sample
curves are not publication evidence and are reassessed with 100 formal
replications.

## Scientific rationale

The paper asks whether TabICLv2 improves PLR-DML nuisance estimation and treatment-effect inference relative to conventional learners. Tuned XGBoost is the principal nonlinear tree baseline, Extra Trees is a randomized tree baseline, and Lasso is the linear baseline. These methods provide the required comparison without the current ensemble's nested DML cross-fitting, ensemble OOF cross-validation, and internal model searches. The ensemble is computationally disproportionate and is not required to answer the primary question.

## Exact experiment universe

The six DGPs and five unique `(n,p)` cells per DGP are unchanged. The two sensitivity sweeps remain:

- fixed `n=1000`, vary `p in [10,50,100]`;
- fixed `p=50`, vary `n in [500,1000,2000]`;
- the center `(1000,50)` is stored and counted once.

With five methods, exact DML result counts are:

- smoke: `6 * 5 * 5 * 1 = 150`;
- preflight: `6 * 5 * 5 * 5 = 750`;
- formal: `6 * 5 * 5 * 100 = 15,000`.

Each DML result uses two nuisance predictions, so the exact nuisance-cache counts are 300, 1,500, and 30,000 respectively. XGBoost tuning remains 720 full-profile candidate records and freezes 12 DGP-target winners.

## Orchestration changes

The Stage 5 controller launches one unsharded GPU worker for both TabICLv2 methods and 5--8 CPU shards for Tuned XGBoost, Extra Trees, and Lasso. There is no ensemble batch. Composition starts only after the exact five-method cache universe validates. Progress, analysis, figures, and the formal gate use the new exact counts.

The controller and analyzer must reject stale six-method outputs, unexpected ensemble results, duplicate keys, non-finite values, failed/OOM/fallback records, and provenance mismatches. The formal profile remains blocked until the user explicitly approves it after a clean 750-result preflight and runtime/cost review.

## Reuse of completed preflight computations

The interrupted six-method preflight produced all 1,500 non-ensemble nuisance predictions plus a small number of ensemble artifacts. The predictions are reusable because the DGP definitions, profile namespace, data seeds, fold seeds, learner identities, target-specific XGBoost winners, estimator settings, and cross-fitting procedure are unchanged.

They cannot be silently reused because the old metadata includes the six-method configuration fingerprint. A one-time migration command must therefore:

1. load and validate the old six-method config and frozen tuning artifact;
2. load and validate the new five-method config and frozen tuning artifact;
3. prove that every computation-affecting field is identical except the method universe and associated fingerprints;
4. enumerate the exact 1,500 required non-ensemble tasks in both protocols;
5. validate every old NPZ and metadata sidecar, including prediction length, finite values, fold count, seeds, learner parameters, devices, and timings;
6. write new task keys and sidecars atomically into a separate five-method cache root;
7. never copy ensemble artifacts;
8. emit an immutable JSON migration manifest containing source/destination fingerprints, exact counts, per-method counts, and hashes of source and destination prediction arrays;
9. fail closed and leave the source cache unchanged on any discrepancy.

The migrated cache is scientifically equivalent to recomputation because method-list membership does not enter the DGP, folds, learner seeds, model parameters, or predictions. The manifest makes this reuse explicit and auditable.

## Outputs and reporting

Five-method artifacts use separate roots such as:

```text
results/stage5_five/preflight/cache
results/stage5_five/preflight/raw
results/stage5_five/preflight/analysis
```

Primary treatment-effect MSE and coverage figures each retain separate fixed-`n`
and fixed-`p` six-panel layouts. Coverage figures mark the nominal 0.95 level.
Legends contain exactly five methods. Supplementary Bias, Coverage, `l`-MSE,
and `m`-MSE figures also contain exactly five methods. Documentation must state
that the prior six-method smoke was an implementation check and is not part of
the final confirmatory protocol.

## Verification and stopping rule

Before formal execution:

- focused and full test suites pass;
- the migration manifest reports exactly 1,500 validated caches and zero violations;
- composition reports exactly 750 successful DML records;
- analysis confirms paired data/fold seeds and center deduplication;
- all figures and exact plotting CSVs are generated;
- the measured preflight runtime is converted into a hardware-dependent formal estimate;
- the user explicitly approves the 15,000-result formal launch.

No formal run is authorized by this amendment.
