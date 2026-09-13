# Stage 5 Sample-size and Dimension Sensitivity Benchmark

## Objective

Add a publication-oriented sensitivity benchmark that separates two questions:

1. At fixed sample size, how does treatment-effect estimation change as feature dimension increases?
2. At fixed feature dimension, how does treatment-effect estimation change as sample size increases?

Stage 5 is a new experiment with an independent seed namespace. It does not append replications to Stage 2 or Stage 4 and does not replace their conclusions.

## Data-generating processes

The benchmark uses six existing PLR DGP implementations without changing their formulas:

- `linear`
- `smooth`
- `tree` (the original complex tree design)
- `tree_stumps`
- `tree_hierarchical`
- `tree_forest_sum`

All designs use `theta0 = 1`, five-fold cross-fitting, standard-normal covariates, unit-scaled structural functions, and unit-variance treatment and outcome noise as implemented by `simulate_plr`.

## Experimental grid

### Dimension sweep

- Fixed sample size: `n = 1000`
- Feature dimensions: `p = [10, 50, 100]`

### Sample-size sweep

- Fixed feature dimension: `p = 50`
- Sample sizes: `n = [500, 1000, 2000]`

The center cell `(n=1000, p=50)` belongs to both sweeps but is generated and stored once. Each DGP therefore has five unique cells, for 30 unique DGP cells overall.

## Learners

The full model set contains six nuisance-learning strategies:

- `tabiclv2_1`
- `tabiclv2_8`
- `xgboost_tuned`
- `extra_trees`
- `lasso`
- `ensemble`

`ensemble` retains the existing convex out-of-fold ensemble of Lasso, random forest, XGBoost, and MLP. It is not replaced by a different accelerated approximation.

## XGBoost tuning and freezing

For each DGP, tune XGBoost separately for the `l` and `m` targets at the center cell `(n=1000, p=50)` using 10 independent tuning replications and the frozen Stage 4 candidate grid. Select the candidate with the lowest mean validation MSE, record deterministic tie-breaking, and freeze the two selected configurations for every cell of that DGP.

The tuning seed namespace must be distinct from smoke, preflight, and formal experiment namespaces. Final treatment-effect results are never used to select XGBoost hyperparameters.

## Replications and phases

Run three isolated phases:

1. Smoke: one replication per cell and method to validate interfaces, output schemas, GPU routing, and resume behavior.
2. Preflight: five independent replications per cell and method using full learner settings to detect failures, OOM, silent fallback, or implausible estimates.
3. Formal: 100 paired Monte Carlo replications per cell and method under a new formal seed namespace.

The formal workload is:

`30 cells x 6 methods x 100 replications = 18,000 DML results`.

Every method at a given DGP cell and replication shares the same data seed and cross-fitting seed. Results are written incrementally and successful existing records are skipped on resume.

## Device routing and parallelism

- One unsharded GPU worker runs `tabiclv2_1` and `tabiclv2_8` sequentially on the available NVIDIA GPU.
- Five to eight CPU workers run `xgboost_tuned`, `extra_trees`, and `lasso` in independent shards.
- `ensemble` runs as a separate CPU batch with two to four workers because its nested cross-validation and four base learners make it substantially more expensive.
- GPU XGBoost is not used. Moving every learner to GPU would require replacing the scikit-learn estimators and would change the benchmark methods.

Statistical accuracy remains comparable across CPU and GPU learners because all methods receive the same data and folds. Runtime comparisons are labeled hardware- and implementation-dependent.

## Stored results

Each formal record stores at least:

- DGP, `n`, `p`, method, replication, and seed provenance
- `theta_hat`, reported standard error, confidence interval, and coverage indicator
- squared treatment-effect error `(theta_hat - theta0)^2`
- `l` MSE and `m` MSE
- nuisance error product and cross term when available
- fit time, total time, peak GPU memory observation, fallback state, and status
- frozen learner configuration identifiers and tuning provenance

Raw results, nuisance caches, worker logs, progress state, summaries, and published artifacts use separate Stage 5 directories.

## Analysis

For each DGP-cell-method combination report:

- Bias
- RMSE and treatment-effect MSE
- empirical standard deviation
- mean reported standard error
- 95% coverage and interval width
- mean `l` MSE and mean `m` MSE
- mean runtime
- success, failure, OOM, missing, and fallback counts

Estimate a 95% interval for treatment-effect MSE by bootstrapping the 100 replication-level squared errors. Paired method comparisons use matched replication identifiers. The plots are treated as sensitivity analyses; they do not override the frozen Stage 4 superiority rule.

## Figures

Create two 2 x 3 publication figures with identical method colors and markers:

1. Fixed `n=1000`: horizontal axis `p = 10, 50, 100`.
2. Fixed `p=50`: horizontal axis `n = 500, 1000, 2000`.

Each panel represents one DGP. The vertical axis shows treatment-effect MSE on a logarithmic scale with original-unit tick labels. Points show mean MSE and translucent ribbons show bootstrap 95% intervals. The center-cell value must be identical in both plotting datasets.

Also export the exact plotting data to CSV and generate supplementary figures for Bias, Coverage, `l` MSE, and `m` MSE.

## Gates

No formal phase starts until:

- all unit and integration tests pass;
- all 180 smoke tasks complete successfully;
- all 900 preflight tasks complete successfully;
- no OOM, missing result, duplicate key, silent fallback, non-finite estimate, or provenance mismatch remains;
- the center-cell deduplication and paired-seed checks pass;
- the user explicitly approves the expensive 18,000-result formal launch after seeing the preflight runtime estimate.

## Interpretation limits

- Curves compare sensitivity within a DGP; vertical differences between DGP panels also reflect different structural functions.
- A two-method ordering at one point is not a universal superiority claim.
- MSE intervals quantify Monte Carlo uncertainty, not uncertainty for a single real-data causal estimate.
- The full ensemble is retained for continuity with Stage 2, but its runtime is not directly comparable to GPU TabICLv2.
