# Treatment-effect MSE versus feature dimension: exploratory figure

## Purpose

Create a descriptive figure showing how treatment-effect estimation error changes with feature dimension `p`, using existing Stage 2 and Stage 4 results only. The figure is exploratory and does not claim that the two stages form one unified controlled benchmark.

## Data

- Stage 2: `results/published/stage2/summary_stage2.csv`, using 100-replication results.
- Stage 4: `results/published/stage4_tree_benchmark/screening_summary.csv`, using 20-replication screening results.
- Plot `rmse ** 2` as treatment-effect MSE.

## Panels

Use a 2 x 3 layout:

1. Linear, Stage 2, `n = 2000` (single point at `p = 50`).
2. Smooth, Stage 2, `n = 1000` (`p = 50, 100`).
3. Original tree, Stage 2, `n = 5000` (`p = 10, 50`).
4. Tree stumps, Stage 4 standard panel, `n = 1000` (`p = 10, 50`).
5. Tree hierarchical, Stage 4 standard panel, `n = 1000` (`p = 10, 50`).
6. Tree forest-sum, Stage 4 standard panel, `n = 1000` (`p = 10, 50`).

Each title states the stage, fixed sample size, and replication count so readers can see the non-uniform design.

## Methods

The shared main methods are:

- TabICLv2-1
- TabICLv2-8
- XGBoost (default)

Stage 4 tuned-XGBoost is shown as an additional Stage 4-only dashed series and is clearly distinguished from default XGBoost. Stage-specific methods are omitted from this exploratory cross-stage figure to avoid changing the meaning of a line between panels.

## Visual encoding

- Horizontal axis: feature dimension `p`, displayed as numeric ticks.
- Vertical axis: treatment-effect MSE on a logarithmic scale; tick labels remain in original MSE units.
- Fixed color, marker, and line style for each method across panels.
- Points show observed cells; lines connect cells only within the same DGP and fixed `n`.
- No uncertainty ribbon, because the published summary files do not contain a directly valid confidence interval for MSE and the stages use different replication counts.
- A single shared legend is placed outside the panels.

## Outputs

- High-resolution PNG for inspection and Word/slide use.
- Vector PDF for paper preparation.
- A compact CSV containing exactly the plotted rows and the derived treatment-effect MSE.

## Interpretation limits

- Do not compare slopes between panels as if sample size and replication count were identical.
- Do not infer a trend from the Stage 2 linear panel, which has only one point.
- Two-point lines are descriptive contrasts, not evidence of linear or monotone dependence on `p`.
- Stage 4 screening curves are exploratory; the 100-replication confirmation results contain only one selected cell per panel and structure and therefore cannot supply a dimension curve.

## Verification

- Check every selected row against its source CSV.
- Confirm `treatment_effect_mse = rmse ** 2` for every plotted point.
- Confirm no line combines different sample sizes.
- Render and visually inspect both output formats for clipped labels, overlapping legends, and misleading axis scaling.
