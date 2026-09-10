# Exploratory Treatment-effect MSE Dimension Figure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible 2 x 3 exploratory figure of treatment-effect MSE against feature dimension `p` from the existing Stage 2 and Stage 4 screening summaries.

**Architecture:** Add data-selection and plotting functions to the existing `tabdml.figures` module, with a small command-line wrapper under `scripts/`. The preparation function produces a tidy plotting table, and the plotting function consumes only that table so selection logic and rendering can be tested independently.

**Tech Stack:** Python 3.11+, pandas, matplotlib, pytest.

## Global Constraints

- Use only existing Stage 2 100-replication summaries and Stage 4 20-replication screening summaries.
- Plot `rmse ** 2` as treatment-effect MSE on a logarithmic vertical axis.
- Never connect points with different DGPs or sample sizes.
- Do not draw an uncertainty ribbon from unavailable aggregate uncertainty information.
- Produce a high-resolution PNG, vector PDF, and compact plotted-data CSV.

---

### Task 1: Select and validate the plotted data

**Files:**
- Modify: `src/tabdml/figures.py`
- Test: `tests/test_figures.py`

**Interfaces:**
- Consumes: Stage 2 and Stage 4 summary `pandas.DataFrame` objects.
- Produces: `prepare_treatment_effect_p_trend_data(stage2, stage4) -> pandas.DataFrame`.

- [ ] **Step 1: Write a failing test**

Add a synthetic-data test that verifies exact Stage 2 and Stage 4 filters, method labels, panel order, fixed sample sizes, replication labels, and `treatment_effect_mse == rmse ** 2`. Include irrelevant rows and assert that they are excluded.

- [ ] **Step 2: Run the test and verify the expected failure**

Run: `python -m pytest tests/test_figures.py -q`

Expected: failure because `prepare_treatment_effect_p_trend_data` does not exist.

- [ ] **Step 3: Implement the preparation function**

Add immutable panel specifications for Stage 2 Linear/Smooth/Original tree and Stage 4 standard Tree stumps/Tree hierarchical/Tree forest-sum. Filter shared methods plus Stage 4-only tuned-XGBoost, coerce numeric fields, reject duplicate method-panel-`p` rows, validate that each panel has one sample size, calculate `rmse ** 2`, and return columns needed by the figure and exported CSV.

- [ ] **Step 4: Run the focused test**

Run: `python -m pytest tests/test_figures.py -q`

Expected: all tests in the file pass.

- [ ] **Step 5: Commit Task 1**

Commit `src/tabdml/figures.py` and `tests/test_figures.py` with message `Add exploratory dimension-plot data preparation`.

### Task 2: Render and export the figure

**Files:**
- Modify: `src/tabdml/figures.py`
- Modify: `tests/test_figures.py`

**Interfaces:**
- Consumes: the tidy frame returned by `prepare_treatment_effect_p_trend_data`.
- Produces: `make_treatment_effect_p_trend_figure(plot_data, output_dir) -> dict[str, pathlib.Path]` with `png`, `pdf`, and `csv` paths.

- [ ] **Step 1: Write a failing output test**

Build a complete synthetic plotting table, call the wished-for rendering function, and assert that all three files exist, the CSV preserves all plotted rows, and every MSE is positive.

- [ ] **Step 2: Run the test and verify the expected failure**

Run: `python -m pytest tests/test_figures.py -q`

Expected: failure because `make_treatment_effect_p_trend_figure` does not exist.

- [ ] **Step 3: Implement minimal rendering**

Create a 2 x 3 matplotlib figure with fixed Okabe-Ito-inspired colors, distinct markers and line styles, per-panel numeric `p` ticks, logarithmic MSE axes, English titles containing stage, fixed `n`, and replication count, and one figure-level legend. Save `treatment_effect_mse_by_p_exploratory.png` at 300 dpi, a same-name PDF, and `treatment_effect_mse_by_p_exploratory_data.csv`.

- [ ] **Step 4: Run the focused test**

Run: `python -m pytest tests/test_figures.py -q`

Expected: all tests in the file pass with no failures.

- [ ] **Step 5: Commit Task 2**

Commit the figure implementation and test with message `Render exploratory treatment-effect dimension figure`.

### Task 3: Add the reproducible command and publish outputs

**Files:**
- Create: `scripts/make_exploratory_p_mse_figure.py`
- Modify: `tests/test_figures.py`
- Create when run: `results/published/figures/treatment_effect_mse_by_p_exploratory.png`
- Create when run: `results/published/figures/treatment_effect_mse_by_p_exploratory.pdf`
- Create when run: `results/published/figures/treatment_effect_mse_by_p_exploratory_data.csv`

**Interfaces:**
- Consumes: optional CLI source paths and output directory, with published-result paths as defaults.
- Produces: the three requested artifacts and prints their paths.

- [ ] **Step 1: Write a failing command test**

Add a subprocess test invoking the script against temporary synthetic Stage 2 and Stage 4 files and assert exit code zero and three output artifacts.

- [ ] **Step 2: Run the command test and verify the expected failure**

Run: `python -m pytest tests/test_figures.py -q`

Expected: failure because the command-line script does not exist.

- [ ] **Step 3: Implement the command wrapper**

Parse `--stage2-summary`, `--stage4-screening`, and `--output-dir`; load the CSV files; call the two tested figure functions; and print each generated artifact path.

- [ ] **Step 4: Run focused and full verification**

Run: `python -m pytest tests/test_figures.py -q`

Run: `python scripts/make_exploratory_p_mse_figure.py`

Run: `python -m pytest -q`

Expected: tests pass, the command exits zero, and all three published artifacts exist.

- [ ] **Step 5: Inspect the generated data and image**

Verify 39 plotted rows, no duplicate panel-method-`p` keys, one sample size per panel, positive MSE values, and visually inspect the PNG for clipping, overlap, and misleading line connections.

- [ ] **Step 6: Commit Task 3**

Commit the script and generated published artifacts with message `Publish exploratory treatment-effect dimension figure`.
