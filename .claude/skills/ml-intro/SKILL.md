---
name: ml-intro
description: Create an introductory Random Forest ML workflow (script + notebook) predicting Bolivia's Municipal Sustainable Development Index from satellite embeddings. Saves figures to images/ and tables to tables/. Use this skill when the user wants to set up a machine learning tutorial, predict development outcomes from remote sensing data, or create an educational RF regression example with the DS4Bolivia dataset.
argument-hint: "[optional: target variable name, default imds]"
disable-model-invocation: true
user-invocable: true
---

# ML Introduction: Random Forest with Satellite Embeddings

Create an educational machine-learning workflow predicting Bolivia's Municipal
Sustainable Development Index (IMDS) from satellite image embeddings using
Random Forest regression. The workflow teaches ML best practices through a
concrete regional development application.

## Deliverables

| Output | Path |
|--------|------|
| Python script | `code/ml_intro_rf.py` |
| Jupyter notebook | `notebooks/notebook-NN.ipynb` |
| Jupytext pair | `notebooks/notebook-NN.md` |
| Figures (7) | `images/ml_*.png` |
| Tables (2) | `tables/ml_*.csv` |

The target variable defaults to `imds` but can be overridden via `$ARGUMENTS`.

---

## Pre-flight

1. Read `config.py`, `_quarto.yml`, `jupytext.toml`, `pyproject.toml`
2. Determine the next available notebook number from `_quarto.yml` (don't hardcode — count existing entries and increment)
3. Check that the target notebook file doesn't already exist; if it does, ask the user how to proceed
4. If `scikit-learn` is missing from `pyproject.toml`, run `uv add scikit-learn`
5. If `scipy` is missing from `pyproject.toml`, run `uv add scipy` (needed for `RandomizedSearchCV` parameter distributions)

---

## Data Source: DS4Bolivia

All data comes from the [DS4Bolivia](https://github.com/quarcs-lab/ds4bolivia) repository.

**Base URL:** `https://raw.githubusercontent.com/quarcs-lab/ds4bolivia/master`

| Dataset | Path | Key columns |
|---------|------|-------------|
| SDG indices | `/sdg/sdg.csv` | `asdf_id`, `imds`, `sdg1`–`sdg15` |
| Satellite embeddings | `/satelliteEmbeddings/satelliteEmbeddings2017.csv` | `asdf_id`, `A00`–`A63` |
| Region names | `/regionNames/regionNames.csv` | `asdf_id`, municipality/department names |

- **Join key:** `asdf_id` (integer, shared across all datasets)
- **Target (y):** `imds` (Municipal Sustainable Development Index, 0–100 scale)
- **Features (X):** `A00` through `A63` (64 satellite embedding dimensions from 2017)
- **Observations:** 339 Bolivian municipalities

**Local caching:** After first download, save merged data to `data/rawData/ds4bolivia_merged.csv`. On subsequent runs, load from cache if file exists. This avoids repeated network requests and enables offline runs.

---

## Step 1: Create `code/ml_intro_rf.py`

A self-contained script runnable via `uv run python code/ml_intro_rf.py` from project root.

### Structure

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import set_seeds, RANDOM_SEED, IMAGES_DIR, TABLES_DIR, DATA_DIR
set_seeds()
```

Then import numpy, pandas, matplotlib, seaborn, scikit-learn modules, `scipy.stats.randint`,
and `IPython.display.Markdown` (for the notebook).

Define two configuration variables near the top of the script:
- `TARGET` — the column name (default `"imds"`, or `$ARGUMENTS` if provided)
- `TARGET_LABEL` — a human-readable label (e.g., `"IMDS"` or `"SDG 1 (No Poverty)"`)

Use `TARGET_LABEL` in all plot titles and axis labels so they adapt automatically
when the target variable changes.

### Workflow sections

The script should implement each section below, with comments explaining the
ML concepts (especially *why* each step matters — not just *what* it does).

**1. Data loading** — Fetch CSVs from the URLs above using `pd.read_csv()`,
merge on `asdf_id`, cache locally. Extract `X` (columns `A00`–`A63`) and
`y` (column `imds`). Drop rows with missing values and report count.

**2. EDA** — Two figures saved to `images/`:
- `ml_target_distribution.png` — histogram of the target variable
- `ml_embedding_correlations.png` — heatmap of top-10 correlated embeddings with target

Use `plt.savefig(..., dpi=300, bbox_inches="tight")` for all figures.

**3. Train/test split** — 80/20 with `random_state=RANDOM_SEED`. Comment
explaining that splitting happens *before* any preprocessing to prevent data leakage.

**4. Baseline model** — `RandomForestRegressor(n_estimators=100)` with 5-fold
cross-validation on training set. Print mean and std of CV R² scores.

**5. Hyperparameter tuning** — `RandomizedSearchCV` with `n_iter=50`:
```python
from scipy.stats import randint
param_distributions = {
    "n_estimators": [100, 200, 300, 500],
    "max_depth": [None, 10, 20, 30],
    "min_samples_split": randint(2, 11),
    "min_samples_leaf": randint(1, 5),
    "max_features": ["sqrt", "log2", None],
}
```

**6. Evaluation** — Compute R², RMSE, MAE on test set. Two figures:
- `ml_actual_vs_predicted.png` — scatter with 45° reference line
- `ml_residuals.png` — residuals vs predicted values

**7. Feature importance** — Two figures:
- `ml_feature_importance_mdi.png` — top-20 by Mean Decrease in Impurity
- `ml_feature_importance_permutation.png` — top-20 by permutation importance on test set

**8. Partial dependence plots** — Identify top-6 features by permutation importance.
Use `PartialDependenceDisplay.from_estimator()` in a 2×3 grid.
- `ml_partial_dependence.png`

**9. Save results** — Two CSVs to `tables/`:
- `ml_rf_results.csv` — columns: `Metric`, `Baseline`, `Tuned`
- `ml_rf_best_params.csv` — best hyperparameters from search

**10. Print summary** — formatted results to stdout.

---

## Step 2: Create the notebook

Write the Jupytext `.md:myst` file first, then sync to `.ipynb`.

### Frontmatter (use exactly this format)

```yaml
---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.19.1
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---
```

Followed by a title block: `title: "NX: Introduction to Machine Learning — Random Forest Regression"` (where X matches the notebook number).

### Cell conventions

- Code cells use ```` ```{code-cell} ipython3 ````
- First cell: config import + `set_seeds()` (see `notebooks/notebook-01.md` for exact pattern)
- Second cell: all library imports including `from IPython.display import Markdown`.
  Also define `TARGET`, `TARGET_LABEL`, and `FEATURE_COLS` here.
- Figure cells: `#| label: fig-name` + `#| fig-cap: "..."` — save to `images/`
- Table cells: `#| label: tbl-name` + `#| tbl-cap: "..."` — only for `Markdown(df.to_markdown())` output — save to `tables/`
- Do NOT use `tbl-` prefix for non-table output (it crashes Quarto's parser)
- **Single-line paragraphs** — Every markdown paragraph must be written as one continuous line with no hard line breaks. Jupytext preserves `\n` characters inside `.ipynb` cell sources, and VS Code's notebook viewer renders them as literal line breaks, making sentences appear "cut" mid-line. This applies to both conceptual explanations before code cells and interpretation paragraphs after them.

### Notebook sections

Each code cell should be sandwiched between two kinds of markdown:

- **Before the cell** — conceptual explanation: *why* this step matters, what
  the technique is, how it fits into the ML pipeline. Written before execution,
  so it does not reference specific output values.
- **After the cell** — interpretation of results: what the actual output *means*
  for a beginner. These interpretation cells are added after execution (see
  Step 4.5) and should reference specific numbers from the cell output (R²
  values, sample counts, best parameters, etc.), explain what they imply, and
  connect to the Bolivian development context where relevant. For figures,
  describe what patterns to look for in the plot. Keep each interpretation to
  2–4 sentences.

This two-layer structure turns the notebook into a self-contained tutorial
where a reader can follow both the reasoning and the findings.

| Section | Content | Figures/Tables |
|---------|---------|----------------|
| Overview | What IMDS is, what satellite embeddings are, learning objectives | — |
| Data Loading | Load from DS4Bolivia, explain the dataset | — |
| EDA | Why EDA before modeling | `fig-target-distribution`, `fig-embedding-correlations` |
| Train/Test Split | Data leakage concept, 80/20 ratio, reproducibility | — |
| Model Training | RF algorithm: bagging, feature subsampling, ensemble | — |
| Cross-Validation | k-fold CV, why more reliable than single split | — |
| Hyperparameter Tuning | RandomizedSearchCV, what each param controls | — |
| Model Evaluation | R², RMSE, MAE explained | `fig-actual-vs-predicted`, `fig-residuals` |
| Feature Importance | MDI vs permutation importance, reliability | `fig-importance-mdi`, `fig-importance-permutation` |
| Partial Dependence | Marginal effect, non-linear relationships | `fig-partial-dependence` |
| Interpretation | Summary, limitations, next steps | `tbl-ml-results` |

Every figure cell must also `plt.savefig()` to `images/`. The final table cell
must also save to `tables/`.

---

## Step 3: Sync and register

```bash
uv run jupytext --sync notebooks/notebook-NN.md
```

Then edit `_quarto.yml` to add the notebook under `manuscript.notebooks`:

```yaml
- notebook: notebooks/notebook-NN.ipynb
  title: "NX: Introduction to Machine Learning — Random Forest Regression"
```

---

## Step 4: Execute

```bash
uv run jupyter execute --inplace notebooks/notebook-NN.ipynb
```

The `--inplace` flag is required — without it, outputs are discarded.

After execution, re-sync to capture outputs in the `.md` pair:

```bash
uv run jupytext --sync notebooks/notebook-NN.ipynb
```

---

## Step 4.5: Interpret results — THIS IS THE MOST IMPORTANT STEP

The interpretation cells are what transform this notebook from a code demo into
a genuine tutorial. Without them, a beginner sees numbers and plots but has no
idea what they mean. Every code cell that produces output needs a markdown cell
immediately after it that explains the result in plain language.

### How to do it

1. **Read the executed `.ipynb`** — open the notebook file and look at each
   cell's printed output (metrics, counts, parameter values). Write down the
   key numbers.
2. **Edit the `.md` file** — after each code cell that produces output, insert
   a new markdown paragraph (not a new section heading) that interprets the
   result. Use the actual numbers from the output. Write each paragraph as a
   single continuous line (no soft wraps) — see the cell conventions rule above.
3. **Re-sync** — run `uv run jupytext --sync notebooks/notebook-NN.md` to
   propagate the interpretation cells back to the `.ipynb`.

### What good interpretation looks like

Here is a concrete example. Suppose the evaluation cell prints:
```
Tuned Test R²:   0.4139
Tuned Test RMSE: 17.31
Tuned Test MAE:  13.07
```

A good interpretation cell inserted after it would be:

```
The tuned model explains about 41% of the variation in IMDS scores across Bolivia's 339 municipalities (R² = 0.41). An RMSE of 17.3 means the model's predictions are typically off by about 17 index points on the 0–100 scale — meaningful but far from precise. The MAE of 13.1 confirms this: on average, a prediction misses by 13 points. This tells us satellite imagery captures real poverty-related patterns, but many factors that drive municipal development (governance, infrastructure, migration) are invisible from space.
```

Notice: it quotes exact numbers, explains what they mean for a non-expert, and
connects to the Bolivian development context. This is the standard for every
interpretation cell.

### Cells that need interpretation

Every code cell that prints output or shows a figure needs an interpretation
cell immediately after it. At minimum, cover these:

| After cell | What to explain |
|------------|-----------------|
| Data loading | How many municipalities loaded, any dropped rows, national coverage |
| Target histogram | Distribution shape, mean vs median, what skewness means for inequality |
| Correlation heatmap | Which embeddings correlate most, what that suggests physically |
| Train/test split | Sample sizes (e.g., "271 train, 68 test"), whether adequate |
| Baseline evaluation | R² in plain language ("explains X% of variation"), RMSE/MAE in target units |
| CV scores | Mean and std dev, what variability tells us about dataset size |
| Tuning results | Best parameters found, how much improvement over baseline |
| Final evaluation | All three metrics with interpretation, overall model quality |
| Actual vs predicted | How tight the scatter is, whether extremes are compressed |
| Residuals | Whether random (good) or patterned (problematic) |
| MDI importance | Top features, what embedding dimensions might represent |
| Permutation importance | Comparison with MDI, which is more reliable |
| Partial dependence | Non-linear effects, threshold patterns, what they suggest |
| Results table | Baseline vs tuned summary, overall takeaway for the research question |

### Verification

Before finishing, count the interpretation cells in the `.md` file. There
should be at least 10 interpretation paragraphs that reference specific numbers
from the executed output. If there are fewer, go back and add more.

After adding all interpretation cells, re-sync to update the `.ipynb`:

```bash
uv run jupytext --sync notebooks/notebook-NN.md
```

---

## Step 5: Verify and report

Check all outputs exist:
- 7 figures in `images/ml_*.png`
- 2 tables in `tables/ml_*.csv`
- Both notebook files (`*.ipynb` + `*.md`)
- Notebook registered in `_quarto.yml`
- At least 10 interpretation paragraphs in the `.md` file that reference
  specific numeric values from the executed output (R², RMSE, sample counts,
  best parameters, etc.). If fewer than 10 exist, go back to Step 4.5.

Run the standalone script to confirm it works independently:

```bash
uv run python code/ml_intro_rf.py
```

Report to user what was created, and remind them to:
1. Review notebook outputs
2. Embed figures in `index.qmd` via `{{< embed >}}` shortcodes
3. Run `bash scripts/render.sh` to rebuild the manuscript
4. Write a handoff via `/project:handoff`
