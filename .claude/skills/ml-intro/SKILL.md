---
name: ml-intro
description: Create an introductory Random Forest ML workflow (script + notebook) predicting Bolivia's Municipal Sustainable Development Index from satellite embeddings. Saves figures to images/ and tables to tables/. Use this skill when the user wants to set up a machine learning tutorial, predict development outcomes from remote sensing data, or create an educational RF regression example with the DS4Bolivia dataset.
argument-hint: "[optional: target variable name, default imds]"
disable-model-invocation: true
user-invocable: true
allowed-tools: Bash, Read, Write, Edit, Glob, Grep, WebFetch
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

Then import numpy, pandas, matplotlib, seaborn, and scikit-learn modules.

### Workflow sections

The script should implement each section below, with comments explaining the
ML concepts (especially *why* each step matters — not just *what* it does).

**1. Data loading** — Fetch CSVs from the URLs above using `pd.read_csv()`,
merge on `asdf_id`, cache locally. Extract `X` (columns `A00`–`A63`) and
`y` (column `imds`). Drop rows with missing values and report count.

**2. EDA** — Two figures saved to `images/`:
- `ml_imds_distribution.png` — histogram of IMDS values
- `ml_embedding_correlations.png` — heatmap of top-10 correlated embeddings with IMDS

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
- Figure cells: `#| label: fig-name` + `#| fig-cap: "..."` — save to `images/`
- Table cells: `#| label: tbl-name` + `#| tbl-cap: "..."` — only for `Markdown(df.to_markdown())` output — save to `tables/`
- Do NOT use `tbl-` prefix for non-table output (it crashes Quarto's parser)

### Notebook sections

Between code cells, include **educational markdown** explaining each ML concept
in the context of regional development. The notebook is a teaching tool — explain
*why* each step matters, not just how to do it.

| Section | Content | Figures/Tables |
|---------|---------|----------------|
| Overview | What IMDS is, what satellite embeddings are, learning objectives | — |
| Data Loading | Load from DS4Bolivia, explain the dataset | — |
| EDA | Why EDA before modeling | `fig-imds-distribution`, `fig-embedding-correlations` |
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

## Step 5: Verify and report

Check all outputs exist:
- 7 figures in `images/ml_*.png`
- 2 tables in `tables/ml_*.csv`
- Both notebook files (`*.ipynb` + `*.md`)
- Notebook registered in `_quarto.yml`

Run the standalone script to confirm it works independently:

```bash
uv run python code/ml_intro_rf.py
```

Report to user what was created, and remind them to:
1. Review notebook outputs
2. Embed figures in `index.qmd` via `{{< embed >}}` shortcodes
3. Run `bash scripts/render.sh` to rebuild the manuscript
4. Write a handoff via `/project:handoff`
