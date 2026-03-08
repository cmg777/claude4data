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

---
title: "N4: Introduction to Machine Learning — Random Forest Regression"
---

## Overview

This notebook introduces machine learning through a concrete application:
predicting Bolivia's **SDG 1 (No Poverty)** index from satellite image
embeddings using Random Forest regression.

**What is SDG 1?** The Sustainable Development Goal 1 measures progress toward
eliminating poverty. For Bolivian municipalities, this index captures the local
poverty situation on a continuous scale, with higher values indicating better
outcomes (less poverty).

**What are satellite embeddings?** Satellite images of each municipality have
been processed through a deep learning model that compresses the visual
information into 64 numerical features (A00--A63). These embeddings capture
land use patterns, urbanization, infrastructure, and vegetation — all of which
correlate with economic development and poverty levels.

**Learning objectives:**

1. Understand the complete ML pipeline: data loading, EDA, splitting, training,
   tuning, and evaluation
2. Learn why each step matters (especially data leakage prevention)
3. Interpret model outputs: metrics, feature importance, partial dependence
4. Connect satellite-derived features to poverty prediction

**Data source:** [DS4Bolivia](https://github.com/quarcs-lab/ds4bolivia) — 339
Bolivian municipalities with SDG indices and 2017 satellite embeddings.

```{code-cell} ipython3
import sys
sys.path.insert(0, "..")
from config import set_seeds, RANDOM_SEED, IMAGES_DIR, TABLES_DIR, DATA_DIR

set_seeds()
```

```{code-cell} ipython3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import (
    cross_val_score,
    RandomizedSearchCV,
    train_test_split,
)
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.inspection import PartialDependenceDisplay, permutation_importance
from scipy.stats import randint
from IPython.display import Markdown

sns.set_theme(style="whitegrid")

# Configuration
TARGET = "sdg1"
TARGET_LABEL = "SDG 1 (No Poverty)"
FEATURE_COLS = [f"A{i:02d}" for i in range(64)]
```

## Data Loading

We load three datasets from the DS4Bolivia repository:

- **SDG indices** — municipal-level sustainable development scores
- **Satellite embeddings** — 64-dimensional vectors derived from 2017 satellite imagery
- **Region names** — municipality and department identifiers

All three are joined on `asdf_id`, a unique identifier for each of Bolivia's 339
municipalities. We cache the merged result locally to avoid repeated downloads.

```{code-cell} ipython3
BASE_URL = "https://raw.githubusercontent.com/quarcs-lab/ds4bolivia/master"
CACHE_PATH = DATA_DIR / "rawData" / "ds4bolivia_merged.csv"

if CACHE_PATH.exists():
    print("Loading cached data from", CACHE_PATH)
    df = pd.read_csv(CACHE_PATH)
else:
    print("Downloading DS4Bolivia datasets ...")
    sdg = pd.read_csv(f"{BASE_URL}/sdg/sdg.csv")
    embeddings = pd.read_csv(
        f"{BASE_URL}/satelliteEmbeddings/satelliteEmbeddings2017.csv"
    )
    names = pd.read_csv(f"{BASE_URL}/regionNames/regionNames.csv")
    df = sdg.merge(embeddings, on="asdf_id").merge(names, on="asdf_id")

    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CACHE_PATH, index=False)
    print(f"Saved merged data to {CACHE_PATH}")

# Drop rows with missing values
n_before = len(df)
df = df.dropna(subset=FEATURE_COLS + [TARGET])
n_dropped = n_before - len(df)
print(f"Dataset: {len(df)} municipalities (dropped {n_dropped} with missing values)")

X = df[FEATURE_COLS].values
y = df[TARGET].values
```

## Exploratory Data Analysis

Before building any model, we need to understand the data. EDA reveals the
distribution of our target variable and helps identify which features might be
most informative. This step can also uncover data quality issues and outliers.

```{code-cell} ipython3
#| label: fig-target-distribution
#| fig-cap: "Distribution of SDG 1 (No Poverty) scores across 339 Bolivian municipalities. The red dashed line marks the mean; orange marks the median."

fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(y, bins=30, edgecolor="white", alpha=0.8)
ax.set_xlabel(TARGET_LABEL)
ax.set_ylabel("Frequency")
ax.set_title(f"Distribution of {TARGET_LABEL} across Bolivian Municipalities")
ax.axvline(y.mean(), color="red", linestyle="--", label=f"Mean = {y.mean():.1f}")
ax.axvline(
    np.median(y), color="orange", linestyle="--", label=f"Median = {np.median(y):.1f}"
)
ax.legend()
plt.tight_layout()
plt.savefig(IMAGES_DIR / "ml_target_distribution.png", dpi=300, bbox_inches="tight")
plt.show()
```

Next we examine which embedding dimensions correlate most strongly with SDG 1.
High correlation does not imply causation, but it highlights which
satellite-derived features capture variation related to poverty outcomes.

```{code-cell} ipython3
#| label: fig-embedding-correlations
#| fig-cap: "Correlation heatmap showing the top-10 satellite embedding dimensions most strongly correlated with SDG 1 (No Poverty)."

corr_with_target = (
    df[FEATURE_COLS + [TARGET]]
    .corr()[TARGET]
    .drop(TARGET)
    .abs()
    .sort_values(ascending=False)
)
top10 = corr_with_target.head(10).index.tolist()
corr_matrix = df[top10 + [TARGET]].corr()

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
ax.set_title(f"Correlation: Top-10 Satellite Embeddings with {TARGET_LABEL}")
plt.tight_layout()
plt.savefig(IMAGES_DIR / "ml_embedding_correlations.png", dpi=300, bbox_inches="tight")
plt.show()
```

## Train/Test Split

We split the data **before** any preprocessing or model fitting. This prevents
**data leakage** — when information from the test set inadvertently influences
the training process, leading to overly optimistic performance estimates that
do not generalize to truly unseen data.

We use an 80/20 split: 80% for training (including cross-validation) and 20%
held out for final evaluation. The `random_state` ensures reproducibility.

```{code-cell} ipython3
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_SEED
)
print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set:     {X_test.shape[0]} samples")
```

## Model Training

We start with a Random Forest using default-ish parameters (100 trees) as our
baseline. This establishes a performance floor that hyperparameter tuning should
improve upon.

### What is a Random Forest?

A Random Forest is an **ensemble** of decision trees. Each tree is trained on a
random subset of the data (**bagging**) and considers only a random subset of
features at each split (**feature subsampling**). The final prediction is the
average of all trees' predictions. This combination of randomness makes the
ensemble more robust and less prone to overfitting than any single tree.

```{code-cell} ipython3
baseline_rf = RandomForestRegressor(
    n_estimators=100, random_state=RANDOM_SEED, n_jobs=-1
)
```

## Cross-Validation

Rather than evaluating on a single train/validation split (which can be noisy,
especially with only ~270 training samples), we use **5-fold cross-validation**.
The training data is partitioned into 5 equal folds. The model trains on 4 folds
and evaluates on the remaining one, repeating 5 times. The mean and standard
deviation of R-squared scores tell us about both performance level and stability.

```{code-cell} ipython3
cv_scores = cross_val_score(baseline_rf, X_train, y_train, cv=5, scoring="r2")
print(f"Baseline 5-fold CV R²: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# Fit on full training set for comparison
baseline_rf.fit(X_train, y_train)
baseline_pred = baseline_rf.predict(X_test)
baseline_r2 = r2_score(y_test, baseline_pred)
baseline_rmse = root_mean_squared_error(y_test, baseline_pred)
baseline_mae = mean_absolute_error(y_test, baseline_pred)

print(f"Baseline test R²:   {baseline_r2:.4f}")
print(f"Baseline test RMSE: {baseline_rmse:.4f}")
print(f"Baseline test MAE:  {baseline_mae:.4f}")
```

## Hyperparameter Tuning

Random Forest has several hyperparameters that control model complexity:

| Parameter | Controls | Effect of increasing |
|-----------|----------|---------------------|
| `n_estimators` | Number of trees | More stable predictions (diminishing returns) |
| `max_depth` | Maximum tree depth | More complex trees, risk of overfitting |
| `min_samples_split` | Min samples to split a node | Regularization (prevents small splits) |
| `min_samples_leaf` | Min samples at leaf nodes | Regularization (smoother predictions) |
| `max_features` | Features per split | More diversity between trees |

`RandomizedSearchCV` samples 50 random parameter combinations and evaluates each
with 5-fold cross-validation. This is more efficient than exhaustive grid search
when the parameter space is large.

```{code-cell} ipython3
param_distributions = {
    "n_estimators": [100, 200, 300, 500],
    "max_depth": [None, 10, 20, 30],
    "min_samples_split": randint(2, 11),
    "min_samples_leaf": randint(1, 5),
    "max_features": ["sqrt", "log2", None],
}

search = RandomizedSearchCV(
    RandomForestRegressor(random_state=RANDOM_SEED, n_jobs=-1),
    param_distributions=param_distributions,
    n_iter=50,
    cv=5,
    scoring="r2",
    random_state=RANDOM_SEED,
    n_jobs=-1,
    verbose=0,
)

search.fit(X_train, y_train)
print(f"Best CV R² (tuned): {search.best_score_:.4f}")
print(f"Best parameters:    {search.best_params_}")

tuned_rf = search.best_estimator_
```

## Model Evaluation

We evaluate the tuned model on the **held-out test set** — data the model has
never seen during training or tuning. This gives an unbiased estimate of how the
model would perform on new municipalities.

**Metrics explained:**

- **R-squared** (coefficient of determination): proportion of variance explained.
  1.0 means perfect prediction; 0.0 means no better than predicting the mean.
- **RMSE** (root mean squared error): average prediction error in the same units
  as the target. Sensitive to large errors.
- **MAE** (mean absolute error): average absolute error. Less sensitive to
  outliers than RMSE.

```{code-cell} ipython3
tuned_pred = tuned_rf.predict(X_test)
tuned_r2 = r2_score(y_test, tuned_pred)
tuned_rmse = root_mean_squared_error(y_test, tuned_pred)
tuned_mae = mean_absolute_error(y_test, tuned_pred)

print(f"Tuned test R²:   {tuned_r2:.4f}")
print(f"Tuned test RMSE: {tuned_rmse:.4f}")
print(f"Tuned test MAE:  {tuned_mae:.4f}")
```

```{code-cell} ipython3
#| label: fig-actual-vs-predicted
#| fig-cap: "Actual vs predicted SDG 1 (No Poverty) scores. Points near the dashed 45-degree line indicate accurate predictions."

fig, ax = plt.subplots(figsize=(7, 7))
ax.scatter(y_test, tuned_pred, alpha=0.6, edgecolors="k", linewidths=0.5)
lims = [
    min(y_test.min(), tuned_pred.min()) - 2,
    max(y_test.max(), tuned_pred.max()) + 2,
]
ax.plot(lims, lims, "--", color="red", linewidth=1.5, label="Perfect prediction")
ax.set_xlabel(f"Actual {TARGET_LABEL}")
ax.set_ylabel(f"Predicted {TARGET_LABEL}")
ax.set_title(f"Actual vs Predicted {TARGET_LABEL} (R² = {tuned_r2:.3f})")
ax.set_xlim(lims)
ax.set_ylim(lims)
ax.set_aspect("equal")
ax.legend()
plt.tight_layout()
plt.savefig(IMAGES_DIR / "ml_actual_vs_predicted.png", dpi=300, bbox_inches="tight")
plt.show()
```

```{code-cell} ipython3
#| label: fig-residuals
#| fig-cap: "Residuals (actual minus predicted) vs predicted values. Randomly scattered residuals around zero suggest no systematic bias."

residuals = y_test - tuned_pred
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(tuned_pred, residuals, alpha=0.6, edgecolors="k", linewidths=0.5)
ax.axhline(0, color="red", linestyle="--", linewidth=1.5)
ax.set_xlabel(f"Predicted {TARGET_LABEL}")
ax.set_ylabel("Residual (Actual - Predicted)")
ax.set_title("Residuals vs Predicted Values")
plt.tight_layout()
plt.savefig(IMAGES_DIR / "ml_residuals.png", dpi=300, bbox_inches="tight")
plt.show()
```

## Feature Importance

Understanding which features drive predictions is crucial for interpretation.
We use two complementary approaches:

### Mean Decrease in Impurity (MDI)

MDI measures how much each feature reduces prediction error across all trees in
the forest. It is fast to compute (already available after fitting) but has a
known bias: it tends to favor high-cardinality continuous features and can be
misleading when features are correlated.

```{code-cell} ipython3
#| label: fig-importance-mdi
#| fig-cap: "Top-20 satellite embedding features ranked by Mean Decrease in Impurity (MDI). MDI is fast but can be biased toward correlated features."

importances_mdi = tuned_rf.feature_importances_
idx_mdi = np.argsort(importances_mdi)[::-1][:20]

fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(
    range(20),
    importances_mdi[idx_mdi][::-1],
    align="center",
)
ax.set_yticks(range(20))
ax.set_yticklabels([FEATURE_COLS[i] for i in idx_mdi][::-1])
ax.set_xlabel("Mean Decrease in Impurity")
ax.set_title(f"Top-20 Feature Importances (MDI) — predicting {TARGET_LABEL}")
plt.tight_layout()
plt.savefig(IMAGES_DIR / "ml_feature_importance_mdi.png", dpi=300, bbox_inches="tight")
plt.show()
```

### Permutation Importance

Permutation importance is more reliable: it measures how much test-set
performance drops when a single feature's values are randomly shuffled. This
approach accounts for feature interactions and is computed on held-out data,
making it less prone to overfitting bias.

```{code-cell} ipython3
#| label: fig-importance-permutation
#| fig-cap: "Top-20 satellite embedding features ranked by permutation importance (decrease in R-squared when feature is shuffled). Error bars show standard deviation across 30 permutation repeats."

perm_result = permutation_importance(
    tuned_rf, X_test, y_test, n_repeats=30, random_state=RANDOM_SEED, n_jobs=-1
)
idx_perm = np.argsort(perm_result.importances_mean)[::-1][:20]

fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(
    range(20),
    perm_result.importances_mean[idx_perm][::-1],
    xerr=perm_result.importances_std[idx_perm][::-1],
    align="center",
)
ax.set_yticks(range(20))
ax.set_yticklabels([FEATURE_COLS[i] for i in idx_perm][::-1])
ax.set_xlabel("Mean Permutation Importance (decrease in R²)")
ax.set_title(f"Top-20 Feature Importances (Permutation) — predicting {TARGET_LABEL}")
plt.tight_layout()
plt.savefig(
    IMAGES_DIR / "ml_feature_importance_permutation.png",
    dpi=300,
    bbox_inches="tight",
)
plt.show()
```

## Partial Dependence

Partial dependence plots show the **marginal effect** of a feature on
predictions, averaging over all other features. Unlike correlation, they
capture non-linear relationships and threshold effects learned by the model.

For example, a partial dependence plot might reveal that a satellite embedding
dimension has no effect below a certain threshold, then sharply increases — this
could correspond to a transition from rural to urbanized land cover that relates
to poverty reduction.

```{code-cell} ipython3
#| label: fig-partial-dependence
#| fig-cap: "Partial dependence plots for the top-6 most important features (by permutation importance). Each plot shows how the predicted SDG 1 score changes as one feature varies, holding all others constant."

top6_perm = [int(i) for i in idx_perm[:6]]
top6_names = [FEATURE_COLS[i] for i in top6_perm]

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
display = PartialDependenceDisplay.from_estimator(
    tuned_rf,
    X_train,
    features=top6_perm,
    feature_names=FEATURE_COLS,
    ax=axes,
    n_jobs=-1,
)
fig.suptitle(
    f"Partial Dependence: Top-6 Features for {TARGET_LABEL}",
    y=1.02,
)
plt.tight_layout()
plt.savefig(IMAGES_DIR / "ml_partial_dependence.png", dpi=300, bbox_inches="tight")
plt.show()
```

## Interpretation and Results

```{code-cell} ipython3
#| label: tbl-ml-results
#| tbl-cap: "Random Forest regression results comparing baseline (100 trees, default parameters) with the tuned model (RandomizedSearchCV, 50 iterations). Target variable: SDG 1 (No Poverty)."

results_df = pd.DataFrame(
    {
        "Metric": ["R_squared", "RMSE", "MAE"],
        "Baseline": [baseline_r2, baseline_rmse, baseline_mae],
        "Tuned": [tuned_r2, tuned_rmse, tuned_mae],
    }
)
results_df.to_csv(TABLES_DIR / "ml_rf_results.csv", index=False)

# Also save best parameters
params_df = pd.DataFrame(
    [{"Parameter": k, "Value": str(v)} for k, v in search.best_params_.items()]
)
params_df.to_csv(TABLES_DIR / "ml_rf_best_params.csv", index=False)

Markdown(results_df.to_markdown(index=False))
```

### Summary

This analysis demonstrates that satellite image embeddings contain meaningful
information about municipal poverty levels in Bolivia. The Random Forest model
learns non-linear relationships between 64 embedding dimensions and SDG 1 (No
Poverty) scores.

**Key findings:**

- Satellite embeddings can predict municipal poverty outcomes with reasonable
  accuracy, confirming that remotely sensed features capture development-relevant
  spatial patterns.
- Hyperparameter tuning via RandomizedSearchCV can improve upon baseline
  performance, though the magnitude of improvement depends on how well the
  default parameters already fit the data.
- Permutation importance identifies which embedding dimensions are most
  predictive — these likely correspond to land use patterns, urbanization, or
  vegetation characteristics visible from space.

**Limitations:**

- 339 municipalities is a relatively small dataset for ML — results may be
  sensitive to the specific train/test split.
- Satellite embeddings from 2017 may not perfectly align with the time period of
  SDG measurements.
- Random Forest captures non-linear relationships but does not provide causal
  estimates — a highly predictive feature is not necessarily a causal driver.

**Next steps:**

1. Compare with other ML models (Gradient Boosting, neural networks)
2. Add spatial cross-validation to account for geographic autocorrelation
3. Incorporate temporal variation using multi-year satellite data
4. Use SHAP values for more detailed feature interpretation
