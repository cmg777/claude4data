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
predicting Bolivia's **Municipal Sustainable Development Index (IMDS)** from
satellite image embeddings using **Random Forest regression**.

### What is IMDS?

The IMDS is a composite index (0--100 scale) that captures the sustainable
development status of each of Bolivia's 339 municipalities. Higher values
indicate better outcomes across multiple dimensions of development.

### What are satellite embeddings?

Satellite embeddings are 64-dimensional numeric representations (A00--A63)
derived from satellite imagery of each municipality. These embeddings compress
visual information about land cover, urbanization, infrastructure, and other
spatial features into a compact vector that machine learning models can use
as input features.

### Learning objectives

1. Understand the end-to-end ML workflow: data loading, EDA, splitting,
   training, tuning, and evaluation
2. Learn why **train/test splitting** must happen before any preprocessing
3. Use **cross-validation** for reliable performance estimates
4. Apply **hyperparameter tuning** with `RandomizedSearchCV`
5. Interpret models using **feature importance** and **partial dependence**

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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import (
    PartialDependenceDisplay,
    permutation_importance,
)
from scipy.stats import randint
from IPython.display import Markdown

sns.set_theme(style="whitegrid")
```

## Data Loading

We use the [DS4Bolivia](https://github.com/quarcs-lab/ds4bolivia) repository,
which provides open data on Bolivian municipalities including SDG indices,
satellite embeddings, and region names. The three datasets are merged on
`asdf_id`, a unique municipality identifier.

After the first download, the merged data is cached locally to avoid repeated
network requests and enable offline runs.

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
    df.to_csv(CACHE_PATH, index=False)
    print(f"Saved merged data to {CACHE_PATH}")

TARGET = "imds"
FEATURE_COLS = [f"A{i:02d}" for i in range(64)]

n_before = len(df)
df = df.dropna(subset=FEATURE_COLS + [TARGET])
n_dropped = n_before - len(df)
print(f"Rows: {len(df)} (dropped {n_dropped} with missing values)")

X = df[FEATURE_COLS].values
y = df[TARGET].values
```

## Exploratory Data Analysis

Before building any model, we need to understand our data. EDA helps us spot
anomalies (e.g., outliers, skewness), check whether the target variable has
a reasonable distribution for regression, and identify which features might
be most informative.

```{code-cell} ipython3
#| label: fig-imds-distribution
#| fig-cap: "Distribution of IMDS across Bolivia's municipalities. The red dashed line marks the mean."

fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(y, bins=30, edgecolor="white", alpha=0.8)
ax.set_xlabel("IMDS (Municipal Sustainable Development Index)")
ax.set_ylabel("Frequency")
ax.set_title("Distribution of IMDS across Bolivian Municipalities")
ax.axvline(y.mean(), color="red", linestyle="--", label=f"Mean = {y.mean():.1f}")
ax.legend()
plt.tight_layout()
plt.savefig(IMAGES_DIR / "ml_imds_distribution.png", dpi=300, bbox_inches="tight")
plt.show()
```

Next, we examine which satellite embedding dimensions are most correlated with
IMDS. Strong correlations suggest features that the Random Forest might find
most useful — although correlation only captures *linear* relationships, and
tree-based models can exploit non-linear patterns too.

```{code-cell} ipython3
#| label: fig-embedding-correlations
#| fig-cap: "Correlation heatmap of the top-10 satellite embeddings most correlated with IMDS."

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
ax.set_title("Correlation: Top-10 Satellite Embeddings with IMDS")
plt.tight_layout()
plt.savefig(
    IMAGES_DIR / "ml_embedding_correlations.png", dpi=300, bbox_inches="tight"
)
plt.show()
```

## Train/Test Split

A critical step in any ML workflow is to **split the data before doing
anything else**. The test set must remain completely untouched during training
and hyperparameter tuning — otherwise we risk **data leakage**, where the
model implicitly learns from the test data and our performance estimates
become overly optimistic.

We use an 80/20 split with a fixed random seed for reproducibility.

```{code-cell} ipython3
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_SEED
)
print(f"Train set: {X_train.shape[0]} samples")
print(f"Test  set: {X_test.shape[0]} samples")
```

## Model Training: Random Forest Basics

A **Random Forest** is an *ensemble* method that builds many decision trees
and averages their predictions. Each tree is trained on a **bootstrap sample**
(random sample with replacement) of the training data, and at each split only
a random subset of features is considered. This process — called **bagging**
combined with **feature subsampling** — reduces overfitting and makes the
ensemble more robust than any single tree.

We start with a baseline model using default hyperparameters.

```{code-cell} ipython3
baseline_rf = RandomForestRegressor(
    n_estimators=100, random_state=RANDOM_SEED, n_jobs=-1
)
```

## Cross-Validation

Rather than evaluating on a single train/validation split, we use **5-fold
cross-validation**. The training set is divided into 5 equal parts; the model
is trained on 4 parts and evaluated on the remaining part, rotating through
all 5 possible held-out folds. The mean and standard deviation of the scores
across folds give us a more reliable performance estimate.

```{code-cell} ipython3
cv_scores = cross_val_score(baseline_rf, X_train, y_train, cv=5, scoring="r2")
print(f"Baseline 5-fold CV R²: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# Fit baseline on full training set for later comparison
baseline_rf.fit(X_train, y_train)
baseline_pred = baseline_rf.predict(X_test)
baseline_r2 = r2_score(y_test, baseline_pred)
baseline_rmse = np.sqrt(mean_squared_error(y_test, baseline_pred))
baseline_mae = mean_absolute_error(y_test, baseline_pred)

print(f"Baseline test R²:   {baseline_r2:.4f}")
print(f"Baseline test RMSE: {baseline_rmse:.4f}")
print(f"Baseline test MAE:  {baseline_mae:.4f}")
```

## Hyperparameter Tuning

The baseline model uses sensible defaults, but we can often improve
performance by tuning key hyperparameters:

- **`n_estimators`**: More trees generally improve performance but increase
  computation time.
- **`max_depth`**: Controls how deep each tree can grow. Shallower trees are
  simpler and less prone to overfitting.
- **`min_samples_split`**: Minimum number of samples required to split an
  internal node. Higher values create simpler trees.
- **`min_samples_leaf`**: Minimum number of samples at a leaf node. Acts as
  a regularization mechanism.
- **`max_features`**: Number of features considered at each split. Fewer
  features increase diversity among trees.

**`RandomizedSearchCV`** samples random combinations from these distributions,
which is more efficient than exhaustive grid search when the parameter space
is large. We use 50 iterations with 5-fold CV.

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

We evaluate the tuned model on the held-out test set. Three complementary
metrics capture different aspects of model performance:

- **R² (coefficient of determination)**: Proportion of variance explained.
  R² = 1 is perfect; R² = 0 means the model is no better than predicting
  the mean.
- **RMSE (Root Mean Squared Error)**: Average prediction error in the same
  units as IMDS. Penalizes large errors more heavily.
- **MAE (Mean Absolute Error)**: Average absolute prediction error. More
  robust to outliers than RMSE.

```{code-cell} ipython3
tuned_pred = tuned_rf.predict(X_test)
tuned_r2 = r2_score(y_test, tuned_pred)
tuned_rmse = np.sqrt(mean_squared_error(y_test, tuned_pred))
tuned_mae = mean_absolute_error(y_test, tuned_pred)

print(f"Tuned test R²:   {tuned_r2:.4f}")
print(f"Tuned test RMSE: {tuned_rmse:.4f}")
print(f"Tuned test MAE:  {tuned_mae:.4f}")
```

```{code-cell} ipython3
#| label: fig-actual-vs-predicted
#| fig-cap: "Actual vs predicted IMDS for the tuned Random Forest model. Points near the dashed 45-degree line indicate accurate predictions."

fig, ax = plt.subplots(figsize=(7, 7))
ax.scatter(y_test, tuned_pred, alpha=0.6, edgecolors="k", linewidths=0.5)
lims = [
    min(y_test.min(), tuned_pred.min()) - 2,
    max(y_test.max(), tuned_pred.max()) + 2,
]
ax.plot(lims, lims, "--", color="red", linewidth=1.5, label="Perfect prediction")
ax.set_xlabel("Actual IMDS")
ax.set_ylabel("Predicted IMDS")
ax.set_title(f"Actual vs Predicted IMDS (R² = {tuned_r2:.3f})")
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
#| fig-cap: "Residuals (actual minus predicted) vs predicted IMDS. Randomly scattered residuals around zero suggest no systematic bias."

residuals = y_test - tuned_pred
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(tuned_pred, residuals, alpha=0.6, edgecolors="k", linewidths=0.5)
ax.axhline(0, color="red", linestyle="--", linewidth=1.5)
ax.set_xlabel("Predicted IMDS")
ax.set_ylabel("Residual (Actual - Predicted)")
ax.set_title("Residuals vs Predicted Values")
plt.tight_layout()
plt.savefig(IMAGES_DIR / "ml_residuals.png", dpi=300, bbox_inches="tight")
plt.show()
```

## Feature Importance

Understanding *which* satellite embedding dimensions drive predictions is
crucial for interpretation. We compare two approaches:

### Mean Decrease in Impurity (MDI)

MDI measures how much each feature reduces prediction error across all trees
in the forest. It is fast to compute (it comes "for free" from model fitting)
but has a known bias: it tends to favor high-cardinality continuous features
and features that appear in many splits due to correlation.

```{code-cell} ipython3
#| label: fig-importance-mdi
#| fig-cap: "Top-20 satellite embedding dimensions ranked by Mean Decrease in Impurity (MDI)."

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
ax.set_title("Top-20 Feature Importances (MDI)")
plt.tight_layout()
plt.savefig(
    IMAGES_DIR / "ml_feature_importance_mdi.png", dpi=300, bbox_inches="tight"
)
plt.show()
```

### Permutation Importance

Permutation importance is a more reliable alternative. For each feature, we
randomly shuffle its values in the test set and measure how much model
performance degrades. Features that cause a large drop in R² when shuffled
are truly important. Unlike MDI, permutation importance is computed on
held-out data and is not biased by the tree-building process.

```{code-cell} ipython3
#| label: fig-importance-permutation
#| fig-cap: "Top-20 satellite embedding dimensions ranked by permutation importance on the test set. Error bars show standard deviation across 30 permutation repeats."

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
ax.set_title("Top-20 Feature Importances (Permutation)")
plt.tight_layout()
plt.savefig(
    IMAGES_DIR / "ml_feature_importance_permutation.png",
    dpi=300,
    bbox_inches="tight",
)
plt.show()
```

## Partial Dependence

Partial dependence plots show the **marginal effect** of a single feature
on the predicted outcome, averaging over the values of all other features.
They reveal non-linear relationships that simple correlation coefficients
cannot capture — an important advantage of tree-based models over linear
regression.

We plot the top-6 features identified by permutation importance.

```{code-cell} ipython3
#| label: fig-partial-dependence
#| fig-cap: "Partial dependence plots for the top-6 most important satellite embedding dimensions. Each curve shows how IMDS predictions change as one feature varies while all others are held at their observed values."

top6_perm = [int(i) for i in idx_perm[:6]]

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
    "Partial Dependence: Top-6 Features (by Permutation Importance)", y=1.02
)
plt.tight_layout()
plt.savefig(
    IMAGES_DIR / "ml_partial_dependence.png", dpi=300, bbox_inches="tight"
)
plt.show()
```

## Interpretation and Results Summary

```{code-cell} ipython3
# Save results tables
results_df = pd.DataFrame(
    {
        "Metric": ["R²", "RMSE", "MAE"],
        "Baseline": [baseline_r2, baseline_rmse, baseline_mae],
        "Tuned": [tuned_r2, tuned_rmse, tuned_mae],
    }
)
results_df.to_csv(TABLES_DIR / "ml_rf_results.csv", index=False)

params_df = pd.DataFrame(
    [{"Parameter": k, "Value": v} for k, v in search.best_params_.items()]
)
params_df.to_csv(TABLES_DIR / "ml_rf_best_params.csv", index=False)
```

```{code-cell} ipython3
#| label: tbl-ml-results
#| tbl-cap: "Random Forest regression results: baseline (default hyperparameters) vs tuned (RandomizedSearchCV with 50 iterations)."

display_df = results_df.copy()
display_df["Baseline"] = display_df["Baseline"].round(4)
display_df["Tuned"] = display_df["Tuned"].round(4)
Markdown(display_df.to_markdown(index=False))
```

### Key Takeaways

- **Satellite embeddings encode development-relevant information.** A Random
  Forest trained on 64 embedding dimensions can predict IMDS with meaningful
  accuracy, demonstrating that satellite imagery captures spatial patterns
  associated with sustainable development outcomes.

- **Hyperparameter tuning matters.** The tuned model generally outperforms the
  baseline, though the magnitude of improvement depends on how far the
  defaults were from optimal for this particular dataset.

- **Feature importance provides interpretability.** Permutation importance
  (computed on held-out data) is more reliable than MDI for identifying which
  embedding dimensions truly drive predictions.

- **Partial dependence reveals non-linear relationships.** Unlike correlation
  coefficients, PD plots can show threshold effects and non-monotonic
  relationships between embeddings and development outcomes.

### Limitations

1. **Cross-sectional analysis only** — we use 2017 embeddings and cannot make
   causal claims about how changes in land cover affect development.
2. **Spatial autocorrelation** — neighboring municipalities may have similar
   embeddings and IMDS values, violating the independence assumption of
   standard cross-validation.
3. **Embedding interpretability** — the 64 dimensions are learned
   representations; connecting specific dimensions to physical features
   requires additional analysis.

### Next Steps

- Try other algorithms (Gradient Boosting, neural networks) and compare
- Add spatial cross-validation to account for geographic autocorrelation
- Explore temporal variation using embeddings from multiple years
- Connect top embedding dimensions to physical features via visualization
