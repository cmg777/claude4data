"""
Introduction to Machine Learning: Random Forest Regression
==========================================================

Predicts Bolivia's SDG 1 (No Poverty) index from satellite image embeddings
using Random Forest regression. Demonstrates a complete ML pipeline including
EDA, train/test split, cross-validation, hyperparameter tuning, evaluation,
feature importance, and partial dependence analysis.

Data source: DS4Bolivia (https://github.com/quarcs-lab/ds4bolivia)

Run from project root:
    uv run python code/ml_intro_rf.py
"""

# ---------------------------------------------------------------------------
# 0. Setup — reproducibility and project paths
# ---------------------------------------------------------------------------
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import set_seeds, RANDOM_SEED, IMAGES_DIR, TABLES_DIR, DATA_DIR

set_seeds()

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
from sklearn.inspection import (
    PartialDependenceDisplay,
    permutation_importance,
)
from scipy.stats import randint

sns.set_theme(style="whitegrid")

# Ensure output directories exist
IMAGES_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)
(DATA_DIR / "rawData").mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Data Loading
# ---------------------------------------------------------------------------
# We load three datasets from the DS4Bolivia repository and merge them on
# `asdf_id`, a unique identifier for each of Bolivia's 339 municipalities.
# After the first download we cache the merged file locally to avoid repeated
# network requests and enable offline runs.

BASE_URL = "https://raw.githubusercontent.com/quarcs-lab/ds4bolivia/master"
CACHE_PATH = DATA_DIR / "rawData" / "ds4bolivia_merged.csv"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
TARGET = "sdg1"
TARGET_LABEL = "SDG 1 (No Poverty)"
FEATURE_COLS = [f"A{i:02d}" for i in range(64)]

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

# Define features (64 satellite embedding dimensions) and target (SDG 1)
# Drop rows with missing values and report count.
n_before = len(df)
df = df.dropna(subset=FEATURE_COLS + [TARGET])
n_dropped = n_before - len(df)
print(f"Rows: {len(df)} (dropped {n_dropped} with missing values)")

X = df[FEATURE_COLS].values
y = df[TARGET].values

# ---------------------------------------------------------------------------
# 2. Exploratory Data Analysis (EDA)
# ---------------------------------------------------------------------------
# Before building any model it is essential to understand the data. EDA helps
# us spot anomalies, check distributions, and identify potentially informative
# features. Understanding how SDG 1 (No Poverty) scores are distributed across
# municipalities reveals geographic inequality patterns.

# Figure 1: Distribution of the target variable (SDG 1)
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
plt.close()
print("Saved: images/ml_target_distribution.png")

# Figure 2: Heatmap of top-10 correlated embeddings with SDG 1
# Identifying which embedding dimensions correlate most strongly with SDG 1
# gives us a first hint about which satellite-derived features capture
# variation related to poverty outcomes.
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
plt.savefig(
    IMAGES_DIR / "ml_embedding_correlations.png", dpi=300, bbox_inches="tight"
)
plt.close()
print("Saved: images/ml_embedding_correlations.png")

# ---------------------------------------------------------------------------
# 3. Train/Test Split
# ---------------------------------------------------------------------------
# We split the data BEFORE any preprocessing or model fitting to prevent
# "data leakage" — the situation where information from the test set
# inadvertently influences training. An 80/20 split is a common default that
# balances having enough data to learn from with enough to evaluate on.

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_SEED
)
print(f"\nTrain set: {X_train.shape[0]} samples")
print(f"Test  set: {X_test.shape[0]} samples")

# ---------------------------------------------------------------------------
# 4. Baseline Model
# ---------------------------------------------------------------------------
# A Random Forest with default parameters serves as our baseline. We evaluate
# it with 5-fold cross-validation on the training set, which gives us a more
# reliable performance estimate than a single train/validation split. Each
# fold uses a different 80/20 partition of the training data.

baseline_rf = RandomForestRegressor(
    n_estimators=100, random_state=RANDOM_SEED, n_jobs=-1
)
cv_scores = cross_val_score(baseline_rf, X_train, y_train, cv=5, scoring="r2")
print(f"\nBaseline 5-fold CV R²: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# Fit baseline on full training set for later comparison
baseline_rf.fit(X_train, y_train)
baseline_pred = baseline_rf.predict(X_test)
baseline_r2 = r2_score(y_test, baseline_pred)
baseline_rmse = root_mean_squared_error(y_test, baseline_pred)
baseline_mae = mean_absolute_error(y_test, baseline_pred)

print(f"Baseline test R²:   {baseline_r2:.4f}")
print(f"Baseline test RMSE: {baseline_rmse:.4f}")
print(f"Baseline test MAE:  {baseline_mae:.4f}")

# ---------------------------------------------------------------------------
# 5. Hyperparameter Tuning
# ---------------------------------------------------------------------------
# Random Forests have several hyperparameters that control tree complexity.
# RandomizedSearchCV samples random combinations from the distributions below,
# which is more efficient than exhaustive grid search when the parameter space
# is large.
#
# Key parameters:
#   n_estimators    — number of trees in the forest
#   max_depth       — maximum depth of each tree (None = fully grown)
#   min_samples_split — minimum samples required to split a node
#   min_samples_leaf  — minimum samples at a leaf node
#   max_features    — number of features considered at each split

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
print(f"\nBest CV R² (tuned): {search.best_score_:.4f}")
print(f"Best parameters:    {search.best_params_}")

tuned_rf = search.best_estimator_

# ---------------------------------------------------------------------------
# 6. Evaluation
# ---------------------------------------------------------------------------
# We evaluate the tuned model on the held-out test set that was never used
# during training or hyperparameter selection. This gives an unbiased estimate
# of how the model will perform on new, unseen municipalities.

tuned_pred = tuned_rf.predict(X_test)
tuned_r2 = r2_score(y_test, tuned_pred)
tuned_rmse = root_mean_squared_error(y_test, tuned_pred)
tuned_mae = mean_absolute_error(y_test, tuned_pred)

print(f"\nTuned test R²:   {tuned_r2:.4f}")
print(f"Tuned test RMSE: {tuned_rmse:.4f}")
print(f"Tuned test MAE:  {tuned_mae:.4f}")

# Figure 3: Actual vs Predicted scatter plot
# A perfect model would place all points on the 45-degree line.
fig, ax = plt.subplots(figsize=(7, 7))
ax.scatter(y_test, tuned_pred, alpha=0.6, edgecolors="k", linewidths=0.5)
lims = [min(y_test.min(), tuned_pred.min()) - 2, max(y_test.max(), tuned_pred.max()) + 2]
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
plt.close()
print("Saved: images/ml_actual_vs_predicted.png")

# Figure 4: Residuals vs Predicted
# Residuals should be randomly scattered around zero with no obvious pattern.
# Systematic patterns would indicate the model is missing some structure.
residuals = y_test - tuned_pred
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(tuned_pred, residuals, alpha=0.6, edgecolors="k", linewidths=0.5)
ax.axhline(0, color="red", linestyle="--", linewidth=1.5)
ax.set_xlabel(f"Predicted {TARGET_LABEL}")
ax.set_ylabel("Residual (Actual - Predicted)")
ax.set_title("Residuals vs Predicted Values")
plt.tight_layout()
plt.savefig(IMAGES_DIR / "ml_residuals.png", dpi=300, bbox_inches="tight")
plt.close()
print("Saved: images/ml_residuals.png")

# ---------------------------------------------------------------------------
# 7. Feature Importance
# ---------------------------------------------------------------------------
# Understanding which satellite embedding dimensions matter most helps us
# interpret the model and connect predictions to physical characteristics
# visible from space that relate to poverty outcomes.

# Figure 5: Mean Decrease in Impurity (MDI)
# MDI measures how much each feature reduces prediction error across all trees.
# Caveat: MDI can be biased toward high-cardinality continuous features.
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
plt.savefig(
    IMAGES_DIR / "ml_feature_importance_mdi.png", dpi=300, bbox_inches="tight"
)
plt.close()
print("Saved: images/ml_feature_importance_mdi.png")

# Figure 6: Permutation Importance
# Permutation importance measures the decrease in model performance when a
# single feature's values are randomly shuffled. It is more reliable than MDI
# because it accounts for feature interactions and is computed on the test set.
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
plt.close()
print("Saved: images/ml_feature_importance_permutation.png")

# ---------------------------------------------------------------------------
# 8. Partial Dependence Plots
# ---------------------------------------------------------------------------
# Partial dependence plots show the marginal effect of a feature on the
# predicted outcome, averaging over the values of all other features. They
# reveal non-linear relationships that simple correlation cannot capture —
# particularly useful for understanding how satellite signatures relate to
# poverty indicators at the municipal level.

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
    f"Partial Dependence: Top-6 Features for {TARGET_LABEL} (Permutation Importance)",
    y=1.02,
)
plt.tight_layout()
plt.savefig(
    IMAGES_DIR / "ml_partial_dependence.png", dpi=300, bbox_inches="tight"
)
plt.close()
print("Saved: images/ml_partial_dependence.png")

# ---------------------------------------------------------------------------
# 9. Save Results
# ---------------------------------------------------------------------------
# Persist key metrics and hyperparameters as CSV files for reproducibility
# and easy inclusion in the manuscript.

results_df = pd.DataFrame(
    {
        "Metric": ["R_squared", "RMSE", "MAE"],
        "Baseline": [baseline_r2, baseline_rmse, baseline_mae],
        "Tuned": [tuned_r2, tuned_rmse, tuned_mae],
    }
)
results_df.to_csv(TABLES_DIR / "ml_rf_results.csv", index=False)
print("\nSaved: tables/ml_rf_results.csv")

params_df = pd.DataFrame(
    [{"Parameter": k, "Value": str(v)} for k, v in search.best_params_.items()]
)
params_df.to_csv(TABLES_DIR / "ml_rf_best_params.csv", index=False)
print("Saved: tables/ml_rf_best_params.csv")

# ---------------------------------------------------------------------------
# 10. Summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print(f"RANDOM FOREST REGRESSION — RESULTS SUMMARY")
print(f"Target: {TARGET} ({TARGET_LABEL})")
print("=" * 60)
print(f"Number of features: {len(FEATURE_COLS)}")
print(f"Training samples:   {X_train.shape[0]}")
print(f"Test samples:       {X_test.shape[0]}")
print("-" * 60)
print(f"{'Metric':<20} {'Baseline':>12} {'Tuned':>12}")
print("-" * 60)
print(f"{'R²':<20} {baseline_r2:>12.4f} {tuned_r2:>12.4f}")
print(f"{'RMSE':<20} {baseline_rmse:>12.4f} {tuned_rmse:>12.4f}")
print(f"{'MAE':<20} {baseline_mae:>12.4f} {tuned_mae:>12.4f}")
print("-" * 60)
print("Best hyperparameters:")
for k, v in search.best_params_.items():
    print(f"  {k}: {v}")
print("=" * 60)
print("\nAll figures saved to images/ml_*.png")
print("All tables saved to tables/ml_*.csv")
