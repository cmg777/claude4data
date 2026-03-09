"""
Introduction to Causal Inference: Double Machine Learning

Estimates the causal effect of a cash bonus on unemployment duration using
the Pennsylvania Bonus dataset and the DoubleML framework. Compares naive
OLS with the debiased Double ML estimator (Partially Linear Regression).

Usage:
    uv run python code/tut_doubleml.py

References:
    - https://docs.doubleml.org/stable/intro/intro.html
    - Chernozhukov et al. (2018). Double/debiased machine learning for
      treatment and structural parameters. The Econometrics Journal.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import set_seeds, RANDOM_SEED, IMAGES_DIR, TABLES_DIR

set_seeds()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV, LinearRegression
from doubleml import DoubleMLData, DoubleMLPLR
from doubleml.datasets import fetch_bonus

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
OUTCOME = "inuidur1"
OUTCOME_LABEL = "Log Unemployment Duration"
TREATMENT = "tg"
TREATMENT_LABEL = "Cash Bonus (tg)"
COVARIATES = [
    "female", "black", "othrace", "dep1", "dep2",
    "q2", "q3", "q4", "q5", "q6",
    "agelt35", "agegt54", "durable", "lusd", "husd",
]

# ---------------------------------------------------------------------------
# 1. Data Loading
# ---------------------------------------------------------------------------
# The Pennsylvania Bonus dataset comes from a labor market experiment.
# Workers who became unemployed were randomly assigned to receive a cash
# bonus if they found a new job within a qualifying period. The outcome
# is log unemployment duration.

df = fetch_bonus("DataFrame")
print(f"Dataset shape: {df.shape}")
print(f"Observations: {len(df)}")
print(f"\nTreatment groups:")
print(df[TREATMENT].value_counts().rename({0: "Control", 1: "Bonus"}))
print(f"\nOutcome ({OUTCOME}) summary:")
print(df[OUTCOME].describe().round(3))

# ---------------------------------------------------------------------------
# 2. Exploratory Data Analysis
# ---------------------------------------------------------------------------
# Before modeling, we explore the relationship between treatment and outcome,
# and check whether covariates differ across treatment groups.

# Outcome distribution by treatment group
fig, ax = plt.subplots(figsize=(8, 5))
for group, label, color in [(0, "Control", "#6a9bcc"), (1, "Bonus", "#d97757")]:
    subset = df[df[TREATMENT] == group][OUTCOME]
    ax.hist(subset, bins=30, alpha=0.6, label=f"{label} (mean={subset.mean():.3f})",
            color=color, edgecolor="white")
ax.set_xlabel(OUTCOME_LABEL)
ax.set_ylabel("Count")
ax.set_title(f"Distribution of {OUTCOME_LABEL} by Treatment Group")
ax.legend()
plt.savefig(IMAGES_DIR / "tut_doubleml_outcome_by_treatment.png", dpi=300, bbox_inches="tight")
plt.close()
print("\nSaved: images/tut_doubleml_outcome_by_treatment.png")

# Covariate balance between treatment groups
covariate_means = df.groupby(TREATMENT)[COVARIATES].mean()
fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(COVARIATES))
width = 0.35
ax.bar(x - width / 2, covariate_means.loc[0], width, label="Control", color="#6a9bcc", edgecolor="white")
ax.bar(x + width / 2, covariate_means.loc[1], width, label="Bonus", color="#d97757", edgecolor="white")
ax.set_xticks(x)
ax.set_xticklabels(COVARIATES, rotation=45, ha="right")
ax.set_ylabel("Mean Value")
ax.set_title("Covariate Balance: Control vs Bonus Group")
ax.legend()
plt.savefig(IMAGES_DIR / "tut_doubleml_covariate_balance.png", dpi=300, bbox_inches="tight")
plt.close()
print("Saved: images/tut_doubleml_covariate_balance.png")

# ---------------------------------------------------------------------------
# 3. Naive OLS Baseline
# ---------------------------------------------------------------------------
# A simple OLS regression of outcome on treatment ignores confounders.
# This gives a biased estimate if covariates affect both treatment and outcome.

ols = LinearRegression()
ols.fit(df[[TREATMENT]], df[OUTCOME])
naive_coef = ols.coef_[0]
print(f"\nNaive OLS coefficient (tg -> inuidur1): {naive_coef:.4f}")

# OLS with covariates for comparison
ols_full = LinearRegression()
ols_full.fit(df[[TREATMENT] + COVARIATES], df[OUTCOME])
ols_full_coef = ols_full.coef_[0]
print(f"OLS with covariates coefficient: {ols_full_coef:.4f}")

# ---------------------------------------------------------------------------
# 4. DoubleML Setup
# ---------------------------------------------------------------------------
# We create a DoubleMLData object specifying outcome, treatment, and
# covariates. Then we set up two Random Forest learners:
# - ml_l: estimates E[Y|X] (how covariates predict the outcome)
# - ml_m: estimates E[D|X] (how covariates predict treatment assignment)

dml_data = DoubleMLData(df, y_col=OUTCOME, d_cols=TREATMENT, x_cols=COVARIATES)
print(f"\nDoubleMLData summary:")
print(dml_data)

learner = RandomForestRegressor(n_estimators=500, max_features="sqrt",
                                max_depth=5, random_state=RANDOM_SEED)
ml_l_rf = clone(learner)
ml_m_rf = clone(learner)

# ---------------------------------------------------------------------------
# 5. PLR Model with Random Forest Learners
# ---------------------------------------------------------------------------
# The Partially Linear Regression (PLR) model:
#   Y = D * theta_0 + g_0(X) + zeta
#   D = m_0(X) + V
# DoubleML estimates theta_0 using cross-fitting to avoid regularization bias.

np.random.seed(RANDOM_SEED)
dml_plr_rf = DoubleMLPLR(dml_data, ml_l_rf, ml_m_rf, n_folds=5)
dml_plr_rf.fit()

print("\n--- DoubleML PLR Results (Random Forest) ---")
print(dml_plr_rf.summary)
rf_coef = dml_plr_rf.coef[0]
rf_se = dml_plr_rf.se[0]
rf_pval = dml_plr_rf.pval[0]
rf_ci = dml_plr_rf.confint().values[0]
print(f"\nCoefficient: {rf_coef:.4f}")
print(f"Std Error:   {rf_se:.4f}")
print(f"p-value:     {rf_pval:.4f}")
print(f"95% CI:      [{rf_ci[0]:.4f}, {rf_ci[1]:.4f}]")

# ---------------------------------------------------------------------------
# 6. Sensitivity to Learner Choice: LassoCV
# ---------------------------------------------------------------------------
# We repeat the analysis with Lasso (linear) learners to check whether
# results are sensitive to the choice of ML method.

ml_l_lasso = LassoCV()
ml_m_lasso = LassoCV()

np.random.seed(RANDOM_SEED)
dml_plr_lasso = DoubleMLPLR(dml_data, ml_l_lasso, ml_m_lasso, n_folds=5)
dml_plr_lasso.fit()

print("\n--- DoubleML PLR Results (Lasso) ---")
print(dml_plr_lasso.summary)
lasso_coef = dml_plr_lasso.coef[0]
lasso_se = dml_plr_lasso.se[0]
lasso_pval = dml_plr_lasso.pval[0]
lasso_ci = dml_plr_lasso.confint().values[0]

# ---------------------------------------------------------------------------
# 7. Comparison Figures
# ---------------------------------------------------------------------------

# Coefficient comparison: Naive OLS vs DoubleML (RF) vs DoubleML (Lasso)
fig, ax = plt.subplots(figsize=(8, 5))
methods = ["Naive OLS", "OLS + Covariates", "DoubleML (RF)", "DoubleML (Lasso)"]
coefs = [naive_coef, ols_full_coef, rf_coef, lasso_coef]
colors = ["#999999", "#666666", "#6a9bcc", "#d97757"]

# Use CIs for DoubleML, no CI for OLS (would need bootstrap)
ax.barh(methods, coefs, color=colors, edgecolor="white", height=0.6)
ax.errorbar(rf_coef, 2, xerr=[[rf_coef - rf_ci[0]], [rf_ci[1] - rf_coef]],
            fmt="none", color="black", capsize=5, linewidth=2)
ax.errorbar(lasso_coef, 3, xerr=[[lasso_coef - lasso_ci[0]], [lasso_ci[1] - lasso_coef]],
            fmt="none", color="black", capsize=5, linewidth=2)
ax.axvline(0, color="black", linewidth=0.5, linestyle="--")
ax.set_xlabel("Estimated Coefficient (Effect on Log Unemployment Duration)")
ax.set_title("Naive OLS vs Double Machine Learning Estimates")
plt.savefig(IMAGES_DIR / "tut_doubleml_coefficient_comparison.png", dpi=300, bbox_inches="tight")
plt.close()
print("\nSaved: images/tut_doubleml_coefficient_comparison.png")

# Confidence interval comparison for DoubleML methods
fig, ax = plt.subplots(figsize=(8, 4))
y_pos = [0, 1]
labels = ["DoubleML (Random Forest)", "DoubleML (Lasso)"]
point_estimates = [rf_coef, lasso_coef]
ci_low = [rf_ci[0], lasso_ci[0]]
ci_high = [rf_ci[1], lasso_ci[1]]

for i, (est, lo, hi, label) in enumerate(zip(point_estimates, ci_low, ci_high, labels)):
    ax.plot([lo, hi], [i, i], color="#6a9bcc" if i == 0 else "#d97757", linewidth=3)
    ax.plot(est, i, "o", color="black", markersize=8, zorder=5)
    ax.text(hi + 0.005, i, f"{est:.4f} [{lo:.4f}, {hi:.4f}]", va="center", fontsize=9)

ax.axvline(0, color="black", linewidth=0.5, linestyle="--")
ax.set_yticks(y_pos)
ax.set_yticklabels(labels)
ax.set_xlabel("Treatment Effect Estimate (95% CI)")
ax.set_title("Confidence Intervals: DoubleML Estimates")
plt.savefig(IMAGES_DIR / "tut_doubleml_confint.png", dpi=300, bbox_inches="tight")
plt.close()
print("Saved: images/tut_doubleml_confint.png")

# ---------------------------------------------------------------------------
# 8. Save Results
# ---------------------------------------------------------------------------
results_df = pd.DataFrame({
    "Method": ["Naive OLS", "OLS + Covariates", "DoubleML (RF)", "DoubleML (Lasso)"],
    "Coefficient": [naive_coef, ols_full_coef, rf_coef, lasso_coef],
    "Std_Error": [np.nan, np.nan, rf_se, lasso_se],
    "p_value": [np.nan, np.nan, rf_pval, lasso_pval],
    "CI_lower": [np.nan, np.nan, rf_ci[0], lasso_ci[0]],
    "CI_upper": [np.nan, np.nan, rf_ci[1], lasso_ci[1]],
})
results_df.to_csv(TABLES_DIR / "tut_doubleml_results.csv", index=False)
print(f"\nSaved: tables/tut_doubleml_results.csv")

# ---------------------------------------------------------------------------
# 9. Summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("  Double Machine Learning — Pennsylvania Bonus Experiment")
print("=" * 60)
print(f"  Observations:          {len(df)}")
print(f"  Treatment variable:    {TREATMENT} ({TREATMENT_LABEL})")
print(f"  Outcome variable:      {OUTCOME} ({OUTCOME_LABEL})")
print(f"  Covariates:            {len(COVARIATES)}")
print(f"  Naive OLS coefficient: {naive_coef:.4f}")
print(f"  OLS + covariates:      {ols_full_coef:.4f}")
print(f"  DoubleML (RF):         {rf_coef:.4f} (p={rf_pval:.4f})")
print(f"  DoubleML (Lasso):      {lasso_coef:.4f} (p={lasso_pval:.4f})")
print("=" * 60)
