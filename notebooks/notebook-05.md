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
title: "N5: Introduction to Causal Inference — Double Machine Learning"
---

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cmg777/claude4data/blob/master/notebooks/notebook-05.ipynb)

## Case Study: Does a Cash Bonus Shorten Unemployment?

Imagine you are a policymaker in Pennsylvania in the 1980s. Unemployment is high, and you want to know: **if we offer cash bonuses to unemployed workers who find new jobs quickly, will they actually find jobs faster?** This is not just a prediction question — it is a *causal* question. We need to know whether the bonus *causes* shorter unemployment, not just whether bonus recipients happen to have shorter spells. This distinction matters because if the bonus only appears effective due to confounding factors (like age or industry), spending money on it would be wasteful. In this tutorial, we use **Double Machine Learning (DoubleML)** to estimate the causal effect of the bonus while rigorously controlling for confounding variables using modern machine learning methods.

**Why not just compare averages?** A common first instinct is to simply compare the average unemployment duration of workers who received the bonus versus those who did not. But this comparison can be misleading. Think of a classic example: ice cream sales and drowning deaths are positively correlated — but ice cream does not *cause* drowning. The hidden confounder is summer heat, which drives both. Similarly, if bonus recipients differ systematically from non-recipients in age, race, or industry, a simple comparison confuses the effect of those characteristics with the effect of the bonus itself.

### Learning Objectives

- Understand the difference between *causal inference* and *prediction* in data science
- Learn the Partially Linear Regression (PLR) model and its key equations
- Understand *cross-fitting* — how DoubleML uses sample splitting to produce valid statistical inference
- Interpret causal estimates: coefficients, standard errors, p-values, and confidence intervals
- Compare naive OLS estimates with debiased Double ML estimates

```{code-cell} ipython3
import sys
if "google.colab" in sys.modules:
    !pip install -q doubleml
    !git clone --depth 1 https://github.com/cmg777/claude4data.git /content/claude4data 2>/dev/null || true
    %cd /content/claude4data/notebooks
sys.path.insert(0, "..")
from config import set_seeds, RANDOM_SEED, IMAGES_DIR, TABLES_DIR

set_seeds()
```

```{code-cell} ipython3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV, LinearRegression
from doubleml import DoubleMLData, DoubleMLPLR
from doubleml.datasets import fetch_bonus
from IPython.display import Markdown

# Configuration
OUTCOME = "inuidur1"
OUTCOME_LABEL = "Log Unemployment Duration"
TREATMENT = "tg"
TREATMENT_LABEL = "Cash Bonus (tg)"
COVARIATES = [
    "female", "black", "othrace", "dep1", "dep2",
    "q2", "q3", "q4", "q5", "q6",
    "agelt35", "agegt54", "durable", "lusd", "husd",
]
```

## Data Loading: The Pennsylvania Bonus Experiment

The Pennsylvania Bonus dataset comes from a real labor market experiment conducted in the 1980s. Unemployed workers were assigned to either receive a cash bonus for finding a new job quickly (treatment group) or receive no bonus (control group). The dataset includes 15 covariates capturing demographic and labor market characteristics. We load it directly from the `doubleml` package using `fetch_bonus()`.

```{code-cell} ipython3
df = fetch_bonus("DataFrame")

print(f"Dataset shape: {df.shape}")
print(f"Observations: {len(df)}")
print(f"\nTreatment groups:")
print(df[TREATMENT].value_counts().rename({0: "Control", 1: "Bonus"}))
print(f"\nOutcome ({OUTCOME}) summary:")
print(df[OUTCOME].describe().round(3))
```

The dataset contains 5,099 unemployed workers: 3,354 in the control group and 1,745 in the bonus group. The outcome variable `inuidur1` is the *log* of unemployment duration, which ranges from 0.0 to 3.95 with a mean of 2.03. Using log-transformed duration means that our estimated coefficients will represent approximate *percentage changes* in unemployment duration — a coefficient of -0.07 corresponds to roughly a 7% reduction.

## Exploratory Data Analysis

### Outcome Distribution by Treatment Group

Before any modeling, let us visualize how unemployment duration differs between the two groups. If the bonus works, we would expect the bonus group to have a lower average duration.

```{code-cell} ipython3
#| label: fig-outcome-by-treatment
#| fig-cap: "Distribution of log unemployment duration by treatment group. The bonus group shows a slightly lower mean."

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
plt.show()
```

The histograms show substantial overlap between the two groups, with the bonus group (orange) having a slightly lower mean (1.98) compared to the control group (2.05). The difference is small relative to the spread of the data, which is why we need formal statistical methods to determine whether this difference is real or just noise.

### Covariate Balance

A critical check in any causal analysis is whether the treatment and control groups are *balanced* — do they look similar on observable characteristics? If the groups differ substantially on covariates, a simple comparison of outcomes would be confounded.

```{code-cell} ipython3
#| label: fig-covariate-balance
#| fig-cap: "Mean values of covariates in control vs bonus groups. Similar heights indicate good balance."

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
plt.show()
```

The covariate means are quite similar across treatment and control groups, which is expected in a randomized experiment. This is good news — it means the simple difference in means is not likely to be severely biased. However, even small imbalances can matter with large samples, and DoubleML provides a principled way to adjust for them.

## The Confounding Problem

Even when covariates look balanced, subtle confounding can bias causal estimates. **Confounding** occurs when a variable influences both the treatment (bonus assignment) and the outcome (unemployment duration). For example, if workers in durable-goods industries are both more likely to receive the bonus *and* tend to have longer unemployment spells, then a naive comparison would underestimate the bonus effect.

The standard approach — adding covariates to a linear regression — works if the relationships are truly linear. But what if the effect of age on unemployment is non-linear, or interactions between covariates matter? This is where **Double Machine Learning** shines: it uses flexible ML methods to model these complex confounding relationships, while still producing valid causal estimates with proper standard errors.

## Naive OLS Baseline

As a benchmark, we first run a simple OLS regression of unemployment duration on the treatment indicator alone (ignoring confounders), and then with all covariates included.

```{code-cell} ipython3
# Naive OLS: no covariates
ols = LinearRegression()
ols.fit(df[[TREATMENT]], df[OUTCOME])
naive_coef = ols.coef_[0]

# OLS with covariates
ols_full = LinearRegression()
ols_full.fit(df[[TREATMENT] + COVARIATES], df[OUTCOME])
ols_full_coef = ols_full.coef_[0]

print(f"Naive OLS coefficient (no covariates): {naive_coef:.4f}")
print(f"OLS with covariates coefficient:       {ols_full_coef:.4f}")
```

The naive OLS gives a coefficient of -0.0855, suggesting the bonus reduces log unemployment duration by about 8.6%. Adding covariates changes the estimate to -0.0717 (about 7.2%). The shift tells us that confounders do matter — the naive estimate was slightly exaggerated because it picked up some confounding variation. But OLS assumes linear relationships between covariates and outcome; DoubleML relaxes this assumption.

## What is Double Machine Learning?

Double Machine Learning (DoubleML) is a framework for estimating causal effects while using flexible machine learning methods to control for confounders. It was developed by Chernozhukov et al. (2018) and solves a key problem: when you use ML to adjust for confounders, the regularization bias from the ML models contaminates your causal estimate, making standard errors and p-values unreliable.

### The Partially Linear Regression (PLR) Model

The PLR model describes the data-generating process with two equations:

**Outcome equation:**

$$Y = D \cdot \theta_0 + g_0(X) + \zeta$$

This says that unemployment duration ($Y$) depends on the bonus ($D$) through the causal effect $\theta_0$, plus a potentially complex function $g_0(X)$ of the covariates (capturing how demographics and labor market conditions affect unemployment), plus unobserved noise $\zeta$. The key parameter $\theta_0$ is what we want to estimate — it tells us the causal effect of receiving the bonus on log unemployment duration.

**Treatment equation:**

$$D = m_0(X) + V$$

This says that bonus assignment ($D$) depends on covariates through a function $m_0(X)$, plus residual variation $V$. Even in a randomized experiment, treatment assignment can be correlated with covariates (especially with imperfect randomization or stratified designs).

### The Partialling-Out Estimator

DoubleML estimates $\theta_0$ by "partialling out" the confounders from both the outcome and the treatment:

$$\hat{\theta}_0 = \frac{\sum_i (Y_i - \hat{g}_0(X_i)) \cdot (D_i - \hat{m}_0(X_i))}{\sum_i (D_i - \hat{m}_0(X_i))^2}$$

In plain language: (1) use ML to predict $Y$ from $X$ and compute the residual — this is $Y$ "cleaned" of confounder effects; (2) use ML to predict $D$ from $X$ and compute the residual — this is $D$ "cleaned" of confounder effects; (3) regress the cleaned outcome on the cleaned treatment. The result is an estimate of $\theta_0$ that is robust to the complexity of $g_0$ and $m_0$.

### Cross-Fitting: Why It Matters

There is a catch: if you use the same data to estimate $\hat{g}_0$ and $\hat{m}_0$ as you use to estimate $\theta_0$, the ML regularization bias leaks into your causal estimate. DoubleML solves this with **cross-fitting** (also called sample splitting): it splits the data into K folds, estimates the nuisance functions ($\hat{g}_0$, $\hat{m}_0$) on K-1 folds, and computes residuals on the held-out fold. This ensures the ML predictions are always out-of-sample, eliminating overfitting bias and producing valid standard errors, p-values, and confidence intervals.

## Setting Up DoubleML

We create a `DoubleMLData` object by specifying the outcome column, treatment column, and covariate columns. Then we set up two Random Forest learners — one for each nuisance function. We use `clone()` from scikit-learn to ensure each learner is an independent copy.

```{code-cell} ipython3
dml_data = DoubleMLData(df, y_col=OUTCOME, d_cols=TREATMENT, x_cols=COVARIATES)
print(dml_data)
```

```{code-cell} ipython3
learner = RandomForestRegressor(n_estimators=500, max_features="sqrt",
                                max_depth=5, random_state=RANDOM_SEED)
ml_l_rf = clone(learner)  # Learner for E[Y|X]
ml_m_rf = clone(learner)  # Learner for E[D|X]

print(f"ml_l (outcome model): {type(ml_l_rf).__name__}")
print(f"ml_m (treatment model): {type(ml_m_rf).__name__}")
print(f"  n_estimators={learner.n_estimators}, max_depth={learner.max_depth}, max_features='{learner.max_features}'")
```

The `DoubleMLData` object confirms we have 5,099 observations with `inuidur1` as outcome, `tg` as treatment, and 15 covariates. The Random Forest learners use 500 trees with max depth 5 — a moderately complex model that can capture non-linear relationships without severe overfitting.

## Fitting the PLR Model

Now we fit the DoubleML PLR model. Under the hood, this performs 5-fold cross-fitting: for each fold, it (1) trains the outcome model `ml_l` on the other 4 folds, (2) trains the treatment model `ml_m` on the other 4 folds, (3) computes residuals on the held-out fold. Then it combines all residuals to estimate $\theta_0$.

```{code-cell} ipython3
np.random.seed(RANDOM_SEED)
dml_plr_rf = DoubleMLPLR(dml_data, ml_l_rf, ml_m_rf, n_folds=5)
dml_plr_rf.fit()

print(dml_plr_rf.summary)
```

The DoubleML estimate of the bonus effect is -0.0736 with a standard error of 0.0354, giving a t-statistic of -2.08 and a p-value of 0.038. Since the p-value is below 0.05, we can reject the null hypothesis that the bonus has no effect at the 5% significance level. The negative sign means the bonus *reduces* unemployment duration — workers who received the bonus found jobs faster.

## Interpreting the Results

Let us unpack the key numbers from the DoubleML output:

```{code-cell} ipython3
rf_coef = dml_plr_rf.coef[0]
rf_se = dml_plr_rf.se[0]
rf_pval = dml_plr_rf.pval[0]
rf_ci = dml_plr_rf.confint().values[0]

print(f"Coefficient (θ₀):  {rf_coef:.4f}")
print(f"Standard Error:    {rf_se:.4f}")
print(f"p-value:           {rf_pval:.4f}")
print(f"95% CI:            [{rf_ci[0]:.4f}, {rf_ci[1]:.4f}]")
print(f"\nInterpretation:")
print(f"  The bonus reduces log unemployment duration by {abs(rf_coef):.4f}.")
print(f"  This corresponds to approximately a {abs(rf_coef)*100:.1f}% reduction.")
print(f"  We are 95% confident the true effect lies between")
print(f"  {abs(rf_ci[1])*100:.1f}% and {abs(rf_ci[0])*100:.1f}% reduction.")
```

The coefficient of -0.0736 means that receiving the cash bonus reduces log unemployment duration by about 7.4%. The 95% confidence interval runs from -0.143 to -0.004, meaning we are 95% confident the true effect is between a 14.3% reduction and a 0.4% reduction. The interval does not contain zero, which is consistent with the statistically significant p-value of 0.038.

## Sensitivity: Does the Choice of ML Learner Matter?

A natural concern is whether the results depend on using Random Forest specifically. We re-run the analysis with Lasso (a linear regularized model) to check robustness.

```{code-cell} ipython3
ml_l_lasso = LassoCV()
ml_m_lasso = LassoCV()

np.random.seed(RANDOM_SEED)
dml_plr_lasso = DoubleMLPLR(dml_data, ml_l_lasso, ml_m_lasso, n_folds=5)
dml_plr_lasso.fit()

print(dml_plr_lasso.summary)
```

The Lasso-based DoubleML gives a very similar estimate of -0.0712 (p=0.044), compared to -0.0736 (p=0.038) with Random Forest. This consistency across learners is reassuring — it suggests the result is robust and not an artifact of a particular ML model choice. Both methods agree: the bonus reduces unemployment duration by about 7%.

## Comparing All Estimates

```{code-cell} ipython3
#| label: fig-coefficient-comparison
#| fig-cap: "Comparison of causal effect estimates across methods. Error bars show 95% confidence intervals for DoubleML methods."

lasso_coef = dml_plr_lasso.coef[0]
lasso_se = dml_plr_lasso.se[0]
lasso_ci = dml_plr_lasso.confint().values[0]

fig, ax = plt.subplots(figsize=(8, 5))
methods = ["Naive OLS", "OLS + Covariates", "DoubleML (RF)", "DoubleML (Lasso)"]
coefs = [naive_coef, ols_full_coef, rf_coef, lasso_coef]
colors = ["#999999", "#666666", "#6a9bcc", "#d97757"]

ax.barh(methods, coefs, color=colors, edgecolor="white", height=0.6)
ax.errorbar(rf_coef, 2, xerr=[[rf_coef - rf_ci[0]], [rf_ci[1] - rf_coef]],
            fmt="none", color="black", capsize=5, linewidth=2)
ax.errorbar(lasso_coef, 3, xerr=[[lasso_coef - lasso_ci[0]], [lasso_ci[1] - lasso_coef]],
            fmt="none", color="black", capsize=5, linewidth=2)
ax.axvline(0, color="black", linewidth=0.5, linestyle="--")
ax.set_xlabel("Estimated Coefficient (Effect on Log Unemployment Duration)")
ax.set_title("Naive OLS vs Double Machine Learning Estimates")
plt.savefig(IMAGES_DIR / "tut_doubleml_coefficient_comparison.png", dpi=300, bbox_inches="tight")
plt.show()
```

The comparison reveals a clear pattern: all four methods find a negative effect (the bonus shortens unemployment), but the magnitude differs. The naive OLS (-0.086) gives the largest estimate because it confounds the bonus effect with covariate effects. Adding covariates linearly (-0.072) brings the estimate closer to the DoubleML results (-0.074 and -0.071), which properly account for potentially non-linear confounding. The DoubleML estimates also come with rigorous confidence intervals — something the naive OLS cannot provide for a causal interpretation.

### Confidence Intervals

```{code-cell} ipython3
#| label: fig-confint
#| fig-cap: "95% confidence intervals for the DoubleML causal effect estimates using two different ML learners."

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
plt.show()
```

Both confidence intervals are entirely to the left of zero, confirming a statistically significant negative effect. The intervals largely overlap, showing that the two ML approaches give consistent results. The RF interval ([-0.143, -0.004]) is slightly wider than the Lasso interval ([-0.141, -0.002]), but both tell the same story: the bonus reduces unemployment duration.

## Summary Table

```{code-cell} ipython3
#| label: tbl-doubleml-results
#| tbl-cap: "Comparison of treatment effect estimates across methods."

results_df = pd.DataFrame({
    "Method": ["Naive OLS", "OLS + Covariates", "DoubleML (RF)", "DoubleML (Lasso)"],
    "Coefficient": [f"{naive_coef:.4f}", f"{ols_full_coef:.4f}", f"{rf_coef:.4f}", f"{lasso_coef:.4f}"],
    "Std Error": ["—", "—", f"{rf_se:.4f}", f"{lasso_se:.4f}"],
    "p-value": ["—", "—", f"{rf_pval:.4f}", f"{dml_plr_lasso.pval[0]:.4f}"],
    "95% CI": ["—", "—", f"[{rf_ci[0]:.4f}, {rf_ci[1]:.4f}]", f"[{lasso_ci[0]:.4f}, {lasso_ci[1]:.4f}]"],
})
results_df.to_csv(TABLES_DIR / "tut_doubleml_results.csv", index=False)

Markdown(results_df.to_markdown(index=False))
```

The table summarizes all four approaches. The key finding is that the DoubleML estimates (around -0.07) are slightly smaller in magnitude than the naive OLS (-0.086), suggesting that naive OLS slightly overestimates the bonus effect. Both DoubleML methods produce statistically significant results (p < 0.05) with proper causal confidence intervals — a major advantage over naive OLS, which cannot claim causal validity.

## Discussion

This case study demonstrates a complete causal inference workflow using Double Machine Learning. The Pennsylvania Bonus experiment provides a clean setting to illustrate the key ideas:

1. **The bonus works:** Our DoubleML estimate suggests that offering a cash bonus reduces unemployment duration by approximately 7%, controlling for demographics and labor market conditions. This effect is statistically significant at the 5% level.

2. **Naive methods overestimate:** The naive OLS estimate (-0.086) is about 17% larger than the DoubleML estimate (-0.074), showing that even in a randomized experiment with good covariate balance, confounding adjustment matters.

3. **Results are robust:** Using two very different ML methods (Random Forest and Lasso) produces nearly identical causal estimates, strengthening our confidence in the finding.

4. **DoubleML provides valid inference:** Unlike prediction-focused ML, DoubleML produces standard errors, p-values, and confidence intervals that are asymptotically valid — meaning they can be used for hypothesis testing and policy decisions.

### Limitations

- The cross-fitting approach relies on the assumption that all confounders are observed. If there are important unobserved confounders (like worker motivation), the estimate may still be biased.
- The dataset is from the 1980s, and labor market conditions have changed substantially since then.
- With 5,099 observations, the confidence intervals are relatively wide, meaning the true effect could be anywhere from a 0.4% to a 14.3% reduction.

## Exercises

1. **Different treatment variable:** Replace `tg` with `dep1` (number of dependents) as the treatment variable. How does the DoubleML estimate differ from naive OLS? What does this tell you about confounding?

2. **Try the IRM model:** DoubleML also implements the Interactive Regression Model (`DoubleMLIRM`), which is designed for binary treatments. Replace `DoubleMLPLR` with `DoubleMLIRM` and compare the results. Hint: you will need to change the learner for `ml_m` to a classifier.

3. **Simulate known effects:** Use `doubleml.datasets.make_plr_CCDDHNR2018(alpha=0.5)` to generate synthetic data where the true causal effect is 0.5. Does DoubleML recover this known effect? How do the confidence intervals compare to the bonus dataset?

## References

1. [DoubleML — An Object-Oriented Implementation of Double Machine Learning in Python](https://docs.doubleml.org/stable/intro/intro.html)
2. [Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen, C., Newey, W., & Robins, J. (2018). Double/debiased machine learning for treatment and structural parameters. *The Econometrics Journal*, 21(1), C1–C68.](https://doi.org/10.1111/ectj.12097)
3. [scikit-learn — Random Forest Regressor Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)
4. [scikit-learn — LassoCV Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LassoCV.html)
