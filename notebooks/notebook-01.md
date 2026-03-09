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
title: "N1: Sample Data Exploration"
toc: true
toc-depth: 3
---

<a href="https://colab.research.google.com/github/cmg777/claude4data/blob/master/notebooks/notebook-01.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>

## Overview

This notebook demonstrates the template workflow: generating synthetic data,
producing labeled figures and tables that are embedded in the manuscript via
Quarto's `{{< embed >}}` shortcode.

```{code-cell} ipython3
import sys
if "google.colab" in sys.modules:
    !git clone --depth 1 https://github.com/cmg777/claude4data.git /content/claude4data 2>/dev/null || true
    %cd /content/claude4data/notebooks
sys.path.insert(0, "..")
from config import set_seeds, DATA_DIR

set_seeds()
```

```{code-cell} ipython3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")
```

```{code-cell} ipython3
#| label: fig-sample
#| fig-cap: "Synthetic regional indicators: GDP per capita vs. life expectancy across 80 simulated regions, colored by geographic cluster."

# Generate synthetic cross-sectional data
rng = np.random.default_rng(42)
n = 80
clusters = np.repeat(["North", "South", "East", "West"], n // 4)

gdp = rng.lognormal(mean=9.5, sigma=0.6, size=n)
life_exp = 50 + 15 * (1 - np.exp(-gdp / 20000)) + rng.normal(0, 2, n)

df = pd.DataFrame({
    "GDP per capita (USD)": gdp,
    "Life expectancy (years)": life_exp,
    "Region": clusters,
})

fig, ax = plt.subplots(figsize=(8, 5))
sns.scatterplot(
    data=df,
    x="GDP per capita (USD)",
    y="Life expectancy (years)",
    hue="Region",
    style="Region",
    s=60,
    alpha=0.8,
    ax=ax,
)
ax.set_title("GDP per Capita vs. Life Expectancy")
ax.legend(title="Region", frameon=True)
plt.tight_layout()
plt.show()
```

```{code-cell} ipython3
#| label: tbl-summary
#| tbl-cap: "Summary statistics by region."

summary = (
    df.groupby("Region")[["GDP per capita (USD)", "Life expectancy (years)"]]
    .agg(["mean", "std", "count"])
    .round(1)
)
summary.columns = [f"{col} ({stat})" for col, stat in summary.columns]
from IPython.display import Markdown
Markdown(summary.to_markdown())
```
