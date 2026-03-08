---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.19.1
kernelspec:
  display_name: R
  language: R
  name: ir
---

---
title: "N2: Sample R Analysis"
---

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cmg777/claude4data/blob/master/notebooks/notebook-02.ipynb)

## Overview

This notebook demonstrates the R workflow: generating synthetic data,
producing labeled figures and tables that are embedded in the manuscript via
Quarto's `{{< embed >}}` shortcode. It mirrors `notebook-01` (Python) using R.

```{code-cell} r
# Reproducibility setup
if (dir.exists("/content")) {
    system("git clone --depth 1 https://github.com/cmg777/claude4data.git /content/claude4data 2>/dev/null || true")
    setwd("/content/claude4data/notebooks")
}
source("../config.R")
set_seeds()
```

```{code-cell} r
library(ggplot2)
```

```{code-cell} r
#| label: fig-r-sample
#| fig-cap: "Synthetic regional indicators: GDP per capita vs. life expectancy across 80 simulated regions, colored by geographic cluster (R)."

# Generate synthetic cross-sectional data
set.seed(RANDOM_SEED)
n <- 80
region <- rep(c("North", "South", "East", "West"), each = n / 4)

gdp <- exp(rnorm(n, mean = 9.5, sd = 0.6))
life_exp <- 50 + 15 * (1 - exp(-gdp / 20000)) + rnorm(n, mean = 0, sd = 2)

df <- data.frame(
  gdp_per_capita = gdp,
  life_expectancy = life_exp,
  region = region
)

ggplot(df, aes(x = gdp_per_capita, y = life_expectancy, color = region, shape = region)) +
  geom_point(size = 2.5, alpha = 0.8) +
  labs(
    title = "GDP per Capita vs. Life Expectancy",
    x = "GDP per capita (USD)",
    y = "Life expectancy (years)",
    color = "Region",
    shape = "Region"
  ) +
  theme_minimal()
```

```{code-cell} r
#| label: tbl-r-summary
#| tbl-cap: "Summary statistics by region (R)."

summary_df <- aggregate(
  cbind(gdp_per_capita, life_expectancy) ~ region,
  data = df,
  FUN = function(x) c(mean = mean(x), sd = sd(x), n = length(x))
)

# Flatten the matrix columns
result <- data.frame(
  Region = summary_df$region,
  "GDP mean" = round(summary_df$gdp_per_capita[, "mean"], 1),
  "GDP sd" = round(summary_df$gdp_per_capita[, "sd"], 1),
  "Life exp. mean" = round(summary_df$life_expectancy[, "mean"], 1),
  "Life exp. sd" = round(summary_df$life_expectancy[, "sd"], 1),
  N = summary_df$gdp_per_capita[, "n"],
  check.names = FALSE
)

knitr::kable(result)
```
