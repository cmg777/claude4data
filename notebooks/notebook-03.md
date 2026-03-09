---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.19.1
kernelspec:
  display_name: Stata (nbstata)
  language: stata
  name: nbstata
---

---
title: "N3: Sample Stata Analysis"
toc: true
toc-depth: 3
---

<a href="https://colab.research.google.com/github/cmg777/claude4data/blob/master/notebooks/notebook-03.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>

## Overview

This notebook demonstrates the Stata workflow: generating synthetic data,
producing labeled figures and tables that are embedded in the manuscript via
Quarto's `{{< embed >}}` shortcode. It mirrors `notebook-01` (Python) using Stata.

**Prerequisites:** Stata must be installed and the `nbstata` Jupyter kernel
must be registered. See `notebooks/README.md` for setup instructions.

```{code-cell}
* Reproducibility setup
clear all
set seed 42
```

```{code-cell}
* Generate synthetic cross-sectional data (80 regions, 4 clusters)
quietly {
    set obs 80
    gen region_id = _n
    gen region = "North" if region_id <= 20
    replace region = "South" if region_id > 20 & region_id <= 40
    replace region = "East"  if region_id > 40 & region_id <= 60
    replace region = "West"  if region_id > 60

    gen log_gdp = rnormal(9.5, 0.6)
    gen gdp_per_capita = exp(log_gdp)
    gen life_expectancy = 50 + 15 * (1 - exp(-gdp_per_capita / 20000)) + rnormal(0, 2)
}
describe
```

```{code-cell}
*| label: fig-stata-sample
*| fig-cap: "Synthetic regional indicators: GDP per capita vs. life expectancy across 80 simulated regions, colored by geographic cluster (Stata)."

twoway ///
    (scatter life_expectancy gdp_per_capita if region == "North", mcolor(blue)    msymbol(circle)) ///
    (scatter life_expectancy gdp_per_capita if region == "South", mcolor(red)     msymbol(diamond)) ///
    (scatter life_expectancy gdp_per_capita if region == "East",  mcolor(green)   msymbol(triangle)) ///
    (scatter life_expectancy gdp_per_capita if region == "West",  mcolor(orange)  msymbol(square)), ///
    title("GDP per Capita vs. Life Expectancy") ///
    xtitle("GDP per capita (USD)") ///
    ytitle("Life expectancy (years)") ///
    legend(order(1 "North" 2 "South" 3 "East" 4 "West") title("Region"))
```

```{code-cell}
*| label: stata-summary

tabstat gdp_per_capita life_expectancy, by(region) stat(mean sd count) format(%9.1f)
```
