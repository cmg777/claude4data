# Region Names

## Overview

This directory contains administrative metadata and identifiers for Bolivia's 339 municipalities. This dataset serves as the foundation for joining all other datasets in the repository.

## Files

### regionNames.csv

Contains official names and identifiers for all municipalities and their parent departments.

## Variable Dictionary

| Variable Name | Description |
|---------------|-------------|
| **poly_id** | Polygon ID: Sequential identifier for each polygon feature |
| **asdf_id** | ASDF ID: **Primary join key** - Unique spatial identifier used across all datasets in this repository |
| **mun** | Municipality name (Spanish) |
| **mun_id** | Official municipality ID code from national government registry |
| **dep** | Department (province) name - Bolivia has 9 departments |
| **dep_id** | Department ID code |
| **dep_mun** | Combined department-municipality identifier |
| **shapeID** | Municipality Geoquery polygon ID: External reference for spatial matching |

## Usage

This dataset is the starting point for any analysis. Use it to:
- **Add labels** to your spatial analysis results
- **Join datasets** using the `asdf_id` key
- **Filter** municipalities by department
- **Group** analysis by administrative regions

## Example Code

You can run the examples below in [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/notebooks/empty.ipynb)

```python
import pandas as pd

# Load region names
url = "https://raw.githubusercontent.com/quarcs-lab/ds4bolivia/master/regionNames/regionNames.csv"
df_names = pd.read_csv(url)

# Join with other datasets using asdf_id
df_sdg = pd.read_csv("../sdg/sdg.csv")
df_merged = pd.merge(df_names, df_sdg, on='asdf_id', how='inner')

# Filter by department
la_paz_municipalities = df_names[df_names['dep'] == 'La Paz']
```

## Key Notes

- There are **339 municipalities** in total
- The **asdf_id** is the primary key for all joins in this repository
- While `mun_id` is the official government code, `asdf_id` ensures consistency across satellite and optimized map files
