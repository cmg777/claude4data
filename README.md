# Claude for Data Science Workflows

A project demonstrating how to use Claude Code for data science tasks with Python and R.

## Project Overview

This repository contains data and analysis tools for exploring Bolivia's sustainable development indicators across 339 municipalities. It serves as a learning resource for using Claude Code in data science workflows.

**Key Analysis:** Predicting public service indicators from satellite imagery embeddings using Random Forest regression.

## Quick Start

### Python Environment

```bash
# Activate the virtual environment
source claude4data/bin/activate

# Run Jupyter
jupyter notebook

# Run analysis scripts
cd code && python 03_rf_public_services.py
```

### R Environment

The project uses `renv` for R package management. When opening R notebooks:

1. Select the **R (claude4data)** kernel in VS Code/Jupyter
2. The kernel automatically loads packages from the project-local `renv/` library

### Quarto Slides

```bash
# Render presentation slides
quarto render slides/rf_public_services_slides.qmd

# Preview with live reload
quarto preview slides/rf_public_services_slides.qmd
```

## Project Structure

```
claude4data/
├── data/                   # Datasets (DO NOT DELETE)
│   ├── regionNames/        # Municipality identifiers
│   ├── sdg/                # SDG index scores (0-100)
│   ├── sdgVariables/       # Detailed SDG indicators
│   ├── ntl/                # Night-time lights data (2012-2020)
│   └── rawData/            # Raw input data
├── notebooks/              # Jupyter notebooks (Python & R)
├── code/                   # Analysis scripts
│   ├── 01_eda.py           # Exploratory data analysis (Python)
│   ├── 01_eda.R            # Exploratory data analysis (R)
│   ├── 02_random_forest.py # IMDS prediction model
│   └── 03_rf_public_services.py  # Public services analysis
├── output/                 # Figures, tables, and results
├── slides/                 # Quarto presentations
├── log/                    # Session progress logs
├── legacy/                 # Original project snapshot
├── docs/                   # Documentation
├── templates/              # Document templates
├── claude4data/            # Python virtual environment
├── renv/                   # R package environment
├── config.py               # Python configuration
├── config.R                # R configuration
└── requirements.txt        # Python dependencies
```

## Data

All datasets contain 339 Bolivian municipalities and can be joined via `asdf_id`:

| Dataset | Description | Key Variables |
|---------|-------------|---------------|
| regionNames | Administrative metadata | mun, dep, mun_id, dep_id |
| sdg | SDG composite indices | imds, index_sdg1-17 |
| sdgVariables | Detailed SDG indicators | 64 variables |
| ntl | Night-time lights | ln_NTLpc2012-2020 |
| satelliteEmbeddings | 64-dim image features | A00-A63 (from 2017 imagery) |

Data is streamed from: [github.com/quarcs-lab/ds4bolivia](https://github.com/quarcs-lab/ds4bolivia)

## Key Findings

The Random Forest analysis reveals which public service indicators can be predicted from satellite imagery:

**Best Predicted (R² > 0.30):**

| Indicator | R² | Category |
|-----------|------|----------|
| Institutional Childbirth | 0.579 | Health |
| Access to 3 Basic Services | 0.499 | Basic Utilities |
| Tuberculosis Incidence | 0.368 | Health |
| Civil Registry Coverage | 0.362 | Institutional |
| Electricity Coverage | 0.361 | Basic Utilities |

**Poorly Predicted (R² < 0):**

| Indicator | R² | Category |
|-----------|--------|----------|
| School Dropout (Female) | -0.588 | Education |
| Mass Transit Seats | -0.346 | Infrastructure |

## Presentations

| Slides                                   | Description                                    |
|------------------------------------------|------------------------------------------------|
| `slides/eda_slides.html`                 | Exploratory Data Analysis results              |
| `slides/rf_public_services_slides.html`  | Random Forest prediction results               |
| `slides/esda_slides.html`                | Spatial autocorrelation analysis (ESDA)        |

## Configuration

Both Python and R use configuration files that provide:
- Reproducibility settings (`RANDOM_SEED = 42`)
- Absolute paths that work from any subdirectory
- Directory constants: `DATA_DIR`, `OUTPUT_DIR`, `NOTEBOOKS_DIR`, etc.

### Python
```python
from config import DATA_DIR, OUTPUT_DIR, set_seeds
set_seeds()
```

### R
```r
source('../config.R')  # or source('config.R') from project root
set_seeds()
```

## Available Jupyter Kernels

| Kernel | Name | Description |
|--------|------|-------------|
| Python | claude4data | Python 3.10 with data science packages |
| R | R (claude4data) | R 4.5.2 with tidyverse, using renv |

## Author

Carlos Mendez

## Notes

- See `CLAUDE.md` for AI assistant instructions
- Progress logs are stored in `log/`
- Never delete data or program files
