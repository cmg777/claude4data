# Notebooks

Jupyter notebooks for data exploration and analysis.

## Available Notebooks

| Notebook | Kernel | Description |
| -------- | ------ | ----------- |
| 01_data_exploration.ipynb | Python | Data loading, merging, summary statistics, visualization |
| 01_data_exploration_R.ipynb | R | Same analysis using tidyverse and ggplot2 |

## Running Notebooks

### Python

```bash
source ../claude4data/bin/activate
jupyter notebook
```

Select the **claude4data (Python 3.10)** kernel.

### R

Select the **R (claude4data)** kernel in VS Code or Jupyter.

The R kernel uses packages from the project-local `renv/` environment.

## Creating New Notebooks

1. Use the project configuration for paths:

   **Python:**

   ```python
   import sys
   sys.path.insert(0, '..')
   from config import DATA_DIR, OUTPUT_DIR, set_seeds
   set_seeds()
   ```

   **R:**

   ```r
   source('../config.R')
   set_seeds()
   ```

2. Save figures and tables to `../output/`

3. Use descriptive filenames with numeric prefixes (e.g., `02_regression_analysis.ipynb`)
