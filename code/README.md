# Code

Scripts for data processing and analysis.

## Supported Languages

- Python (`.py`)
- R (`.R`)
- Stata (`.do`)

## Guidelines

1. Use the project configuration at the top of each script:

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

2. Use descriptive filenames with numeric prefixes for ordering

3. Save outputs to `../output/`

4. Document the purpose of each script at the top

## Current Scripts

| Script | Description |
|--------|-------------|
| `01_eda.py` | Exploratory data analysis of IMDS and SDG indicators |
| `02_random_forest.py` | Random Forest prediction of IMDS from satellite embeddings |
| `03_rf_public_services.py` | Comparative analysis of 20 public service indicators |

## Running Scripts

Scripts should be run from the `code/` directory:

```bash
cd code
source ../claude4data/bin/activate
python 03_rf_public_services.py
```

## Data Sources

Scripts stream data directly from GitHub:

- Repository: [quarcs-lab/ds4bolivia](https://github.com/quarcs-lab/ds4bolivia)
- Satellite embeddings: 64 features (A00-A63) from 2017 imagery
- SDG variables: 20 public service indicators across 5 categories

## Output

Results are saved to `../output/`:

- `rf_public_services_results.csv` - Model performance metrics
- `rf_public_services_comparison.png` - Visualization of R² scores
- `rf_imds_satelliteEmbeddings2017.png` - IMDS prediction results
