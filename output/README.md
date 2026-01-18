# Output

This directory contains figures, tables, and results generated from analysis scripts and notebooks.

## Current Files

### Figures

| File                                  | Source                       | Description                                      |
|---------------------------------------|------------------------------|--------------------------------------------------|
| `eda_overview.png`                    | `01_eda.py`                  | EDA overview (Python)                            |
| `eda_overview_R.png`                  | `01_eda.R`                   | EDA overview with 4-panel visualization (R)      |
| `rf_imds_satelliteEmbeddings2017.png` | `02_random_forest.py`        | IMDS prediction: actual vs predicted             |
| `rf_public_services_comparison.png`   | `03_rf_public_services.py`   | R² scores for 20 public service indicators       |
| `imds_distribution.png`               | `01_data_exploration.ipynb`  | IMDS histogram and boxplot (Python notebook)     |
| `imds_distribution_R.png`             | `01_data_exploration_R.ipynb`| IMDS histogram and boxplot (R notebook)          |

### Data Files

| File                             | Source                     | Description                                         |
|----------------------------------|----------------------------|-----------------------------------------------------|
| `rf_public_services_results.csv` | `03_rf_public_services.py` | Model results with R², RMSE, MAE for each indicator |
| `imds_by_department.csv`         | `01_eda.R`                 | IMDS statistics by department                       |
| `public_services_summary.csv`    | `01_eda.R`                 | Public service indicators summary                   |

## Key Results

The `rf_public_services_results.csv` contains prediction performance for 20 public service indicators sorted by R²:

- **Best predicted:** Institutional Childbirth (R² = 0.579)
- **Worst predicted:** School Dropout Female (R² = -0.588)
- **6 indicators** with R² > 0.30 (moderate-good predictability)

## Guidelines

- Only save final figures and tables here (not intermediate outputs)
- Use descriptive filenames
- Reference the source script/notebook in this README when adding new files
