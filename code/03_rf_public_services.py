"""
03_rf_public_services.py - Comparative Analysis of Satellite Embeddings for Public Services

Evaluate how well satellite embeddings can predict various public service indicators.

Data is streamed directly from GitHub repository:
https://github.com/quarcs-lab/ds4bolivia
"""

import sys
sys.path.insert(0, '..')
from config import OUTPUT_DIR, RANDOM_SEED, set_seeds

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Set seeds for reproducibility
set_seeds()

# -----------------------------------------------------------------------------
# 1. Load data (streaming from GitHub)
# -----------------------------------------------------------------------------
GITHUB_RAW = "https://raw.githubusercontent.com/quarcs-lab/ds4bolivia/master"

print("Loading data from GitHub...")
sdg_vars = pd.read_csv(f"{GITHUB_RAW}/sdgVariables/sdgVariables.csv")
embeddings = pd.read_csv(f"{GITHUB_RAW}/satelliteEmbeddings/satelliteEmbeddings2017.csv")

df = sdg_vars.merge(embeddings, on='asdf_id')
print(f"Dataset: {df.shape[0]} rows, {df.shape[1]} columns")

# -----------------------------------------------------------------------------
# 2. Define target variables by category with descriptive names
# -----------------------------------------------------------------------------
TARGET_VARIABLES = {
    'Basic Utilities': [
        'sdg1_4_abs',   # Access to 3 basic services
        'sdg6_1_dwc',   # Drinking water coverage
        'sdg6_2_sc',    # Sanitation coverage
        'sdg6_3_wwt',   # Wastewater treatment
        'sdg7_1_ec',    # Electricity coverage
    ],
    'Education': [
        'sdg4_c_qti',   # Qualified teachers (initial)
        'sdg4_c_qts',   # Qualified teachers (secondary)
        'sdg9_5_cd',    # Kuaa computers delivered
        'sdg9_5_eutf',  # Educational units with tech floors
        'sdg4_1_ssdrm', # School dropout rate (male)
        'sdg4_1_ssdrf', # School dropout rate (female)
    ],
    'Health': [
        'sdg3_1_idca',  # Institutional childbirth coverage
        'sdg3_3_ti',    # Tuberculosis incidence
        'sdg3_3_hivi',  # HIV incidence
    ],
    'Infrastructure': [
        'sdg9_c_mnc',   # Network coverage
        'sdg11_2_samt', # Mass transit seats
        'sdg9_1_routes',# Railways/primary roads
    ],
    'Institutional': [
        'sdg16_6_pbec', # Budget execution capacity
        'sdg16_9_cr',   # Civil registry coverage
        'sdg17_5_pipc', # Public investment per capita
    ],
}

# Descriptive names for visualization
VARIABLE_LABELS = {
    'sdg1_4_abs': 'Access to 3 Basic Services',
    'sdg6_1_dwc': 'Drinking Water Coverage',
    'sdg6_2_sc': 'Sanitation Coverage',
    'sdg6_3_wwt': 'Wastewater Treatment',
    'sdg7_1_ec': 'Electricity Coverage',
    'sdg4_c_qti': 'Qualified Teachers (Initial)',
    'sdg4_c_qts': 'Qualified Teachers (Secondary)',
    'sdg9_5_cd': 'Computers Delivered',
    'sdg9_5_eutf': 'Schools with Tech Floors',
    'sdg4_1_ssdrm': 'School Dropout Rate (Male)',
    'sdg4_1_ssdrf': 'School Dropout Rate (Female)',
    'sdg3_1_idca': 'Institutional Childbirth',
    'sdg3_3_ti': 'Tuberculosis Incidence',
    'sdg3_3_hivi': 'HIV Incidence',
    'sdg9_c_mnc': 'Network Coverage',
    'sdg11_2_samt': 'Mass Transit Seats',
    'sdg9_1_routes': 'Roads/Railways',
    'sdg16_6_pbec': 'Budget Execution Capacity',
    'sdg16_9_cr': 'Civil Registry Coverage',
    'sdg17_5_pipc': 'Public Investment per Capita',
}

# Feature columns (satellite embeddings)
feature_cols = [col for col in df.columns if col.startswith('A')]
print(f"Features: {len(feature_cols)} satellite embeddings")

# -----------------------------------------------------------------------------
# 3. Train Random Forest for each target variable
# -----------------------------------------------------------------------------
print("\nTraining models for each public service indicator...")
print("=" * 70)

results = []

for category, variables in TARGET_VARIABLES.items():
    print(f"\n{category}:")
    print("-" * 40)

    for var in variables:
        if var not in df.columns:
            print(f"  {var}: NOT FOUND in dataset")
            continue

        # Prepare data (drop missing values)
        data = df[[var] + feature_cols].dropna()

        if len(data) < 50:
            print(f"  {var}: Insufficient data ({len(data)} samples)")
            continue

        X = data[feature_cols]
        y = data[var]

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_SEED
        )

        # Train Random Forest
        rf = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=RANDOM_SEED,
            n_jobs=-1
        )
        rf.fit(X_train, y_train)

        # Evaluate
        y_pred = rf.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)

        results.append({
            'Category': category,
            'Variable': var,
            'N_samples': len(data),
            'R2': r2,
            'RMSE': rmse,
            'MAE': mae
        })

        print(f"  {var}: R²={r2:.3f}, RMSE={rmse:.2f}, MAE={mae:.2f} (n={len(data)})")

# -----------------------------------------------------------------------------
# 4. Create summary DataFrame
# -----------------------------------------------------------------------------
results_df = pd.DataFrame(results)

print("\n" + "=" * 70)
print("SUMMARY: R² Performance by Category")
print("=" * 70)

summary = results_df.groupby('Category')['R2'].agg(['mean', 'std', 'min', 'max', 'count'])
summary = summary.round(3).sort_values('mean', ascending=False)
print(summary)

# -----------------------------------------------------------------------------
# 5. Visualizations
# -----------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Color mapping for categories
colors = {
    'Basic Utilities': '#2ecc71',
    'Education': '#3498db',
    'Health': '#e74c3c',
    'Infrastructure': '#9b59b6',
    'Institutional': '#f39c12'
}

# Plot 1: Bar chart of R² by variable (using descriptive names)
results_df_sorted = results_df.sort_values('R2', ascending=True)
results_df_sorted['Label'] = results_df_sorted['Variable'].map(VARIABLE_LABELS)
bar_colors = [colors[cat] for cat in results_df_sorted['Category']]

axes[0].barh(results_df_sorted['Label'], results_df_sorted['R2'], color=bar_colors)
axes[0].set_xlabel('R² Score')
axes[0].set_ylabel('Public Service Indicator')
axes[0].set_title('Random Forest Prediction Performance\n(Satellite Embeddings → Public Services)')
axes[0].axvline(x=0, color='black', linestyle='-', linewidth=0.5)
axes[0].set_xlim(-0.7, max(0.7, results_df_sorted['R2'].max() + 0.1))

# Plot 2: Category summary
cat_summary = results_df.groupby('Category')['R2'].mean().sort_values(ascending=True)
cat_colors = [colors[cat] for cat in cat_summary.index]

axes[1].barh(cat_summary.index, cat_summary.values, color=cat_colors)
axes[1].set_xlabel('Mean R² Score')
axes[1].set_ylabel('Category')
axes[1].set_title('Average Prediction Performance by Category')
axes[1].axvline(x=0, color='black', linestyle='-', linewidth=0.5)

# Add value labels
for i, (idx, val) in enumerate(cat_summary.items()):
    axes[1].text(val + 0.01, i, f'{val:.3f}', va='center', fontsize=10)

plt.tight_layout()

# Save figure
fig_path = OUTPUT_DIR / 'rf_public_services_comparison.png'
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
print(f"\nFigure saved to: {fig_path}")

# Save results to CSV (with description column, sorted by R² descending)
csv_path = OUTPUT_DIR / 'rf_public_services_results.csv'
results_df['Description'] = results_df['Variable'].map(VARIABLE_LABELS)
# Reorder columns to put Description after Variable and sort by R² descending
cols = ['Category', 'Variable', 'Description', 'N_samples', 'R2', 'RMSE', 'MAE']
results_df[cols].sort_values('R2', ascending=False).to_csv(csv_path, index=False)
print(f"Results saved to: {csv_path}")

# plt.show()  # Commented out for non-interactive execution

# -----------------------------------------------------------------------------
# 6. Print final insights
# -----------------------------------------------------------------------------
print("\n" + "=" * 70)
print("KEY INSIGHTS")
print("=" * 70)

best = results_df.loc[results_df['R2'].idxmax()]
worst = results_df.loc[results_df['R2'].idxmin()]

print(f"\nBest predicted variable:")
print(f"  {best['Variable']} ({best['Category']}): R² = {best['R2']:.3f}")

print(f"\nWorst predicted variable:")
print(f"  {worst['Variable']} ({worst['Category']}): R² = {worst['R2']:.3f}")

print(f"\nVariables with R² > 0.2 (moderate predictability):")
good_vars = results_df[results_df['R2'] > 0.2].sort_values('R2', ascending=False)
for _, row in good_vars.iterrows():
    print(f"  {row['Variable']}: R² = {row['R2']:.3f}")
