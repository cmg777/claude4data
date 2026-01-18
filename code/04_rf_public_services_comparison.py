"""
04_rf_public_services_comparison.py - Comparative Analysis of Regular vs Population-Weighted Embeddings

Compare prediction performance between:
- Regular satellite embeddings
- Population-weighted satellite embeddings

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
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set seeds for reproducibility
set_seeds()

# -----------------------------------------------------------------------------
# 1. Load data (streaming from GitHub)
# -----------------------------------------------------------------------------
GITHUB_RAW = "https://raw.githubusercontent.com/quarcs-lab/ds4bolivia/master"

print("Loading data from GitHub...")
print("=" * 70)
sdg_vars = pd.read_csv(f"{GITHUB_RAW}/sdgVariables/sdgVariables.csv")
embeddings_regular = pd.read_csv(f"{GITHUB_RAW}/satelliteEmbeddings/satelliteEmbeddings2017.csv")
embeddings_popweighted = pd.read_csv(f"{GITHUB_RAW}/satelliteEmbeddings/satelliteEmbeddings2017popWeighted.csv")

print(f"SDG Variables: {sdg_vars.shape[0]} municipalities, {sdg_vars.shape[1]} variables")
print(f"Regular embeddings: {embeddings_regular.shape[0]} municipalities, {embeddings_regular.shape[1]} features")
print(f"Pop-weighted embeddings: {embeddings_popweighted.shape[0]} municipalities, {embeddings_popweighted.shape[1]} features")

# Create datasets for both embedding types
df_regular = sdg_vars.merge(embeddings_regular, on='asdf_id')
df_popweighted = sdg_vars.merge(embeddings_popweighted, on='asdf_id')

print(f"\nMerged dataset (regular): {df_regular.shape[0]} rows, {df_regular.shape[1]} columns")
print(f"Merged dataset (pop-weighted): {df_popweighted.shape[0]} rows, {df_popweighted.shape[1]} columns")

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
feature_cols = [col for col in df_regular.columns if col.startswith('A')]
print(f"\nFeatures: {len(feature_cols)} satellite embeddings")

# -----------------------------------------------------------------------------
# 3. Train Random Forest for each target variable (BOTH embedding types)
# -----------------------------------------------------------------------------
print("\n" + "=" * 70)
print("TRAINING MODELS FOR BOTH EMBEDDING TYPES")
print("=" * 70)

def train_and_evaluate(df, var, feature_cols, embedding_type):
    """Train RF model and return performance metrics"""

    if var not in df.columns:
        return None

    # Prepare data (drop missing values)
    data = df[[var] + feature_cols].dropna()

    if len(data) < 50:
        return None

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

    return {
        'Variable': var,
        'Embedding_Type': embedding_type,
        'N_samples': len(data),
        'R2': r2,
        'RMSE': rmse,
        'MAE': mae
    }

results = []

for category, variables in TARGET_VARIABLES.items():
    print(f"\n{category}:")
    print("-" * 70)

    for var in variables:
        # Train with regular embeddings
        result_reg = train_and_evaluate(df_regular, var, feature_cols, 'Regular')
        if result_reg:
            result_reg['Category'] = category
            results.append(result_reg)

        # Train with population-weighted embeddings
        result_pop = train_and_evaluate(df_popweighted, var, feature_cols, 'Pop-weighted')
        if result_pop:
            result_pop['Category'] = category
            results.append(result_pop)

        # Print comparison
        if result_reg and result_pop:
            r2_reg = result_reg['R2']
            r2_pop = result_pop['R2']
            diff = r2_pop - r2_reg
            better = "Pop-weighted" if diff > 0 else "Regular"
            symbol = "✓" if diff > 0 else "✗"
            print(f"  {var}:")
            print(f"    Regular:       R²={r2_reg:.3f}")
            print(f"    Pop-weighted:  R²={r2_pop:.3f} (Δ={diff:+.3f}) {symbol} {better}")
        elif result_reg:
            print(f"  {var}: Regular only - R²={result_reg['R2']:.3f}")
        elif result_pop:
            print(f"  {var}: Pop-weighted only - R²={result_pop['R2']:.3f}")
        else:
            print(f"  {var}: Insufficient data")

# -----------------------------------------------------------------------------
# 4. Create summary DataFrames
# -----------------------------------------------------------------------------
results_df = pd.DataFrame(results)
results_df['Description'] = results_df['Variable'].map(VARIABLE_LABELS)

# Pivot to compare side-by-side
comparison_df = results_df.pivot_table(
    index=['Category', 'Variable', 'Description'],
    columns='Embedding_Type',
    values='R2'
).reset_index()

comparison_df.columns.name = None
comparison_df['Difference'] = comparison_df['Pop-weighted'] - comparison_df['Regular']
comparison_df['Better_Method'] = comparison_df['Difference'].apply(
    lambda x: 'Pop-weighted' if x > 0 else ('Regular' if x < 0 else 'Tie')
)

print("\n" + "=" * 70)
print("COMPARATIVE SUMMARY: R² Performance")
print("=" * 70)
print(comparison_df[['Description', 'Regular', 'Pop-weighted', 'Difference', 'Better_Method']].to_string(index=False))

# Category-level summary
print("\n" + "=" * 70)
print("CATEGORY SUMMARY")
print("=" * 70)

cat_summary = results_df.groupby(['Category', 'Embedding_Type'])['R2'].mean().unstack()
cat_summary['Difference'] = cat_summary['Pop-weighted'] - cat_summary['Regular']
cat_summary = cat_summary.sort_values('Difference', ascending=False)
print(cat_summary.round(3))

# Overall summary
print("\n" + "=" * 70)
print("OVERALL SUMMARY")
print("=" * 70)
overall = results_df.groupby('Embedding_Type')['R2'].agg(['mean', 'std', 'min', 'max', 'count'])
print(overall.round(3))

# Count wins
wins = comparison_df['Better_Method'].value_counts()
print(f"\nMethod Performance:")
print(f"  Pop-weighted better: {wins.get('Pop-weighted', 0)} variables")
print(f"  Regular better:      {wins.get('Regular', 0)} variables")
print(f"  Ties:                {wins.get('Tie', 0)} variables")

# -----------------------------------------------------------------------------
# 5. Visualizations
# -----------------------------------------------------------------------------
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

# Color mapping for categories
colors = {
    'Basic Utilities': '#2ecc71',
    'Education': '#3498db',
    'Health': '#e74c3c',
    'Infrastructure': '#9b59b6',
    'Institutional': '#f39c12'
}

# Plot 1: Comparison of R² by variable (sorted by difference)
ax1 = fig.add_subplot(gs[0:2, 0])
comparison_sorted = comparison_df.sort_values('Difference', ascending=True)
x_pos = np.arange(len(comparison_sorted))
width = 0.35

bar_colors_cat = [colors[cat] for cat in comparison_sorted['Category']]

ax1.barh(x_pos - width/2, comparison_sorted['Regular'], width,
         label='Regular', color='lightgray', edgecolor='black', linewidth=0.5)
ax1.barh(x_pos + width/2, comparison_sorted['Pop-weighted'], width,
         label='Pop-weighted', color=bar_colors_cat, edgecolor='black', linewidth=0.5)

ax1.set_yticks(x_pos)
ax1.set_yticklabels(comparison_sorted['Description'], fontsize=9)
ax1.set_xlabel('R² Score', fontsize=11)
ax1.set_title('Prediction Performance Comparison\n(Regular vs. Pop-weighted Embeddings)',
              fontsize=12, fontweight='bold')
ax1.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
ax1.legend(loc='lower right')
ax1.grid(axis='x', alpha=0.3)

# Plot 2: Difference (Pop-weighted - Regular)
ax2 = fig.add_subplot(gs[0:2, 1])
diff_colors = ['green' if d > 0 else 'red' for d in comparison_sorted['Difference']]
ax2.barh(x_pos, comparison_sorted['Difference'], color=diff_colors, alpha=0.6, edgecolor='black', linewidth=0.5)
ax2.set_yticks(x_pos)
ax2.set_yticklabels(comparison_sorted['Description'], fontsize=9)
ax2.set_xlabel('Difference in R² (Pop-weighted - Regular)', fontsize=11)
ax2.set_title('Performance Improvement with Pop-weighted Embeddings\n(Green = Better, Red = Worse)',
              fontsize=12, fontweight='bold')
ax2.axvline(x=0, color='black', linestyle='-', linewidth=1)
ax2.grid(axis='x', alpha=0.3)

# Plot 3: Category summary
ax3 = fig.add_subplot(gs[2, 0])
cat_summary_sorted = cat_summary.sort_values('Difference', ascending=True)
x_cat = np.arange(len(cat_summary_sorted))
cat_colors_list = [colors[cat] for cat in cat_summary_sorted.index]

ax3.barh(x_cat - width/2, cat_summary_sorted['Regular'], width,
         label='Regular', color='lightgray', edgecolor='black', linewidth=0.5)
ax3.barh(x_cat + width/2, cat_summary_sorted['Pop-weighted'], width,
         label='Pop-weighted', color=cat_colors_list, edgecolor='black', linewidth=0.5)

ax3.set_yticks(x_cat)
ax3.set_yticklabels(cat_summary_sorted.index, fontsize=10)
ax3.set_xlabel('Mean R² Score', fontsize=11)
ax3.set_title('Average Performance by Category', fontsize=12, fontweight='bold')
ax3.legend()
ax3.grid(axis='x', alpha=0.3)

# Plot 4: Overall distribution
ax4 = fig.add_subplot(gs[2, 1])
regular_r2 = results_df[results_df['Embedding_Type'] == 'Regular']['R2']
popw_r2 = results_df[results_df['Embedding_Type'] == 'Pop-weighted']['R2']

ax4.hist([regular_r2, popw_r2], bins=15, label=['Regular', 'Pop-weighted'],
         color=['lightgray', 'steelblue'], edgecolor='black', alpha=0.7)
ax4.set_xlabel('R² Score', fontsize=11)
ax4.set_ylabel('Frequency', fontsize=11)
ax4.set_title('Distribution of R² Scores', fontsize=12, fontweight='bold')
ax4.axvline(regular_r2.mean(), color='gray', linestyle='--', linewidth=2, label=f'Regular mean: {regular_r2.mean():.3f}')
ax4.axvline(popw_r2.mean(), color='steelblue', linestyle='--', linewidth=2, label=f'Pop-weighted mean: {popw_r2.mean():.3f}')
ax4.legend()
ax4.grid(axis='y', alpha=0.3)

# Save figure
fig_path = OUTPUT_DIR / 'rf_embeddings_comparison.png'
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
print(f"\nFigure saved to: {fig_path}")

# Save results to CSV
csv_path = OUTPUT_DIR / 'rf_embeddings_comparison.csv'
comparison_df.to_csv(csv_path, index=False)
print(f"Comparison results saved to: {csv_path}")

# Save detailed results
csv_detailed_path = OUTPUT_DIR / 'rf_embeddings_comparison_detailed.csv'
cols = ['Category', 'Variable', 'Description', 'Embedding_Type', 'N_samples', 'R2', 'RMSE', 'MAE']
results_df[cols].sort_values(['Variable', 'Embedding_Type']).to_csv(csv_detailed_path, index=False)
print(f"Detailed results saved to: {csv_detailed_path}")

# plt.show()  # Commented out for non-interactive execution

# -----------------------------------------------------------------------------
# 6. Print final insights
# -----------------------------------------------------------------------------
print("\n" + "=" * 70)
print("KEY INSIGHTS")
print("=" * 70)

# Best performing variables for each method
print("\nBest predicted variables (Regular embeddings):")
best_reg = results_df[results_df['Embedding_Type'] == 'Regular'].nlargest(3, 'R2')
for _, row in best_reg.iterrows():
    print(f"  {row['Description']}: R² = {row['R2']:.3f}")

print("\nBest predicted variables (Pop-weighted embeddings):")
best_pop = results_df[results_df['Embedding_Type'] == 'Pop-weighted'].nlargest(3, 'R2')
for _, row in best_pop.iterrows():
    print(f"  {row['Description']}: R² = {row['R2']:.3f}")

# Biggest improvements
print("\nBiggest improvements with pop-weighted embeddings:")
top_improvements = comparison_df.nlargest(5, 'Difference')
for _, row in top_improvements.iterrows():
    print(f"  {row['Description']}: Δ = {row['Difference']:+.3f} (Regular: {row['Regular']:.3f} → Pop-weighted: {row['Pop-weighted']:.3f})")

# Biggest decreases
print("\nBiggest decreases with pop-weighted embeddings:")
top_decreases = comparison_df.nsmallest(5, 'Difference')
for _, row in top_decreases.iterrows():
    print(f"  {row['Description']}: Δ = {row['Difference']:+.3f} (Regular: {row['Regular']:.3f} → Pop-weighted: {row['Pop-weighted']:.3f})")

# Statistical test
reg_values = comparison_df['Regular'].values
pop_values = comparison_df['Pop-weighted'].values
t_stat, p_value = stats.ttest_rel(pop_values, reg_values)

print(f"\nPaired t-test (Pop-weighted vs Regular):")
print(f"  Mean difference: {(pop_values - reg_values).mean():.4f}")
print(f"  t-statistic: {t_stat:.3f}")
print(f"  p-value: {p_value:.4f}")
if p_value < 0.05:
    if t_stat > 0:
        conclusion = "Pop-weighted significantly better (p < 0.05)"
    else:
        conclusion = "Regular significantly better (p < 0.05)"
else:
    conclusion = "No significant difference (p >= 0.05)"
print(f"  Conclusion: {conclusion}")

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)
