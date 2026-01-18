"""
01_eda.py - Exploratory Data Analysis

Basic exploration of Bolivia's sustainable development data.
"""

import sys
sys.path.insert(0, '..')
from config import DATA_DIR, OUTPUT_DIR, set_seeds

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set seeds for reproducibility
set_seeds()

# Load datasets
print("Loading data...")
regions = pd.read_csv(DATA_DIR / 'regionNames' / 'regionNames.csv')
sdg = pd.read_csv(DATA_DIR / 'sdg' / 'sdg.csv')
ntl = pd.read_csv(DATA_DIR / 'ntl' / 'ln_NTLpc.csv')

# Merge datasets
df = regions.merge(sdg, on='asdf_id').merge(ntl, on='asdf_id')
print(f"Combined dataset: {df.shape[0]} rows, {df.shape[1]} columns")

# Basic statistics
print("\n=== IMDS Summary ===")
print(df['imds'].describe())

# Top and bottom municipalities
print("\n=== Top 5 Municipalities by IMDS ===")
print(df.nlargest(5, 'imds')[['mun', 'dep', 'imds']])

print("\n=== Bottom 5 Municipalities by IMDS ===")
print(df.nsmallest(5, 'imds')[['mun', 'dep', 'imds']])

# Department summary
print("\n=== IMDS by Department ===")
dept_stats = df.groupby('dep')['imds'].agg(['mean', 'std', 'count'])
dept_stats = dept_stats.sort_values('mean', ascending=False).round(2)
print(dept_stats)

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 1. IMDS distribution
axes[0, 0].hist(df['imds'], bins=30, edgecolor='black', alpha=0.7)
axes[0, 0].set_xlabel('IMDS Score')
axes[0, 0].set_ylabel('Count')
axes[0, 0].set_title('Distribution of IMDS')

# 2. IMDS by department
df.boxplot(column='imds', by='dep', ax=axes[0, 1], rot=45)
axes[0, 1].set_xlabel('Department')
axes[0, 1].set_ylabel('IMDS Score')
axes[0, 1].set_title('IMDS by Department')
plt.suptitle('')

# 3. SDG indices correlation
sdg_cols = [col for col in df.columns if col.startswith('index_sdg')]
corr = df[sdg_cols].corr()
sns.heatmap(corr, ax=axes[1, 0], cmap='coolwarm', center=0, annot=False)
axes[1, 0].set_title('SDG Indices Correlation')

# 4. IMDS vs Night-time lights (2020)
axes[1, 1].scatter(df['ln_NTLpc2020'], df['imds'], alpha=0.5)
axes[1, 1].set_xlabel('Log Night-time Lights per Capita (2020)')
axes[1, 1].set_ylabel('IMDS Score')
axes[1, 1].set_title('IMDS vs Economic Activity')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'eda_overview.png', dpi=150, bbox_inches='tight')
print(f"\nFigure saved to: {OUTPUT_DIR / 'eda_overview.png'}")

plt.show()
