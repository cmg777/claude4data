"""
02_random_forest.py - Random Forest Prediction

Predict the Municipal Sustainable Development Index (IMDS)
using satellite embeddings as features.

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

# Set seeds for reproducibility
set_seeds()

# -----------------------------------------------------------------------------
# 1. Load and merge data (streaming from GitHub)
# -----------------------------------------------------------------------------
GITHUB_RAW = "https://raw.githubusercontent.com/quarcs-lab/ds4bolivia/master"

print("Loading data from GitHub...")
sdg = pd.read_csv(f"{GITHUB_RAW}/sdg/sdg.csv")
embeddings = pd.read_csv(f"{GITHUB_RAW}/satelliteEmbeddings/satelliteEmbeddings2017.csv")

df = sdg.merge(embeddings, on='asdf_id')
print(f"Dataset: {df.shape[0]} rows, {df.shape[1]} columns")

# -----------------------------------------------------------------------------
# 2. Prepare features and target
# -----------------------------------------------------------------------------
feature_cols = [col for col in df.columns if col.startswith('A')]
X = df[feature_cols]
y = df['imds']

print(f"Features: {len(feature_cols)} satellite embeddings (A00-A63)")
print(f"Target: imds (mean={y.mean():.2f}, std={y.std():.2f})")

# -----------------------------------------------------------------------------
# 3. Train/test split
# -----------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_SEED
)
print(f"\nTrain set: {len(X_train)} samples")
print(f"Test set: {len(X_test)} samples")

# -----------------------------------------------------------------------------
# 4. Train Random Forest model
# -----------------------------------------------------------------------------
print("\nTraining Random Forest...")
rf = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    random_state=RANDOM_SEED,
    n_jobs=-1
)
rf.fit(X_train, y_train)

# -----------------------------------------------------------------------------
# 5. Evaluate performance
# -----------------------------------------------------------------------------
y_pred_train = rf.predict(X_train)
y_pred_test = rf.predict(X_test)

print("\n=== Model Performance ===")
print(f"{'Metric':<15} {'Train':>10} {'Test':>10}")
print("-" * 37)
print(f"{'R²':<15} {r2_score(y_train, y_pred_train):>10.3f} {r2_score(y_test, y_pred_test):>10.3f}")
print(f"{'RMSE':<15} {np.sqrt(mean_squared_error(y_train, y_pred_train)):>10.3f} {np.sqrt(mean_squared_error(y_test, y_pred_test)):>10.3f}")
print(f"{'MAE':<15} {mean_absolute_error(y_train, y_pred_train):>10.3f} {mean_absolute_error(y_test, y_pred_test):>10.3f}")

# -----------------------------------------------------------------------------
# 6. Feature importance
# -----------------------------------------------------------------------------
importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

print("\n=== Top 10 Most Important Features ===")
print(importance.head(10).to_string(index=False))

# -----------------------------------------------------------------------------
# 7. Visualizations
# -----------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Actual vs Predicted
axes[0].scatter(y_test, y_pred_test, alpha=0.6, edgecolors='black', linewidth=0.5)
axes[0].plot([y.min(), y.max()], [y.min(), y.max()], 'r--', linewidth=2)
axes[0].set_xlabel('Actual IMDS')
axes[0].set_ylabel('Predicted IMDS')
axes[0].set_title(f'Actual vs Predicted (Test R² = {r2_score(y_test, y_pred_test):.3f})')

# Feature importance (top 15)
top_features = importance.head(15)
axes[1].barh(top_features['feature'], top_features['importance'])
axes[1].set_xlabel('Importance')
axes[1].set_ylabel('Feature')
axes[1].set_title('Top 15 Feature Importances')
axes[1].invert_yaxis()

plt.tight_layout()
output_path = OUTPUT_DIR / 'rf_imds_satelliteEmbeddings2017.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\nFigure saved to: {output_path}")

plt.show()
