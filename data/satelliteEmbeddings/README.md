# README: Bolivia Satellite Embeddings (2017)

**Dataset Title:** Aggregated Google Satellite Embeddings for Bolivia Administrative Units (2017)  
**Date Generated:** October 2023  
**Source Platform:** Google Earth Engine (GEE)  
**Source Collection:** `GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL`  

## 1. Context & Overview

This dataset contains spatially aggregated "embedding" vectors derived from high-resolution satellite imagery for administrative units in Bolivia for the year 2017.

### Files

**satelliteEmbeddings2017.csv** - Contains 65 columns: one identifier (asdf_id) and 64 embedding dimensions (A00-A63) for all 339 municipalities.

### What are Satellite Embeddings?
Unlike traditional satellite data that provides physical values (e.g., surface reflectance, temperature), these embeddings are the output of a deep learning model (a self-supervised Convolutional Neural Network). 

* **The Input:** The model processes raw Sentinel-2 and Landsat imagery.
* **The Process:** It learns to compress visual information (texture, shapes, colors, road networks, building density) into a compact numerical representation.
* **The Output:** A 64-dimensional vector (an array of 64 numbers) that uniquely characterizes the landscape of a specific location.

### Why use this data?
These 64 variables capture complex spatial patterns that simple indices like NDVI (vegetation) or NTL (nightlights) miss. They are effective for:
* **Poverty Mapping:** Predicting economic indicators based on the built-up environment.
* **Population Estimation:** Correlating texture with population density.
* **Land Use Classification:** Distinguishing between different types of agriculture or urban zoning without manual labeling.

---

## 2. Methodology

1.  **Source Data:** Google Satellite Embeddings V1 (Annual Mosaic 2017).
2.  **Resolution:** The native embeddings were computed at approximately **10m resolution**.
3.  **Aggregation:** We calculated the **spatial mean** of each embedding band for every polygon in the Bolivia administrative boundaries dataset.
4.  **Spatial Scope:** 339 Administrative Units (Municipalities) in Bolivia.

---

## 3. Variable Dictionary

The dataset contains **65 columns**.

| Variable Name | Type | Description |
| :--- | :--- | :--- |
| **asdf_id** | String/Int | **Unique Spatial Identifier.** Matches the ID from the source shapefile (`bolivia339geoqueryFull`). Use this key to join this data back to your original shapefile or socio-economic datasets. |
| **A00** | Float | **Embedding Dimension 0.** An abstract feature learned by the neural network. |
| **A01** | Float | **Embedding Dimension 1.** An abstract feature learned by the neural network. |
| **...** | ... | ... |
| **A63** | Float | **Embedding Dimension 63.** An abstract feature learned by the neural network. |

### Interpreting the "A" Variables
* **Abstract Nature:** You cannot intuitively say "A05 is vegetation" or "A12 is water." These are latent variables representing high-level visual features.
* **Usage:** They are designed to be used as **features (X)** in machine learning models (e.g., Ridge Regression, Random Forest).
    * *Example Model:* $GDP_{capita} = f(A00, A01, ..., A63)$
* **Similarity:** Locations with similar vectors (Euclidean distance) look similar from space.

---

## 4. Usage Notes

* **Data Completeness:** Always check for `null` values. If a polygon is extremely small or falls outside the satellite coverage (e.g., heavy cloud masking), it may not have a computed mean.
* **Scale:** The values typically range between **-1.0 and 1.0** (standardized output from the neural network), though the mean aggregation may shrink this range.
* **Dimensionality Reduction:** Because 64 variables can be a lot for simple regressions, it is common practice to perform **PCA (Principal Component Analysis)** on columns A00-A63 to reduce them to the top 3-5 "Principal Components" before analysis.

---

## 5. Example Code

You can run the examples below in [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/notebooks/empty.ipynb)

```python
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor

# Load satellite embeddings
url = "https://raw.githubusercontent.com/quarcs-lab/ds4bolivia/master/satelliteEmbeddings/satelliteEmbeddings2017.csv"
df_emb = pd.read_csv(url)

# Select the 64 embedding dimensions
embedding_cols = [f'A{str(i).zfill(2)}' for i in range(64)]
X = df_emb[embedding_cols]

# Option 1: Use all 64 dimensions directly
# Load SDG target variable
df_sdg = pd.read_csv("../sdg/sdg.csv")
df_merged = df_emb.merge(df_sdg, on='asdf_id')
y = df_merged['index_sdg1']  # Poverty index

# Train a model
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X, y)

# Option 2: Dimensionality reduction with PCA
pca = PCA(n_components=5)
X_pca = pca.fit_transform(X)
print(f"Variance explained by 5 components: {pca.explained_variance_ratio_.sum():.2%}")
```

---

## 6. Citation

If using this data, please cite the source model:

> Google Earth Engine. (2020). *Satellite Embeddings V1*. Available at: https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_SATELLITE_EMBEDDING_V1_ANNUAL

## 7. Google Earth Engine Processing Code

```
/**
 * AGGREGATE SATELLITE EMBEDDINGS TO ADM BOUNDARIES
 * ---------------------------------------------------------------------
 * Objective: Compute 64-dimensional mean embeddings for administrative units.
 *
 * 1. Performance: Removed .clip() to prevent timeouts.
 * 2. Logic: Generated band names client-side to fix "Invalid value (selectors)" error.
 * 3. Best Practice: Strict column selection to minimize memory usage.
 */

// --- 1. DATA INPUTS & CLEANING ---

// Administrative Boundaries
// We strictly select 'asdf_id' to drop heavy attributes we don't need.
var admBoundaries = ee.FeatureCollection("projects/ee-carlosmendez777/assets/bolivia/bolivia339geoqueryFull")
  .select(['asdf_id']); 

// Satellite Embeddings Collection
var satEmbeddingsCol = ee.ImageCollection('GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL');

// --- 2. CLIENT-SIDE SETUP (THE FIX) ---

// We generate the list of band names (A00...A63) using standard JavaScript.
// This prevents the "Invalid value (selectors)" error by ensuring the 
// Export function gets a concrete list of strings, not a server-side object.
var bandNames = [];
for (var i = 0; i < 64; i++) {
  // Adds a leading zero for single digits (e.g., 'A05') to match GEE naming
  var str = i < 10 ? 'A0' + i : 'A' + i; 
  bandNames.push(str);
}

// Prepare the column order for the CSV: ID first, then the 64 bands
var exportSelectors = ['asdf_id'].concat(bandNames);


// --- 3. PRE-PROCESSING ---

// Step A: Define Timeframe
var start = ee.Date('2017-01-01');
var end   = ee.Date('2018-01-01');

// Step B: Create Seamless Composite
// We filter and mosaic. We do NOT clip here; the reducer handles the geometry.
var embeddings2017 = satEmbeddingsCol
  .filterDate(start, end)
  .mosaic(); 


// --- 4. SPATIAL AGGREGATION ---

// We compute the mean of all 64 bands for each polygon.
// - scale: 10 forces the analysis to native resolution.
// - tileScale: 8 helps process complex geometries without memory errors.
var aggregatedData = embeddings2017.reduceRegions({
  collection: admBoundaries,
  reducer: ee.Reducer.mean(), 
  scale: 10,  
  tileScale: 8 
});


// --- 5. EXPORT ---

// Export to Google Drive
Export.table.toDrive({
  collection: aggregatedData,
  description: 'Bolivia_Embeddings_Mean_2017',
  folder: 'gee', 
  fileNamePrefix: 'bolivia_embeddings_2017',
  fileFormat: 'CSV',
  selectors: exportSelectors 
});

// --- DIAGNOSTICS ---

print('Workflow complete. Open the "Tasks" tab to run the export.');
print('Columns to be exported:', exportSelectors);

// Visual Sanity Check
Map.centerObject(admBoundaries, 6);
Map.addLayer(embeddings2017, {bands: ['A00', 'A01', 'A02'], min: -0.1, max: 0.1}, 'Embeddings (RGB)');
Map.addLayer(admBoundaries.style({color: 'red', fillColor: '00000000'}), {}, 'Boundaries');
```