# Slides

Quarto presentations for the project.

## Current Presentations

| File                                  | Format        | Description                              |
|---------------------------------------|---------------|------------------------------------------|
| `eda_slides.qmd`                      | Quarto source | Exploratory Data Analysis                |
| `eda_slides.html`                     | RevealJS HTML | Rendered EDA presentation (18 slides)    |
| `rf_public_services_slides.qmd`       | Quarto source | Random Forest analysis of public services|
| `rf_public_services_slides.html`      | RevealJS HTML | Rendered RF presentation (18 slides)     |
| `esda_slides.qmd`                     | Quarto source | Spatial autocorrelation analysis (ESDA)  |
| `esda_slides.html`                    | RevealJS HTML | Rendered ESDA presentation (20 slides)   |
| `solow_model_slides.qmd`              | Quarto source | Solow Growth Model and convergence       |
| `solow_model_slides.html`             | RevealJS HTML | Rendered Solow presentation (26 slides)  |
| `embeddings_comparison_slides.qmd`    | Quarto source | Regular vs pop-weighted embeddings       |
| `embeddings_comparison_slides.html`   | RevealJS HTML | Rendered embeddings comparison (24 slides)|

## Viewing Slides

Open the HTML file directly in a browser:

```bash
open rf_public_services_slides.html
```

Or use Quarto preview for live editing:

```bash
quarto preview rf_public_services_slides.qmd
```

## Presentation Contents

### EDA Slides (`eda_slides.html`)

**Title:** Exploratory Data Analysis - Bolivia's Municipal Development Indicators

**Slides include:**

1. Dataset overview (339 municipalities, 152 variables)
2. IMDS distribution and statistics
3. IMDS by department comparison
4. Public service indicators summary
5. Satellite embeddings overview
6. Key findings and implications

### RF Public Services Slides (`rf_public_services_slides.html`)

**Title:** Predicting Public Services from Satellite Imagery

**Slides include:**

1. Overview and research question
2. Data sources (satellite embeddings + SDG variables)
3. Categories of public services (5 categories, 20 indicators)
4. Model performance visualization
5. Best predicted variables with explanations:
   - Institutional Childbirth (R² = 0.579)
   - Access to Basic Services (R² = 0.499)
   - Tuberculosis Incidence (R² = 0.368)
6. Worst predicted variables with explanations:
   - School Dropout Female (R² = -0.588)
   - Mass Transit Seats (R² = -0.346)
7. Category summary and key insights
8. Implications for researchers and policymakers
9. Methodology notes and data availability

### ESDA Slides (`esda_slides.html`)

**Title:** Spatial Autocorrelation in Bolivia's Municipal Development

**Slides include:**

1. Overview and research question
2. Spatial distribution of development (interactive map)
3. Spatial weights matrix (K-nearest neighbors)
4. Global spatial autocorrelation (Moran's I = 0.39)
5. Moran scatterplot interpretation
6. Local spatial autocorrelation (LISA)
7. LISA cluster map visualization
8. Combined LISA analysis
9. High-High clusters (hotspots) analysis
10. Low-Low clusters (coldspots) analysis
11. Spatial outliers (High-Low and Low-High)
12. Statistical summary of cluster distribution
13. Policy implications for each cluster type
14. Methodology notes (spatial statistics)
15. Key findings and conclusions
16. Future research directions
17. Data and reproducibility information

### Solow Model Slides (`solow_model_slides.html`)

**Title:** The Solow Growth Model and Economic Convergence

**Replication of:** Mankiw, Romer, and Weil (1992)

**Slides include:**

1. Research questions and framework
2. The Solow model theoretical foundation
3. Steady-state predictions
4. Data and samples (121 countries, 1960-1985)
5. Textbook Solow: Unrestricted regression results
6. Textbook Solow: Implied alpha (α ≈ 0.60, too high)
7. Augmented Solow model with human capital
8. Augmented Solow: Results (R² improved to 0.78)
9. Implied alpha and beta (α ≈ 1/3, β ≈ 1/3)
10. Model comparison (textbook vs. augmented)
11. Unconditional convergence (only OECD)
12. OECD unconditional convergence analysis
13. Conditional convergence (all samples)
14. Augmented conditional convergence
15. Convergence speeds and half-lives
16. Convergence visualization
17. Main conclusions
18. Policy implications
19. Limitations and extensions
20. Key takeaways
21. Data and reproducibility
22. References

### Embeddings Comparison Slides (`embeddings_comparison_slides.html`)

**Title:** Satellite Embeddings for Public Service Prediction: Comparing Regular vs. Population-Weighted Embeddings

**Slides include:**

1. Research question and approach comparison
2. Embedding types explained
3. Overall performance comparison (p = 0.007)
4. Performance by category
5. Top 5 improvements with pop-weighted
6. Best predicted variables (both methods)
7. Education category: Largest gains
8. Health category performance
9. Infrastructure category
10. Basic utilities category
11. Institutional category
12. Where regular is better (5 variables)
13. Visualization: Main results
14. Statistical significance
15. Why pop-weighted performs better
16. Implications for researchers
17. Implications for policymakers
18. Limitations
19. Future research directions
20. Comparison with previous study
21. Practical recommendations
22. Key takeaways
23. Data and reproducibility

## Re-rendering Slides

After editing the `.qmd` file:

```bash
quarto render rf_public_services_slides.qmd --to revealjs
```

## Design Features

- RevealJS format with slide transitions
- Chalkboard enabled (press 'c' to toggle)
- Slide numbers
- Embedded figures from `output/`
- **Professional two-color palette (Navy + Teal):**
  - **Navy (#1E3A8A):** Titles, headers, table headers (authority & trust) - WCAG AAA (9.7:1)
  - **Teal (#0D9488):** Bold text, subtitles, emphasis (clarity & innovation) - WCAG AA (3.8:1)
  - Clean, readable typography for academic presentations
  - Enhanced with callout boxes, badges, and improved table styling
  - Full accessibility compliance with excellent contrast ratios
