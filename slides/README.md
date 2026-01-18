# Slides

Quarto presentations for the project.

## Current Presentations

| File                             | Format        | Description                              |
|----------------------------------|---------------|------------------------------------------|
| `eda_slides.qmd`                 | Quarto source | Exploratory Data Analysis                |
| `eda_slides.html`                | RevealJS HTML | Rendered EDA presentation (18 slides)    |
| `rf_public_services_slides.qmd`  | Quarto source | Random Forest analysis of public services|
| `rf_public_services_slides.html` | RevealJS HTML | Rendered RF presentation (18 slides)     |
| `esda_slides.qmd`                | Quarto source | Spatial autocorrelation analysis (ESDA)  |
| `esda_slides.html`               | RevealJS HTML | Rendered ESDA presentation (20 slides)   |

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
- **Professional color scheme:**
  - Blue titles (#2874A6)
  - Green bold text (#229954)
  - Clean, readable typography for academic presentations
