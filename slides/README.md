# Slides

Quarto presentations for the project.

## Current Presentations

| File | Format | Description |
| ---- | ------ | ----------- |
| `rf_public_services_slides.qmd` | Quarto source | Random Forest analysis of public services |
| `rf_public_services_slides.html` | RevealJS HTML | Rendered presentation (18 slides) |

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

## Re-rendering Slides

After editing the `.qmd` file:

```bash
quarto render rf_public_services_slides.qmd --to revealjs
```

## Features

- RevealJS format with slide transitions
- Chalkboard enabled (press 'c' to toggle)
- Slide numbers
- Embedded figures from `output/`
