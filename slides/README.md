# Slides

Quarto presentations for showcasing analysis results.

## Style Guide

### Colors

- **Titles:** Blue (`#2874A6`)
- **Bold emphasis:** Green (`#229954`)
- **Body text:** Default (dark gray/black)

### Format

- Engine: Quarto with `revealjs` theme
- Theme: `simple` (clean academic look)
- Custom CSS: Embed directly in `.qmd` files for portability

### Template

```yaml
---
title: "Your Analysis Title"
subtitle: "[FILL: Subtitle if needed]"
author: "[FILL: Author name]"
date: today
format:
  revealjs:
    theme: simple
    slide-number: true
    css: |
      .reveal h1, .reveal h2, .reveal h3 {
        color: #2874A6;
      }
      .reveal strong {
        color: #229954;
      }
---
```

### Design Principles

- Clean, professional design suitable for academic conferences
- One key idea per slide
- Figures and tables should be self-explanatory with clear captions
- Effective communication of data analysis findings

### Naming Convention

- Use descriptive names: `01-introduction.qmd`, `analysis-results.qmd`
- Rendered output (HTML) can be kept alongside source files

### Rendering

```bash
quarto render slides/your-presentation.qmd
```
