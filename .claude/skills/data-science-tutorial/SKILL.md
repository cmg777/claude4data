---
name: data-science-tutorial
description: Create a Colab-ready, pedagogical data science case-study tutorial (script + notebook) based on a user-specified topic, dataset, and reference materials. Common use case — user provides a link to a library's documentation (e.g., DoubleML, PySAL, XGBoost) and a dataset, and the skill creates a complete case-study tutorial with a motivating problem, implementation, interpretation, and references.
argument-hint: "<topic> dataset: <dataset name or URL> [references: <URLs, papers, or notes>]"
disable-model-invocation: true
user-invocable: true
---

# Data Science Tutorial: Case-Study Generator

Create a self-contained, pedagogical data science tutorial framed as a
**case study** with a clear motivating problem. The user specifies the topic,
dataset, and (optionally) reference materials such as library documentation
URLs. The skill produces a standalone Python script and a Colab-ready Jupyter
notebook with conceptual explanations before every code cell and interpretation
of actual results after every code cell.

## Example invocations

```
/project:data-science-tutorial double machine learning dataset: DS4Bolivia references: https://docs.doubleml.org/stable/intro/intro.html
/project:data-science-tutorial k-means clustering dataset: DS4Bolivia
/project:data-science-tutorial spatial regression dataset: DS4Bolivia references: https://pysal.org/spreg/
/project:data-science-tutorial gradient boosting dataset: https://example.com/housing.csv references: https://scikit-learn.org/stable/modules/ensemble.html#gradient-boosting
```

## Deliverables

| Output | Path |
|--------|------|
| Python script | `code/tut_<topic-slug>.py` |
| Jupyter notebook | `notebooks/notebook-NN.ipynb` |
| Jupytext pair | `notebooks/notebook-NN.md` |
| Figures (≥3) | `images/tut_<topic-slug>_*.png` |
| Tables (if applicable) | `tables/tut_<topic-slug>_*.csv` |

---

## Pre-flight

1. Read `config.py`, `_quarto.yml`, `jupytext.toml`, `pyproject.toml`
2. Parse `$ARGUMENTS`:
   - **Topic** — everything before `dataset:` (e.g., "double machine learning")
   - **Dataset** — everything between `dataset:` and `references:` (or end of string). Can be `DS4Bolivia` or a URL/description of another dataset
   - **References** — everything after `references:` (optional). Typically URLs to library docs, papers, or tutorials
   - **Topic slug** — lowercase, no spaces, for file naming (e.g., "double machine learning" → `doubleml`, "k-means clustering" → `kmeans`)
3. **Fetch reference URLs** — if references are provided, use WebFetch to read each URL and understand the library's API, key classes/functions, and recommended usage patterns. This is critical for producing accurate, idiomatic code
4. Determine the next available notebook number from `_quarto.yml` (count existing entries under `manuscript.notebooks` and increment)
5. Check that the target notebook and script files don't already exist; if they do, ask the user how to proceed
6. Identify which Python packages the topic requires beyond the base stack in `pyproject.toml`. For each missing package, run `uv add <package>`. Common examples:
   - DoubleML: `uv add doubleml`
   - Spatial analysis: `uv add libpysal esda geopandas`
   - XGBoost: `uv add xgboost`
   - Deep learning intro: `uv add torch`
   - Time series: `uv add statsmodels`
7. If dataset is `DS4Bolivia`, use the standard loading/caching pattern (see Data Source section below). If it's another dataset, design appropriate loading code

---

## Data Source: DS4Bolivia (default)

When the user specifies `dataset: DS4Bolivia`, use this standard data source.

**Base URL:** `https://raw.githubusercontent.com/quarcs-lab/ds4bolivia/master`

| Dataset | Path | Key columns |
|---------|------|-------------|
| SDG indices | `/sdg/sdg.csv` | `asdf_id`, `imds`, `sdg1`–`sdg15` |
| Satellite embeddings | `/satelliteEmbeddings/satelliteEmbeddings2017.csv` | `asdf_id`, `A00`–`A63` |
| Region names | `/regionNames/regionNames.csv` | `asdf_id`, municipality/department names |

- **Join key:** `asdf_id` (integer, shared across all datasets)
- **Observations:** 339 Bolivian municipalities
- **Local caching:** Save merged data to `data/rawData/ds4bolivia_merged.csv`

Select appropriate columns for the topic:
- **Supervised methods:** use an SDG index as target (default `imds`), embeddings `A00`–`A63` as features
- **Unsupervised methods:** use `A00`–`A63` as features (or SDG indices if more appropriate)
- **Causal inference:** use an SDG index as outcome, select treatment and controls from embeddings or other SDG indices
- **Spatial methods:** incorporate region identifiers + geographic context

---

## Step 1: Create `code/tut_<topic-slug>.py`

A self-contained script runnable via `uv run python code/tut_<topic-slug>.py` from project root. Follow the same structure as `code/ml_intro_rf.py`.

### Structure

```python
"""
<Tutorial Title>: <Topic> Case Study

<One-paragraph description of what this script does, what method it
implements, and what dataset it uses.>

Usage:
    uv run python code/tut_<topic-slug>.py

References:
    - <URL 1>
    - <URL 2>
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import set_seeds, RANDOM_SEED, IMAGES_DIR, TABLES_DIR, DATA_DIR
set_seeds()
```

Then import all needed libraries. Define topic-specific configuration
variables near the top (e.g., `TARGET`, `N_CLUSTERS`, `TREATMENT_VAR`, etc.).

### Workflow sections

The script should implement the full analysis pipeline with clear section
comments explaining *why* each step matters (not just *what* it does).
Adapt sections to the topic, but generally include:

1. **Data loading** — Load from the specified dataset with caching
2. **EDA** — At least 1 figure exploring the data relevant to the case study
3. **Data preparation** — Feature selection, scaling, encoding as needed
4. **Core method** — The main technique implementation. Use the library and
   API patterns from the reference materials. Include comments explaining
   key parameters and choices
5. **Evaluation** — Topic-appropriate metrics and at least 1 evaluation figure
6. **Feature/variable analysis** — Importance, coefficients, loadings, etc.
7. **Save results** — Figures to `images/tut_<slug>_*.png`, tables to
   `tables/tut_<slug>_*.csv`
8. **Print summary** — Formatted results to stdout

All `plt.savefig()` calls use `dpi=300, bbox_inches="tight"`.

Use `IMAGES_DIR / "tut_<topic-slug>_<name>.png"` for figure paths.
Use `TABLES_DIR / "tut_<topic-slug>_<name>.csv"` for table paths.

---

## Step 2: Create the notebook

Write the Jupytext `.md:myst` file first, then sync to `.ipynb`.

### Frontmatter (use exactly this format)

```yaml
---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.19.1
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---
```

Followed by a title block: `title: "NX: <Tutorial Title>"` (where X matches the notebook number).

### Colab badge (immediately after title block)

```markdown
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cmg777/claude4data/blob/master/notebooks/notebook-NN.ipynb)
```

### First code cell: Colab setup + config

```python
import sys
if "google.colab" in sys.modules:
    !git clone --depth 1 https://github.com/cmg777/claude4data.git /content/claude4data 2>/dev/null || true
    %cd /content/claude4data/notebooks
sys.path.insert(0, "..")
from config import set_seeds, RANDOM_SEED, IMAGES_DIR, TABLES_DIR, DATA_DIR

set_seeds()
```

### Second code cell: imports and configuration

Import all needed libraries. Define topic-specific configuration variables.
Include `from IPython.display import Markdown` for table rendering.

### Cell conventions

- Code cells use ```` ```{code-cell} ipython3 ````
- Figure cells: `#| label: fig-<slug>` + `#| fig-cap: "..."` — save to `images/`
- Table cells: `#| label: tbl-<slug>` + `#| tbl-cap: "..."` — only for `Markdown(df.to_markdown())` output
- Do NOT use `tbl-` prefix for non-table output (crashes Quarto)
- **Single-line paragraphs** — Every markdown paragraph must be written as one continuous line with no hard line breaks. Jupytext preserves `\n` characters inside `.ipynb` cell sources, and VS Code's notebook viewer renders them as literal line breaks. This applies to all markdown cells.
- All `plt.savefig()` calls use `dpi=300, bbox_inches="tight"`

### Notebook structure — framed as a case study

The notebook must tell a coherent story: a real-world question motivates
the analysis, the method addresses that question, and the results answer it.

| Section | Content | Required |
|---------|---------|----------|
| **Case Study Introduction** | Motivate the problem with a compelling question. Why does this method matter? What real-world decision could the results inform? Connect to the dataset's domain (e.g., Bolivian development for DS4Bolivia). Frame as: "We want to know X. Method Y can help because Z." | Yes |
| **Learning Objectives** | 3–5 bullet points of what the reader will learn | Yes |
| **Setup & Imports** | Colab cell + imports + configuration | Yes |
| **Data Loading & Description** | Load dataset, explain its structure and relevance to the case study question | Yes |
| **EDA** | Topic-appropriate exploration, at least 1 figure, connected to the case study question | Yes |
| **Data Preparation** | Scaling, encoding, feature selection, train/test split as needed | If needed |
| **Core Method** (1–3 sections) | Main technique implementation with conceptual explanations. Each section gets its own heading. At least 1 figure | Yes |
| **Evaluation & Results** | Metrics, visualizations, at least 1 figure | Yes |
| **Discussion** | What the results mean for the case study question. Connect findings to the real-world context | Yes |
| **Summary & Next Steps** | Key takeaways, limitations, suggestions for further exploration | Yes |
| **Exercises** | 2–3 self-study challenges for the reader | Encouraged |
| **References** | Numbered list of clickable markdown links to all sources used (library docs, papers, tutorials, dataset sources) | Yes |

### Sandwich structure

Every code cell that produces output must be sandwiched:

- **Before**: Conceptual explanation — what the technique is, why this step
  matters, how it connects to the case study question. Written generically
  (does not reference specific output values since it hasn't been executed yet).

- **After**: Interpretation — what the actual output means. Added after
  execution (Step 4.5). References specific numbers, explains their
  significance, connects to the case study context. 2–4 sentences each.

### References section

The final section must be a numbered list of all references used:

```markdown
## References

1. [Library Name — Documentation Title](https://full-url-to-docs)
2. [Author(s) (Year). Paper Title. Journal.](https://doi-or-url)
3. [DS4Bolivia — Dataset Repository](https://github.com/quarcs-lab/ds4bolivia)
```

Include at minimum:
- The library documentation URL (if provided in references)
- The dataset source
- Any papers or tutorials that informed the implementation

---

## Step 3: Sync and register

```bash
uv run jupytext --sync notebooks/notebook-NN.md
```

Then edit `_quarto.yml` to add the notebook under `manuscript.notebooks`:

```yaml
- notebook: notebooks/notebook-NN.ipynb
  title: "NX: <Tutorial Title>"
```

---

## Step 4: Execute

```bash
uv run jupyter execute --inplace notebooks/notebook-NN.ipynb
```

The `--inplace` flag is REQUIRED — without it, outputs are discarded.

After execution, re-sync to capture outputs in the `.md` pair:

```bash
uv run jupytext --sync notebooks/notebook-NN.ipynb
```

---

## Step 4.5: Interpret results — THIS IS THE MOST IMPORTANT STEP

The interpretation cells are what transform this notebook from a code demo
into a genuine case-study tutorial. Without them, a beginner sees numbers
and plots but has no idea what they mean. Every code cell that produces output
needs a markdown cell immediately after it that explains the result in plain
language and connects it back to the case study question.

### How to do it

1. **Read the executed `.ipynb`** — open the notebook file and look at each
   cell's printed output (metrics, counts, parameter values). Write down the
   key numbers.
2. **Edit the `.md` file** — after each code cell that produces output, insert
   a new markdown paragraph (not a new section heading) that interprets the
   result. Use the actual numbers from the output. Write each paragraph as a
   single continuous line (no soft wraps).
3. **Re-sync** — run `uv run jupytext --sync notebooks/notebook-NN.md` to
   propagate the interpretation cells back to the `.ipynb`.

### What good interpretation looks like

Each interpretation paragraph must:
1. Quote specific numbers from the output
2. Explain what those numbers mean in plain language
3. Connect findings to the case study question and real-world context
4. Be written as a single continuous line (no hard wraps)
5. Be 2–4 sentences

### Verification

Count the interpretation cells in the `.md` file. There must be at least
8 interpretation paragraphs that reference specific numeric values from the
executed output. If fewer than 8, go back and add more.

After adding all interpretation cells, re-sync:

```bash
uv run jupytext --sync notebooks/notebook-NN.md
```

---

## Step 5: Verify and report

Check all outputs exist:
- Python script at `code/tut_<topic-slug>.py`
- Both notebook files (`*.ipynb` + `*.md`)
- At least 3 figures in `images/tut_<topic-slug>_*.png`
- Tables in `tables/tut_<topic-slug>_*.csv` (if applicable)
- Notebook registered in `_quarto.yml`
- At least 8 interpretation paragraphs referencing specific numeric values
- Case study introduction with clear problem motivation
- References section at the end with all source URLs
- Colab badge present with correct URL

Run the standalone script to confirm it works independently:

```bash
uv run python code/tut_<topic-slug>.py
```

Report to user what was created, and remind them to:
1. Review notebook outputs
2. Embed figures in `index.qmd` via `{{< embed >}}` shortcodes if desired
3. Run `bash scripts/render.sh` to rebuild the manuscript
4. Write a handoff via `/project:handoff`
