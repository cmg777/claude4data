# CLAUDE.md -- AI Assistant Instructions

**Your role:** Research assistant and workflow orchestrator for an academic research project.
For full documentation (installation, workflows, Overleaf sync, reproducibility), see `README.md`.

**First actions for every session:**

1. Read this file completely
2. Check `./handoffs/` for the most recent entry to understand prior work
3. Review the project context below for current status
4. Begin work or ask clarifying questions

---

## Project Context

| Field | Value |
| ----- | ----- |
| **Title** | Predicting Municipal Sustainable Development from Satellite Imagery in Bolivia |
| **Authors** | Carlos Mendez (Nagoya University) |
| **Stage** | Analysis |
| **Primary tools** | Python, R, Stata, Quarto, LaTeX |
| **Reference manager** | Zotero (exports to `references.bib`) |
| **Manuscript** | `index.qmd` (Quarto manuscript project) |
| **Environment** | `uv` + `pyproject.toml` (Python 3.12) |
| **Data source** | [DS4Bolivia](https://github.com/quarcs-lab/ds4bolivia) — SDG indices, satellite embeddings, region names for 339 municipalities |

---

## Critical Rules

These are non-negotiable behavioral constraints.

1. **Never delete data or code** -- Do not delete files in `data/`, `code/`, `notebooks/`, `references/`, or `templates/`. Move old versions to `legacy/`.
2. **Stay within this directory** -- All work must remain inside this project folder. Ask before accessing external resources.
3. **Preserve raw data** -- Files in `data/rawData/` are source-of-truth inputs. Never modify them.
4. **Document progress** -- Write a handoff report to `./handoffs/` after significant work or before ending a session.

---

## Key Paths

| Path | Purpose |
| ---- | ------- |
| `index.qmd` | Main manuscript source |
| `_quarto.yml` | Quarto config (formats, notebook registrations) |
| `references.bib` | Bibliography (from Zotero) |
| `config.py` / `config.R` | Reproducibility config (seed = 42, project paths) |
| `pyproject.toml` / `uv.lock` | Python dependencies |
| `jupytext.toml` | Cell metadata filter |
| `notebooks/` | Jupyter notebooks (`.ipynb` + `.md:myst` pairs) |
| `code/ml_intro_rf.py` | Standalone Random Forest ML script |
| `code/tut_doubleml.py` | Standalone DoubleML causal inference script |
| `data/rawData/` | Raw source data (never modify) |
| `scripts/render.sh` | Clean render + Overleaf staging |
| `handoffs/` | Session handoff reports |
| `.claude/skills/ml-intro/` | ML intro skill (Random Forest tutorial generation) |
| `.claude/skills/data-science-tutorial/` | Data science tutorial skill (case-study generator) |
| `.env` | API keys and secrets (gitignored, never commit) |

---

## Skills

Invoke with `/project:<name>`.

| Skill | What It Does |
| ----- | ------------ |
| `/project:ml-intro` | Create an introductory Random Forest ML workflow (script + notebook) predicting Bolivia's IMDS from satellite embeddings. Accepts optional target variable argument. |
| `/project:data-science-tutorial` | Create a pedagogical case-study tutorial (script + notebook) on any data science topic. User provides topic, dataset, and reference materials (e.g., library docs URL). |

---

## Session Management

**Handoff reports** go in `./handoffs/` as `YYYYMMDD_HHMM.md`. Write one when you complete significant work, make major decisions, or end a session. Every handoff must include:

- Current project state (one paragraph)
- Work completed (bullet list)
- Decisions made and rationale
- Open issues or blockers
- Concrete next steps

---

## Essential Commands

See `README.md` § Manuscript Workflow and § Notebook Workflow for full details.

```bash
bash scripts/render.sh                                      # clean render (all formats)
uv run jupyter execute --inplace notebooks/<file>.ipynb      # execute notebook (--inplace REQUIRED)
uv add <package>                                             # add Python package (NEVER use pip)
```

---

## Workflow Gotchas

These are non-obvious pitfalls. See `README.md` for full context.

- **`--inplace` is required** for `jupyter execute` -- without it, outputs are discarded
- **Register new notebooks** in `_quarto.yml` under `manuscript.notebooks`
- **Single-line markdown paragraphs** in notebooks -- Jupytext preserves `\n` in `.ipynb` cell sources and VS Code renders them as literal line breaks. Write each paragraph as one continuous line.
- **Stata cell directives** use `*|` prefix (not `#|`), e.g., `*| label: fig-name`
- **Never use `tbl-` prefix** for Stata text output cells -- it triggers Quarto's table parser and crashes. Use a plain label (e.g., `stata-summary`)
- **Never use `pip install`** -- it bypasses the lockfile. Always use `uv add`
- **Avoid `@fig-`/`@tbl-` cross-refs** to `{{< embed >}}`-ed notebook content -- use plain prose instead (avoids "Unable to resolve crossref" warnings)
- **Jupytext metadata filter** in `jupytext.toml` strips `_sphinx_cell_id`, `execution`, and `vscode` keys. Do not remove it.
- **Credentials** go in `.env` only. Never commit secrets to git.
