# Notebooks

Computational notebooks that provide inputs (figures, tables) for the manuscript.

## Quick Reference

```bash
# Execute a notebook and save outputs
uv run jupyter execute --inplace notebooks/notebook-01.ipynb

# Sync Jupytext pair after editing
uv run jupytext --sync notebooks/notebook-01.md

# List installed kernels
jupyter kernelspec list
```

## Selecting the Right Kernel

Each notebook is written in a specific language and requires the matching Jupyter
kernel. Use this table to know which kernel to select when you open a notebook:

| Notebook | Language | Kernel to select |
| --- | --- | --- |
| `notebook-01.ipynb` | Python | **Python 3 (ipykernel)** — use the project `.venv` |
| `notebook-02.ipynb` | R | **IR** |
| `notebook-03.ipynb` | Stata | **Stata (nbstata)** |

> **Tip:** You can identify a notebook's language from its first code cell:
> `import` → Python, `library()` → R, `set seed` → Stata.

### In VSCode

1. Open the notebook and click **"Select Kernel"** in the top-right corner.
2. **Python notebooks:** Choose **"Python Environments..."** → select the project
   `.venv` (recommended, so you have access to all project dependencies).
3. **R / Stata notebooks:** Choose **"Jupyter Kernel..."** → pick **IR** or
   **Stata (nbstata)** from the list.
4. If a kernel doesn't appear, press **Cmd+Shift+P** (macOS) or **Ctrl+Shift+P**
   (Windows/Linux) → run **"Developer: Reload Window"** to refresh the kernel list.

### In Jupyter (browser)

Go to **Kernel → Change Kernel** in the menu bar and select the matching kernel
from the dropdown.

---

## Kernel Setup

The template supports three Jupyter kernels. Each must be installed before its
notebooks can be executed.

### Python (automatic)

The Python kernel is installed automatically when you run `uv sync`. No extra
setup is needed.

### R (IRkernel)

**Requirements:** R must be installed on the system.

From an R console (or terminal):

```bash
R -e "install.packages(c('IRkernel', 'ggplot2', 'knitr'), repos='https://cloud.r-project.org')"
R -e "IRkernel::installspec()"
```

This installs the kernel and the R packages used by the sample notebook. Add any
additional R packages your project needs to the first `install.packages()` call.

**Verify:** `jupyter kernelspec list` should show `ir`.

### Stata (nbstata)

**Requirements:** Stata must be installed and licensed on the system.

> **Important:** Use [nbstata](https://github.com/hugetim/nbstata), not the
> older `stata_kernel`. The legacy `stata_kernel` (v1.14.2) has a graph-capture
> bug that crashes with Stata 19+.

**Step 1 — Register the nbstata kernel:**

`nbstata` is already included in `pyproject.toml`, so `uv sync` installs it
into the project's `.venv`. You only need to register the Jupyter kernel:

```bash
uv run python -m nbstata.install
```

**Step 2 — Configure Stata path:**

Create `~/.config/nbstata/nbstata.conf`:

```ini
[nbstata]
stata_dir = /Applications/Stata
edition = se
```

Adjust `stata_dir` and `edition` for your system:

| OS | Typical `stata_dir` |
| --- | --- |
| macOS | `/Applications/Stata` |
| Linux | `/usr/local/stata19` |
| Windows | `C:\Program Files\Stata19` |

Valid editions: `be` (Basic), `se` (Standard), `mp` (Multiprocessor).

**Verify:** `jupyter kernelspec list` should show `nbstata`.

## Conventions

### Naming

- Use sequential numbering: `notebook-01.ipynb`, `notebook-02.ipynb`, etc.
- Each `.ipynb` file has a paired `.md` file (MyST Markdown) managed by Jupytext.

### Jupytext Pairing

Every notebook is stored in two formats:

- `.ipynb` — Standard Jupyter format (for execution and interactive use)
- `.md` — MyST Markdown format (for version control, clean diffs in git)

Sync changes between formats:

```bash
# After editing the .md file
uv run jupytext --sync notebooks/notebook-01.md

# After editing the .ipynb file
uv run jupytext --sync notebooks/notebook-01.ipynb
```

### Labeling Outputs for the Manuscript

To embed a figure or table in `index.qmd`, add a label in the notebook cell.

**Python / R cells** — use `#|` prefix:

```python
#| label: fig-my-figure
#| fig-cap: "Description of the figure"

plt.show()
```

**Stata cells** — use `*|` prefix (because `*` is Stata's comment character):

```stata
*| label: fig-my-figure
*| fig-cap: "Description of the figure"

twoway scatter y x
```

Then reference it in the manuscript:

```markdown
{{< embed notebooks/notebook-01.ipynb#fig-my-figure >}}
```

### Stata Label Restriction

Do **not** use the `tbl-` prefix for Stata cell labels that produce stream text
output (e.g., `tabstat`, `table`, `summarize`). The `tbl-` prefix triggers
Quarto's table parser, which expects a markdown-formatted table and will crash
on Stata's plain-text output.

Instead, use a plain label:

```stata
*| label: stata-summary

tabstat gdp_per_capita life_expectancy, by(region) stat(mean sd count)
```

This restriction does not apply to Python or R cells that output proper markdown
tables (e.g., via `pandas` or `knitr::kable()`).

### Registering Notebooks

New notebooks must be registered in `_quarto.yml` under `manuscript.notebooks`
(so Quarto links them in the manuscript sidebar):

```yaml
manuscript:
  notebooks:
    - notebook: notebooks/notebook-04.ipynb
      title: "N4: Your notebook title"
```

The `project.render` section already includes a `notebooks/*.ipynb` wildcard,
so new notebooks are automatically picked up for rendering without additional
configuration.

### Quarto Configuration Notes

> **Do NOT set `output-dir: .` in `_quarto.yml`.** This prevents Quarto from
> generating the notebook preview HTML pages (`*-preview.html`) that are linked
> from the manuscript sidebar. Use the default output directory (`_manuscript/`).

Rendered outputs are placed in `_manuscript/`:

```text
_manuscript/
  index.html                          # Main manuscript
  index.pdf
  index.docx
  notebooks/
    notebook-01-preview.html          # Clickable notebook previews
    notebook-02-preview.html
    notebook-03-preview.html
```

### Reproducibility

Always import and call `set_seeds()` at the top of every notebook:

**Python:**

```python
import sys
sys.path.insert(0, "..")
from config import set_seeds, DATA_DIR
set_seeds()
```

**R:**

```r
source("../config.R")
set_seeds()
```

**Stata:**

```stata
clear all
set seed 42
```

### Executing Notebooks

To execute notebooks and save their outputs (required before rendering):

```bash
# Python
uv run jupyter execute --inplace notebooks/notebook-01.ipynb

# R (requires IRkernel)
uv run jupyter execute --inplace notebooks/notebook-02.ipynb

# Stata (requires nbstata)
uv run jupyter execute --inplace notebooks/notebook-03.ipynb
```

The `--inplace` flag is required to write outputs back into the `.ipynb` file.
Without it, the notebook executes but outputs are discarded.

After execution, sync the Jupytext pairs:

```bash
uv run jupytext --sync notebooks/notebook-01.ipynb
uv run jupytext --sync notebooks/notebook-02.ipynb
uv run jupytext --sync notebooks/notebook-03.ipynb
```
