#!/usr/bin/env bash
set -euo pipefail

# Always run from the project root, regardless of where the script is called from
cd "$(dirname "$0")/.."

command -v quarto >/dev/null 2>&1 || { echo "ERROR: quarto not found on PATH"; exit 1; }

echo "=== Clean render: $(basename "$(pwd)") ==="

echo "1. Cleaning Quarto caches..."
rm -rf _freeze/
rm -rf _manuscript/
rm -rf .quarto/

echo "2. Rendering manuscript (HTML + PDF + Word)..."
quarto render

echo "3. Staging LaTeX for Overleaf sync..."
mkdir -p latex
if [ -f "_manuscript/_tex/index.tex" ]; then
    cp _manuscript/_tex/index.tex latex/index.tex
    cp _manuscript/_tex/index.tex latex/.baseline.tex
    echo "LaTeX staged in latex/index.tex (baseline saved to latex/.baseline.tex)"
else
    echo "WARNING: _manuscript/_tex/index.tex not found — skipping Overleaf staging"
    echo "  (PDF format may be disabled or keep-tex is false in _quarto.yml)"
fi

echo "=== Done ==="
echo "Outputs in _manuscript/: index.html, index.pdf, index.docx"
