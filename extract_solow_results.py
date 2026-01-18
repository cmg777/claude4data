#!/usr/bin/env python
"""Extract key results from Solow Model notebook"""

import json
import re

notebook_path = "notebooks/03_[R]_Solow_Model_and_Convergence_.ipynb"

with open(notebook_path) as f:
    nb = json.load(f)

print("=== SOLOW MODEL NOTEBOOK - KEY RESULTS ===\n")

# Extract markdown cells with key content
key_sections = []
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'markdown':
        source = ''.join(cell['source'])
        # Look for section headers
        if source.startswith('##'):
            print(f"\n--- Cell {i}: {source[:80]} ---")
            key_sections.append((i, source[:200]))

    # Extract regression table outputs
    if cell['cell_type'] == 'code' and 'stargazer' in ''.join(cell.get('source', [])):
        print(f"\n--- Cell {i}: REGRESSION TABLE ---")
        for output in cell.get('outputs', []):
            if 'text' in output:
                text = ''.join(output['text']) if isinstance(output['text'], list) else output['text']
                # Print first 500 chars of regression table
                print(text[:800])

print("\n\n=== SECTIONS FOUND ===")
for i, section in key_sections:
    print(f"Cell {i}: {section}")
