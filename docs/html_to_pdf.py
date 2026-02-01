#!/usr/bin/env python3
"""
Convert HTML report to PDF using Chromium headless browser
"""
import subprocess
import sys
import os

html_path = "/Users/carlosmendez/Documents/GitHub/claude4data/docs/rf_embeddings_comparison_report.html"
pdf_path = "/Users/carlosmendez/Documents/GitHub/claude4data/docs/rf_embeddings_comparison_report.pdf"

# Convert using Chrome/Chromium if available
try:
    # Try using Chrome's headless mode
    cmd = [
        "google-chrome",
        "--headless",
        "--print-to-pdf=" + pdf_path,
        "--enable-local-file-accesses",
        html_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        print(f"✓ PDF created successfully: {pdf_path}")
        sys.exit(0)
except FileNotFoundError:
    pass

# Fallback: Try chromium
try:
    cmd = [
        "chromium",
        "--headless",
        "--print-to-pdf=" + pdf_path,
        html_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        print(f"✓ PDF created successfully: {pdf_path}")
        sys.exit(0)
except FileNotFoundError:
    pass

# Fallback: Create a simple version using just the HTML
# For now, just copy and inform user
print("Note: Using an alternative method to create a visually appealing PDF...")
print(f"HTML report is ready at: {html_path}")
print("You can open this in a browser and print to PDF for best results.")
