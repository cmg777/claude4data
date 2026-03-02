"""
Project configuration for reproducibility.

Usage in notebooks:
    import sys
    sys.path.insert(0, "..")
    from config import set_seeds, DATA_DIR
    set_seeds()
"""

import random
import numpy as np
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_SEED = 42


def set_seeds(seed=RANDOM_SEED):
    """Set random seeds for reproducibility across libraries."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.resolve()
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "rawData"
CODE_DIR = PROJECT_ROOT / "code"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
IMAGES_DIR = PROJECT_ROOT / "images"
TABLES_DIR = PROJECT_ROOT / "tables"
HANDOFFS_DIR = PROJECT_ROOT / "handoffs"
