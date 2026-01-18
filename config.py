import random
import numpy as np
import os
from pathlib import Path

# Reproducibility
RANDOM_SEED = 42

def set_seeds(seed=RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    # Add torch.manual_seed(seed) if using PyTorch
    # Add tf.random.set_seed(seed) if using TensorFlow
    os.environ['PYTHONHASHSEED'] = str(seed)

# Project root (directory containing this config file)
PROJECT_ROOT = Path(__file__).parent.resolve()

# Paths (as Path objects for easy joining with /)
DATA_DIR = PROJECT_ROOT / 'data'
OUTPUT_DIR = PROJECT_ROOT / 'output'
NOTEBOOKS_DIR = PROJECT_ROOT / 'notebooks'
LOG_DIR = PROJECT_ROOT / 'log'
CODE_DIR = PROJECT_ROOT / 'code'