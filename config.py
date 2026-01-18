import random
import numpy as np
import os

# Reproducibility
RANDOM_SEED = 42

def set_seeds(seed=RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    # Add torch.manual_seed(seed) if using PyTorch
    # Add tf.random.set_seed(seed) if using TensorFlow
    os.environ['PYTHONHASHSEED'] = str(seed)

# Paths
DATA_DIR = 'data'
OUTPUT_DIR = 'output'
NOTEBOOKS_DIR = 'notebooks'
LOG_DIR = 'log'