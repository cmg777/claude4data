# Reproducibility
RANDOM_SEED <- 42

set_seeds <- function(seed = RANDOM_SEED) {
  set.seed(seed)
  # For parallel processing
  RNGkind("L'Ecuyer-CMRG")
}

# Paths
DATA_DIR <- "data"
OUTPUT_DIR <- "output"
NOTEBOOKS_DIR <- "notebooks"
LOG_DIR <- "log"
CODE_DIR <- "code"

# Create directories if they don't exist
dirs <- c(DATA_DIR, OUTPUT_DIR, NOTEBOOKS_DIR, LOG_DIR, CODE_DIR)
invisible(lapply(dirs, function(d) if (!dir.exists(d)) dir.create(d, recursive = TRUE)))