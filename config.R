# Project configuration for reproducibility
#
# Usage in notebooks:
#   source("../config.R")
#   set_seeds()

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_SEED <- 42

set_seeds <- function(seed = RANDOM_SEED) {
  set.seed(seed)
  RNGkind("L'Ecuyer-CMRG")
}

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
get_project_root <- function() {
  # Try common locations relative to where the script is run
  candidates <- c(".", "..", "../..")
  for (cand in candidates) {
    if (file.exists(file.path(cand, "config.R"))) {
      return(normalizePath(cand))
    }
  }
  return(getwd())
}

PROJECT_ROOT <- get_project_root()
DATA_DIR     <- file.path(PROJECT_ROOT, "data")
RAW_DATA_DIR <- file.path(PROJECT_ROOT, "data", "rawData")
CODE_DIR     <- file.path(PROJECT_ROOT, "code")
NOTEBOOKS_DIR <- file.path(PROJECT_ROOT, "notebooks")
IMAGES_DIR   <- file.path(PROJECT_ROOT, "images")
TABLES_DIR   <- file.path(PROJECT_ROOT, "tables")
HANDOFFS_DIR <- file.path(PROJECT_ROOT, "handoffs")
