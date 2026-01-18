# Reproducibility
RANDOM_SEED <- 42

set_seeds <- function(seed = RANDOM_SEED) {
  set.seed(seed)
  # For parallel processing
  RNGkind("L'Ecuyer-CMRG")
}

# Project root (directory containing this config file)
# This works whether sourced from project root or from subdirectories
get_project_root <- function() {
  # Try to find config.R location
  if (exists("ofile")) {
    # Running via source() with chdir = FALSE
    return(dirname(normalizePath(ofile)))
  }
  # Check common locations
  candidates <- c(
    ".",           # Current directory
    "..",          # Parent directory (if in notebooks/)
    "../.."        # Grandparent (if deeper)
  )
  for (cand in candidates) {
    if (file.exists(file.path(cand, "config.R"))) {
      return(normalizePath(cand))
    }
  }
  # Fallback to current directory
  return(getwd())
}

PROJECT_ROOT <- get_project_root()

# Paths (as absolute paths)
DATA_DIR <- file.path(PROJECT_ROOT, "data")
OUTPUT_DIR <- file.path(PROJECT_ROOT, "output")
NOTEBOOKS_DIR <- file.path(PROJECT_ROOT, "notebooks")
LOG_DIR <- file.path(PROJECT_ROOT, "log")
CODE_DIR <- file.path(PROJECT_ROOT, "code")

# Create directories if they don't exist
dirs <- c(DATA_DIR, OUTPUT_DIR, NOTEBOOKS_DIR, LOG_DIR, CODE_DIR)
invisible(lapply(dirs, function(d) if (!dir.exists(d)) dir.create(d, recursive = TRUE)))