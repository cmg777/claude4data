# =============================================================================
# 01_eda.R - Exploratory Data Analysis of Bolivia's Municipal Indicators
# =============================================================================
#
# This script performs basic exploratory data analysis including:
# - Loading and merging datasets from GitHub
# - Descriptive statistics tables
# - Distribution visualizations
# - Correlation analysis
#
# Data: 339 Bolivian municipalities with SDG indicators and satellite embeddings
# =============================================================================

# -----------------------------------------------------------------------------
# 1. Setup
# -----------------------------------------------------------------------------

# Load configuration
source("../config.R")
set_seeds()

# Ensure renv library is in path
.libPaths(c(file.path(PROJECT_ROOT, "renv/library/macos/R-4.5/x86_64-apple-darwin20"), .libPaths()))

# Load required packages
library(tidyverse)
library(knitr)       # For nice tables
library(scales)      # For formatting
library(patchwork)   # For combining plots

# GitHub base URL for data
GITHUB_BASE <- "https://raw.githubusercontent.com/quarcs-lab/ds4bolivia/master/"

# -----------------------------------------------------------------------------
# 2. Load Data from GitHub
# -----------------------------------------------------------------------------

cat("Loading data from GitHub...\n")

# Load datasets (CSV files from subdirectories)
region_names <- read_csv(paste0(GITHUB_BASE, "regionNames/regionNames.csv"), show_col_types = FALSE)
sdg <- read_csv(paste0(GITHUB_BASE, "sdg/sdg.csv"), show_col_types = FALSE)
sdg_variables <- read_csv(paste0(GITHUB_BASE, "sdgVariables/sdgVariables.csv"), show_col_types = FALSE)
satellite <- read_csv(paste0(GITHUB_BASE, "satelliteEmbeddings/satelliteEmbeddings2017.csv"), show_col_types = FALSE)

# Merge all datasets
df <- region_names %>%
  left_join(sdg, by = "asdf_id") %>%
  left_join(sdg_variables, by = "asdf_id") %>%
  left_join(satellite, by = "asdf_id")

cat(sprintf("Dataset: %d rows, %d columns\n\n", nrow(df), ncol(df)))

# -----------------------------------------------------------------------------
# 3. Basic Dataset Overview
# -----------------------------------------------------------------------------

cat("=" %>% rep(70) %>% paste(collapse = ""), "\n")
cat("DATASET OVERVIEW\n")
cat("=" %>% rep(70) %>% paste(collapse = ""), "\n\n")

# Dataset dimensions
cat(sprintf("Observations: %d municipalities\n", nrow(df)))
cat(sprintf("Variables: %d\n", ncol(df)))

# Variable types
var_types <- df %>%
  summarise(across(everything(), class)) %>%
  pivot_longer(everything(), names_to = "variable", values_to = "type") %>%
  count(type)

cat("\nVariable types:\n")
print(var_types)

# -----------------------------------------------------------------------------
# 4. Descriptive Statistics - Main Development Index
# -----------------------------------------------------------------------------

cat("\n")
cat("=" %>% rep(70) %>% paste(collapse = ""), "\n")
cat("DESCRIPTIVE STATISTICS: IMDS (Municipal Development Index)\n")
cat("=" %>% rep(70) %>% paste(collapse = ""), "\n\n")

# IMDS summary statistics
imds_stats <- df %>%
  summarise(
    N = n(),
    Mean = mean(imds, na.rm = TRUE),
    SD = sd(imds, na.rm = TRUE),
    Min = min(imds, na.rm = TRUE),
    Q1 = quantile(imds, 0.25, na.rm = TRUE),
    Median = median(imds, na.rm = TRUE),
    Q3 = quantile(imds, 0.75, na.rm = TRUE),
    Max = max(imds, na.rm = TRUE),
    Missing = sum(is.na(imds))
  )

cat("Overall IMDS Statistics:\n")
print(imds_stats %>% mutate(across(where(is.numeric), ~round(.x, 2))))

# IMDS by department
cat("\nIMDS by Department:\n")
imds_by_dept <- df %>%
  group_by(dep) %>%
  summarise(
    N = n(),
    Mean = mean(imds, na.rm = TRUE),
    SD = sd(imds, na.rm = TRUE),
    Min = min(imds, na.rm = TRUE),
    Median = median(imds, na.rm = TRUE),
    Max = max(imds, na.rm = TRUE)
  ) %>%
  arrange(desc(Mean))

print(imds_by_dept %>% mutate(across(where(is.numeric), ~round(.x, 2))))

# -----------------------------------------------------------------------------
# 5. Descriptive Statistics - Public Service Indicators
# -----------------------------------------------------------------------------

cat("\n")
cat("=" %>% rep(70) %>% paste(collapse = ""), "\n")
cat("DESCRIPTIVE STATISTICS: PUBLIC SERVICE INDICATORS\n")
cat("=" %>% rep(70) %>% paste(collapse = ""), "\n\n")

# Select key public service variables
public_services <- c(
  "sdg1_4_abs",   # Access to 3 Basic Services
  "sdg6_1_dwc",   # Drinking Water Coverage
  "sdg6_2_sc",    # Sanitation Coverage
  "sdg7_1_ec",    # Electricity Coverage
  "sdg3_1_idca",  # Institutional Childbirth
  "sdg4_c_qts",   # Qualified Teachers (Secondary)
  "sdg9_c_mnc",   # Network Coverage
  "sdg16_9_cr"    # Civil Registry Coverage
)

# Labels for public services
service_labels <- c(
  "sdg1_4_abs" = "Access to 3 Basic Services",
  "sdg6_1_dwc" = "Drinking Water Coverage",
  "sdg6_2_sc" = "Sanitation Coverage",
  "sdg7_1_ec" = "Electricity Coverage",
  "sdg3_1_idca" = "Institutional Childbirth",
  "sdg4_c_qts" = "Qualified Teachers (Secondary)",
  "sdg9_c_mnc" = "Network Coverage",
  "sdg16_9_cr" = "Civil Registry Coverage"
)

# Calculate statistics for each public service
service_stats <- df %>%
  select(all_of(public_services)) %>%
  pivot_longer(everything(), names_to = "Variable", values_to = "Value") %>%
  group_by(Variable) %>%
  summarise(
    N = sum(!is.na(Value)),
    Mean = mean(Value, na.rm = TRUE),
    SD = sd(Value, na.rm = TRUE),
    Min = min(Value, na.rm = TRUE),
    Median = median(Value, na.rm = TRUE),
    Max = max(Value, na.rm = TRUE)
  ) %>%
  mutate(Description = service_labels[Variable]) %>%
  select(Variable, Description, everything()) %>%
  arrange(desc(Mean))

cat("Public Service Indicators Summary:\n")
print(service_stats %>% mutate(across(where(is.numeric), ~round(.x, 2))))

# -----------------------------------------------------------------------------
# 6. Satellite Embeddings Overview
# -----------------------------------------------------------------------------

cat("\n")
cat("=" %>% rep(70) %>% paste(collapse = ""), "\n")
cat("DESCRIPTIVE STATISTICS: SATELLITE EMBEDDINGS\n")
cat("=" %>% rep(70) %>% paste(collapse = ""), "\n\n")

# Select satellite embedding columns (A00 to A63)
embedding_cols <- paste0("A", sprintf("%02d", 0:63))

# Summary of embeddings
embedding_stats <- df %>%
  select(all_of(embedding_cols)) %>%
  pivot_longer(everything(), names_to = "Feature", values_to = "Value") %>%
  summarise(
    N_features = n_distinct(Feature),
    Overall_Mean = mean(Value, na.rm = TRUE),
    Overall_SD = sd(Value, na.rm = TRUE),
    Overall_Min = min(Value, na.rm = TRUE),
    Overall_Max = max(Value, na.rm = TRUE)
  )

cat(sprintf("Number of embedding features: %d\n", embedding_stats$N_features))
cat(sprintf("Overall mean: %.4f\n", embedding_stats$Overall_Mean))
cat(sprintf("Overall SD: %.4f\n", embedding_stats$Overall_SD))
cat(sprintf("Range: [%.4f, %.4f]\n\n", embedding_stats$Overall_Min, embedding_stats$Overall_Max))

# Top 5 features by variance (most informative)
cat("Top 10 Embedding Features by Variance:\n")
feature_variance <- df %>%
  select(all_of(embedding_cols)) %>%
  summarise(across(everything(), ~var(.x, na.rm = TRUE))) %>%
  pivot_longer(everything(), names_to = "Feature", values_to = "Variance") %>%
  arrange(desc(Variance)) %>%
  head(10)

print(feature_variance %>% mutate(Variance = round(Variance, 4)))

# -----------------------------------------------------------------------------
# 7. Visualizations
# -----------------------------------------------------------------------------

cat("\n")
cat("=" %>% rep(70) %>% paste(collapse = ""), "\n")
cat("CREATING VISUALIZATIONS\n")
cat("=" %>% rep(70) %>% paste(collapse = ""), "\n\n")

# Set theme
theme_set(theme_minimal(base_size = 11))

# Create multi-panel figure
fig <- ggplot() + theme_void()

# Plot 1: IMDS Distribution
p1 <- ggplot(df, aes(x = imds)) +
  geom_histogram(bins = 30, fill = "steelblue", color = "white", alpha = 0.8) +
  geom_vline(aes(xintercept = mean(imds, na.rm = TRUE)),
             color = "red", linetype = "dashed", size = 1) +
  labs(
    title = "Distribution of IMDS",
    subtitle = "Municipal Development Index (0-100 scale)",
    x = "IMDS Score",
    y = "Count"
  ) +
  annotate("text", x = mean(df$imds, na.rm = TRUE) + 5, y = Inf,
           label = sprintf("Mean: %.1f", mean(df$imds, na.rm = TRUE)),
           vjust = 2, color = "red", size = 3.5)

# Plot 2: IMDS by Department (boxplot)
p2 <- df %>%
  mutate(dep = fct_reorder(dep, imds, .fun = median, na.rm = TRUE)) %>%
  ggplot(aes(x = dep, y = imds, fill = dep)) +
  geom_boxplot(alpha = 0.7, show.legend = FALSE) +
  coord_flip() +
  labs(
    title = "IMDS by Department",
    subtitle = "Ordered by median score",
    x = NULL,
    y = "IMDS Score"
  ) +
  scale_fill_brewer(palette = "Set3")

# Plot 3: Public Services Comparison
service_long <- df %>%
  select(all_of(public_services)) %>%
  pivot_longer(everything(), names_to = "Variable", values_to = "Value") %>%
  mutate(Description = service_labels[Variable]) %>%
  mutate(Description = fct_reorder(Description, Value, .fun = median, na.rm = TRUE))

p3 <- ggplot(service_long, aes(x = Description, y = Value, fill = Description)) +
  geom_boxplot(alpha = 0.7, show.legend = FALSE) +
  coord_flip() +
  labs(
    title = "Public Service Coverage",
    subtitle = "Distribution across 339 municipalities",
    x = NULL,
    y = "Coverage (%)"
  ) +
  scale_fill_brewer(palette = "Paired")

# Plot 4: Correlation between IMDS and key indicators
key_vars <- c("imds", "sdg1_4_abs", "sdg3_1_idca", "sdg7_1_ec", "sdg9_c_mnc")
cor_data <- df %>%
  select(all_of(key_vars)) %>%
  drop_na()

cor_matrix <- cor(cor_data)

# Convert to long format for plotting
cor_long <- cor_matrix %>%
  as.data.frame() %>%
  rownames_to_column("Var1") %>%
  pivot_longer(-Var1, names_to = "Var2", values_to = "Correlation")

# Variable labels for correlation plot
var_labels <- c(
  "imds" = "IMDS",
  "sdg1_4_abs" = "Basic Services",
  "sdg3_1_idca" = "Inst. Childbirth",
  "sdg7_1_ec" = "Electricity",
  "sdg9_c_mnc" = "Network Coverage"
)

cor_long <- cor_long %>%
  mutate(
    Var1 = var_labels[Var1],
    Var2 = var_labels[Var2]
  )

p4 <- ggplot(cor_long, aes(x = Var1, y = Var2, fill = Correlation)) +
  geom_tile(color = "white") +
  geom_text(aes(label = sprintf("%.2f", Correlation)), size = 3) +
  scale_fill_gradient2(low = "blue", mid = "white", high = "red",
                       midpoint = 0, limits = c(-1, 1)) +
  labs(
    title = "Correlation Matrix",
    subtitle = "IMDS and key public service indicators",
    x = NULL,
    y = NULL
  ) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# Combine plots
combined_plot <- (p1 + p2) / (p3 + p4) +
  plot_annotation(
    title = "Exploratory Data Analysis: Bolivia Municipal Development",
    subtitle = "339 municipalities with SDG indicators and satellite embeddings",
    caption = "Data source: github.com/quarcs-lab/ds4bolivia"
  )

# Save figure
output_path <- file.path(OUTPUT_DIR, "eda_overview_R.png")
ggsave(output_path, combined_plot, width = 14, height = 10, dpi = 150)
cat(sprintf("Figure saved to: %s\n", output_path))

# -----------------------------------------------------------------------------
# 8. Save Descriptive Statistics Tables
# -----------------------------------------------------------------------------

# Save IMDS by department table
imds_table_path <- file.path(OUTPUT_DIR, "imds_by_department.csv")
write_csv(imds_by_dept, imds_table_path)
cat(sprintf("Table saved to: %s\n", imds_table_path))

# Save public services summary table
services_table_path <- file.path(OUTPUT_DIR, "public_services_summary.csv")
write_csv(service_stats, services_table_path)
cat(sprintf("Table saved to: %s\n", services_table_path))

# -----------------------------------------------------------------------------
# 9. Final Summary
# -----------------------------------------------------------------------------

cat("\n")
cat("=" %>% rep(70) %>% paste(collapse = ""), "\n")
cat("EDA SUMMARY\n")
cat("=" %>% rep(70) %>% paste(collapse = ""), "\n\n")

cat("Key Findings:\n\n")

cat("1. IMDS Distribution:\n")
cat(sprintf("   - Mean: %.2f, Median: %.2f\n",
            mean(df$imds, na.rm = TRUE),
            median(df$imds, na.rm = TRUE)))
cat(sprintf("   - Range: %.2f to %.2f\n",
            min(df$imds, na.rm = TRUE),
            max(df$imds, na.rm = TRUE)))

cat("\n2. Department with highest mean IMDS: ",
    imds_by_dept$dep[1],
    sprintf(" (%.2f)\n", imds_by_dept$Mean[1]))

cat("   Department with lowest mean IMDS: ",
    imds_by_dept$dep[nrow(imds_by_dept)],
    sprintf(" (%.2f)\n", imds_by_dept$Mean[nrow(imds_by_dept)]))

cat("\n3. Public Service with highest coverage: ",
    service_stats$Description[1],
    sprintf(" (%.1f%%)\n", service_stats$Mean[1]))

cat("   Public Service with lowest coverage: ",
    service_stats$Description[nrow(service_stats)],
    sprintf(" (%.1f%%)\n", service_stats$Mean[nrow(service_stats)]))

cat("\n4. Satellite Embeddings: 64 features available for predictive modeling\n")

cat("\n")
cat("=" %>% rep(70) %>% paste(collapse = ""), "\n")
cat("EDA COMPLETE\n")
cat("=" %>% rep(70) %>% paste(collapse = ""), "\n")
