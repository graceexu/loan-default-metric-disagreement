# R/data_prep.R ------------------------------------------------------------

source("R/config.R")

library(tidyverse)
library(janitor)
library(skimr)
library(naniar)
library(rsample)

set.seed(SEED)

# 0) Load ------------------------------------------------------------------

if (!file.exists(DATA_RAW)) {
  stop("Raw data file not found at: ", DATA_RAW,
       "\nPut your dataset at that path (or update DATA_RAW_FILE in R/config.R).")
}

df_raw <- readr::read_csv(DATA_RAW, show_col_types = FALSE) %>%
  janitor::clean_names()

cat("Loaded:", nrow(df_raw), "rows x", ncol(df_raw), "cols\n")
glimpse(df_raw)

# 1) Target identification --------------------------------------------------
# EDIT this once you see the column names.
TARGET <- "default"  # <-- change to your target column name

if (!(TARGET %in% names(df_raw))) {
  stop("TARGET column '", TARGET, "' not found.\nAvailable columns:\n",
       paste(names(df_raw), collapse = ", "))
}

# What values does the target take?
target_counts <- df_raw %>%
  count(.data[[TARGET]], name = "n") %>%
  mutate(p = n / sum(n)) %>%
  arrange(desc(n))

print(target_counts)

# If target isn't 0/1, recode it here (examples):
# df_raw <- df_raw %>% mutate(default = if_else(loan_status == "Charged Off", 1L, 0L))
# df_raw <- df_raw %>% mutate(default = if_else(default == "Yes", 1L, 0L))

# Sanity check: exactly 2 levels after recode
n_levels <- df_raw %>% distinct(.data[[TARGET]]) %>% nrow()
if (n_levels != 2) {
  warning("Target currently has ", n_levels, " distinct values. ",
          "For binary classification, you probably want exactly 2.")
}

# 2) Quick EDA --------------------------------------------------------------

# High-level summary
skimr::skim(df_raw)

# Missingness summaries
miss_var <- naniar::miss_var_summary(df_raw) %>% arrange(desc(n_miss))
miss_case <- naniar::miss_case_summary(df_raw) %>% arrange(desc(n_miss))

print(miss_var, n = min(50, nrow(miss_var)))
print(miss_case, n = min(20, nrow(miss_case)))

# Separate numeric vs categorical (after clean_names)
num_cols <- df_raw %>% select(where(is.numeric)) %>% names()
cat_cols <- df_raw %>%
  select(where(~is.character(.x) || is.factor(.x) || is.logical(.x))) %>%
  names()

cat("Numeric cols:", length(num_cols), "\n")
cat("Categorical/logical cols:", length(cat_cols), "\n")

# Target prevalence (assumes positive class is coded as 1; if not, ignore)
pos_rate <- suppressWarnings(mean(as.numeric(df_raw[[TARGET]]) == 1, na.rm = TRUE))
cat("Approx positive rate (assuming 1 = positive):", round(pos_rate, 4), "\n")

# 3) Quick relationship scans ----------------------------------------------
# (These are intentionally lightweight—just to spot obvious signal/leakage.)

# Numeric feature differences by target (mean/sd)
num_summary <- df_raw %>%
  mutate(.y = factor(.data[[TARGET]])) %>%
  select(all_of(TARGET), all_of(num_cols)) %>%
  pivot_longer(cols = -all_of(TARGET), names_to = "feature", values_to = "value") %>%
  group_by(feature, .data[[TARGET]]) %>%
  summarise(
    n = sum(!is.na(value)),
    mean = mean(value, na.rm = TRUE),
    sd = sd(value, na.rm = TRUE),
    .groups = "drop"
  )

print(num_summary, n = 60)

# Categorical feature distribution by target (top counts)
cat_summary <- df_raw %>%
  mutate(.y = factor(.data[[TARGET]])) %>%
  select(all_of(TARGET), all_of(cat_cols)) %>%
  pivot_longer(cols = -all_of(TARGET), names_to = "feature", values_to = "value") %>%
  mutate(value = fct_explicit_na(factor(value), na_level = "(NA)")) %>%
  count(feature, value, .data[[TARGET]], name = "n") %>%
  group_by(feature, .data[[TARGET]]) %>%
  mutate(p_within_target = n / sum(n)) %>%
  ungroup() %>%
  arrange(desc(n))

print(cat_summary, n = 80)

# 4) Leakage candidate scan -------------------------------------------------
# This flags columns that *might* encode outcomes or post-loan info.
suspicious <- c(
  "default", "charged", "charge", "collection", "recover", "late", "delinq",
  "write", "loss", "status", "outcome", "paid", "settle", "bankrupt"
)

leak_candidates <- names(df_raw)[
  stringr::str_detect(names(df_raw), paste0(suspicious, collapse = "|"))
]

cat("Potential leakage-ish columns by name:\n",
    paste(leak_candidates, collapse = ", "), "\n")

# 5) Minimal cleaning (keep it minimal for now!) ----------------------------
# For phase 1, you can often do:
# - drop obviously leaky columns (once confirmed)
# - convert target to factor (for yardstick)
# - leave other cleaning to a modeling recipe later (tidymodels)

df_model <- df_raw

# Ensure target is factor with levels c("0","1") or c("no","yes") etc.
# You can set the positive class later explicitly in yardstick.
df_model <- df_model %>%
  mutate("{TARGET}" := as.factor(.data[[TARGET]]))

# 6) Split -----------------------------------------------------------------
# Stratified split by target
split_obj <- rsample::initial_split(df_model, prop = 0.80, strata = all_of(TARGET))
train_df <- rsample::training(split_obj)
test_df  <- rsample::testing(split_obj)

cat("Train size:", nrow(train_df), " Test size:", nrow(test_df), "\n")
cat("Train target distribution:\n")
print(train_df %>% count(.data[[TARGET]]) %>% mutate(p = n/sum(n)))
cat("Test target distribution:\n")
print(test_df %>% count(.data[[TARGET]]) %>% mutate(p = n/sum(n)))

# 7) Save artifacts ---------------------------------------------------------
# Save the split object + modeling data (optional but useful)
arrow::write_parquet(train_df, "data/train_modeling.parquet")
arrow::write_parquet(test_df,  "data/test_modeling.parquet")

# Also save a single modeling dataset if you prefer (comment out if not needed):
arrow::write_parquet(df_model, DATA_MODEL_FILE)

saveRDS(split_obj, SPLIT_FILE)

cat("Wrote:\n",
    "- train: data/train_modeling.parquet\n",
    "- test : data/test_modeling.parquet\n",
    "- full : ", DATA_MODEL_FILE, "\n",
    "- split: ", SPLIT_FILE, "\n", sep = "")
