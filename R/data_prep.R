# R/data_prep.R

source("R/config.R")

library(tidyverse)
library(janitor)
library(skimr)
library(naniar)
library(rsample)

set.seed(SEED)

##### Load #####

df_raw <- readr::read_csv(DATA_RAW, show_col_types = FALSE) %>%
  janitor::clean_names()

cat("Loaded:", nrow(df_raw), "rows x", ncol(df_raw), "cols\n")
glimpse(df_raw)

##### Target identification #####
TARGET <- "default"

# What values does the target take?
target_counts <- df_raw %>%
  count(.data[[TARGET]], name = "n") %>%
  mutate(p = n / sum(n)) %>%
  arrange(desc(n))

print(target_counts)

##### Quick EDA #####

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

##### Numeric feature differences by target (mean/sd) #####

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

##### Minor cleaning #####

df_model <- df_raw
df_model <- df_model %>% select(-loan_id)

##### Stratified split by target #####

split_obj <- rsample::initial_split(df_model, prop = 0.80, strata = all_of(TARGET))
train_df <- rsample::training(split_obj)
test_df  <- rsample::testing(split_obj)

cat("Train size:", nrow(train_df), " Test size:", nrow(test_df), "\n")
print(train_df %>% count(.data[[TARGET]]) %>% mutate(p = n/sum(n)))
print(test_df %>% count(.data[[TARGET]]) %>% mutate(p = n/sum(n)))

##### Save artifacts #####

saveRDS(train_df, "data/modeling_train.rds")
saveRDS(test_df, "data/modeling_test.rds")

saveRDS(df_model, DATA_MODEL)

saveRDS(split_obj, SPLIT_FILE)
