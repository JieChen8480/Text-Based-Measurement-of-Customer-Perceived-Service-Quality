# ==============================================================================
# Yelp Review Sentiment and Rating Analysis
# Revised, leakage-safe, reproducible version
#
# Research question:
# To what extent can interpretable sentiment measures recover ordinal Yelp
# ratings, and does a restaurant-specific lexicon improve out-of-sample results?
#
# Input:
#   all_CM_restaurants_Columbus.csv
#
# Required columns:
#   rating_value, review_text
#
# Outputs:
#   outputs/model_metrics.csv
#   outputs/confusion_matrices.csv
#   outputs/test_predictions.csv
#   outputs/error_analysis.csv
#   outputs/custom_lexicon.csv
#   outputs/model_coefficients.csv
#   outputs/rating_distribution.png
#   outputs/confusion_matrix_combined.png
#   outputs/predicted_probability_example.csv
#
# Important design correction:
# The restaurant-specific lexicon is constructed using TRAINING DATA ONLY.
# The test set is never used to choose sentiment words or estimate models.
# ==============================================================================

# ----------------------------- 0. Configuration -------------------------------

SEED <- 3632
TRAIN_PROP <- 0.80
MIN_WORD_FREQUENCY <- 20
LOG_ODDS_THRESHOLD <- log(3)
INPUT_FILE <- "all_CM_restaurants_Columbus.csv"
OUTPUT_DIR <- "outputs"

set.seed(SEED)
dir.create(OUTPUT_DIR, showWarnings = FALSE, recursive = TRUE)

# ----------------------------- 1. Packages ------------------------------------

required_packages <- c(
  "dplyr",
  "tidyr",
  "readr",
  "stringr",
  "ggplot2",
  "tidytext",
  "MASS",
  "broom"
)

missing_packages <- required_packages[
  !vapply(required_packages, requireNamespace, logical(1), quietly = TRUE)
]

if (length(missing_packages) > 0) {
  stop(
    paste0(
      "Missing required packages: ",
      paste(missing_packages, collapse = ", "),
      "\nInstall them with:\ninstall.packages(c(",
      paste(sprintf('"%s"', missing_packages), collapse = ", "),
      "))"
    )
  )
}

suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
  library(readr)
  library(stringr)
  library(ggplot2)
  library(tidytext)
})

# ----------------------------- 2. Helper functions ----------------------------

rating_to_category <- function(rating) {
  factor(
    dplyr::case_when(
      rating <= 2 ~ "low",
      rating == 3 ~ "medium",
      rating >= 4 ~ "high",
      TRUE ~ NA_character_
    ),
    levels = c("low", "medium", "high"),
    ordered = TRUE
  )
}

stratified_split <- function(data, outcome, train_prop = 0.80, seed = 3632) {
  set.seed(seed)

  outcome_name <- rlang::as_name(rlang::ensym(outcome))

  train_ids <- data %>%
    group_by(.data[[outcome_name]]) %>%
    group_modify(~ {
      n_train <- max(1L, floor(nrow(.x) * train_prop))
      slice_sample(.x, n = min(n_train, nrow(.x)))
    }) %>%
    ungroup() %>%
    pull(id)

  list(
    train = data %>% filter(id %in% train_ids),
    test = data %>% filter(!id %in% train_ids)
  )
}

tokenize_reviews <- function(review_data) {
  review_data %>%
    select(id, rating_value, rating_category, review_text) %>%
    unnest_tokens(
      output = word,
      input = review_text,
      token = "words",
      to_lower = TRUE,
      drop = FALSE
    ) %>%
    filter(
      !is.na(word),
      str_detect(word, "[a-z]"),
      !str_detect(word, "^\\d+$")
    )
}

build_custom_lexicon <- function(
    train_tokens,
    min_frequency = 20,
    threshold = log(3),
    smoothing = 0.5
) {
  # Ratings 1-2 define the negative group; ratings 4-5 define the positive
  # group. Three-star reviews are omitted only during lexicon construction.
  # Laplace smoothing prevents infinite log odds for words absent in one group.

  class_counts <- train_tokens %>%
    filter(rating_category %in% c("low", "high")) %>%
    mutate(class = as.character(rating_category)) %>%
    count(word, class, name = "n") %>%
    complete(word, class = c("low", "high"), fill = list(n = 0)) %>%
    group_by(word) %>%
    mutate(total_word_count = sum(n)) %>%
    ungroup() %>%
    filter(total_word_count >= min_frequency)

  class_totals <- train_tokens %>%
    filter(rating_category %in% c("low", "high")) %>%
    mutate(class = as.character(rating_category)) %>%
    count(class, name = "class_total")

  vocabulary_size <- n_distinct(class_counts$word)

  custom <- class_counts %>%
    left_join(class_totals, by = "class") %>%
    mutate(
      probability = (n + smoothing) /
        (class_total + smoothing * vocabulary_size)
    ) %>%
    select(word, class, probability, total_word_count) %>%
    pivot_wider(
      names_from = class,
      values_from = probability,
      names_prefix = "prob_"
    ) %>%
    mutate(
      log_odds = log(prob_high / prob_low),
      sentiment = case_when(
        log_odds >= threshold ~ "positive",
        log_odds <= -threshold ~ "negative",
        TRUE ~ NA_character_
      ),
      source = "training_specific"
    ) %>%
    filter(!is.na(sentiment)) %>%
    arrange(desc(abs(log_odds))) %>%
    select(word, sentiment, source, log_odds, total_word_count)

  custom
}

make_bing_lexicon <- function() {
  get_sentiments("bing") %>%
    transmute(
      word,
      sentiment,
      source = "bing",
      log_odds = NA_real_,
      total_word_count = NA_integer_
    )
}

combine_lexicons <- function(bing_lexicon, custom_lexicon) {
  # Training-specific classifications override Bing classifications where both
  # lexicons contain the same word.
  bind_rows(
    custom_lexicon,
    bing_lexicon %>% filter(!word %in% custom_lexicon$word)
  ) %>%
    distinct(word, .keep_all = TRUE)
}

score_reviews <- function(review_data, lexicon) {
  # The first word is retained so that a sentiment word immediately following a
  # negator can have its sign reversed. For example, "not good" changes +1 to -1.
  negators <- c(
    "not", "no", "never", "none", "nothing", "neither", "nor",
    "without", "hardly", "scarcely", "barely", "cannot", "can't",
    "isn't", "wasn't", "weren't", "don't", "doesn't", "didn't",
    "won't", "wouldn't", "shouldn't", "couldn't"
  )

  tokens <- review_data %>%
    select(id, rating_value, rating_category, review_text) %>%
    unnest_tokens(
      output = word,
      input = review_text,
      token = "words",
      to_lower = TRUE,
      drop = FALSE
    ) %>%
    group_by(id) %>%
    mutate(
      token_position = row_number(),
      previous_word = lag(word)
    ) %>%
    ungroup()

  scored_tokens <- tokens %>%
    left_join(
      lexicon %>% select(word, sentiment),
      by = "word"
    ) %>%
    mutate(
      raw_sentiment = case_when(
        sentiment == "positive" ~ 1,
        sentiment == "negative" ~ -1,
        TRUE ~ 0
      ),
      negated = !is.na(previous_word) & previous_word %in% negators,
      adjusted_sentiment = if_else(
        negated & raw_sentiment != 0,
        -raw_sentiment,
        raw_sentiment
      ),
      positive_word = as.integer(adjusted_sentiment > 0),
      negative_word = as.integer(adjusted_sentiment < 0)
    )

  review_features <- scored_tokens %>%
    group_by(id, rating_value, rating_category) %>%
    summarise(
      sentiment_score = sum(adjusted_sentiment, na.rm = TRUE),
      positive_count = sum(positive_word, na.rm = TRUE),
      negative_count = sum(negative_word, na.rm = TRUE),
      sentiment_word_count = positive_count + negative_count,
      review_length = n(),
      negated_sentiment_count = sum(negated & raw_sentiment != 0, na.rm = TRUE),
      .groups = "drop"
    )

  review_data %>%
    select(id, review_text) %>%
    left_join(review_features, by = "id") %>%
    mutate(
      sentiment_score = coalesce(sentiment_score, 0),
      positive_count = coalesce(positive_count, 0L),
      negative_count = coalesce(negative_count, 0L),
      sentiment_word_count = coalesce(sentiment_word_count, 0L),
      review_length = coalesce(review_length, 0L),
      negated_sentiment_count = coalesce(negated_sentiment_count, 0L)
    )
}

safe_polr <- function(formula, data) {
  tryCatch(
    MASS::polr(
      formula = formula,
      data = data,
      Hess = TRUE,
      method = "logistic"
    ),
    error = function(e) {
      stop("Ordered logistic regression failed: ", conditionMessage(e))
    }
  )
}

macro_f1 <- function(actual, predicted) {
  classes <- union(levels(actual), levels(predicted))

  f1_values <- vapply(classes, function(class_name) {
    tp <- sum(actual == class_name & predicted == class_name, na.rm = TRUE)
    fp <- sum(actual != class_name & predicted == class_name, na.rm = TRUE)
    fn <- sum(actual == class_name & predicted != class_name, na.rm = TRUE)

    precision <- if ((tp + fp) == 0) 0 else tp / (tp + fp)
    recall <- if ((tp + fn) == 0) 0 else tp / (tp + fn)

    if ((precision + recall) == 0) {
      0
    } else {
      2 * precision * recall / (precision + recall)
    }
  }, numeric(1))

  mean(f1_values)
}

multiclass_log_loss <- function(actual, probability_matrix) {
  actual_index <- match(as.character(actual), colnames(probability_matrix))

  if (any(is.na(actual_index))) {
    stop("Probability matrix columns do not match observed classes.")
  }

  chosen_probabilities <- probability_matrix[
    cbind(seq_len(nrow(probability_matrix)), actual_index)
  ]

  -mean(log(pmax(chosen_probabilities, 1e-15)))
}

evaluate_predictions <- function(
    actual,
    predicted,
    probability_matrix,
    model_name
) {
  actual_num <- as.integer(actual)
  predicted_num <- as.integer(predicted)

  tibble(
    model = model_name,
    accuracy = mean(actual == predicted),
    macro_f1 = macro_f1(actual, predicted),
    ordinal_mae = mean(abs(actual_num - predicted_num)),
    within_one_category = mean(abs(actual_num - predicted_num) <= 1),
    log_loss = multiclass_log_loss(actual, probability_matrix)
  )
}

make_confusion_long <- function(actual, predicted, model_name) {
  as.data.frame(table(actual = actual, predicted = predicted)) %>%
    as_tibble() %>%
    mutate(model = model_name, .before = 1)
}

majority_probabilities <- function(train_outcome, test_n) {
  proportions <- prop.table(table(train_outcome))
  classes <- levels(train_outcome)

  probability_vector <- setNames(
    vapply(classes, function(x) {
      if (x %in% names(proportions)) unname(proportions[[x]]) else 0
    }, numeric(1)),
    classes
  )

  matrix(
    rep(probability_vector, each = test_n),
    nrow = test_n,
    byrow = FALSE,
    dimnames = list(NULL, classes)
  )
}

extract_polr_coefficients <- function(model, model_name) {
  coefficient_table <- coef(summary(model)) %>%
    as.data.frame() %>%
    tibble::rownames_to_column("term") %>%
    as_tibble() %>%
    rename(
      estimate = Value,
      std_error = `Std. Error`,
      statistic = `t value`
    ) %>%
    mutate(
      p_value_approx = 2 * pnorm(abs(statistic), lower.tail = FALSE),
      odds_ratio = if_else(
        str_detect(term, "\\|"),
        NA_real_,
        exp(estimate)
      ),
      model = model_name,
      .before = 1
    )

  coefficient_table
}

# ----------------------------- 3. Read and validate data -----------------------

if (!file.exists(INPUT_FILE)) {
  stop(
    "Input file not found: ", INPUT_FILE,
    "\nPlace the CSV in the repository root or change INPUT_FILE."
  )
}

raw_data <- read_csv(INPUT_FILE, show_col_types = FALSE)

required_columns <- c("rating_value", "review_text")
missing_columns <- setdiff(required_columns, names(raw_data))

if (length(missing_columns) > 0) {
  stop(
    "The input file is missing required columns: ",
    paste(missing_columns, collapse = ", ")
  )
}

reviews <- raw_data %>%
  transmute(
    id = row_number(),
    rating_value = suppressWarnings(as.integer(rating_value)),
    review_text = as.character(review_text)
  ) %>%
  filter(
    !is.na(rating_value),
    rating_value %in% 1:5,
    !is.na(review_text),
    str_squish(review_text) != ""
  ) %>%
  mutate(
    review_text = str_squish(review_text),
    rating_category = rating_to_category(rating_value)
  )

if (nrow(reviews) < 30) {
  stop("Too few valid reviews remain after cleaning.")
}

if (n_distinct(reviews$rating_category) < 3) {
  stop("All three rating categories are required: low, medium, and high.")
}

message("Valid reviews: ", nrow(reviews))

# ----------------------------- 4. Descriptive output --------------------------

rating_plot <- reviews %>%
  count(rating_category) %>%
  ggplot(aes(x = rating_category, y = n)) +
  geom_col() +
  geom_text(aes(label = n), vjust = -0.3) +
  labs(
    title = "Distribution of Yelp Rating Categories",
    x = "Rating category",
    y = "Number of reviews"
  ) +
  theme_minimal(base_size = 12)

ggsave(
  filename = file.path(OUTPUT_DIR, "rating_distribution.png"),
  plot = rating_plot,
  width = 7,
  height = 5,
  dpi = 300
)

# ----------------------------- 5. Leakage-safe split --------------------------

split_data <- stratified_split(
  data = reviews,
  outcome = rating_category,
  train_prop = TRAIN_PROP,
  seed = SEED
)

train_reviews <- split_data$train
test_reviews <- split_data$test

if (nrow(test_reviews) == 0) {
  stop("The test set is empty. Reduce TRAIN_PROP.")
}

message("Training reviews: ", nrow(train_reviews))
message("Testing reviews: ", nrow(test_reviews))

# ----------------------------- 6. Lexicon construction ------------------------

train_tokens <- tokenize_reviews(train_reviews)

custom_lexicon <- build_custom_lexicon(
  train_tokens = train_tokens,
  min_frequency = MIN_WORD_FREQUENCY,
  threshold = LOG_ODDS_THRESHOLD
)

bing_lexicon <- make_bing_lexicon()
combined_lexicon <- combine_lexicons(bing_lexicon, custom_lexicon)

write_csv(
  custom_lexicon,
  file.path(OUTPUT_DIR, "custom_lexicon.csv")
)

message("Training-specific sentiment words: ", nrow(custom_lexicon))

# ----------------------------- 7. Feature construction ------------------------

train_bing <- score_reviews(train_reviews, bing_lexicon)
test_bing <- score_reviews(test_reviews, bing_lexicon)

train_combined <- score_reviews(train_reviews, combined_lexicon)
test_combined <- score_reviews(test_reviews, combined_lexicon)

# ----------------------------- 8. Model estimation ----------------------------

# OLS is retained only as an interpretable association baseline.
ols_combined <- lm(
  rating_value ~ sentiment_score + log1p(review_length),
  data = train_combined
)

# Ordered logistic models respect the ordered low/medium/high outcome.
olr_bing <- safe_polr(
  rating_category ~ sentiment_score + log1p(review_length),
  data = train_bing
)

olr_combined <- safe_polr(
  rating_category ~ sentiment_score + log1p(review_length),
  data = train_combined
)

# ----------------------------- 9. Test predictions ----------------------------

category_levels <- levels(reviews$rating_category)

# Majority-class baseline
majority_probs <- majority_probabilities(
  train_outcome = train_combined$rating_category,
  test_n = nrow(test_combined)
)

majority_pred <- factor(
  colnames(majority_probs)[max.col(majority_probs, ties.method = "first")],
  levels = category_levels,
  ordered = TRUE
)

# Bing model
bing_probs <- predict(olr_bing, newdata = test_bing, type = "probs")
bing_probs <- bing_probs[, category_levels, drop = FALSE]

bing_pred <- factor(
  colnames(bing_probs)[max.col(bing_probs, ties.method = "first")],
  levels = category_levels,
  ordered = TRUE
)

# Combined model
combined_probs <- predict(
  olr_combined,
  newdata = test_combined,
  type = "probs"
)
combined_probs <- combined_probs[, category_levels, drop = FALSE]

combined_pred <- factor(
  colnames(combined_probs)[max.col(combined_probs, ties.method = "first")],
  levels = category_levels,
  ordered = TRUE
)

# ----------------------------- 10. Evaluation ---------------------------------

metrics <- bind_rows(
  evaluate_predictions(
    actual = test_combined$rating_category,
    predicted = majority_pred,
    probability_matrix = majority_probs,
    model_name = "Majority baseline"
  ),
  evaluate_predictions(
    actual = test_bing$rating_category,
    predicted = bing_pred,
    probability_matrix = bing_probs,
    model_name = "Bing sentiment ordered logit"
  ),
  evaluate_predictions(
    actual = test_combined$rating_category,
    predicted = combined_pred,
    probability_matrix = combined_probs,
    model_name = "Combined lexicon ordered logit"
  )
) %>%
  arrange(desc(accuracy))

write_csv(metrics, file.path(OUTPUT_DIR, "model_metrics.csv"))

confusion_matrices <- bind_rows(
  make_confusion_long(
    test_combined$rating_category,
    majority_pred,
    "Majority baseline"
  ),
  make_confusion_long(
    test_bing$rating_category,
    bing_pred,
    "Bing sentiment ordered logit"
  ),
  make_confusion_long(
    test_combined$rating_category,
    combined_pred,
    "Combined lexicon ordered logit"
  )
)

write_csv(
  confusion_matrices,
  file.path(OUTPUT_DIR, "confusion_matrices.csv")
)

combined_confusion_plot <- confusion_matrices %>%
  filter(model == "Combined lexicon ordered logit") %>%
  ggplot(aes(x = predicted, y = actual, fill = Freq)) +
  geom_tile() +
  geom_text(aes(label = Freq)) +
  labs(
    title = "Test-Set Confusion Matrix: Combined Lexicon Model",
    x = "Predicted category",
    y = "Observed category",
    fill = "Count"
  ) +
  theme_minimal(base_size = 12)

ggsave(
  filename = file.path(OUTPUT_DIR, "confusion_matrix_combined.png"),
  plot = combined_confusion_plot,
  width = 7,
  height = 5,
  dpi = 300
)

# ----------------------------- 11. Predictions and error analysis -------------

test_predictions <- test_combined %>%
  transmute(
    id,
    review_text,
    observed_rating = rating_value,
    observed_category = rating_category,
    sentiment_score,
    positive_count,
    negative_count,
    review_length,
    negated_sentiment_count,
    predicted_category = combined_pred,
    probability_low = combined_probs[, "low"],
    probability_medium = combined_probs[, "medium"],
    probability_high = combined_probs[, "high"],
    correct = observed_category == predicted_category,
    ordinal_error = abs(
      as.integer(observed_category) - as.integer(predicted_category)
    ),
    confidence = pmax(
      probability_low,
      probability_medium,
      probability_high
    )
  )

write_csv(
  test_predictions,
  file.path(OUTPUT_DIR, "test_predictions.csv")
)

error_analysis <- test_predictions %>%
  filter(!correct) %>%
  arrange(desc(ordinal_error), desc(confidence)) %>%
  slice_head(n = 50)

write_csv(
  error_analysis,
  file.path(OUTPUT_DIR, "error_analysis.csv")
)

# ----------------------------- 12. Model coefficients -------------------------

ols_coefficients <- broom::tidy(ols_combined) %>%
  mutate(
    model = "OLS combined lexicon",
    odds_ratio = NA_real_,
    .before = 1
  ) %>%
  rename(
    std_error = std.error,
    statistic = statistic,
    p_value = p.value
  ) %>%
  select(
    model,
    term,
    estimate,
    std_error,
    statistic,
    p_value,
    odds_ratio
  )

olr_bing_coefficients <- extract_polr_coefficients(
  olr_bing,
  "Bing sentiment ordered logit"
) %>%
  rename(p_value = p_value_approx) %>%
  select(
    model,
    term,
    estimate,
    std_error,
    statistic,
    p_value,
    odds_ratio
  )

olr_combined_coefficients <- extract_polr_coefficients(
  olr_combined,
  "Combined lexicon ordered logit"
) %>%
  rename(p_value = p_value_approx) %>%
  select(
    model,
    term,
    estimate,
    std_error,
    statistic,
    p_value,
    odds_ratio
  )

model_coefficients <- bind_rows(
  ols_coefficients,
  olr_bing_coefficients,
  olr_combined_coefficients
)

write_csv(
  model_coefficients,
  file.path(OUTPUT_DIR, "model_coefficients.csv")
)

# ----------------------------- 13. Probability example ------------------------

sentiment_grid <- tibble(
  sentiment_score = seq(
    floor(quantile(train_combined$sentiment_score, 0.05, na.rm = TRUE)),
    ceiling(quantile(train_combined$sentiment_score, 0.95, na.rm = TRUE)),
    length.out = 25
  ),
  review_length = median(train_combined$review_length, na.rm = TRUE)
)

example_probs <- predict(
  olr_combined,
  newdata = sentiment_grid,
  type = "probs"
)

probability_example <- bind_cols(
  sentiment_grid,
  as_tibble(example_probs)
)

write_csv(
  probability_example,
  file.path(OUTPUT_DIR, "predicted_probability_example.csv")
)

# ----------------------------- 14. Console summary ----------------------------

cat("\n====================== ANALYSIS COMPLETE ======================\n")
cat("Input reviews:", nrow(reviews), "\n")
cat("Training reviews:", nrow(train_reviews), "\n")
cat("Testing reviews:", nrow(test_reviews), "\n")
cat("Custom sentiment words:", nrow(custom_lexicon), "\n\n")

print(metrics)

cat("\nCombined-model coefficients:\n")
print(
  olr_combined_coefficients %>%
    filter(!str_detect(term, "\\|"))
)

cat("\nFiles written to:", OUTPUT_DIR, "\n")
cat("===============================================================\n")
