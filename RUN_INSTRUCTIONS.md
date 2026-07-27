# Running the revised analysis

1. Keep `FinalCode_revised.R` and `all_CM_restaurants_Columbus.csv` in the repository root.
2. Open the project in RStudio.
3. Install packages once:

```r
install.packages(c(
  "dplyr", "tidyr", "readr", "stringr",
  "ggplot2", "tidytext", "MASS", "broom"
))
```

4. Run:

```r
source("FinalCode.R")
```

The script creates an `outputs/` folder containing model metrics, predictions,
error cases, coefficients, the training-only custom lexicon, and figures.

## Main methodological correction

The restaurant-specific sentiment lexicon is created only from the training
sample. The held-out test set is not used to select words or estimate models,
which prevents the target leakage in the original exploratory code.
