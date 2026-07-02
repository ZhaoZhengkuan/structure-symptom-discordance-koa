#!/usr/bin/env Rscript

required <- c(
  "data.table", "dplyr", "tidyr", "lcmm", "cmprsk", "survival",
  "mice", "broom", "purrr", "readr", "ggplot2"
)

repos <- "https://cloud.r-project.org"
installed <- rownames(installed.packages())
missing <- setdiff(required, installed)
if (length(missing) > 0) {
  install.packages(missing, repos = repos)
}

cat("R package check complete.\n")
cat("Installed/available packages:\n")
print(required)
