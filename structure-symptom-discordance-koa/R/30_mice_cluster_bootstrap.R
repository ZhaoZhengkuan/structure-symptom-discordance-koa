#!/usr/bin/env Rscript

# MICE + participant-cluster bootstrap skeleton for final submission.
# This script is intentionally conservative: it pools logistic 96-month TKR
# models with Rubin's rules and provides a bootstrap wrapper for Fine-Gray.
# Run after 20_fine_gray_cmprsk.R works locally.

suppressPackageStartupMessages({
  library(data.table)
  library(dplyr)
  library(mice)
  library(cmprsk)
})

args <- commandArgs(trailingOnly = TRUE)
get_arg <- function(flag, default) {
  hit <- which(args == flag)
  if (length(hit) == 0 || hit == length(args)) return(default)
  as.numeric(args[hit + 1])
}
M <- get_arg("--m", 20)
B <- get_arg("--B", 50)

root <- getwd()
tables <- file.path(root, "outputs", "tables")
outdir <- file.path(root, "outputs", "upgrades")
dir.create(outdir, showWarnings = FALSE, recursive = TRUE)

baseline <- fread(file.path(tables, "oai_baseline_knee.csv"))
pheno <- fread(file.path(tables, "oai_discordance_phenotypes.csv"))
outcomes <- fread(file.path(tables, "oai_outcomes_knee.csv"))
horizon_days <- 96 * 30.4375

d <- baseline %>%
  inner_join(pheno[, .(kid, phenotype)], by = "kid") %>%
  inner_join(outcomes, by = c("kid", "id", "side")) %>%
  mutate(
    participant_id = id,
    phenotype = factor(phenotype, levels = c("concordant_low", "structural_dominant", "symptom_dominant", "concordant_high")),
    event96 = as.integer(tkr_event == 1 & !is.na(tkr_days) & tkr_days <= horizon_days),
    death96 = as.integer(death_event == 1 & !is.na(death_days) & death_days <= horizon_days)
  )

vars <- c("event96", "phenotype", "age", "sex", "race", "site", "bmi", "kl_base", "mjsw_base", "fta_base",
          "cesd", "comorbidity", "income", "education", "pase", "nsaid")
vars <- vars[vars %in% names(d)]
imp_dat <- d[, ..vars]
imp <- mice(imp_dat, m = M, maxit = 20, seed = 20250620, printFlag = FALSE)
fit <- with(imp, glm(event96 ~ phenotype + age + sex + race + site + bmi + kl_base + mjsw_base + fta_base +
                       cesd + comorbidity + income + nsaid,
                     family = binomial()))
pooled <- pool(fit)
pooled_tab <- summary(pooled, conf.int = TRUE, exponentiate = TRUE)
write.csv(pooled_tab, file.path(outdir, "mice_pooled_logistic_tkr96.csv"), row.names = FALSE)

boot_fine_gray <- function(data, B = 200, seed = 20250620) {
  set.seed(seed)
  ids <- unique(data$participant_id)
  out <- vector("list", B)
  covars <- c("phenotype", "age", "sex", "race", "site", "bmi", "kl_base", "mjsw_base", "fta_base", "cesd", "comorbidity", "income", "nsaid")
  covars <- covars[covars %in% names(data)]
  for (b in seq_len(B)) {
    samp <- sample(ids, length(ids), replace = TRUE)
    bd <- bind_rows(lapply(seq_along(samp), function(i) {
      tmp <- data[data$participant_id == samp[i], ]
      tmp$boot_cluster <- i
      tmp
    }))
    bd <- bd %>%
      mutate(
        tkr_time = ifelse(!is.na(tkr_days), tkr_days, Inf),
        death_time = ifelse(!is.na(death_days), death_days, Inf),
        ftime = pmin(tkr_time, death_time, horizon_days, na.rm = TRUE),
        fstatus = case_when(
          tkr_event == 1 & tkr_time <= death_time & tkr_time <= horizon_days ~ 1L,
          death_event == 1 & death_time < tkr_time & death_time <= horizon_days ~ 2L,
          TRUE ~ 0L
        ),
        ftime = pmax(ftime, 1)
      )
    cc <- bd %>% select(all_of(c("ftime", "fstatus", covars))) %>% na.omit()
    x <- model.matrix(as.formula(paste("~", paste(covars, collapse = "+"))), cc)[, -1, drop = FALSE]
    fit <- tryCatch(crr(cc$ftime, cc$fstatus, cov1 = x, failcode = 1, cencode = 0), error = function(e) NULL)
    if (!is.null(fit)) {
      out[[b]] <- data.frame(iter = b, term = names(fit$coef), log_subdistribution_hr = as.numeric(fit$coef))
    }
  }
  bind_rows(out)
}

boot <- boot_fine_gray(d, B = B)
write.csv(boot, file.path(outdir, "cluster_bootstrap_fine_gray_coefficients.csv"), row.names = FALSE)
boot_ci <- boot %>%
  group_by(term) %>%
  summarise(
    boot_n = n(),
    subdistribution_hr = exp(median(log_subdistribution_hr, na.rm = TRUE)),
    ci_low = exp(quantile(log_subdistribution_hr, 0.025, na.rm = TRUE)),
    ci_high = exp(quantile(log_subdistribution_hr, 0.975, na.rm = TRUE)),
    .groups = "drop"
  )
write.csv(boot_ci, file.path(outdir, "cluster_bootstrap_fine_gray_ci.csv"), row.names = FALSE)

cat("MICE and cluster bootstrap complete. m=", M, " B=", B, "\n")
