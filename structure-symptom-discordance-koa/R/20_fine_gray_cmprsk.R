#!/usr/bin/env Rscript

# Fine-Gray competing-risk model for confirmed TKR with death as competing event.
# Requires: cmprsk, survival, data.table, dplyr, broom.
# Run:
#   Rscript code/r/20_fine_gray_cmprsk.R

suppressPackageStartupMessages({
  library(data.table)
  library(dplyr)
  library(cmprsk)
})

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
    phenotype = factor(phenotype, levels = c("concordant_low", "structural_dominant", "symptom_dominant", "concordant_high")),
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

covars <- c("phenotype", "age", "sex", "race", "site", "bmi", "kl_base", "mjsw_base", "fta_base", "cesd", "comorbidity", "income", "nsaid")
covars <- covars[covars %in% names(d)]
complete <- d %>% select(all_of(c("ftime", "fstatus", covars))) %>% na.omit()
x <- model.matrix(as.formula(paste("~", paste(covars, collapse = "+"))), complete)[, -1, drop = FALSE]

fg <- crr(ftime = complete$ftime, fstatus = complete$fstatus, cov1 = x, failcode = 1, cencode = 0)
coef <- fg$coef
se <- sqrt(diag(fg$var))
res <- data.frame(
  term = names(coef),
  log_subdistribution_hr = as.numeric(coef),
  se = as.numeric(se),
  subdistribution_hr = exp(coef),
  ci_low = exp(coef - 1.96 * se),
  ci_high = exp(coef + 1.96 * se),
  z = coef / se,
  p = 2 * pnorm(-abs(coef / se))
)
write.csv(res, file.path(outdir, "fine_gray_cmprsk_results.csv"), row.names = FALSE)
capture.output(summary(fg), file = file.path(outdir, "fine_gray_cmprsk_summary.txt"))

# Non-parametric cumulative incidence by phenotype.
ci <- cuminc(d$ftime, d$fstatus, group = d$phenotype, cencode = 0)
capture.output(print(ci), file = file.path(outdir, "cuminc_by_phenotype_summary.txt"))

cat("Fine-Gray model complete. N complete cases:", nrow(complete), "\n")
