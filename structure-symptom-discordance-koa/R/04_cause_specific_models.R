#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(data.table)
  library(dplyr)
  library(survival)
})

root <- getwd()
tables <- file.path(root, "outputs", "tables")
outdir <- file.path(root, "outputs", "optimization")
dir.create(outdir, showWarnings = FALSE, recursive = TRUE)

baseline <- fread(file.path(tables, "oai_baseline_knee.csv"))
pheno <- fread(file.path(tables, "oai_discordance_phenotypes.csv"))
outcomes <- fread(file.path(tables, "oai_outcomes_knee.csv"))
panel <- fread(file.path(tables, "oai_panel_long.csv"))

horizon_days <- 96 * 30.4375
p0 <- panel[month == 0, .(kid, pain0 = first(womac_pain), func0 = first(womac_func))]
d <- baseline %>%
  inner_join(pheno[, .(kid, phenotype)], by = "kid") %>%
  inner_join(outcomes, by = c("kid", "id", "side")) %>%
  left_join(p0, by = "kid") %>%
  mutate(
    phenotype = factor(phenotype, levels = c("concordant_low", "structural_dominant", "symptom_dominant", "concordant_high")),
    participant_id = id,
    time = ifelse(!is.na(tkr_days), pmin(tkr_days, horizon_days), horizon_days),
    time = pmax(time, 1),
    event = as.integer(tkr_event == 1 & !is.na(tkr_days) & tkr_days <= horizon_days),
    event96 = event
  )

cox_formula <- Surv(time, event) ~ phenotype + age + sex + race + site + bmi + kl_base + mjsw_base + fta_base + cesd + comorbidity + income + nsaid + pain0 + func0 + cluster(participant_id)
cox_fit <- coxph(cox_formula, data = d, ties = "efron", robust = TRUE)
cox_sum <- summary(cox_fit)
cox_tab <- data.frame(
  term = rownames(cox_sum$coefficients),
  log_cause_specific_hr = cox_sum$coefficients[, "coef"],
  robust_se = cox_sum$coefficients[, "robust se"],
  cause_specific_hr = cox_sum$coefficients[, "exp(coef)"],
  ci_low = cox_sum$conf.int[, "lower .95"],
  ci_high = cox_sum$conf.int[, "upper .95"],
  p = cox_sum$coefficients[, "Pr(>|z|)"],
  row.names = NULL
)
write.csv(cox_tab, file.path(outdir, "A7_cause_specific_cox_cluster_robust.csv"), row.names = FALSE)
capture.output(summary(cox_fit), file = file.path(outdir, "A7_cause_specific_cox_cluster_robust_summary.txt"))

logit_df <- d %>% select(event96, phenotype, age, sex, race, site, bmi, kl_base, mjsw_base, fta_base, cesd, comorbidity, income, nsaid, pain0, func0, participant_id) %>% na.omit()
logit_fit <- glm(event96 ~ phenotype + age + sex + race + site + bmi + kl_base + mjsw_base + fta_base + cesd + comorbidity + income + nsaid + pain0 + func0,
                 data = logit_df, family = binomial())
# Cluster-robust covariance via sandwich formula implemented directly. Drop
# aliased/non-estimable columns to avoid singular design matrices.
X_full <- model.matrix(logit_fit)
keep <- !is.na(coef(logit_fit))
X <- X_full[, keep, drop = FALSE]
coef <- coef(logit_fit)[keep]
mu <- fitted(logit_fit)
u <- logit_df$event96 - mu
XtWX <- t(X) %*% (X * as.numeric(mu * (1 - mu)))
bread <- solve(XtWX)
ids <- as.factor(logit_df$participant_id)
meat <- matrix(0, ncol(X), ncol(X))
for (lev in levels(ids)) {
  idx <- which(logit_df$participant_id == lev)
  if (length(idx) == 0) next
  Xi <- X[idx, , drop = FALSE]
  ui <- u[idx]
  score <- t(Xi) %*% ui
  meat <- meat + score %*% t(score)
}
vc <- bread %*% meat %*% bread
se <- sqrt(diag(vc))
z <- coef / se
logit_tab <- data.frame(
  term = names(coef),
  log_or = as.numeric(coef),
  cluster_robust_se = as.numeric(se),
  or = exp(coef),
  ci_low = exp(coef - 1.96 * se),
  ci_high = exp(coef + 1.96 * se),
  p = 2 * pnorm(-abs(z)),
  row.names = NULL
)
write.csv(logit_tab, file.path(outdir, "A4_logistic_tkr96_cluster_robust.csv"), row.names = FALSE)

cat("A4/A7 cluster robust models complete.\n")
