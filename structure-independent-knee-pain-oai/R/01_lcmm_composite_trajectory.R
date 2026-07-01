#!/usr/bin/env Rscript

# Practical joint lcmm upgrade:
# two longitudinal processes are modelled jointly with multlcmm:
#   1) structural severity composite = z(-mJSW) + z(KL)
#   2) symptom severity composite    = z(WOMAC pain) + z(WOMAC function)
#
# This is substantially more stable than the four-marker multlcmm, while still
# estimating a joint structure-symptom latent trajectory model.

suppressPackageStartupMessages({
  library(data.table)
  library(dplyr)
  library(lcmm)
})

args <- commandArgs(trailingOnly = TRUE)
get_arg <- function(flag, default) {
  hit <- which(args == flag)
  if (length(hit) == 0 || hit == length(args)) return(default)
  as.numeric(args[hit + 1])
}
KMAX <- get_arg("--kmax", 4)
REP <- get_arg("--rep", 5)
MAXITER <- get_arg("--maxiter", 120)

root <- getwd()
tables <- file.path(root, "outputs", "tables")
outdir <- file.path(root, "outputs", "upgrades")
dir.create(outdir, showWarnings = FALSE, recursive = TRUE)

panel <- fread(file.path(tables, "oai_panel_long.csv"))

dat <- panel %>%
  filter(!is.na(mjsw), !is.na(kl), !is.na(womac_pain), !is.na(womac_func)) %>%
  mutate(
    kid_num = as.integer(factor(kid)),
    time_y = month / 12,
    mjsw_sev = -as.numeric(mjsw),
    kl = as.numeric(kl),
    womac_pain = as.numeric(womac_pain),
    womac_func = as.numeric(womac_func)
  ) %>%
  group_by(kid) %>%
  filter(n_distinct(month) >= 2) %>%
  ungroup()

dat <- dat %>%
  mutate(
    structure_comp = as.numeric(scale(mjsw_sev)) + as.numeric(scale(kl)),
    symptom_comp = as.numeric(scale(womac_pain)) + as.numeric(scale(womac_func))
  )
dat$structure_comp <- as.numeric(scale(dat$structure_comp))
dat$symptom_comp <- as.numeric(scale(dat$symptom_comp))

fit_base <- multlcmm(
  fixed = structure_comp + symptom_comp ~ time_y,
  random = ~ time_y,
  subject = "kid_num",
  link = rep("linear", 2),
  data = dat,
  maxiter = MAXITER,
  verbose = FALSE
)

fits <- list(ng1 = fit_base)
sel <- list(data.frame(
  ng = 1,
  loglik = fit_base$loglik,
  AIC = fit_base$AIC,
  BIC = fit_base$BIC,
  conv = fit_base$conv,
  error = NA
))

if (KMAX >= 2) {
  for (k in 2:KMAX) {
    message("Fitting composite multlcmm ng=", k)
    fit <- tryCatch(
      gridsearch(
        rep = REP,
        maxiter = 40,
        minit = fit_base,
        multlcmm(
          fixed = structure_comp + symptom_comp ~ time_y,
          mixture = ~ time_y,
          random = ~ time_y,
          subject = "kid_num",
          ng = k,
          nwg = TRUE,
          link = rep("linear", 2),
          data = dat,
          maxiter = MAXITER,
          verbose = FALSE
        )
      ),
      error = function(e) e
    )
    fits[[paste0("ng", k)]] <- fit
    if (inherits(fit, "error")) {
      sel[[length(sel) + 1]] <- data.frame(ng = k, loglik = NA, AIC = NA, BIC = NA, conv = NA, error = fit$message)
    } else {
      sel[[length(sel) + 1]] <- data.frame(ng = k, loglik = fit$loglik, AIC = fit$AIC, BIC = fit$BIC, conv = fit$conv, error = NA)
    }
  }
}

selection <- bind_rows(sel)
write.csv(selection, file.path(outdir, "lcmm_composite_joint_model_selection.csv"), row.names = FALSE)

ok <- selection %>% filter(!is.na(BIC), conv %in% c(1, 2)) %>% arrange(BIC)
if (nrow(ok) == 0) {
  stop("No converged composite lcmm model. Inspect lcmm_composite_joint_model_selection.csv.")
}
best_k <- ok$ng[1]
best <- fits[[paste0("ng", best_k)]]
saveRDS(best, file.path(outdir, sprintf("lcmm_composite_joint_best_ng%s.rds", best_k)))
capture.output(summary(best), file = file.path(outdir, "lcmm_composite_joint_best_summary.txt"))

pp <- postprob(best)
if (!"kid_num" %in% names(pp)) pp$kid_num <- as.integer(rownames(pp))
key <- unique(dat[, c("kid", "kid_num")])
pp <- pp %>% left_join(key, by = "kid_num")
write.csv(pp, file.path(outdir, "lcmm_composite_joint_posterior_classes.csv"), row.names = FALSE)

cat("Composite joint lcmm complete. Best ng:", best_k, "\n")
