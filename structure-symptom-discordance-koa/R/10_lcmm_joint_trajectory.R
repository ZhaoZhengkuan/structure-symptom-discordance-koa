#!/usr/bin/env Rscript

# Joint latent class mixed model upgrade for C1.
# Requires: lcmm, data.table, dplyr, tidyr.
# Run from project root:
#   Rscript code/r/00_install_r_packages.R
#   Rscript code/r/10_lcmm_joint_trajectory.R

suppressPackageStartupMessages({
  library(data.table)
  library(dplyr)
  library(lcmm)
})

root <- getwd()
tables <- file.path(root, "outputs", "tables")
outdir <- file.path(root, "outputs", "upgrades")
dir.create(outdir, showWarnings = FALSE, recursive = TRUE)

panel <- fread(file.path(tables, "oai_panel_long.csv"))
baseline <- fread(file.path(tables, "oai_baseline_knee.csv"))

dat <- panel %>%
  filter(!is.na(mjsw), !is.na(womac_pain)) %>%
  mutate(
    kid_num = as.integer(factor(kid)),
    time_y = month / 12,
    # Direction: larger means worse structural state.
    mjsw_sev = -mjsw,
    kl = as.numeric(kl),
    womac_pain = as.numeric(womac_pain),
    womac_func = as.numeric(womac_func)
  ) %>%
  group_by(kid) %>%
  filter(n_distinct(month[!is.na(mjsw_sev) & !is.na(womac_pain)]) >= 2) %>%
  ungroup()

dat <- dat %>%
  mutate(
    mjsw_z = as.numeric(scale(mjsw_sev)),
    kl_z = as.numeric(scale(kl)),
    pain_z = as.numeric(scale(womac_pain)),
    func_z = as.numeric(scale(womac_func))
  )

fit_one <- function(k) {
  message("Fitting multlcmm ng=", k)
  if (k == 1) {
    multlcmm(
      fixed = mjsw_z + kl_z + pain_z + func_z ~ time_y,
      random = ~ time_y,
      subject = "kid_num",
      link = rep("linear", 4),
      data = dat,
      maxiter = 200,
      verbose = FALSE
    )
  } else {
    base <- multlcmm(
      fixed = mjsw_z + kl_z + pain_z + func_z ~ time_y,
      random = ~ time_y,
      subject = "kid_num",
      link = rep("linear", 4),
      data = dat,
      maxiter = 80,
      verbose = FALSE
    )
    gridsearch(
      rep = 25,
      maxiter = 80,
      minit = base,
      multlcmm(
        fixed = mjsw_z + kl_z + pain_z + func_z ~ time_y,
        mixture = ~ time_y,
        random = ~ time_y,
        subject = "kid_num",
        ng = k,
        nwg = TRUE,
        link = rep("linear", 4),
        data = dat,
        maxiter = 200,
        verbose = FALSE
      )
    )
  }
}

ks <- 1:5
fits <- vector("list", length(ks))
names(fits) <- paste0("ng", ks)
sel <- list()
for (k in ks) {
  fit <- tryCatch(fit_one(k), error = function(e) e)
  fits[[paste0("ng", k)]] <- fit
  if (inherits(fit, "error")) {
    sel[[length(sel) + 1]] <- data.frame(ng = k, loglik = NA, AIC = NA, BIC = NA, conv = NA, error = fit$message)
  } else {
    sel[[length(sel) + 1]] <- data.frame(ng = k, loglik = fit$loglik, AIC = fit$AIC, BIC = fit$BIC, conv = fit$conv, error = NA)
  }
}
selection <- bind_rows(sel)
write.csv(selection, file.path(outdir, "lcmm_joint_model_selection.csv"), row.names = FALSE)

ok <- selection %>% filter(!is.na(BIC), conv %in% c(1, 2)) %>% arrange(BIC)
if (nrow(ok) == 0) {
  stop("No converged lcmm model. Inspect lcmm_joint_model_selection.csv.")
}
best_k <- ok$ng[1]
best <- fits[[paste0("ng", best_k)]]
saveRDS(best, file.path(outdir, sprintf("lcmm_joint_best_ng%s.rds", best_k)))

pp <- postprob(best)
if (!"kid_num" %in% names(pp)) {
  pp$kid_num <- as.integer(rownames(pp))
}
key <- unique(dat[, c("kid", "kid_num")])
pp <- pp %>% left_join(key, by = "kid_num")
write.csv(pp, file.path(outdir, "lcmm_joint_posterior_classes.csv"), row.names = FALSE)

capture.output(summary(best), file = file.path(outdir, "lcmm_joint_best_summary.txt"))
cat("Best ng:", best_k, "\n")
