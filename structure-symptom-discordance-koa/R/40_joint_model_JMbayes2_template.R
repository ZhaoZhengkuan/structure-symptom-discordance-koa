# Template only: requires JMbayes2 and multi-hour/server computation.
# Inputs: oai_panel_long.csv, oai_outcomes_knee.csv, oai_discordance_phenotypes.csv.
# Fit separate longitudinal mixed models for WOMAC pain and mJSW, then jointModelBayes()
# with cause-specific TKR hazard. Check Rhat/effective sample size before using.
library(nlme)
library(JMbayes2)
panel <- read.csv("../tables/oai_panel_long.csv")
out <- read.csv("../tables/oai_outcomes_knee.csv")
lme_pain <- lme(womac_pain ~ ns(month, 3) + phenotype, random = ~ month | kid, data = panel, na.action = na.omit)
# cox_tkr <- coxph(Surv(tkr_days, tkr_event) ~ phenotype + age + sex + bmi, data = ...)
# jm <- jm(cox_tkr, list(lme_pain), time_var = "month", n_iter = 20000)
