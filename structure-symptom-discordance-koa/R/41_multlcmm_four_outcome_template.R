# Template only: requires lcmm and convergence screening across random starts.
library(lcmm)
panel <- read.csv("../tables/oai_panel_long.csv")
# multlcmm(cbind(mjsw, kl, womac_pain, womac_func) ~ poly(month, 2),
#          random = ~ month, subject = "kid", ng = 2:5, data = panel)
# Report GRoLTS: entropy, class size, posterior probability, random-start reproducibility.
