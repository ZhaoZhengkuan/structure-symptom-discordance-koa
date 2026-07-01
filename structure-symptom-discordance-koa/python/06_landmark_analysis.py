"""
B1_landmark_analysis.py  (skeleton — run on raw OAI longitudinal data)
=====================================================================
Removes the look-ahead bias in the prognostic claim. In the current design the
phenotype uses the full 0-96 month trajectory yet is used to predict TKR that can
occur *within* that same window. A landmark design fixes this.

ANALYSIS PLAN
-------------
Estimand: among knees still TKR-free and under follow-up at the landmark time
t_L, does the discordance phenotype (defined using ONLY visits up to t_L)
predict TKR occurring AFTER t_L?

Steps:
  1. Choose landmark t_L (primary 24 mo; sensitivity 12, 36 mo).
  2. Restrict the trajectory panel to visits with month <= t_L; refit the
     trajectory features and the discordance phenotype on this early window only.
  3. Landmark cohort = knees TKR-free and in follow-up at t_L.
  4. Reset the clock at t_L: time-to-TKR measured from t_L, death competing.
  5. Fine-Gray / cause-specific models for TKR after t_L by phenotype, adjusted
     for the same covariates (incl. baseline pain). Report SHR + absolute risk.
  6. Sensitivity: vary t_L; compare with the non-landmark result to show how much
     of the original association was look-ahead.

Expected columns:
  panel_long: id, side, kid, visit, month, mjsw, kl, womac_pain, womac_func
  outcomes:   kid, id, tkr_months, tkr_event, death_months, death_event
  baseline:   kid, id, + covariates incl. womac_pain_base, womac_func_base
"""
import numpy as np
import pandas as pd

# reuse your existing trajectory + phenotype code on the truncated panel
# from c1_trajectories import fit_dimension
# from c1_phenotypes import cross_classify

LANDMARKS = [24, 12, 36]   # months; primary first


def landmark_dataset(panel_long, baseline, outcomes, t_L):
    early = panel_long[panel_long["month"] <= t_L].copy()
    # require >=2 visits with both markers within the early window
    cnt = (early.assign(s=early.mjsw.notna(), y=early.womac_pain.notna())
                .groupby("kid")[["s", "y"]].sum())
    elig = cnt[(cnt.s >= 2) & (cnt.y >= 2)].index
    early = early[early.kid.isin(elig)]

    # landmark risk set: TKR-free and still in follow-up at t_L
    o = outcomes.copy()
    tkr_before = (o["tkr_event"] == 1) & (o["tkr_months"] <= t_L)
    death_before = (o["death_event"] == 1) & (o["death_months"] <= t_L)
    at_risk = o[~(tkr_before | death_before) & o.kid.isin(elig)].copy()

    # reset clock at t_L (death competing)
    at_risk["t"] = np.where(at_risk.tkr_event == 1, at_risk.tkr_months,
                    np.where(at_risk.death_event == 1, at_risk.death_months,
                             np.maximum(at_risk.tkr_months, at_risk.death_months))) - t_L
    at_risk["status"] = np.where(at_risk.tkr_event == 1, 1,
                         np.where(at_risk.death_event == 1, 2, 0))
    at_risk = at_risk[at_risk["t"] > 0]
    return early, at_risk


def run_landmark(panel_long, baseline, outcomes, fit_dimension, cross_classify):
    results = {}
    for t_L in LANDMARKS:
        early, atrisk = landmark_dataset(panel_long, baseline, outcomes, t_L)
        struct = fit_dimension(early, "structure")          # refit on early window
        sympt  = fit_dimension(early, "symptom")
        pheno  = cross_classify(struct, sympt)
        d = (atrisk.merge(pheno[["kid", "phenotype"]], on="kid")
                   .merge(baseline, on="kid"))
        # -> Fine-Gray / cause-specific of (t, status) ~ phenotype + covariates
        #    (use A_competing_risk_and_clustering.R or lifelines for cause-specific)
        results[t_L] = d
        print(f"landmark {t_L} mo: n={len(d)}, TKR after t_L={(d.status==1).sum()}")
    return results


if __name__ == "__main__":
    print(__doc__)
    print("Load raw panel/baseline/outcomes, then call run_landmark(...).")
