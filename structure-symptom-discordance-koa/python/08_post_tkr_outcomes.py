"""
D1_post_tkr_outcomes.py  (skeleton — needs OAI post-replacement WOMAC)
=====================================================================
The high-impact clinical hook: is the symptom-dominant phenotype a pre-operatively
identifiable "TKR but still in pain" group? Pre-operative pain sensitisation is a
known predictor of poor TKR response, and SyD is exactly the preserved-structure /
high-pain / high-comorbidity group.

ANALYSIS PLAN
  Cohort   : knees that underwent confirmed TKR during follow-up.
  Exposure : pre-operative phenotype (esp. SyD vs others), defined from visits
             BEFORE the replacement date (use the landmark/early-window phenotype
             so it is genuinely pre-operative).
  Outcomes : post-TKR WOMAC pain at the first 1-2 visits after surgery;
             non-response by an OMERACT-OARSI-style rule (e.g., < clinically
             important improvement in pain/function), and absolute residual pain.
  Models   : mixed model for post-TKR WOMAC pain ~ phenotype + age + sex + BMI +
             pre-op pain + comorbidity + depression; logistic for non-response.
             Cluster by participant.
  Key test : does SyD predict higher residual pain / higher non-response after
             adjustment for pre-operative pain (i.e., beyond just "started higher")?

DATA
  - identify TKR knees and replacement date (OUTCOMES99)
  - post-TKR visit WOMAC pain/function (visits with month > replacement month)
  - pre-operative phenotype (early-window / landmark definition)
"""
import numpy as np
import pandas as pd

# response thresholds (adapt to your scaling; WOMAC pain here on 0-20)
MCII_PAIN_ABS = 2.0       # minimal clinically important improvement (example)
PASS_PAIN = 4.0           # patient-acceptable symptom state (example)


def assemble_post_tkr(panel_long, outcomes, phenotype_preop):
    tkr = outcomes[outcomes["tkr_event"] == 1][["kid", "id", "tkr_months"]]
    post = panel_long.merge(tkr, on=["kid", "id"])
    post = post[post["month"] > post["tkr_months"]]                 # post-op visits
    # first post-op WOMAC per knee
    first_post = (post.sort_values("month").groupby("kid").first()
                      .reset_index()[["kid", "womac_pain", "womac_func"]]
                      .rename(columns={"womac_pain": "post_pain",
                                       "womac_func": "post_func"}))
    pre = panel_long.merge(tkr, on=["kid", "id"])
    pre = pre[pre["month"] <= pre["tkr_months"]]
    last_pre = (pre.sort_values("month").groupby("kid").last()
                   .reset_index()[["kid", "womac_pain"]]
                   .rename(columns={"womac_pain": "pre_pain"}))
    d = (first_post.merge(last_pre, on="kid")
                   .merge(phenotype_preop[["kid", "phenotype"]], on="kid"))
    d["improvement"] = d["pre_pain"] - d["post_pain"]
    d["non_response"] = ((d["improvement"] < MCII_PAIN_ABS) &
                         (d["post_pain"] > PASS_PAIN)).astype(int)
    return d


def analyse(d):
    """TODO: mixed model post_pain ~ phenotype + pre_pain + covariates (cluster id);
    logistic non_response ~ phenotype + covariates. Report SyD vs others."""
    summ = (d.groupby("phenotype")
             .agg(n=("kid", "size"), post_pain=("post_pain", "mean"),
                  non_response=("non_response", "mean")).round(2))
    print(summ)
    return summ


if __name__ == "__main__":
    print(__doc__)
    print("Provide post-TKR WOMAC + pre-operative phenotype, then assemble + analyse.")
