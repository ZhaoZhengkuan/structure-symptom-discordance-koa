"""
C1_moaks_mediation.py  (skeleton — needs OAI MRI MOAKS files)
=============================================================
Turns the "symptoms beyond structure" finding from description into MECHANISM:
do MRI-visible inflammatory features (effusion-synovitis, Hoffa-synovitis) and
bone-marrow lesions (BMLs) mediate the symptom excess of the symptom-dominant
phenotype?

DATA (OAI MRI subset)
  - kMRI_SQ_MOAKS_* readings -> effusion-synovitis grade, Hoffa-synovitis grade,
    BML size/number (summed), cartilage MOAKS as needed
  - merge to the analytic knees by (id, side); only the MRI subset has these
  - because the MRI subset is selected, fit inverse-probability-of-selection
    weights (IPSW): P(in MRI subset | baseline covariates), and weight the
    mediation models.

ANALYSIS PLAN
  Exposure A : symptom-dominant vs concordant-low (or the continuous discordance
               index; SyD has preserved structure so structure is held ~constant)
  Mediators M: effusion-synovitis, Hoffa-synovitis, BML burden
  Outcome  Y : WOMAC pain (level or slope)
  Confounders: age, sex, BMI, KL, comorbidity, depression
  Estimate natural direct/indirect effects (g-computation or the
  regression-based mediation of Valeri & VanderWeele), IPSW-weighted, with
  bootstrap CIs; report proportion of the symptom gap mediated by MRI inflammation.
"""
import numpy as np
import pandas as pd

# from sklearn.linear_model import LogisticRegression  # for IPSW
# (reuse c1_mediation.gcomp_mediation, adapted for weights)

MEDIATORS = ["effusion_synovitis", "hoffa_synovitis", "bml_burden"]
CONFOUNDERS = ["age", "sex", "bmi", "kl_base", "comorbidity", "cesd"]


def load_moaks(moaks_paths):
    """Read kMRI_SQ_MOAKS_* files; return per-knee (id, side, mediators).
    Map MOAKS columns to effusion-synovitis, Hoffa-synovitis and summed BML."""
    raise NotImplementedError("Map your MOAKS release columns here.")


def selection_weights(analytic, mri_subset_kids):
    """IPSW: P(knee in MRI subset | baseline covariates); return weights."""
    analytic = analytic.copy()
    analytic["in_mri"] = analytic["kid"].isin(mri_subset_kids).astype(int)
    # fit logistic P(in_mri ~ covariates); w = 1/phat for those in subset
    raise NotImplementedError("Fit IPSW logistic and return 1/phat for MRI knees.")


def moaks_mediation(df_mri, weights):
    """IPSW-weighted natural direct/indirect effects of SyD on WOMAC pain through
    MRI inflammation (effusion-synovitis, Hoffa, BML). Bootstrap CIs by id."""
    # A = (phenotype == 'symptom_dominant'); Y = womac_pain; M = MEDIATORS
    # 1. weighted mediator models  M_j ~ A + C
    # 2. weighted outcome model    Y ~ A + M + C
    # 3. g-computation NDE/NIE; proportion mediated; E-value
    raise NotImplementedError("Implement weighted g-computation (see c1_mediation).")


if __name__ == "__main__":
    print(__doc__)
    print("Provide MOAKS files + MRI-subset ids, then run IPSW + mediation.")
