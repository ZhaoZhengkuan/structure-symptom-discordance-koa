"""
A_fix_incremental_value.py
==========================
Turnkey replacement for the incremental-value analysis that fixes the two
blocking issues flagged in review:

  (1) the base model now INCLUDES baseline WOMAC pain + function, so the
      increment reflects the longitudinal phenotype, not omitted pain;
  (2) discrimination/reclassification are estimated OUT-OF-FOLD with
      participant-grouped cross-validation (honest, not in-sample), so the
      optimism that inflated the apparent NRI is removed.

Also reports categorical NRI at pre-specified thresholds (instead of the
upwardly biased continuous NRI), IDI, and calibration (intercept + slope).

USAGE
-----
Provide one knee-level dataframe `df` with (at least) these columns:
    id                 participant id (for grouped CV + clustering)
    event96            1 = TKR by 96 months, 0 otherwise
    phenotype          CL / SD / SyD / CH  (or your labels)
    age, sex, race, site, bmi, kl_base, mjsw_base, fta_base,
    cesd, comorbidity, income, nsaid,
    womac_pain_base, womac_func_base        # <-- the pain variables that were missing

Then:
    res = evaluate_incremental_value(df)
    print(res["summary"]); res["calibration"].to_csv(...)

The __main__ block runs a synthetic smoke test so you can confirm it executes
before pointing it at the real analytic table.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score

# ---- model specification ----------------------------------------------------
NUM = ["age", "bmi", "kl_base", "mjsw_base", "fta_base", "cesd", "comorbidity",
       "income", "womac_pain_base", "womac_func_base"]          # incl. baseline pain
CAT = ["sex", "race", "site", "nsaid"]
RISK_CATEGORIES = [0.0, 0.05, 0.15, 1.01]    # pre-specified reclassification bands


def _make_pipe(num, cat):
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), num),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat),
    ])
    return Pipeline([("pre", pre),
                     ("lr", LogisticRegression(max_iter=4000, class_weight="balanced"))])


def _oof_predictions(df, cols_num, cols_cat, use_phenotype, n_splits=10, n_repeats=5, seed=0):
    """Participant-grouped, stratified, repeated CV out-of-fold risk."""
    y = df["event96"].to_numpy(int)
    groups = df["id"].to_numpy()
    num = cols_num + (["__pheno_dummy__"] if False else [])
    cat = cols_cat + (["phenotype"] if use_phenotype else [])
    X = df[cols_num + cat].copy()
    oof = np.zeros((len(df), n_repeats))
    for r in range(n_repeats):
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed + r)
        for tr, te in sgkf.split(X, y, groups):
            pipe = _make_pipe(cols_num, cat)
            pipe.fit(X.iloc[tr], y[tr])
            oof[te, r] = pipe.predict_proba(X.iloc[te])[:, 1]
    return oof.mean(axis=1)                # average risk across repeats


def _categorical_nri(y, p_old, p_new, edges):
    cat_old = np.digitize(p_old, edges[1:-1])
    cat_new = np.digitize(p_new, edges[1:-1])
    up = cat_new > cat_old
    down = cat_new < cat_old
    ev, ne = y == 1, y == 0
    nri_ev = up[ev].mean() - down[ev].mean()
    nri_ne = down[ne].mean() - up[ne].mean()
    return nri_ev + nri_ne, nri_ev, nri_ne


def _calibration(y, p, bins=10):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    # logistic recalibration: intercept (CITL) and slope
    from numpy.linalg import lstsq
    logit = np.log(p / (1 - p))
    X = np.column_stack([np.ones_like(logit), logit])
    # Newton steps for logistic regression of y on logit
    beta = np.zeros(2)
    for _ in range(50):
        eta = X @ beta; mu = 1 / (1 + np.exp(-eta))
        W = mu * (1 - mu)
        z = eta + (y - mu) / np.clip(W, 1e-6, None)
        beta = lstsq(X * W[:, None], z * W, rcond=None)[0]
    citl, slope = float(beta[0]), float(beta[1])
    # binned calibration table
    q = np.quantile(p, np.linspace(0, 1, bins + 1)); q[-1] += 1e-9
    rows = []
    for b in range(bins):
        m = (p >= q[b]) & (p < q[b + 1])
        if m.sum():
            rows.append((float(p[m].mean()), float(y[m].mean()), int(m.sum())))
    return citl, slope, pd.DataFrame(rows, columns=["pred_mean", "obs_rate", "n"])


def evaluate_incremental_value(df, n_splits=10, n_repeats=5, seed=0):
    df = df.copy()
    y = df["event96"].to_numpy(int)
    p_base = _oof_predictions(df, NUM, CAT, use_phenotype=False,
                              n_splits=n_splits, n_repeats=n_repeats, seed=seed)
    p_full = _oof_predictions(df, NUM, CAT, use_phenotype=True,
                              n_splits=n_splits, n_repeats=n_repeats, seed=seed)
    auc_b, auc_f = roc_auc_score(y, p_base), roc_auc_score(y, p_full)
    nri, nri_ev, nri_ne = _categorical_nri(y, p_base, p_full, RISK_CATEGORIES)
    idi = ((p_full[y == 1].mean() - p_base[y == 1].mean())
           - (p_full[y == 0].mean() - p_base[y == 0].mean()))
    citl_b, slope_b, _ = _calibration(y, p_base)
    citl_f, slope_f, cal_f = _calibration(y, p_full)
    summary = pd.DataFrame([{
        "auc_base_cv": round(auc_b, 3), "auc_full_cv": round(auc_f, 3),
        "delta_auc_cv": round(auc_f - auc_b, 3),
        "categorical_NRI": round(nri, 3), "NRI_event": round(nri_ev, 3),
        "NRI_nonevent": round(nri_ne, 3), "IDI": round(float(idi), 4),
        "calib_intercept_full": round(citl_f, 3), "calib_slope_full": round(slope_f, 3),
        "events": int(y.sum()), "n": int(len(df)),
    }])
    return {"summary": summary, "calibration": cal_f,
            "preds": {"y": y, "p_base": p_base, "p_full": p_full}}


# --------------------------------------------------------------------------- #
# synthetic smoke test (proves the script runs; replace with real df)          #
# --------------------------------------------------------------------------- #
def _synthetic(n=3000, seed=1):
    rng = np.random.default_rng(seed)
    idp = rng.integers(0, n // 2, n)                       # ~2 knees per person
    pheno = rng.choice(["CL", "SD", "SyD", "CH"], n, p=[.30, .20, .20, .30])
    pain = {"CL": 1, "SD": 1.3, "SyD": 3.7, "CH": 4.5}
    womac_pain = np.array([pain[p] for p in pheno]) + rng.normal(0, 1, n)
    kl = rng.integers(0, 4, n) + (np.isin(pheno, ["SD", "CH"]).astype(int))
    lp = (0.9 * np.isin(pheno, ["CH"]).astype(float) + 0.5 * np.isin(pheno, ["SyD"]).astype(float)
          + 0.3 * (kl - 1) + 0.2 * (womac_pain - 2))
    y = rng.binomial(1, 1 / (1 + np.exp(-(lp - 2.0))))
    return pd.DataFrame(dict(
        id=idp, event96=y, phenotype=pheno, age=rng.normal(62, 9, n),
        sex=rng.integers(0, 2, n), race=rng.integers(0, 3, n), site=rng.integers(0, 4, n),
        bmi=rng.normal(29, 5, n), kl_base=kl, mjsw_base=rng.normal(4, 1, n),
        fta_base=rng.normal(0, 2, n), cesd=rng.normal(8, 4, n), comorbidity=rng.poisson(1, n),
        income=rng.integers(0, 5, n), nsaid=rng.integers(0, 2, n),
        womac_pain_base=np.clip(womac_pain, 0, 20),
        womac_func_base=np.clip(womac_pain * 1.4 + rng.normal(0, 2, n), 0, 68)))


if __name__ == "__main__":
    res = evaluate_incremental_value(_synthetic())
    print("SMOKE TEST (synthetic) — cross-validated, pain-inclusive base model:")
    print(res["summary"].to_string(index=False))
    print("\nCalibration (full model):\n", res["calibration"].to_string(index=False))
    print("\nReplace _synthetic() with your analytic dataframe to get real estimates.")
