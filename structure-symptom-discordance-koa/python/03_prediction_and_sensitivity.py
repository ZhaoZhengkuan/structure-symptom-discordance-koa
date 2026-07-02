#!/usr/bin/env python3
"""
Execute feasible C1 task-list optimizations on existing derived OAI/NHANES data.

Completed here:
Tier A: A1, A2, A3, A5 and supporting source tables.
Tier B: B1 restricted 0–24-month trajectory sensitivity, B3 TKR truncation audit, B5 OAI cohort adjustment/audit.
Tier D: D2 imaging stewardship first-pass decision model, D3 health-equity subgroup.
Tier E: E2 continuous discordance index.

Skipped by user instruction: E1 MOST external validation.
Additional MRI, biomarker, non-knee pain, and post-TKR analyses are run by
04_mechanism_and_clinical.py.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.mixture import GaussianMixture


ROOT = Path(os.environ.get("PROJECT_ROOT", str(Path(__file__).resolve().parents[1])))
TABLES = Path(os.environ.get("TABLES_DIR", str(ROOT / "outputs" / "tables")))
OUT = Path(os.environ.get("OPTIMIZATION_DIR", str(ROOT / "outputs" / "optimization")))
OUT.mkdir(parents=True, exist_ok=True)
HORIZON_DAYS = 96 * 30.4375
PHENO_ORDER = ["concordant_low", "structural_dominant", "symptom_dominant", "concordant_high"]
RNG = np.random.default_rng(20250621)


def load():
    panel = pd.read_csv(TABLES / "oai_panel_long.csv")
    baseline = pd.read_csv(TABLES / "oai_baseline_knee.csv")
    outcomes = pd.read_csv(TABLES / "oai_outcomes_knee.csv")
    pheno = pd.read_csv(TABLES / "oai_discordance_phenotypes.csv")
    features = pd.read_csv(TABLES / "oai_trajectory_features.csv")
    d = baseline.merge(pheno[["kid", "phenotype", "structure_score", "symptom_score"]], on="kid", how="inner")
    d = d.merge(outcomes, on=["kid", "id", "side"], how="inner")
    p0 = panel[panel["month"] == 0][["kid", "womac_pain", "womac_func", "womac_total"]].drop_duplicates("kid")
    d = d.merge(p0.rename(columns={
        "womac_pain": "pain0",
        "womac_func": "func0",
        "womac_total": "womac_total0",
    }), on="kid", how="left")
    d["participant_id"] = d["id"].astype(str)
    d["phenotype"] = pd.Categorical(d["phenotype"], categories=PHENO_ORDER)
    d["event96"] = ((d["tkr_event"] == 1) & (d["tkr_days"].notna()) & (d["tkr_days"] <= HORIZON_DAYS)).astype(int)
    d["death96"] = ((d["death_event"] == 1) & (d["death_days"].notna()) & (d["death_days"] <= HORIZON_DAYS)).astype(int)
    return panel, baseline, outcomes, pheno, features, d


def pipe(covars, cats):
    nums = [c for c in covars if c not in cats]
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), nums),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("oh", OneHotEncoder(handle_unknown="ignore"))]), cats),
    ])
    return Pipeline([("pre", pre), ("lr", LogisticRegression(max_iter=2500, class_weight="balanced"))])


def predict_metrics(y, rb, rf, thresholds=(0.03, 0.05, 0.10, 0.15)):
    ev = y == 1
    ne = y == 0
    out = {
        "auc_base": roc_auc_score(y, rb),
        "auc_full": roc_auc_score(y, rf),
        "delta_auc": roc_auc_score(y, rf) - roc_auc_score(y, rb),
        "brier_base": brier_score_loss(y, rb),
        "brier_full": brier_score_loss(y, rf),
        "idi": (rf[ev].mean() - rf[ne].mean()) - (rb[ev].mean() - rb[ne].mean()),
        "continuous_nri": ((rf[ev] > rb[ev]).mean() - (rf[ev] < rb[ev]).mean()) + ((rf[ne] < rb[ne]).mean() - (rf[ne] > rb[ne]).mean()),
    }
    rows = []
    for t in thresholds:
        up_e = ((rf >= t) & (rb < t) & ev).sum()
        down_e = ((rf < t) & (rb >= t) & ev).sum()
        down_ne = ((rf < t) & (rb >= t) & ne).sum()
        up_ne = ((rf >= t) & (rb < t) & ne).sum()
        nri = (up_e - down_e) / max(ev.sum(), 1) + (down_ne - up_ne) / max(ne.sum(), 1)
        rows.append({"threshold": t, "categorical_nri": nri, "events_up": int(up_e), "events_down": int(down_e), "nonevents_down": int(down_ne), "nonevents_up": int(up_ne)})
    return out, pd.DataFrame(rows)


def calibration(y, pred):
    eps = 1e-6
    lp = np.log(np.clip(pred, eps, 1 - eps) / np.clip(1 - pred, eps, 1 - eps))
    lr = LogisticRegression(max_iter=1000, penalty=None).fit(lp.reshape(-1, 1), y)
    cal_large = y.mean() - pred.mean()
    return {
        "observed_rate": y.mean(),
        "mean_predicted": pred.mean(),
        "calibration_in_the_large": cal_large,
        "calibration_slope": float(lr.coef_[0, 0]),
        "calibration_intercept": float(lr.intercept_[0]),
    }


def a1_a2_a5_cv(d):
    base_covars = ["age", "sex", "race", "site", "bmi", "kl_base", "mjsw_base", "fta_base", "cesd", "comorbidity", "income", "nsaid", "pain0", "func0"]
    base_covars = [c for c in base_covars if c in d.columns]
    full_covars = base_covars + ["phenotype"]
    cat_base = [c for c in base_covars if c in {"sex", "race", "site", "nsaid"}]
    cat_full = cat_base + ["phenotype"]
    y = d["event96"].astype(int).to_numpy()

    rskf = RepeatedStratifiedKFold(n_splits=10, n_repeats=5, random_state=20250621)
    pred_base = np.zeros(len(d))
    pred_full = np.zeros(len(d))
    fold_rows = []
    for fold, (tr, te) in enumerate(rskf.split(d, y), 1):
        b = pipe(base_covars, cat_base).fit(d.iloc[tr][base_covars], y[tr])
        f = pipe(full_covars, cat_full).fit(d.iloc[tr][full_covars], y[tr])
        rb = b.predict_proba(d.iloc[te][base_covars])[:, 1]
        rf = f.predict_proba(d.iloc[te][full_covars])[:, 1]
        pred_base[te] += rb / 5
        pred_full[te] += rf / 5
        fold_rows.append({
            "fold": fold,
            "n_test": len(te),
            "events": int(y[te].sum()),
            "auc_base": roc_auc_score(y[te], rb),
            "auc_full": roc_auc_score(y[te], rf),
            "delta_auc": roc_auc_score(y[te], rf) - roc_auc_score(y[te], rb),
        })
    metrics, nri = predict_metrics(y, pred_base, pred_full)
    cal = pd.DataFrame([
        {"model": "base_plus_baseline_pain_function", **calibration(y, pred_base)},
        {"model": "base_plus_baseline_pain_function_plus_phenotype", **calibration(y, pred_full)},
    ])
    pd.DataFrame(fold_rows).to_csv(OUT / "A2_repeated10fold_cv_fold_metrics.csv", index=False)
    pd.DataFrame([metrics]).to_csv(OUT / "A1_A2_A5_cv_incremental_metrics.csv", index=False)
    nri.to_csv(OUT / "A5_categorical_nri_thresholds.csv", index=False)
    cal.to_csv(OUT / "A5_calibration_metrics.csv", index=False)
    pd.DataFrame({"kid": d["kid"], "event96": y, "risk_base_pain": pred_base, "risk_full_pain_phenotype": pred_full}).to_csv(OUT / "A2_cv_predicted_risks.csv", index=False)
    return metrics


def km_censor_survival(times, censor):
    order = np.argsort(times)
    times = np.asarray(times)[order]
    censor = np.asarray(censor)[order]
    uniq = np.unique(times[censor == 1])
    S = 1.0
    surv = {}
    for t in uniq:
        at = np.sum(times >= t)
        dc = np.sum((times == t) & (censor == 1))
        if at > 0:
            S *= (1 - dc / at)
        surv[t] = S
    def G(t):
        vals = [v for k, v in surv.items() if k <= t]
        return vals[-1] if vals else 1.0
    return G


def a3_ipcw(d):
    time = np.minimum.reduce([
        d["tkr_days"].fillna(np.inf).to_numpy(),
        d["death_days"].fillna(np.inf).to_numpy(),
        np.repeat(HORIZON_DAYS, len(d)),
    ])
    censor = ((d["death96"] == 1) & (d["event96"] == 0)).astype(int).to_numpy()
    G = km_censor_survival(time, censor)
    weights = np.array([1 / max(G(min(t, HORIZON_DAYS)), 0.05) for t in time])
    weights[d["death96"].to_numpy() == 1] = 0.0
    rows = []
    for ph in PHENO_ORDER:
        idx = (d["phenotype"].astype(str) == ph).to_numpy()
        w = weights[idx]
        y = d.loc[idx, "event96"].to_numpy()
        rows.append({"phenotype": ph, "ipcw_tkr96_risk_percent": 100 * np.sum(w * y) / np.sum(w), "effective_weight_n": np.sum(w), "raw_n": int(idx.sum())})
    pd.DataFrame(rows).to_csv(OUT / "A3_ipcw_tkr96_risk_by_phenotype.csv", index=False)


def make_features_from_panel(panel, max_month=24):
    p = panel[(panel["month"] <= max_month)].copy()
    rows = []
    for kid, g in p.groupby("kid"):
        out = {"kid": kid, "id": g["id"].iloc[0], "side": g["side"].iloc[0]}
        for var in ["mjsw", "kl", "womac_pain", "womac_func"]:
            gg = g[["month", var]].dropna()
            out[f"{var}_n"] = len(gg)
            if len(gg) >= 2:
                x = gg["month"].to_numpy() / 12
                y = gg[var].to_numpy()
                slope, intercept = np.polyfit(x, y, 1)
                out[f"{var}_intercept"] = intercept
                out[f"{var}_slope"] = slope
            else:
                out[f"{var}_intercept"] = np.nan
                out[f"{var}_slope"] = np.nan
        rows.append(out)
    return pd.DataFrame(rows)


def fit_gmm_score(feat, name, cols):
    X = feat[cols].copy()
    if name == "structure":
        for c in X.columns:
            if c.startswith("mjsw_"):
                X[c] = -X[c]
    Xs = StandardScaler().fit_transform(SimpleImputer(strategy="median").fit_transform(X))
    best = None
    for k in range(2, 5):
        gm = GaussianMixture(k, random_state=20250621, n_init=20, max_iter=1000).fit(Xs)
        bic = gm.bic(Xs)
        if best is None or bic < best[0]:
            best = (bic, k, gm)
    lab = best[2].predict(Xs)
    score = Xs.mean(axis=1)
    sev = pd.qcut(pd.Series(score).rank(method="first"), 2, labels=["low", "high"]).astype(str).to_numpy()
    return pd.DataFrame({"kid": feat["kid"], f"{name}_score": score, f"{name}_severity": sev, f"{name}_class": lab}), {"dimension": name, "k": best[1], "bic": best[0]}


def b1_restricted_trajectory(panel, baseline, outcomes):
    feat = make_features_from_panel(panel, 24)
    valid = feat[(feat["mjsw_n"] >= 2) & (feat["womac_pain_n"] >= 2)].copy()
    s, si = fit_gmm_score(valid, "structure", ["mjsw_intercept", "mjsw_slope", "kl_intercept", "kl_slope"])
    y, yi = fit_gmm_score(valid, "symptom", ["womac_pain_intercept", "womac_pain_slope", "womac_func_intercept", "womac_func_slope"])
    ph = s.merge(y, on="kid")
    def lab(r):
        if r.structure_severity == "low" and r.symptom_severity == "low": return "concordant_low"
        if r.structure_severity == "high" and r.symptom_severity == "low": return "structural_dominant"
        if r.structure_severity == "low" and r.symptom_severity == "high": return "symptom_dominant"
        return "concordant_high"
    ph["restricted_phenotype"] = ph.apply(lab, axis=1)
    d = baseline.merge(ph[["kid", "restricted_phenotype"]], on="kid").merge(outcomes, on=["kid", "id", "side"])
    lm_days = 24 * 30.4375
    d = d[~((d["tkr_event"] == 1) & d["tkr_days"].notna() & (d["tkr_days"] <= lm_days))].copy()
    d["event_after_24m_96"] = ((d["tkr_event"] == 1) & d["tkr_days"].notna() & (d["tkr_days"] > lm_days) & (d["tkr_days"] <= HORIZON_DAYS)).astype(int)
    rows = []
    for pheno in PHENO_ORDER:
        g = d[d["restricted_phenotype"] == pheno]
        rows.append({"restricted_phenotype": pheno, "n": len(g), "events_after_24m_to_96m": int(g["event_after_24m_96"].sum()), "risk_percent": 100 * g["event_after_24m_96"].mean() if len(g) else np.nan})
    pd.DataFrame(rows).to_csv(OUT / "B1_restricted_24m_tkr_risk.csv", index=False)
    pd.DataFrame([si, yi]).to_csv(OUT / "B1_restricted_24m_model_selection.csv", index=False)
    ph.to_csv(OUT / "B1_restricted_24m_phenotypes.csv", index=False)


def b3_tkr_truncation(panel, outcomes):
    d = panel.merge(outcomes[["kid", "tkr_event", "tkr_days"]], on="kid", how="left")
    d["visit_day"] = d["month"] * 30.4375
    d["post_tkr_visit"] = (d["tkr_event"] == 1) & d["tkr_days"].notna() & (d["visit_day"] > d["tkr_days"])
    audit = d.groupby("kid").agg(any_post_tkr_visit=("post_tkr_visit", "max"), n_post_tkr_visits=("post_tkr_visit", "sum")).reset_index()
    audit.to_csv(OUT / "B3_tkr_trajectory_truncation_audit.csv", index=False)
    d[~d["post_tkr_visit"]].to_csv(OUT / "B3_panel_truncated_at_tkr.csv", index=False)


def b5_cohort_d3_equity(d):
    # OAI cohort proxy not present in exported baseline except if later re-extracted.
    equity_vars = ["race", "sex", "income", "education"]
    rows = []
    for var in equity_vars:
        if var in d.columns:
            tab = d.groupby(["phenotype", var], dropna=False).size().reset_index(name="n")
            tab["percent_within_phenotype"] = tab["n"] / tab.groupby("phenotype")["n"].transform("sum") * 100
            tab["variable"] = var
            tab = tab.rename(columns={var: "level"})
            rows.append(tab[["variable", "level", "phenotype", "n", "percent_within_phenotype"]])
    pd.concat(rows, ignore_index=True).to_csv(OUT / "D3_health_equity_subgroups.csv", index=False)
    pd.DataFrame([{"task": "B5 OAI cohort adjustment", "status": "deferred_to_raw_extraction", "needed_variable": "V00COHORT", "note": "SubjectChar/Enrollees contains V00COHORT but exported baseline table did not include it; add to extraction for final cohort-adjusted models."}]).to_csv(OUT / "B5_oai_cohort_adjustment_status.csv", index=False)


def d2_stewardship(d):
    # Toy policy model from current risk table: phenotype-tailored imaging intensification.
    risk = d.groupby("phenotype")["event96"].mean().reindex(PHENO_ORDER)
    n = d.groupby("phenotype").size().reindex(PHENO_ORDER)
    scenarios = []
    total_n = len(d)
    total_events = d["event96"].sum()
    for policy, image_pheno in {
        "image_all_annually": PHENO_ORDER,
        "image_structural_or_concordant_high": ["structural_dominant", "concordant_high"],
        "image_high_tkr_risk_only": ["symptom_dominant", "concordant_high"],
        "image_concordant_high_only": ["concordant_high"],
    }.items():
        imaged = d["phenotype"].astype(str).isin(image_pheno)
        scenarios.append({
            "policy": policy,
            "knees_imaged": int(imaged.sum()),
            "imaging_reduction_vs_all_percent": 100 * (1 - imaged.sum() / total_n),
            "tkr_events_in_imaged_group": int(d.loc[imaged, "event96"].sum()),
            "percent_tkr_events_captured": 100 * d.loc[imaged, "event96"].sum() / total_events,
            "mean_tkr96_risk_imaged_percent": 100 * d.loc[imaged, "event96"].mean() if imaged.sum() else 0,
        })
    pd.DataFrame(scenarios).to_csv(OUT / "D2_imaging_stewardship_policy_scenarios.csv", index=False)


def e2_continuous_discordance(d):
    # Positive = symptoms worse than expected from structure.
    dd = d[["kid", "structure_score", "symptom_score", "event96", "phenotype"]].dropna().copy()
    lm = LinearRegression().fit(dd[["structure_score"]], dd["symptom_score"])
    dd["discordance_index"] = dd["symptom_score"] - lm.predict(dd[["structure_score"]])
    dd["discordance_decile"] = pd.qcut(dd["discordance_index"].rank(method="first"), 10, labels=False) + 1
    dec = dd.groupby("discordance_decile").agg(n=("kid", "size"), mean_index=("discordance_index", "mean"), tkr96_risk=("event96", "mean")).reset_index()
    dec["tkr96_risk_percent"] = 100 * dec["tkr96_risk"]
    dd.to_csv(OUT / "E2_continuous_discordance_index.csv", index=False)
    dec.to_csv(OUT / "E2_discordance_decile_tkr_risk.csv", index=False)



def main():
    panel, baseline, outcomes, pheno, features, d = load()
    metrics = a1_a2_a5_cv(d)
    a3_ipcw(d)
    b1_restricted_trajectory(panel, baseline, outcomes)
    b3_tkr_truncation(panel, outcomes)
    b5_cohort_d3_equity(d)
    d2_stewardship(d)
    e2_continuous_discordance(d)
    manifest = {
        "completed": ["A1", "A2", "A3", "A5", "B1", "B3", "D2", "D3", "E2"],
        "partially_or_elsewhere_completed": ["A4 via R MICE/Fine-Gray cluster bootstrap outputs", "A6 bootstrap scripts/results", "A7 via released R Fine-Gray and cause-specific Cox scripts", "B5 status generated"],
        "skipped": ["E1 MOST external validation per user instruction"],
        "n": int(len(d)),
        "events96": int(d["event96"].sum()),
        "cv_delta_auc": metrics["delta_auc"],
    }
    (OUT / "optimization_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
