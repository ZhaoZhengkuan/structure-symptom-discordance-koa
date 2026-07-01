#!/usr/bin/env python3
"""
Runnable statistical upgrades when R is not available locally.

Outputs:
1) participant-cluster bootstrap confidence intervals for 96-month phenotype
   risks and incremental prediction metrics;
2) multiple-imputation sensitivity analysis using sklearn IterativeImputer;
3) non-parametric competing-risk cumulative incidence functions with death as a
   competing event.

The gold-standard R lcmm/cmprsk/mice scripts live under code/r/.
"""

from __future__ import annotations

import json
import argparse
import os
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


ROOT = Path(os.environ.get("PROJECT_ROOT", str(Path(__file__).resolve().parents[1])))
TABLES = Path(os.environ.get("TABLES_DIR", str(ROOT / "outputs" / "tables")))
OUT = Path(os.environ.get("UPGRADES_DIR", str(ROOT / "outputs" / "upgrades")))
OUT.mkdir(parents=True, exist_ok=True)

PHENO_ORDER = ["concordant_low", "structural_dominant", "symptom_dominant", "concordant_high"]
HORIZON_DAYS = 96 * 30.4375
RNG = np.random.default_rng(20250620)


def load_analysis() -> pd.DataFrame:
    baseline = pd.read_csv(TABLES / "oai_baseline_knee.csv")
    pheno = pd.read_csv(TABLES / "oai_discordance_phenotypes.csv")
    outcomes = pd.read_csv(TABLES / "oai_outcomes_knee.csv")
    d = baseline.merge(pheno[["kid", "phenotype"]], on="kid", how="inner")
    d = d.merge(outcomes, on=["kid", "id", "side"], how="inner")
    d["participant_id"] = d["id"].astype(str)
    d["phenotype"] = pd.Categorical(d["phenotype"], categories=PHENO_ORDER, ordered=False)
    d["tkr_time"] = d["tkr_days"].fillna(np.inf)
    d["death_time"] = d["death_days"].fillna(np.inf)
    d["ftime"] = np.minimum.reduce([d["tkr_time"], d["death_time"], np.repeat(HORIZON_DAYS, len(d))])
    d["ftime"] = np.maximum(d["ftime"], 1)
    d["fstatus"] = 0
    d.loc[(d["tkr_event"] == 1) & (d["tkr_time"] <= d["death_time"]) & (d["tkr_time"] <= HORIZON_DAYS), "fstatus"] = 1
    d.loc[(d["death_event"] == 1) & (d["death_time"] < d["tkr_time"]) & (d["death_time"] <= HORIZON_DAYS), "fstatus"] = 2
    d["event96"] = (d["fstatus"] == 1).astype(int)
    d["death96"] = (d["fstatus"] == 2).astype(int)
    return d


def model_pipeline(covars: list[str], categorical: list[str]) -> Pipeline:
    numeric = [c for c in covars if c not in categorical]
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), numeric),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("oh", OneHotEncoder(handle_unknown="ignore"))]), categorical),
    ])
    return Pipeline([("pre", pre), ("lr", LogisticRegression(max_iter=2000, class_weight="balanced"))])


def incremental_metrics(d: pd.DataFrame) -> dict:
    base_covars = ["age", "sex", "race", "site", "bmi", "kl_base", "mjsw_base", "fta_base", "cesd", "comorbidity", "income", "nsaid"]
    base_covars = [c for c in base_covars if c in d.columns]
    full_covars = base_covars + ["phenotype"]
    cat_base = [c for c in base_covars if c in {"sex", "race", "site", "nsaid"}]
    cat_full = cat_base + ["phenotype"]
    y = d["event96"].astype(int)
    if y.nunique() < 2:
        return {"auc_base": np.nan, "auc_full": np.nan, "delta_auc": np.nan, "continuous_nri": np.nan, "idi": np.nan}
    base = model_pipeline(base_covars, cat_base).fit(d[base_covars], y)
    full = model_pipeline(full_covars, cat_full).fit(d[full_covars], y)
    rb = base.predict_proba(d[base_covars])[:, 1]
    rf = full.predict_proba(d[full_covars])[:, 1]
    ev = y == 1
    ne = y == 0
    auc_b = roc_auc_score(y, rb)
    auc_f = roc_auc_score(y, rf)
    nri = ((rf[ev] > rb[ev]).mean() - (rf[ev] < rb[ev]).mean()) + ((rf[ne] < rb[ne]).mean() - (rf[ne] > rb[ne]).mean())
    idi = (rf[ev].mean() - rf[ne].mean()) - (rb[ev].mean() - rb[ne].mean())
    return {"auc_base": auc_b, "auc_full": auc_f, "delta_auc": auc_f - auc_b, "continuous_nri": nri, "idi": idi}


def phenotype_risks(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ph in PHENO_ORDER:
        g = d[d["phenotype"] == ph]
        rows.append({
            "phenotype": ph,
            "n": len(g),
            "tkr96_risk": g["event96"].mean(),
            "death96_risk": g["death96"].mean(),
        })
    return pd.DataFrame(rows)


def cluster_bootstrap(d: pd.DataFrame, B: int = 200) -> tuple[pd.DataFrame, pd.DataFrame]:
    ids = d["participant_id"].drop_duplicates().to_numpy()
    by_id = {pid: g.copy() for pid, g in d.groupby("participant_id", sort=False)}
    risk_rows = []
    metric_rows = []
    for b in range(B):
        sampled = RNG.choice(ids, size=len(ids), replace=True)
        parts = []
        for j, pid in enumerate(sampled):
            tmp = by_id[pid].copy()
            tmp["boot_cluster"] = j
            parts.append(tmp)
        bd = pd.concat(parts, ignore_index=True)
        r = phenotype_risks(bd)
        r["iter"] = b + 1
        risk_rows.append(r)
        m = incremental_metrics(bd)
        m["iter"] = b + 1
        metric_rows.append(m)
    risk_boot = pd.concat(risk_rows, ignore_index=True)
    metric_boot = pd.DataFrame(metric_rows)
    return risk_boot, metric_boot


def summarise_bootstrap(d: pd.DataFrame, risk_boot: pd.DataFrame, metric_boot: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    point = phenotype_risks(d)
    rows = []
    for ph in PHENO_ORDER:
        p = point[point["phenotype"] == ph].iloc[0]
        rb = risk_boot[risk_boot["phenotype"] == ph]
        rows.append({
            "phenotype": ph,
            "n": int(p["n"]),
            "tkr96_risk_percent": 100 * p["tkr96_risk"],
            "tkr96_ci_low": 100 * rb["tkr96_risk"].quantile(0.025),
            "tkr96_ci_high": 100 * rb["tkr96_risk"].quantile(0.975),
            "death96_risk_percent": 100 * p["death96_risk"],
            "death96_ci_low": 100 * rb["death96_risk"].quantile(0.025),
            "death96_ci_high": 100 * rb["death96_risk"].quantile(0.975),
        })
    risk_ci = pd.DataFrame(rows)
    point_m = incremental_metrics(d)
    metric_rows = []
    for k, v in point_m.items():
        metric_rows.append({
            "metric": k,
            "estimate": v,
            "ci_low": metric_boot[k].quantile(0.025),
            "ci_high": metric_boot[k].quantile(0.975),
            "boot_n": metric_boot[k].notna().sum(),
        })
    return risk_ci, pd.DataFrame(metric_rows)


def cif_at_times(time: np.ndarray, status: np.ndarray, grid: np.ndarray) -> pd.DataFrame:
    order = np.argsort(time)
    time = time[order]
    status = status[order]
    uniq = np.unique(time[time <= grid.max()])
    S = 1.0
    cif1 = 0.0
    cif2 = 0.0
    out = []
    gi = 0
    n = len(time)
    for t in uniq:
        while gi < len(grid) and grid[gi] < t:
            out.append({"days": grid[gi], "cif_tkr": cif1, "cif_death": cif2})
            gi += 1
        at_risk = np.sum(time >= t)
        d1 = np.sum((time == t) & (status == 1))
        d2 = np.sum((time == t) & (status == 2))
        if at_risk > 0:
            cif1 += S * d1 / at_risk
            cif2 += S * d2 / at_risk
            S *= (1 - (d1 + d2) / at_risk)
    while gi < len(grid):
        out.append({"days": grid[gi], "cif_tkr": cif1, "cif_death": cif2})
        gi += 1
    return pd.DataFrame(out)


def competing_risk_cif(d: pd.DataFrame) -> pd.DataFrame:
    grid = np.array([12, 24, 36, 48, 60, 72, 84, 96]) * 30.4375
    rows = []
    for ph in PHENO_ORDER:
        g = d[d["phenotype"] == ph]
        c = cif_at_times(g["ftime"].to_numpy(), g["fstatus"].to_numpy(), grid)
        c["phenotype"] = ph
        c["month"] = c["days"] / 30.4375
        rows.append(c)
    return pd.concat(rows, ignore_index=True)


def multiple_imputation_sensitivity(d: pd.DataFrame, m: int = 20) -> pd.DataFrame:
    # Impute numeric covariates, keep categorical via most-frequent imputation.
    covars = ["age", "sex", "race", "site", "bmi", "kl_base", "mjsw_base", "fta_base", "cesd", "comorbidity", "income", "education", "pase", "nsaid", "phenotype"]
    covars = [c for c in covars if c in d.columns]
    numeric = [c for c in covars if c not in {"sex", "race", "site", "nsaid", "phenotype"}]
    categorical = [c for c in covars if c not in numeric]
    y = d["event96"].astype(int).to_numpy()
    rows = []
    base_num = d[numeric].copy()
    for i in range(m):
        imp = IterativeImputer(random_state=20250620 + i, sample_posterior=True, max_iter=15, initial_strategy="median")
        imputed_num = pd.DataFrame(imp.fit_transform(base_num), columns=numeric, index=d.index)
        dd = pd.concat([imputed_num, d[categorical].reset_index(drop=True)], axis=1)
        pipe = model_pipeline(covars, categorical).fit(dd[covars], y)
        names_num = numeric
        oh = pipe.named_steps["pre"].named_transformers_["cat"].named_steps["oh"]
        names = names_num + list(oh.get_feature_names_out(categorical))
        coefs = pipe.named_steps["lr"].coef_[0]
        for term, coef in zip(names, coefs):
            if term.startswith("phenotype"):
                rows.append({"imputation": i + 1, "term": term, "log_or": coef, "or": np.exp(coef)})
    raw = pd.DataFrame(rows)
    pooled = raw.groupby("term").agg(
        m=("log_or", "count"),
        log_or_mean=("log_or", "mean"),
        log_or_sd_between=("log_or", "std"),
        or_median=("or", "median"),
    ).reset_index()
    pooled["or_mean"] = np.exp(pooled["log_or_mean"])
    # This is a sensitivity summary; exact Rubin SE requires model covariance from
    # each imputed fit, not exposed by sklearn's penalised logistic regression.
    return raw, pooled


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap", type=int, default=50, help="participant-cluster bootstrap iterations; use 200+ for final run")
    parser.add_argument("--imputations", type=int, default=10, help="multiple imputations; use 20+ for final run")
    args = parser.parse_args()
    d = load_analysis()
    risk_boot, metric_boot = cluster_bootstrap(d, B=args.bootstrap)
    risk_ci, metric_ci = summarise_bootstrap(d, risk_boot, metric_boot)
    cif = competing_risk_cif(d)
    mi_raw, mi_pool = multiple_imputation_sensitivity(d, m=args.imputations)

    risk_boot.to_csv(OUT / "cluster_bootstrap_risk_raw.csv", index=False)
    metric_boot.to_csv(OUT / "cluster_bootstrap_incremental_metrics_raw.csv", index=False)
    risk_ci.to_csv(OUT / "cluster_bootstrap_tkr_death_risk_ci.csv", index=False)
    metric_ci.to_csv(OUT / "cluster_bootstrap_incremental_metrics_ci.csv", index=False)
    cif.to_csv(OUT / "competing_risk_cif_by_phenotype.csv", index=False)
    mi_raw.to_csv(OUT / "mice_like_imputation_phenotype_log_or_raw.csv", index=False)
    mi_pool.to_csv(OUT / "mice_like_imputation_phenotype_log_or_summary.csv", index=False)

    manifest = {
        "n_knees": int(len(d)),
        "n_participants": int(d["participant_id"].nunique()),
        "bootstrap_iterations": args.bootstrap,
        "multiple_imputations": args.imputations,
        "note": "Python outputs provide runnable cluster-bootstrap, MI-like sensitivity, and non-parametric competing-risk CIF. Gold-standard R lcmm/cmprsk/mice scripts are in code/r/ but require Rscript.",
    }
    (OUT / "python_upgrade_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
