#!/usr/bin/env python3
"""Run the remaining C1 OA discordance enhancement analyses.

Inputs are deliberately local-only:
  * existing derived C1 OAI tables under projects/c1_oa_discordance/outputs/tables
  * the raw OAI complete ASCII zip

The script writes a self-contained enhancement folder under
outputs/c1_oa_discordance_remaining_tasks/remaining_tasks.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import textwrap
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# Project root; override with the C1_PROJECT environment variable. All default
# paths below sit under this root and can also be overridden via the CLI flags
# defined in the argument parser near the bottom of this file.
PROJECT = Path(os.environ.get("C1_PROJECT", Path(__file__).resolve().parents[2]))
DEFAULT_DERIVED = Path(os.environ.get("C1_DERIVED", PROJECT / "outputs" / "tables"))
DEFAULT_OPT = Path(os.environ.get("C1_OPT", PROJECT / "outputs" / "optimization"))
DEFAULT_OAI_ZIP = Path(os.environ.get("OAI_ZIP", PROJECT / "data" / "OAICompleteData_ASCII.zip"))
DEFAULT_OUT = Path(os.environ.get("C1_OUT", PROJECT / "outputs" / "remaining_tasks"))
HORIZON_DAYS = 96 * 30.4375
PHENO_ORDER = ["concordant_low", "structural_dominant", "symptom_dominant", "concordant_high"]
PHENO_SHORT = {
    "concordant_low": "CL",
    "structural_dominant": "SD",
    "symptom_dominant": "SyD",
    "concordant_high": "CH",
}
VISITS = {"00": 0, "01": 12, "03": 24, "05": 36, "06": 48, "08": 72, "10": 96}
VISIT_FILE = {
    "00": "AllClinical00.txt",
    "01": "AllClinical01.txt",
    "03": "AllClinical03.txt",
    "05": "AllClinical05.txt",
    "06": "AllClinical06.txt",
    "08": "AllClinical08.txt",
    "10": "AllClinical10.txt",
}
RNG = np.random.default_rng(20260621)


def first_numeric(value):
    if pd.isna(value):
        return np.nan
    m = re.match(r"^\s*([-+]?\d+(?:\.\d+)?)", str(value))
    return float(m.group(1)) if m else np.nan


def numeric_series(s: pd.Series) -> pd.Series:
    return s.map(first_numeric).astype(float)


def clean_id(s: pd.Series) -> pd.Series:
    return s.astype(str).str.extract(r"(\d+)")[0]


def side_num(s: pd.Series) -> pd.Series:
    return numeric_series(s).astype("Int64")


def std_col_map(cols) -> dict[str, str]:
    return {c.upper(): c for c in cols}


def read_oai_member(z: zipfile.ZipFile, basename: str, usecols: list[str] | None = None) -> pd.DataFrame:
    member = f"OAI Complete Data_ASCII/{basename}"
    with z.open(member) as fh:
        header = fh.readline().decode("utf-8", errors="replace").strip("\r\n").split("|")
    actual = None
    if usecols is not None:
        cmap = std_col_map(header)
        actual = [cmap[c.upper()] for c in usecols if c.upper() in cmap]
    with z.open(member) as fh:
        return pd.read_csv(fh, sep="|", dtype=str, usecols=actual, low_memory=False, encoding="latin1")


def write_csv(df: pd.DataFrame, path: Path) -> pd.DataFrame:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def load_core(derived: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    panel = pd.read_csv(derived / "oai_panel_long.csv")
    baseline = pd.read_csv(derived / "oai_baseline_knee.csv")
    outcomes = pd.read_csv(derived / "oai_outcomes_knee.csv")
    pheno = pd.read_csv(derived / "oai_discordance_phenotypes.csv")
    features = pd.read_csv(derived / "oai_trajectory_features.csv")
    for df in [panel, baseline, outcomes, pheno, features]:
        if "id" in df.columns:
            df["id"] = df["id"].astype(str)
        if "kid" in df.columns:
            df["kid"] = df["kid"].astype(str)
    return panel, baseline, outcomes, pheno, features


def analysis_frame(panel, baseline, outcomes, pheno):
    d = baseline.merge(pheno[["kid", "phenotype", "structure_score", "symptom_score"]], on="kid", how="inner")
    d = d.merge(outcomes, on=["kid", "id", "side"], how="inner")
    p0 = panel.loc[panel["month"] == 0, ["kid", "womac_pain", "womac_func", "womac_total"]].drop_duplicates("kid")
    d = d.merge(p0.rename(columns={"womac_pain": "pain0", "womac_func": "func0", "womac_total": "total0"}), on="kid", how="left")
    d["event96"] = ((d["tkr_event"] == 1) & d["tkr_days"].notna() & (d["tkr_days"] <= HORIZON_DAYS)).astype(int)
    d["event96_self_report"] = ((d["tkr_event_including_self_report"] == 1) & d["tkr_days"].notna() & (d["tkr_days"] <= HORIZON_DAYS)).astype(int)
    d["death96"] = ((d["death_event"] == 1) & d["death_days"].notna() & (d["death_days"] <= HORIZON_DAYS)).astype(int)
    d["phenotype"] = pd.Categorical(d["phenotype"], categories=PHENO_ORDER)
    return d


def model_pipeline(covars: list[str], cats: list[str], balanced: bool = False):
    nums = [c for c in covars if c not in cats]
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), nums),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("oh", OneHotEncoder(handle_unknown="ignore"))]), cats),
    ])
    return Pipeline([
        ("pre", pre),
        ("lr", LogisticRegression(max_iter=3000, class_weight="balanced" if balanced else None)),
    ])


def calibration(y, pred):
    eps = 1e-6
    pred = np.clip(pred, eps, 1 - eps)
    lp = np.log(pred / (1 - pred)).reshape(-1, 1)
    lr = LogisticRegression(max_iter=1000, penalty=None).fit(lp, y)
    return {
        "observed_rate": float(np.mean(y)),
        "mean_predicted": float(np.mean(pred)),
        "calibration_in_the_large": float(np.mean(y) - np.mean(pred)),
        "calibration_slope": float(lr.coef_[0, 0]),
        "calibration_intercept": float(lr.intercept_[0]),
        "brier": float(brier_score_loss(y, pred)),
    }


def task6_unbalanced_calibration(d: pd.DataFrame, out: Path):
    base_covars = [c for c in ["age", "sex", "race", "site", "bmi", "kl_base", "mjsw_base", "fta_base", "cesd", "comorbidity", "income", "nsaid", "pain0", "func0"] if c in d.columns]
    full_covars = base_covars + ["phenotype"]
    cat_base = [c for c in base_covars if c in {"sex", "race", "site", "nsaid"}]
    cat_full = cat_base + ["phenotype"]
    use = d.dropna(subset=["event96"]).copy()
    y = use["event96"].astype(int).to_numpy()
    pred_base = np.zeros(len(use))
    pred_full = np.zeros(len(use))
    folds = StratifiedKFold(n_splits=10, shuffle=True, random_state=20260621)
    rows = []
    for fold, (tr, te) in enumerate(folds.split(use, y), 1):
        b = model_pipeline(base_covars, cat_base, balanced=False).fit(use.iloc[tr][base_covars], y[tr])
        f = model_pipeline(full_covars, cat_full, balanced=False).fit(use.iloc[tr][full_covars], y[tr])
        rb = b.predict_proba(use.iloc[te][base_covars])[:, 1]
        rf = f.predict_proba(use.iloc[te][full_covars])[:, 1]
        pred_base[te] = rb
        pred_full[te] = rf
        rows.append({"fold": fold, "n": len(te), "events": int(y[te].sum()), "auc_base": roc_auc_score(y[te], rb), "auc_full": roc_auc_score(y[te], rf)})
    cal = pd.DataFrame([
        {"model": "natural_scale_base", "auc": roc_auc_score(y, pred_base), **calibration(y, pred_base)},
        {"model": "natural_scale_base_plus_phenotype", "auc": roc_auc_score(y, pred_full), **calibration(y, pred_full)},
    ])
    write_csv(cal, out / "task6_unbalanced_calibration_metrics.csv")
    write_csv(pd.DataFrame(rows), out / "task6_unbalanced_cv_folds.csv")
    write_csv(pd.DataFrame({"kid": use["kid"], "event96": y, "risk_base": pred_base, "risk_full": pred_full}), out / "task6_unbalanced_cv_predictions.csv")
    fig, ax = plt.subplots(figsize=(5, 4), dpi=180)
    for label, pred in [("Base", pred_base), ("+ phenotype", pred_full)]:
        bins = pd.qcut(pd.Series(pred).rank(method="first"), 10, labels=False)
        tmp = pd.DataFrame({"pred": pred, "y": y, "bin": bins}).groupby("bin").agg(pred=("pred", "mean"), obs=("y", "mean")).reset_index()
        ax.plot(tmp["pred"], tmp["obs"], marker="o", label=label)
    ax.plot([0, max(pred_full.max(), pred_base.max())], [0, max(pred_full.max(), pred_base.max())], color="0.5", lw=1)
    ax.set_xlabel("Mean predicted 96-month TKR risk")
    ax.set_ylabel("Observed 96-month TKR risk")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "task6_unbalanced_calibration_plot.png")
    plt.close(fig)
    return cal


def cluster_bootstrap_metrics(pred: pd.DataFrame, ids: pd.Series, B: int):
    pred = pred.copy()
    pred["id"] = ids.astype(str).to_numpy()
    people = pred["id"].drop_duplicates().to_numpy()
    group_idx = {pid: np.flatnonzero(pred["id"].to_numpy() == pid) for pid in people}
    y_all = pred["event96"].astype(int).to_numpy()
    rb_all = pred["risk_base"].to_numpy()
    rf_all = pred["risk_full"].to_numpy()
    rows = []
    for b in range(B):
        samp = RNG.choice(people, size=len(people), replace=True)
        idx = np.concatenate([group_idx[pid] for pid in samp])
        y = y_all[idx]
        if len(np.unique(y)) < 2:
            continue
        rb = rb_all[idx]
        rf = rf_all[idx]
        ev = y == 1
        ne = y == 0
        rows.append({
            "bootstrap": b + 1,
            "auc_base": roc_auc_score(y, rb),
            "auc_full": roc_auc_score(y, rf),
            "delta_auc": roc_auc_score(y, rf) - roc_auc_score(y, rb),
            "idi": (rf[ev].mean() - rf[ne].mean()) - (rb[ev].mean() - rb[ne].mean()),
            "continuous_nri": ((rf[ev] > rb[ev]).mean() - (rf[ev] < rb[ev]).mean()) + ((rf[ne] < rb[ne]).mean() - (rf[ne] > rb[ne]).mean()),
        })
    return pd.DataFrame(rows)


def task7_bootstrap_1000(d: pd.DataFrame, out: Path, B: int):
    pred = pd.read_csv(out / "task6_unbalanced_cv_predictions.csv")
    ids = d.set_index("kid").loc[pred["kid"].astype(str), "id"].reset_index(drop=True)
    boot = cluster_bootstrap_metrics(pred, ids, B)
    ci = boot.drop(columns=["bootstrap"]).agg(["mean", lambda x: x.quantile(0.025), lambda x: x.quantile(0.975)]).T.reset_index()
    ci.columns = ["metric", "mean", "ci_low", "ci_high"]
    write_csv(boot, out / f"task7_cluster_bootstrap_B{B}_raw.csv")
    write_csv(ci, out / f"task7_cluster_bootstrap_B{B}_ci.csv")
    return ci


def read_cohort(z: zipfile.ZipFile) -> pd.DataFrame:
    e = read_oai_member(z, "Enrollees.txt", ["ID", "V00COHORT"])
    c = std_col_map(e.columns)
    return pd.DataFrame({"id": clean_id(e[c["ID"]]), "v00cohort": numeric_series(e[c["V00COHORT"]])})


def phenotype_risk_table(d, y_col="event96", group_col="phenotype"):
    rows = []
    for ph in PHENO_ORDER:
        g = d[d[group_col] == ph]
        rows.append({"phenotype": ph, "n": len(g), "events": int(g[y_col].sum()), "risk_percent": 100 * g[y_col].mean() if len(g) else np.nan})
    return pd.DataFrame(rows)


def log_or_by_phenotype(d, y_col="event96", extra_covars=None):
    extra_covars = extra_covars or []
    use = d.copy()
    covars = ["phenotype"] + [c for c in extra_covars if c in use.columns]
    cats = ["phenotype"] + [c for c in extra_covars if c in {"sex", "race", "site", "nsaid", "v00cohort"}]
    y = use[y_col].astype(int).to_numpy()
    pipe = model_pipeline(covars, cats, balanced=False).fit(use[covars], y)
    names = pipe.named_steps["pre"].get_feature_names_out()
    coefs = pipe.named_steps["lr"].coef_[0]
    rows = []
    for nm, beta in zip(names, coefs):
        if "phenotype_" in nm:
            rows.append({"term": nm, "log_or": beta, "or": math.exp(beta)})
    return pd.DataFrame(rows)


def task5_cohort(d: pd.DataFrame, z: zipfile.ZipFile, out: Path):
    cohort = read_cohort(z)
    dd = d.merge(cohort, on="id", how="left")
    dist = dd.groupby(["v00cohort", "phenotype"], observed=False).size().reset_index(name="n")
    dist["percent_within_cohort"] = 100 * dist["n"] / dist.groupby("v00cohort")["n"].transform("sum")
    risk = dd.groupby(["v00cohort", "phenotype"], observed=False).agg(n=("kid", "size"), events=("event96", "sum"), risk=("event96", "mean")).reset_index()
    risk["risk_percent"] = 100 * risk["risk"]
    adj = log_or_by_phenotype(dd, "event96", ["age", "sex", "race", "site", "bmi", "kl_base", "mjsw_base", "pain0", "v00cohort"])
    write_csv(dist, out / "task5_v00cohort_phenotype_distribution.csv")
    write_csv(risk, out / "task5_v00cohort_tkr_risk.csv")
    write_csv(adj, out / "task5_v00cohort_adjusted_logistic_terms.csv")
    return dd


def task8_sensitivities(d: pd.DataFrame, panel: pd.DataFrame, out: Path):
    pain0 = panel.loc[panel["month"] == 0, ["kid", "womac_pain"]].drop_duplicates("kid")
    dd = d.merge(pain0.rename(columns={"womac_pain": "baseline_pain_for_index"}), on="kid", how="left")
    idx = dd.sort_values(["id", "baseline_pain_for_index", "kl_base"], ascending=[True, False, False]).groupby("id").head(1)
    kl2 = dd[dd["kl_base"] >= 2].copy()
    post_visits = panel.groupby("kid")["month"].max().rename("last_observed_month").reset_index()
    dd2 = dd.merge(post_visits, on="kid", how="left")
    dd2["observed_96"] = (dd2["last_observed_month"] >= 96).astype(int)
    covars = [c for c in ["age", "sex", "race", "site", "bmi", "kl_base", "mjsw_base", "pain0", "phenotype"] if c in dd2.columns]
    cats = [c for c in covars if c in {"sex", "race", "site", "phenotype"}]
    if dd2["observed_96"].nunique() < 2:
        dd2["visit_ipw"] = 1.0
        dd2["visit_ipw_note"] = "All analytic knees had an observed 96-month panel record; visit-selection IPW not estimable and set to 1."
    else:
        obs_model = model_pipeline(covars, cats, balanced=False).fit(dd2[covars], dd2["observed_96"])
        p_obs = np.clip(obs_model.predict_proba(dd2[covars])[:, 1], 0.05, 1)
        dd2["visit_ipw"] = 1 / p_obs
        dd2["visit_ipw_note"] = "Estimated as inverse probability of observed 96-month panel record."
    rows = []
    for name, frame, ycol in [
        ("8a_single_more_symptomatic_index_knee", idx, "event96"),
        ("8b_kl_ge_2_definite_oa", kl2, "event96"),
        ("8c_confirmed_plus_self_report_tkr", dd, "event96_self_report"),
        ("8d_observed_visit_ipw_unweighted_reference", dd2, "event96"),
    ]:
        tab = phenotype_risk_table(frame, ycol)
        tab.insert(0, "analysis", name)
        rows.append(tab)
    ipw_rows = []
    for ph in PHENO_ORDER:
        g = dd2[dd2["phenotype"] == ph]
        w = g["visit_ipw"].to_numpy()
        ipw_rows.append({"analysis": "8d_informative_visit_ipw", "phenotype": ph, "n": len(g), "events": int(g["event96"].sum()), "risk_percent": 100 * np.average(g["event96"], weights=w)})
    all_risk = pd.concat(rows + [pd.DataFrame(ipw_rows)], ignore_index=True)
    write_csv(all_risk, out / "task8_sensitivity_risk_by_phenotype.csv")
    write_csv(dd2[["kid", "id", "phenotype", "last_observed_month", "observed_96", "visit_ipw", "visit_ipw_note"]], out / "task8d_visit_ipw_weights.csv")
    return all_risk


def read_moaks_baseline(z: zipfile.ZipFile) -> pd.DataFrame:
    m = read_oai_member(z, "kMRI_SQ_MOAKS_BICL00.txt")
    c = std_col_map(m.columns)
    out = pd.DataFrame({"id": clean_id(m[c["ID"]]), "side": side_num(m[c["SIDE"]]).astype(float)})
    out["kid"] = out["id"].astype(str) + "_" + out["side"].astype(int).astype(str)
    out["effusion_synovitis"] = numeric_series(m[c["V00MEFFWK"]]) if "V00MEFFWK" in c else np.nan
    out["hoffa_synovitis_proxy"] = numeric_series(m[c["V00MSYIC"]]) if "V00MSYIC" in c else np.nan
    bml_cols = [col for col in m.columns if col.upper().startswith("V00MBMS")]
    cart_cols = [col for col in m.columns if col.upper().startswith("V00MCM")]
    out["bml_size_sum"] = sum(numeric_series(m[col]).fillna(0) for col in bml_cols)
    out["cartilage_damage_sum"] = sum(numeric_series(m[col]).fillna(0) for col in cart_cols)
    return out.drop_duplicates("kid")


def weighted_linear_coef(x, y, w):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    w = np.asarray(w, float)
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(w) & (w > 0)
    if ok.sum() < 10:
        return np.nan, np.nan
    X = np.c_[np.ones(ok.sum()), x[ok]]
    sw = np.sqrt(w[ok])
    beta = np.linalg.lstsq(X * sw[:, None], y[ok] * sw, rcond=None)[0]
    return beta[0], beta[1]


def product_mediation(df, exposure_col, mediator_col, y_col, covars, weight_col):
    use = df[[exposure_col, mediator_col, y_col, weight_col] + covars].dropna().copy()
    if use[exposure_col].nunique() < 2 or len(use) < 50:
        return {"n": len(use), "total_effect": np.nan, "nde": np.nan, "nie": np.nan, "proportion_explained": np.nan}
    w = use[weight_col].to_numpy()
    # residualize exposure, mediator and outcome on covariates, then use product of coefficients.
    C = use[covars].copy()
    C = pd.get_dummies(C, columns=[c for c in C.columns if str(C[c].dtype) == "category" or C[c].dtype == object], drop_first=True)
    C = C.apply(pd.to_numeric, errors="coerce")
    C = C.fillna(C.median(numeric_only=True)).fillna(0)
    def resid(v):
        lr = LinearRegression().fit(C, use[v], sample_weight=w)
        return use[v].to_numpy() - lr.predict(C)
    xr = resid(exposure_col)
    mr = resid(mediator_col)
    yr = resid(y_col)
    _, a = weighted_linear_coef(xr, mr, w)
    _, total = weighted_linear_coef(xr, yr, w)
    X = np.c_[xr, mr]
    ok = np.isfinite(X).all(axis=1) & np.isfinite(yr) & np.isfinite(w)
    beta = LinearRegression().fit(X[ok], yr[ok], sample_weight=w[ok]).coef_
    direct, b = beta[0], beta[1]
    nie = a * b
    return {"n": len(use), "total_effect": total, "nde": direct, "nie": nie, "proportion_explained": nie / total if total not in [0, np.nan] and np.isfinite(total) else np.nan}


def task1_moaks(d: pd.DataFrame, z: zipfile.ZipFile, out: Path, B: int):
    moaks = read_moaks_baseline(z)
    dd = d.merge(moaks, on="kid", how="left", suffixes=("", "_moaks"))
    dd["has_moaks"] = dd["effusion_synovitis"].notna().astype(int)
    covars = [c for c in ["age", "sex", "race", "site", "bmi", "kl_base", "mjsw_base", "cesd", "comorbidity", "income"] if c in dd.columns]
    cats = [c for c in covars if c in {"sex", "race", "site"}]
    sel = model_pipeline(covars, cats, balanced=False).fit(dd[covars], dd["has_moaks"])
    p = np.clip(sel.predict_proba(dd[covars])[:, 1], 0.05, 1)
    dd["mri_selection_ipw"] = np.where(dd["has_moaks"] == 1, 1 / p, np.nan)
    dd["mri_selection_ipw"] = dd["mri_selection_ipw"].clip(upper=np.nanpercentile(dd["mri_selection_ipw"], 99))
    mri = dd[dd["has_moaks"] == 1].copy()
    desc = mri.groupby("phenotype", observed=False).agg(
        n=("kid", "size"),
        effusion_synovitis_mean=("effusion_synovitis", "mean"),
        hoffa_synovitis_proxy_mean=("hoffa_synovitis_proxy", "mean"),
        bml_size_sum_mean=("bml_size_sum", "mean"),
        cartilage_damage_sum_mean=("cartilage_damage_sum", "mean"),
        pain0_mean=("pain0", "mean"),
    ).reset_index()
    write_csv(desc, out / "task1_moaks_by_phenotype.csv")
    med_rows = []
    mri["syd_vs_cl"] = np.where(mri["phenotype"].astype(str) == "symptom_dominant", 1, np.where(mri["phenotype"].astype(str) == "concordant_low", 0, np.nan))
    med_covars = [c for c in ["age", "sex", "race", "site", "bmi", "kl_base", "mjsw_base", "cartilage_damage_sum"] if c in mri.columns]
    for mediator in ["effusion_synovitis", "hoffa_synovitis_proxy", "bml_size_sum"]:
        for exposure in ["syd_vs_cl", "structure_score"]:
            res = product_mediation(mri, exposure, mediator, "pain0", med_covars, "mri_selection_ipw")
            res.update({"exposure": exposure, "mediator": mediator})
            med_rows.append(res)
    med = pd.DataFrame(med_rows)
    # Bootstrap CIs for the primary SyD-vs-CL inflammation rows.
    boot_rows = []
    people = mri["id"].drop_duplicates().to_numpy()
    for b in range(B):
        samp = RNG.choice(people, size=len(people), replace=True)
        boot = pd.concat([mri[mri["id"] == sid] for sid in samp], ignore_index=True)
        for mediator in ["effusion_synovitis", "hoffa_synovitis_proxy", "bml_size_sum"]:
            res = product_mediation(boot, "syd_vs_cl", mediator, "pain0", med_covars, "mri_selection_ipw")
            boot_rows.append({"bootstrap": b + 1, "mediator": mediator, **res})
    boot = pd.DataFrame(boot_rows)
    ci = boot.groupby("mediator").agg(
        nie_low=("nie", lambda x: x.quantile(0.025)),
        nie_high=("nie", lambda x: x.quantile(0.975)),
        prop_low=("proportion_explained", lambda x: x.quantile(0.025)),
        prop_high=("proportion_explained", lambda x: x.quantile(0.975)),
    ).reset_index()
    ci["exposure"] = "syd_vs_cl"
    med = med.merge(ci, on=["exposure", "mediator"], how="left")
    write_csv(med, out / "task1_ipw_mediation_decomposition.csv")
    write_csv(boot, out / f"task1_mediation_bootstrap_B{B}.csv")
    fig, ax = plt.subplots(1, 2, figsize=(8, 3.6), dpi=180)
    x = np.arange(len(desc))
    ax[0].bar(x - 0.25, desc["effusion_synovitis_mean"], width=0.25, label="Effusion-synovitis")
    ax[0].bar(x, desc["hoffa_synovitis_proxy_mean"], width=0.25, label="Hoffa/intercondylar synovitis proxy")
    ax[0].bar(x + 0.25, desc["bml_size_sum_mean"] / max(desc["bml_size_sum_mean"].max(), 1) * 3, width=0.25, label="BML load (scaled)")
    ax[0].set_xticks(x, [PHENO_SHORT.get(str(p), str(p)) for p in desc["phenotype"]])
    ax[0].set_ylabel("Mean MOAKS burden")
    ax[0].legend(fontsize=7, frameon=False)
    primary = med[med["exposure"] == "syd_vs_cl"].copy()
    ax[1].bar(primary["mediator"], 100 * primary["proportion_explained"])
    ax[1].set_ylabel("% pain effect explained")
    ax[1].tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(out / "task1_moaks_mediation_figure.png")
    plt.close(fig)
    return med


def task2_post_tkr(panel: pd.DataFrame, d: pd.DataFrame, out: Path, B: int):
    tkr = d[(d["tkr_event"] == 1) & d["tkr_days"].notna()].copy()
    rows = []
    for _, r in tkr.iterrows():
        g = panel[panel["kid"] == r["kid"]].copy()
        g["visit_day"] = g["month"] * 30.4375
        pre = g[g["visit_day"] < r["tkr_days"]].sort_values("visit_day")
        post = g[g["visit_day"] > r["tkr_days"]].sort_values("visit_day")
        pre_pain = pre["womac_pain"].dropna().iloc[-1] if pre["womac_pain"].notna().any() else np.nan
        pre_func = pre["womac_func"].dropna().iloc[-1] if pre["womac_func"].notna().any() else np.nan
        for _, v in post.iterrows():
            rows.append({
                "kid": r["kid"], "id": r["id"], "side": r["side"], "phenotype": r["phenotype"],
                "tkr_days": r["tkr_days"], "post_month": v["month"], "days_after_tkr": v["visit_day"] - r["tkr_days"],
                "preop_pain": pre_pain, "preop_func": pre_func,
                "post_pain": v["womac_pain"], "post_func": v["womac_func"],
            })
    post = pd.DataFrame(rows)
    post["residual_pain_ge4"] = (post["post_pain"] >= 4).astype(int)
    coverage = tkr.assign(has_post_womac=tkr["kid"].isin(post.loc[post["post_pain"].notna(), "kid"])).groupby("phenotype", observed=False).agg(
        tkr_knees=("kid", "size"), tkr_knees_with_post_womac=("has_post_womac", "sum")
    ).reset_index()
    summary = post.groupby("phenotype", observed=False).agg(
        n_post_visits=("kid", "size"),
        n_knees=("kid", "nunique"),
        median_days_after_tkr=("days_after_tkr", "median"),
        post_pain_mean=("post_pain", "mean"),
        post_func_mean=("post_func", "mean"),
        residual_pain_ge4_percent=("residual_pain_ge4", lambda x: 100 * x.mean()),
        preop_pain_mean=("preop_pain", "mean"),
    ).reset_index()
    write_csv(coverage, out / "task2_post_tkr_womac_coverage.csv")
    write_csv(summary, out / "task2_post_tkr_outcomes_by_preop_phenotype.csv")
    write_csv(post, out / "task2_post_tkr_long.csv")
    use = post.dropna(subset=["post_pain", "preop_pain", "phenotype"]).copy()
    use = use[use["phenotype"].isin(PHENO_ORDER)]
    X = pd.get_dummies(use[["phenotype", "preop_pain", "days_after_tkr"]], columns=["phenotype"], drop_first=True)
    lr = LinearRegression().fit(X, use["post_pain"])
    coef = pd.DataFrame({"term": X.columns, "coef": lr.coef_})
    coef.loc[len(coef)] = ["intercept", lr.intercept_]
    boot_rows = []
    people = use["id"].drop_duplicates().to_numpy()
    for b in range(B):
        samp = RNG.choice(people, size=len(people), replace=True)
        boot = pd.concat([use[use["id"] == sid] for sid in samp], ignore_index=True)
        Xb = pd.get_dummies(boot[["phenotype", "preop_pain", "days_after_tkr"]], columns=["phenotype"], drop_first=True).reindex(columns=X.columns, fill_value=0)
        lb = LinearRegression().fit(Xb, boot["post_pain"])
        for term, val in zip(X.columns, lb.coef_):
            boot_rows.append({"bootstrap": b + 1, "term": term, "coef": val})
    boot = pd.DataFrame(boot_rows)
    ci = boot.groupby("term")["coef"].quantile([0.025, 0.975]).unstack().reset_index().rename(columns={0.025: "ci_low", 0.975: "ci_high"})
    coef = coef.merge(ci, on="term", how="left")
    write_csv(coef, out / "task2_post_tkr_adjusted_linear_model.csv")
    fig, ax = plt.subplots(figsize=(5, 3.6), dpi=180)
    ax.bar(summary["phenotype"].map(PHENO_SHORT), summary["post_pain_mean"])
    ax.set_xlabel("Pre-TKR phenotype")
    ax.set_ylabel("Mean post-TKR WOMAC pain")
    fig.tight_layout()
    fig.savefig(out / "task2_post_tkr_pain_by_phenotype.png")
    plt.close(fig)
    return coverage, summary


def task3_fnih(d: pd.DataFrame, z: zipfile.ZipFile, out: Path):
    lab = read_oai_member(z, "Biospec_FNIH_Labcorp00.txt")
    lc = std_col_map(lab.columns)
    markers = [
        "V00Urine_CTXII_NUMCA", "V00Serum_Comp_NUM", "V00Serum_HA_NUM", "V00Serum_CTXI_NUM",
        "V00Serum_NTXI_NUM", "V00Serum_PIIANP_NUM", "V00Serum_C2C_NUM", "V00Serum_CPII_NUM",
        "V00Serum_MMP_3_NUM", "V00Serum_CS846_NUM",
    ]
    bio = pd.DataFrame({"id": clean_id(lab[lc["ID"]])})
    for m in markers:
        if m.upper() in lc:
            bio[m] = numeric_series(lab[lc[m.upper()]])
    cf = read_oai_member(z, "Clinical_FNIH.txt")
    cc = std_col_map(cf.columns)
    clin = pd.DataFrame({"id": clean_id(cf[cc["ID"]]), "side": side_num(cf[cc["SIDE"]]).astype(float)})
    clin["kid"] = clin["id"].astype(str) + "_" + clin["side"].astype(int).astype(str)
    clin["fnih_case"] = numeric_series(cf[cc["CASE"]]) if "CASE" in cc else np.nan
    dd = d.merge(clin[["kid", "fnih_case"]], on="kid", how="inner").merge(bio, on="id", how="left")
    long_rows = []
    assoc = []
    for m in [c for c in bio.columns if c != "id"]:
        dd[f"log_{m}"] = np.log(dd[m].where(dd[m] > 0))
        for ph in PHENO_ORDER:
            g = dd[dd["phenotype"] == ph]
            long_rows.append({"marker": m, "phenotype": ph, "n": int(g[m].notna().sum()), "median": g[m].median(), "iqr": g[m].quantile(0.75) - g[m].quantile(0.25)})
        use = dd[[f"log_{m}", "phenotype", "age", "sex", "race", "bmi", "kl_base", "pain0"]].dropna()
        if len(use) > 50:
            y = stats.zscore(use[f"log_{m}"])
            X = pd.get_dummies(use[["phenotype", "age", "sex", "race", "bmi", "kl_base", "pain0"]], columns=["phenotype"], drop_first=True)
            beta = LinearRegression().fit(X, y).coef_
            for term, val in zip(X.columns, beta):
                if term.startswith("phenotype_"):
                    assoc.append({"marker": m, "term": term, "standardized_beta": val, "n": len(use)})
    prof = pd.DataFrame(long_rows)
    assoc = pd.DataFrame(assoc)
    if not assoc.empty:
        assoc["p_fdr_note"] = "Exploratory; use BH/FDR downstream if retaining all markers."
    write_csv(prof, out / "task3_fnih_biomarkers_by_phenotype.csv")
    write_csv(assoc, out / "task3_fnih_standardized_associations.csv")
    if not assoc.empty:
        plot = assoc[assoc["term"].str.contains("symptom_dominant")]
        fig, ax = plt.subplots(figsize=(5, 4), dpi=180)
        ax.barh(plot["marker"], plot["standardized_beta"])
        ax.axvline(0, color="0.4", lw=1)
        ax.set_xlabel("Std beta for SyD vs reference")
        fig.tight_layout()
        fig.savefig(out / "task3_fnih_syd_forest.png")
        plt.close(fig)
    return prof, assoc


def task4_pain_signature(d: pd.DataFrame, z: zipfile.ZipFile, out: Path):
    ac = read_oai_member(z, "AllClinical00.txt")
    c = std_col_map(ac.columns)
    base = pd.DataFrame({"id": clean_id(ac[c["ID"]])})
    pain_cols = [col for col in ac.columns if re.search(r"(HPN|BP30|OJPN|TMJ|P7[RL]KFR)", col, re.I)]
    for col in pain_cols:
        base[col] = numeric_series(ac[col])
    # Conservative proxy: count labelled pain-presence items coded as 1.
    count_cols = [col for col in pain_cols if not re.search(r"DK|NO|OFT|FR$", col)]
    base["non_knee_widespread_pain_count_proxy"] = sum((base[col] == 1).astype(int) for col in count_cols)
    base["right_knee_pain_freq"] = base["V00P7RKFR"] if "V00P7RKFR" in base else np.nan
    base["left_knee_pain_freq"] = base["V00P7LKFR"] if "V00P7LKFR" in base else np.nan
    rows = []
    for side in [1, 2]:
        tmp = base[["id", "non_knee_widespread_pain_count_proxy", "right_knee_pain_freq", "left_knee_pain_freq"]].copy()
        tmp["side"] = side
        tmp["kid"] = tmp["id"].astype(str) + "_" + str(side)
        tmp["knee_pain_frequency"] = tmp["right_knee_pain_freq"] if side == 1 else tmp["left_knee_pain_freq"]
        rows.append(tmp.drop(columns=["right_knee_pain_freq", "left_knee_pain_freq"]))
    pain = pd.concat(rows, ignore_index=True)
    dd = d.merge(pain, on=["kid", "id", "side"], how="left")
    prof = dd.groupby("phenotype", observed=False).agg(
        n=("kid", "size"),
        widespread_pain_count_mean=("non_knee_widespread_pain_count_proxy", "mean"),
        widespread_pain_count_median=("non_knee_widespread_pain_count_proxy", "median"),
        knee_pain_frequency_mean=("knee_pain_frequency", "mean"),
        womac_pain_mean=("pain0", "mean"),
    ).reset_index()
    write_csv(prof, out / "task4_central_pain_signature_by_phenotype.csv")
    fig, ax = plt.subplots(figsize=(5, 3.6), dpi=180)
    ax.bar(prof["phenotype"].map(PHENO_SHORT), prof["widespread_pain_count_mean"])
    ax.set_ylabel("Mean non-knee pain count proxy")
    ax.set_xlabel("Phenotype")
    fig.tight_layout()
    fig.savefig(out / "task4_widespread_pain_signature.png")
    plt.close(fig)
    return prof


def write_optional_model_scripts(out: Path):
    scripts = out / "optional_server_models"
    scripts.mkdir(parents=True, exist_ok=True)
    (scripts / "task9_JMbayes2_shared_random_effects_template.R").write_text(textwrap.dedent("""
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
    """).strip() + "\n")
    (scripts / "task10_multlcmm_four_outcome_template.R").write_text(textwrap.dedent("""
        # Template only: requires lcmm and convergence screening across random starts.
        library(lcmm)
        panel <- read.csv("../tables/oai_panel_long.csv")
        # multlcmm(cbind(mjsw, kl, womac_pain, womac_func) ~ poly(month, 2),
        #          random = ~ month, subject = "kid", ng = 2:5, data = panel)
        # Report GRoLTS: entropy, class size, posterior probability, random-start reproducibility.
    """).strip() + "\n")


def write_report(out: Path, manifest: dict):
    lines = [
        "# C1 OA discordance remaining-task completion report",
        "",
        "External MOST validation was not run, per instruction.",
        "",
        "## Material Passport",
        "",
        f"- Generated: 2026-06-21",
        f"- Raw OAI zip: `{manifest['oai_zip']}`",
        f"- Derived C1 tables: `{manifest['derived_tables']}`",
        f"- Bootstrap B for final incremental metrics: {manifest['bootstrap_B']}",
        f"- Bootstrap B for mediation/post-TKR exploratory CIs: {manifest['small_bootstrap_B']}",
        "",
        "## Task status",
        "",
        "| Task | Status | Primary outputs |",
        "|---|---|---|",
    ]
    for row in manifest["tasks"]:
        lines.append(f"| {row['task']} | {row['status']} | {row['outputs']} |")
    lines += [
        "",
        "## Interpretation cautions",
        "",
        "- MOAKS mediation uses baseline MRI only and IPSW for MRI-read subset selection; natural-effect estimates are regression/product approximations, not randomized intervention effects.",
        "- Post-TKR WOMAC coverage is explicitly reported before inference; estimates are conditional on TKR knees with post-operative OAI visits.",
        "- FNIH analyses are exploratory because the biomarker subset is a case-control/progression-enriched ancillary cohort.",
        "- The central sensitization analysis uses available OAI widespread/non-knee pain proxies, not a dedicated quantitative sensory testing battery.",
        "- Tasks 9 and 10 are supplied as server-side R templates because full joint models/multlcmm random-start grids require long-running convergence diagnostics.",
    ]
    (out / "remaining_tasks_completion_report.md").write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--derived", type=Path, default=DEFAULT_DERIVED)
    ap.add_argument("--optimization", type=Path, default=DEFAULT_OPT)
    ap.add_argument("--oai-zip", type=Path, default=DEFAULT_OAI_ZIP)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--small-bootstrap", type=int, default=250)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    panel, baseline, outcomes, pheno, features = load_core(args.derived)
    d = analysis_frame(panel, baseline, outcomes, pheno)

    manifest = {
        "oai_zip": str(args.oai_zip),
        "derived_tables": str(args.derived),
        "bootstrap_B": args.bootstrap,
        "small_bootstrap_B": args.small_bootstrap,
        "n_knees": int(d["kid"].nunique()),
        "n_participants": int(d["id"].nunique()),
        "tasks": [],
    }
    with zipfile.ZipFile(args.oai_zip) as z:
        task6_unbalanced_calibration(d, args.out)
        manifest["tasks"].append({"task": "6 calibration without class_weight", "status": "completed", "outputs": "task6_unbalanced_*"})
        task7_bootstrap_1000(d, args.out, args.bootstrap)
        manifest["tasks"].append({"task": "7 participant-cluster bootstrap >=1000", "status": "completed", "outputs": f"task7_cluster_bootstrap_B{args.bootstrap}_*.csv"})
        task5_cohort(d, z, args.out)
        manifest["tasks"].append({"task": "5 V00COHORT adjustment", "status": "completed", "outputs": "task5_v00cohort_*.csv"})
        task8_sensitivities(d, panel, args.out)
        manifest["tasks"].append({"task": "8a-8d sensitivity analyses", "status": "completed", "outputs": "task8_sensitivity_*.csv; task8d_visit_ipw_weights.csv"})
        task1_moaks(d, z, args.out, args.small_bootstrap)
        manifest["tasks"].append({"task": "1 baseline MOAKS IPSW mediation", "status": "completed", "outputs": "task1_moaks_*; task1_ipw_mediation_decomposition.csv"})
        task2_post_tkr(panel, d, args.out, args.small_bootstrap)
        manifest["tasks"].append({"task": "2 post-TKR residual pain", "status": "completed", "outputs": "task2_post_tkr_*"})
        task3_fnih(d, z, args.out)
        manifest["tasks"].append({"task": "3 FNIH biomarker subcohort", "status": "completed exploratory", "outputs": "task3_fnih_*"})
        task4_pain_signature(d, z, args.out)
        manifest["tasks"].append({"task": "4 central/widespread pain signature", "status": "completed proxy analysis", "outputs": "task4_*"})
    write_optional_model_scripts(args.out)
    manifest["tasks"].append({"task": "9 shared random-effects joint model", "status": "template supplied; not executed locally", "outputs": "optional_server_models/task9_*.R"})
    manifest["tasks"].append({"task": "10 multlcmm four-outcome trajectory model", "status": "template supplied; not executed locally", "outputs": "optional_server_models/task10_*.R"})
    (args.out / "remaining_tasks_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    write_report(args.out, manifest)


if __name__ == "__main__":
    main()
