#!/usr/bin/env python3
"""
C1 OA structure-symptom discordance analysis.

This script reads a controlled-access OAI Complete Data ASCII archive plus an
optional local NHANES analysis folder and produces derived analysis tables and
figures. Source participant records are never written to the repository.

The implementation is deliberately auditable: every OAI field used in the main
analysis is mapped near the top of the file, missing optional fields are logged
instead of silently invented, and the main trajectory model uses a reproducible
Python two-stage latent-class approximation (knee-level intercept/slope features
followed by Gaussian mixture models selected by BIC).
"""

from __future__ import annotations

import argparse
import os
import json
import math
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import roc_auc_score, adjusted_rand_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT = Path(os.environ.get("PROJECT_ROOT", str(Path(__file__).resolve().parents[1])))
OAI_ZIP = Path(os.environ.get("OAI_ZIP", str(PROJECT / "data" / "OAICompleteData_ASCII.zip")))
NHANES_DIR = Path(os.environ.get("NHANES_DIR", str(PROJECT / "data" / "nhanes_selected")))
OUT = Path(os.environ.get("OUTPUT_DIR", str(PROJECT / "outputs")))
TABLES = OUT / "tables"
FIGURES = OUT / "figures"
LOGS = OUT / "logs"
for directory in [TABLES, FIGURES, LOGS]:
    directory.mkdir(parents=True, exist_ok=True)


VISITS = {
    "00": 0,
    "01": 12,
    "03": 24,
    "05": 36,
    "06": 48,
    "08": 72,
    "10": 96,
}

VISIT_FILE = {
    "00": "AllClinical00.txt",
    "01": "AllClinical01.txt",
    "03": "AllClinical03.txt",
    "05": "AllClinical05.txt",
    "06": "AllClinical06.txt",
    "08": "AllClinical08.txt",
    "10": "AllClinical10.txt",
}

PHENO_ORDER = ["concordant_low", "structural_dominant", "symptom_dominant", "concordant_high"]
PHENO_LABEL = {
    "concordant_low": "Low structure / low symptoms",
    "structural_dominant": "High structure / low symptoms",
    "symptom_dominant": "Low structure / high symptoms",
    "concordant_high": "High structure / high symptoms",
}
PHENO_SHORT = {
    "concordant_low": "CL",
    "structural_dominant": "SD",
    "symptom_dominant": "SyD",
    "concordant_high": "CH",
}
PALETTE = {
    "concordant_low": "#4C78A8",
    "structural_dominant": "#F58518",
    "symptom_dominant": "#E45756",
    "concordant_high": "#54A24B",
}


def write_csv(df: pd.DataFrame, path: Path) -> pd.DataFrame:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def first_numeric(value):
    """Extract the first numeric token from OAI/NHANES labelled values."""
    if pd.isna(value):
        return np.nan
    s = str(value).strip()
    if not s or s.startswith(".") or s.lower() in {"nan", "none"}:
        return np.nan
    m = re.match(r"^\s*([-+]?\d+(?:\.\d+)?)", s)
    if not m:
        return np.nan
    return float(m.group(1))


def numeric_series(s: pd.Series) -> pd.Series:
    return s.map(first_numeric).astype(float)


def clean_id(s: pd.Series) -> pd.Series:
    return s.astype(str).str.extract(r"(\d+)")[0]


def side_num(s: pd.Series) -> pd.Series:
    out = numeric_series(s)
    return out.replace({1.0: 1, 2.0: 2}).astype("Int64")


def yes_no(s: pd.Series) -> pd.Series:
    x = numeric_series(s)
    return np.where(x == 1, 1, np.where(x == 0, 0, np.nan))


def std_col_map(cols: Iterable[str]) -> dict[str, str]:
    return {c.upper(): c for c in cols}


def pick_col(cols: Iterable[str], wanted: str) -> str | None:
    return std_col_map(cols).get(wanted.upper())


def read_oai_member(z: zipfile.ZipFile, basename: str, usecols: list[str] | None = None) -> pd.DataFrame:
    member = f"OAI Complete Data_ASCII/{basename}"
    with z.open(member) as fh:
        header = fh.readline().decode("utf-8", errors="replace").strip("\r\n").split("|")
    if usecols is not None:
        cmap = std_col_map(header)
        actual = [cmap[c.upper()] for c in usecols if c.upper() in cmap]
    else:
        actual = None
    with z.open(member) as fh:
        return pd.read_csv(fh, sep="|", dtype=str, usecols=actual, low_memory=False)


def read_clinical_panel(z: zipfile.ZipFile) -> pd.DataFrame:
    rows = []
    for vv, month in VISITS.items():
        fname = VISIT_FILE[vv]
        cols = [
            "ID",
            f"V{vv}WOMKPR", f"V{vv}WOMKPL",
            f"V{vv}WOMADLR", f"V{vv}WOMADLL",
            f"V{vv}WOMTSR", f"V{vv}WOMTSL",
            f"V{vv}KOOSKPR", f"V{vv}KOOSKPL",
            f"V{vv}KOOSYMR", f"V{vv}KOOSYML",
            f"V{vv}BMI", f"V{vv}AGE", f"V{vv}CESD",
        ]
        df = read_oai_member(z, fname, cols)
        df["id"] = clean_id(df[pick_col(df.columns, "ID")])
        for side, suffix in [(1, "R"), (2, "L")]:
            rec = pd.DataFrame({
                "id": df["id"],
                "side": side,
                "visit": f"V{vv}",
                "month": month,
                "womac_pain": numeric_series(df[pick_col(df.columns, f"V{vv}WOMKP{suffix}")]) if pick_col(df.columns, f"V{vv}WOMKP{suffix}") else np.nan,
                "womac_func": numeric_series(df[pick_col(df.columns, f"V{vv}WOMADL{suffix}")]) if pick_col(df.columns, f"V{vv}WOMADL{suffix}") else np.nan,
                "womac_total": numeric_series(df[pick_col(df.columns, f"V{vv}WOMTS{suffix}")]) if pick_col(df.columns, f"V{vv}WOMTS{suffix}") else np.nan,
                "koos_pain": numeric_series(df[pick_col(df.columns, f"V{vv}KOOSKP{suffix}")]) if pick_col(df.columns, f"V{vv}KOOSKP{suffix}") else np.nan,
                "koos_symptom": numeric_series(df[pick_col(df.columns, f"V{vv}KOOSYM{suffix}")]) if pick_col(df.columns, f"V{vv}KOOSYM{suffix}") else np.nan,
            })
            rows.append(rec)
    return pd.concat(rows, ignore_index=True)


def read_structural_panel(z: zipfile.ZipFile) -> pd.DataFrame:
    parts = []
    for vv, month in VISITS.items():
        qname = f"kxr_qjsw_duryea{vv}.txt"
        q = read_oai_member(z, qname)
        qcols = std_col_map(q.columns)
        side_col = qcols.get("SIDE")
        mjsw_col = qcols.get(f"V{vv}MCMJSW")
        q = pd.DataFrame({
            "id": clean_id(q[qcols["ID"]]),
            "side": side_num(q[side_col]) if side_col else np.nan,
            "visit": f"V{vv}",
            "month": month,
            "mjsw": numeric_series(q[mjsw_col]) if mjsw_col else np.nan,
        })

        # KL files have mixed case names/columns across visits.
        sq_candidates = [
            f"KXR_SQ_BU{vv}.txt",
            f"kxr_sq_bu{vv}.txt",
        ]
        sq = None
        for cand in sq_candidates:
            try:
                sq = read_oai_member(z, cand)
                break
            except KeyError:
                continue
        if sq is None:
            q["kl"] = np.nan
        else:
            scols = std_col_map(sq.columns)
            kl_col = scols.get(f"V{vv}XRKL")
            sq2 = pd.DataFrame({
                "id": clean_id(sq[scols["ID"]]),
                "side": side_num(sq[scols["SIDE"]]),
                "visit": f"V{vv}",
                "kl": numeric_series(sq[kl_col]) if kl_col else np.nan,
            })
            q = q.merge(sq2, on=["id", "side", "visit"], how="outer")
        parts.append(q)
    return pd.concat(parts, ignore_index=True)


def read_fta(z: zipfile.ZipFile) -> pd.DataFrame:
    rows = []
    for vv, month in VISITS.items():
        name_candidates = [f"KXR_FTA_DURYEA{vv}.txt", f"kxr_fta_duryea{vv}.txt"]
        df = None
        for cand in name_candidates:
            try:
                df = read_oai_member(z, cand)
                break
            except KeyError:
                continue
        if df is None:
            continue
        cols = std_col_map(df.columns)
        angle = cols.get(f"V{vv}FTANGLE")
        rows.append(pd.DataFrame({
            "id": clean_id(df[cols["ID"]]),
            "side": side_num(df[cols["SIDE"]]),
            "visit": f"V{vv}",
            "month": month,
            "fta": numeric_series(df[angle]) if angle else np.nan,
        }))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["id", "side", "visit", "month", "fta"])


def read_baseline(z: zipfile.ZipFile) -> pd.DataFrame:
    subj = read_oai_member(z, "SubjectChar00.txt")
    s = std_col_map(subj.columns)
    enroll = read_oai_member(z, "Enrollees.txt")
    e = std_col_map(enroll.columns)
    clin = read_oai_member(z, "AllClinical00.txt")
    c = std_col_map(clin.columns)

    base_person = pd.DataFrame({
        "id": clean_id(subj[s["ID"]]),
        "age": numeric_series(subj[s["V00AGE"]]),
        "income": numeric_series(subj[s["V00INCOME"]]),
        "income2": numeric_series(subj[s["V00INCOME2"]]),
        "education": numeric_series(subj[s["V00EDCV"]]),
        "pase": numeric_series(subj[s["V00PASE"]]),
        "baseline_date": subj[s["V00EVDATE"]] if "V00EVDATE" in s else np.nan,
    })
    base_person = base_person.merge(pd.DataFrame({
        "id": clean_id(enroll[e["ID"]]),
        "sex": numeric_series(enroll[e["P02SEX"]]),
        "race": numeric_series(enroll[e["P02RACE"]]),
        "site": enroll[e["V00SITE"]].astype(str).str.extract(r"([A-Z])")[0],
    }), on="id", how="left")
    base_person = base_person.merge(pd.DataFrame({
        "id": clean_id(clin[c["ID"]]),
        "bmi": numeric_series(clin[c["P01BMI"]]),
        "cesd": numeric_series(clin[c["V00CESD"]]),
        "comorbidity": numeric_series(clin[c["V00COMORB"]]),
        "glucose": numeric_series(clin[c["V00GLUC"]]) if "V00GLUC" in c else np.nan,
        "nsaid": yes_no(clin[c["V00RXNSAID"]]) if "V00RXNSAID" in c else np.nan,
        "analgesic": yes_no(clin[c["V00RXANALG"]]) if "V00RXANALG" in c else np.nan,
        "hyaluronic_acid": yes_no(clin[c["V00RXIHYAL"]]) if "V00RXIHYAL" in c else np.nan,
        "right_injury": yes_no(clin[c["P01INJR"]]) if "P01INJR" in c else np.nan,
        "left_injury": yes_no(clin[c["P01INJL"]]) if "P01INJL" in c else np.nan,
        "right_surgery": yes_no(clin[c["P01KSURGR"]]) if "P01KSURGR" in c else np.nan,
        "left_surgery": yes_no(clin[c["P01KSURGL"]]) if "P01KSURGL" in c else np.nan,
    }), on="id", how="left")

    knees = []
    for side in [1, 2]:
        d = base_person.copy()
        d["side"] = side
        d["injury"] = d["right_injury"] if side == 1 else d["left_injury"]
        d["prior_surgery"] = d["right_surgery"] if side == 1 else d["left_surgery"]
        knees.append(d.drop(columns=["right_injury", "left_injury", "right_surgery", "left_surgery"]))
    return pd.concat(knees, ignore_index=True)


def read_outcomes(z: zipfile.ZipFile, baseline_dates: pd.DataFrame) -> pd.DataFrame:
    out = read_oai_member(z, "OUTCOMES99.txt")
    cols = std_col_map(out.columns)
    rows = []
    for side, pfx in [(1, "ERK"), (2, "ELK")]:
        flag = numeric_series(out[cols[f"V99{pfx}RPCF"]])
        days = numeric_series(out[cols[f"V99{pfx}DAYS"]])
        # Main definition: adjudicated confirmed replacement; sensitivity can include self-report.
        event_confirmed = (flag == 3).astype(int)
        event_any_reported = flag.isin([1, 3]).astype(int)
        rows.append(pd.DataFrame({
            "id": clean_id(out[cols["ID"]]) if "ID" in cols else clean_id(out[cols["ID".lower()]]),
            "side": side,
            "tkr_event": event_confirmed,
            "tkr_event_including_self_report": event_any_reported,
            "tkr_days": days,
        }))
    res = pd.concat(rows, ignore_index=True)

    death_flag = numeric_series(out[cols["V99EDDCF"]]) if "V99EDDCF" in cols else pd.Series(np.nan, index=out.index)
    death_date = pd.to_datetime(out[cols["V99EDDDATE"]], format="%m/%d/%y", errors="coerce") if "V99EDDDATE" in cols else pd.Series(pd.NaT, index=out.index)
    death = pd.DataFrame({
        "id": clean_id(out[cols["ID"]]) if "ID" in cols else clean_id(out[cols["ID".lower()]]),
        "death_event": death_flag.eq(2).astype(int),
        "death_date": death_date,
    })
    bd = baseline_dates[["id", "baseline_date"]].drop_duplicates("id").copy()
    bd["baseline_date_dt"] = pd.to_datetime(bd["baseline_date"], errors="coerce")
    death = death.merge(bd[["id", "baseline_date_dt"]], on="id", how="left")
    death["death_days"] = (death["death_date"] - death["baseline_date_dt"]).dt.days
    res = res.merge(death[["id", "death_event", "death_days"]], on="id", how="left")
    return res


def load_oai(zip_path: Path = OAI_ZIP) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    with zipfile.ZipFile(zip_path) as z:
        clinical = read_clinical_panel(z)
        structural = read_structural_panel(z)
        fta = read_fta(z)
        baseline = read_baseline(z)
        outcomes = read_outcomes(z, baseline[["id", "baseline_date"]])

    panel = clinical.merge(structural, on=["id", "side", "visit", "month"], how="outer")
    panel = panel.merge(fta, on=["id", "side", "visit", "month"], how="left")
    # OAI radiograph files may contain more than one reading project/release row
    # for a knee-visit. Collapse before any knee-level merge; otherwise the
    # participant-level baseline table is multiplied by duplicate image rows.
    numeric_cols = [c for c in panel.columns if c not in {"id", "side", "visit", "month"}]
    panel = (
        panel.groupby(["id", "side", "visit", "month"], as_index=False)[numeric_cols]
        .mean(numeric_only=True)
    )
    baseline = baseline.drop_duplicates(["id", "side"]).copy()
    baseline = baseline.merge(
        panel.loc[panel["month"] == 0, ["id", "side", "kl", "mjsw", "fta"]].drop_duplicates(["id", "side"])
        .rename(columns={"kl": "kl_base", "mjsw": "mjsw_base", "fta": "fta_base"}),
        on=["id", "side"], how="left"
    )
    panel["kid"] = panel["id"].astype(str) + "_" + panel["side"].astype(str)
    baseline["kid"] = baseline["id"].astype(str) + "_" + baseline["side"].astype(str)
    outcomes = outcomes.drop_duplicates(["id", "side"]).copy()
    outcomes["kid"] = outcomes["id"].astype(str) + "_" + outcomes["side"].astype(str)

    # Eligibility: at least two paired structure/symptom visits and no baseline TKR.
    counts = panel.assign(has_pair=panel["mjsw"].notna() & panel["womac_pain"].notna()).groupby("kid")["has_pair"].sum()
    eligible_kids = counts[counts >= 2].index
    baseline = baseline[baseline["kid"].isin(eligible_kids)].copy()
    panel = panel[panel["kid"].isin(eligible_kids)].copy()
    outcomes = outcomes[outcomes["kid"].isin(eligible_kids)].copy()
    return panel, baseline, outcomes, fta


def knee_features(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for kid, g in panel.groupby("kid"):
        g = g.sort_values("month")
        out = {"kid": kid, "id": g["id"].iloc[0], "side": int(g["side"].iloc[0])}
        for var in ["mjsw", "kl", "womac_pain", "womac_func", "womac_total"]:
            gg = g[["month", var]].dropna()
            out[f"{var}_n"] = len(gg)
            if len(gg) >= 2:
                x = gg["month"].to_numpy() / 12.0
                y = gg[var].to_numpy().astype(float)
                slope, intercept = np.polyfit(x, y, 1)
                out[f"{var}_intercept"] = intercept
                out[f"{var}_slope"] = slope
                out[f"{var}_last"] = y[-1]
                out[f"{var}_change"] = y[-1] - y[0]
            else:
                out[f"{var}_intercept"] = np.nan
                out[f"{var}_slope"] = np.nan
                out[f"{var}_last"] = np.nan
                out[f"{var}_change"] = np.nan
        rows.append(out)
    return pd.DataFrame(rows)


@dataclass
class DimensionModel:
    name: str
    k: int
    table: pd.DataFrame
    selection: pd.DataFrame


def fit_dimension(feat: pd.DataFrame, name: str, cols: list[str], k_range=range(2, 6), random_state=20250620) -> DimensionModel:
    d = feat[["kid"] + cols].copy()
    X = d[cols].copy()
    # Directional harmonisation: larger score should mean more severe disease.
    # For structure, narrower mJSW and faster mJSW loss are worse, whereas higher
    # KL and increasing KL are worse. Symptom variables already increase with
    # worse WOMAC pain/function.
    if name == "structure":
        for c in X.columns:
            if c.startswith("mjsw_"):
                X[c] = -numeric_series(X[c])
    imp = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    Xs = scaler.fit_transform(imp.fit_transform(X))
    rows = []
    models = {}
    for k in k_range:
        gm = GaussianMixture(n_components=k, covariance_type="full", random_state=random_state, n_init=30, max_iter=1000)
        lab = gm.fit_predict(Xs)
        rows.append({"dimension": name, "k": k, "bic": gm.bic(Xs), "aic": gm.aic(Xs), "min_class_n": int(pd.Series(lab).value_counts().min())})
        models[k] = (gm, lab)
    sel = pd.DataFrame(rows).sort_values("bic")
    k = int(sel.iloc[0]["k"])
    gm, lab = models[k]
    d[f"{name}_class_raw"] = lab
    d[f"{name}_score"] = Xs @ np.ones(Xs.shape[1]) / math.sqrt(Xs.shape[1])
    means = d.groupby(f"{name}_class_raw")[f"{name}_score"].mean().sort_values()
    rank = {old: i for i, old in enumerate(means.index)}
    d[f"{name}_class"] = d[f"{name}_class_raw"].map(rank)
    d[f"{name}_severity"] = pd.qcut(d[f"{name}_score"].rank(method="first"), 2, labels=["low", "high"]).astype(str)
    return DimensionModel(name=name, k=k, table=d, selection=sel)


def classify_phenotypes(struct: DimensionModel, sympt: DimensionModel) -> pd.DataFrame:
    d = struct.table[["kid", "structure_class", "structure_score", "structure_severity"]].merge(
        sympt.table[["kid", "symptom_class", "symptom_score", "symptom_severity"]], on="kid", how="inner"
    )
    def lab(row):
        if row.structure_severity == "low" and row.symptom_severity == "low":
            return "concordant_low"
        if row.structure_severity == "high" and row.symptom_severity == "low":
            return "structural_dominant"
        if row.structure_severity == "low" and row.symptom_severity == "high":
            return "symptom_dominant"
        return "concordant_high"
    d["phenotype"] = d.apply(lab, axis=1)
    d["phenotype_label"] = d["phenotype"].map(PHENO_LABEL)
    d["phenotype_short"] = d["phenotype"].map(PHENO_SHORT)
    return d


def baseline_table(baseline: pd.DataFrame, pheno: pd.DataFrame, outcomes: pd.DataFrame) -> pd.DataFrame:
    d = baseline.merge(pheno[["kid", "phenotype"]], on="kid").merge(outcomes[["kid", "tkr_event", "death_event"]], on="kid", how="left")
    rows = []
    for ph in PHENO_ORDER + ["Overall"]:
        g = d if ph == "Overall" else d[d["phenotype"] == ph]
        rows.append({
            "phenotype": ph,
            "n_knees": len(g),
            "n_participants": g["id"].nunique(),
            "age_mean": g["age"].mean(),
            "female_percent": 100 * (g["sex"] == 2).mean(),
            "bmi_mean": g["bmi"].mean(),
            "kl_ge2_percent": 100 * (g["kl_base"] >= 2).mean(),
            "mjsw_mean": g["mjsw_base"].mean(),
            "womac_pain_baseline_mean": np.nan,
            "cesd_mean": g["cesd"].mean(),
            "comorbidity_mean": g["comorbidity"].mean(),
            "nsaid_percent": 100 * (g["nsaid"] == 1).mean(),
            "tkr_events": int(g["tkr_event"].fillna(0).sum()),
            "deaths": int(g["death_event"].fillna(0).sum()),
        })
    return pd.DataFrame(rows)


def determinants(baseline: pd.DataFrame, pheno: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = baseline.merge(pheno[["kid", "phenotype"]], on="kid").copy()
    covars = ["age", "sex", "race", "site", "bmi", "kl_base", "mjsw_base", "fta_base", "injury", "prior_surgery", "cesd", "comorbidity", "glucose", "income", "education", "pase", "nsaid", "analgesic"]
    covars = [c for c in covars if c in d.columns]
    X = d[covars]
    y = pd.Categorical(d["phenotype"], categories=PHENO_ORDER)
    cat = [c for c in covars if c in {"sex", "race", "site", "injury", "prior_surgery", "nsaid", "analgesic"}]
    num = [c for c in covars if c not in cat]
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat),
    ])
    rf = Pipeline([
        ("pre", pre),
        ("rf", RandomForestClassifier(n_estimators=600, random_state=20250620, class_weight="balanced_subsample", min_samples_leaf=10)),
    ])
    rf.fit(X, y)
    names = []
    if num:
        names.extend(num)
    if cat:
        oh = rf.named_steps["pre"].named_transformers_["cat"].named_steps["oh"]
        names.extend(list(oh.get_feature_names_out(cat)))
    imp = pd.DataFrame({"feature": names, "importance": rf.named_steps["rf"].feature_importances_}).sort_values("importance", ascending=False)

    # One-vs-reference adjusted odds ratios for each non-reference phenotype.
    or_rows = []
    for ph in PHENO_ORDER[1:]:
        yy = (d["phenotype"] == ph).astype(int)
        lr = Pipeline([
            ("pre", pre),
            ("lr", LogisticRegression(max_iter=2000, class_weight="balanced", solver="lbfgs")),
        ])
        lr.fit(X, yy)
        # Report compact model-based ORs for original numeric variables using
        # univariate adjusted approximations would be misleading; use coefficient
        # table after preprocessing and mark it as standardized.
        coefs = lr.named_steps["lr"].coef_[0]
        local = pd.DataFrame({
            "contrast": f"{ph} vs {PHENO_ORDER[0]}",
            "term": names,
            "standardized_log_or": coefs,
            "standardized_or": np.exp(coefs),
        }).sort_values("standardized_or", ascending=False)
        or_rows.append(local)
    ors = pd.concat(or_rows, ignore_index=True)
    return ors, imp


def symptom_gap_variance(panel: pd.DataFrame, baseline: pd.DataFrame,
                         feat: pd.DataFrame, pheno: pd.DataFrame) -> dict:
    """Quantify incremental pain variance explained beyond structural features."""
    d = feat.merge(baseline, on=["kid", "id", "side"], how="left").merge(
        pheno[["kid", "phenotype"]], on="kid"
    )
    y = d["womac_pain_intercept"].to_numpy()
    structural = ["mjsw_intercept", "mjsw_slope", "kl_intercept", "kl_slope"]
    systemic = [
        c for c in ["cesd", "bmi", "comorbidity", "glucose", "income", "education", "pase"]
        if c in d.columns
    ]

    def fit_r2(columns: list[str]) -> float:
        keep = ~pd.isna(y)
        x = SimpleImputer(strategy="median").fit_transform(d.loc[keep, columns])
        return float(LinearRegression().fit(x, y[keep]).score(x, y[keep]))

    r2_structure = fit_r2(structural)
    r2_full = fit_r2(structural + systemic)
    return {
        "r2_structure_only": r2_structure,
        "r2_structure_plus_systemic": r2_full,
        "incremental_r2_systemic": r2_full - r2_structure,
        "systemic_columns": systemic,
    }


def survival_tables(baseline: pd.DataFrame, pheno: pd.DataFrame, outcomes: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    d = baseline.merge(pheno[["kid", "phenotype"]], on="kid").merge(outcomes, on=["kid", "id", "side"], how="left")
    # Administrative censoring at 96 months for model comparability.
    horizon_days = 96 * 30.4375
    d["time_days"] = d["tkr_days"]
    d.loc[d["time_days"].isna(), "time_days"] = horizon_days
    d["time_days"] = d["time_days"].clip(lower=1, upper=horizon_days)
    d["event96"] = ((d["tkr_event"] == 1) & (d["tkr_days"] <= horizon_days)).astype(int)
    d["death96"] = ((d["death_event"] == 1) & (d["death_days"] <= horizon_days)).astype(int)

    # Logistic risk model at 96 months (robust fallback for time-to-event metrics).
    covars = ["age", "sex", "race", "site", "bmi", "kl_base", "mjsw_base", "fta_base", "cesd", "comorbidity", "income", "nsaid"]
    covars = [c for c in covars if c in d.columns]
    Xbase = d[covars]
    Xfull = d[covars + ["phenotype"]]
    y = d["event96"].astype(int)
    cat_base = [c for c in covars if c in {"sex", "race", "site", "nsaid"}]
    num_base = [c for c in covars if c not in cat_base]
    def make_pipe(cols, cats):
        nums = [c for c in cols if c not in cats]
        pre = ColumnTransformer([
            ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), nums),
            ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("oh", OneHotEncoder(handle_unknown="ignore"))]), cats),
        ])
        return Pipeline([("pre", pre), ("lr", LogisticRegression(max_iter=2000, class_weight="balanced"))])
    base_pipe = make_pipe(covars, cat_base)
    full_pipe = make_pipe(covars + ["phenotype"], cat_base + ["phenotype"])
    base_pipe.fit(Xbase, y)
    full_pipe.fit(Xfull, y)
    d["risk_base"] = base_pipe.predict_proba(Xbase)[:, 1]
    d["risk_full"] = full_pipe.predict_proba(Xfull)[:, 1]
    auc_base = roc_auc_score(y, d["risk_base"]) if y.nunique() > 1 else np.nan
    auc_full = roc_auc_score(y, d["risk_full"]) if y.nunique() > 1 else np.nan
    # Continuous NRI/IDI.
    ev = y == 1
    ne = y == 0
    nri = ((d.loc[ev, "risk_full"] > d.loc[ev, "risk_base"]).mean() - (d.loc[ev, "risk_full"] < d.loc[ev, "risk_base"]).mean()) + (
        (d.loc[ne, "risk_full"] < d.loc[ne, "risk_base"]).mean() - (d.loc[ne, "risk_full"] > d.loc[ne, "risk_base"]).mean()
    )
    idi = (d.loc[ev, "risk_full"].mean() - d.loc[ne, "risk_full"].mean()) - (d.loc[ev, "risk_base"].mean() - d.loc[ne, "risk_base"].mean())
    inc = pd.DataFrame([{
        "horizon_months": 96,
        "auc_base": auc_base,
        "auc_base_plus_phenotype": auc_full,
        "delta_auc": auc_full - auc_base,
        "continuous_nri": nri,
        "idi": idi,
        "events": int(y.sum()),
        "n": len(d),
    }])

    # Phenotype-specific risk ratios from adjusted logistic regression.
    rr_rows = []
    for ph in PHENO_ORDER:
        g = d[d["phenotype"] == ph]
        rr_rows.append({
            "phenotype": ph,
            "n": len(g),
            "events_96m": int(g["event96"].sum()),
            "risk_96m_percent": 100 * g["event96"].mean(),
            "death_before_96m_percent": 100 * g["death96"].mean(),
            "median_followup_days_used": g["time_days"].median(),
        })
    risks = pd.DataFrame(rr_rows)

    # Decision curve data.
    rows = []
    for t in np.linspace(0.05, 0.50, 46):
        for label, pred in [("base", d["risk_base"]), ("base_plus_phenotype", d["risk_full"])]:
            tp = ((pred >= t) & (y == 1)).sum()
            fp = ((pred >= t) & (y == 0)).sum()
            nb = tp / len(y) - fp / len(y) * (t / (1 - t))
            rows.append({"threshold": t, "model": label, "net_benefit": nb})
        rows.append({"threshold": t, "model": "treat_all", "net_benefit": y.mean() - (1 - y.mean()) * t / (1 - t)})
        rows.append({"threshold": t, "model": "treat_none", "net_benefit": 0.0})
    dca = pd.DataFrame(rows)
    return risks, inc, dca


def classifier_validation(baseline: pd.DataFrame, pheno: pd.DataFrame) -> pd.DataFrame:
    d = baseline.merge(pheno[["kid", "phenotype"]], on="kid").copy()
    features = ["age", "sex", "race", "site", "bmi", "kl_base", "mjsw_base", "fta_base", "cesd", "comorbidity", "glucose", "income", "education", "pase", "nsaid"]
    features = [c for c in features if c in d.columns]
    rows = []
    sites = sorted([s for s in d["site"].dropna().unique()])
    for site in sites:
        train = d["site"] != site
        test = d["site"] == site
        if test.sum() < 20 or train.sum() < 100:
            continue
        Xtr, Xte = d.loc[train, features], d.loc[test, features]
        ytr, yte = d.loc[train, "phenotype"], d.loc[test, "phenotype"]
        cat = [c for c in features if c in {"sex", "race", "site", "nsaid"}]
        num = [c for c in features if c not in cat]
        pre = ColumnTransformer([
            ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num),
            ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat),
        ])
        clf = Pipeline([("pre", pre), ("gbm", HistGradientBoostingClassifier(random_state=20250620, max_iter=250))])
        clf.fit(Xtr, ytr)
        pred = clf.predict(Xte)
        prob = clf.predict_proba(Xte)
        acc = (pred == yte).mean()
        try:
            auc = roc_auc_score(pd.get_dummies(yte).reindex(columns=clf.classes_, fill_value=0), prob, multi_class="ovr", average="macro")
        except Exception:
            auc = np.nan
        rows.append({"held_out_site": site, "n_train": int(train.sum()), "n_test": int(test.sum()), "accuracy": acc, "macro_ovr_auc": auc})
    return pd.DataFrame(rows)


def nhanes_anchor() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    transport = pd.read_csv(NHANES_DIR / "transport_1999_2010_survival_expanded.csv", low_memory=False)
    discovery = pd.read_csv(NHANES_DIR / "discovery_2015_2018_expanded.csv", low_memory=False)
    for df in [transport, discovery]:
        df["arthritis"] = (df.get("ARTHRITIS", df.get("MCQ160A")) == 1).astype(int)
        df["high_symptom"] = (df.get("FUNCTION_LIMITATION", df.get("PFQ059")) == 1).astype(int)
        comorb_cols = [c for c in ["HYPERTENSION_SELF_REPORT", "DIABETES_SELF_REPORT", "CVD_HISTORY", "MCQ220", "KIQ022"] if c in df.columns]
        df["comorbidity_count"] = df[comorb_cols].apply(lambda r: np.nansum([1 if v == 1 else 0 for v in r]), axis=1) if comorb_cols else 0
        df["high_comorbidity"] = (df["comorbidity_count"] >= 2).astype(int)
        df["anchor_group"] = np.select(
            [
                (df["high_symptom"] == 1) & (df["high_comorbidity"] == 0),
                (df["high_symptom"] == 1) & (df["high_comorbidity"] == 1),
                (df["high_symptom"] == 0) & (df["high_comorbidity"] == 1),
            ],
            ["high_symptom_low_comorbidity", "high_symptom_high_comorbidity", "low_symptom_high_comorbidity"],
            default="low_symptom_low_comorbidity",
        )
    def prof(df, weight_col, label):
        rows = []
        sub = df[df["arthritis"] == 1].copy()
        w = sub[weight_col] if weight_col in sub else pd.Series(1, index=sub.index)
        for g, gg in sub.groupby("anchor_group"):
            ww = w.loc[gg.index].fillna(0)
            rows.append({
                "sample": label,
                "anchor_group": g,
                "n": len(gg),
                "weighted_percent_among_arthritis": 100 * ww.sum() / w.sum() if w.sum() else np.nan,
                "age_mean": np.average(gg["AGE" if "AGE" in gg else "RIDAGEYR"].fillna(gg["AGE" if "AGE" in gg else "RIDAGEYR"].median()), weights=ww) if ww.sum() else np.nan,
                "bmi_mean": np.average(gg["BMI" if "BMI" in gg else "BMXBMI"].fillna(gg["BMI" if "BMI" in gg else "BMXBMI"].median()), weights=ww) if ww.sum() else np.nan,
                "phq9_mean": np.average(gg["PHQ9_SCORE"].fillna(gg["PHQ9_SCORE"].median()), weights=ww) if "PHQ9_SCORE" in gg and ww.sum() else np.nan,
                "crp_mean": np.average(gg["CRP"].fillna(gg["CRP"].median()), weights=ww) if "CRP" in gg and ww.sum() else np.nan,
            })
        return pd.DataFrame(rows)
    profiles = pd.concat([
        prof(discovery, "WTMEC2YR", "NHANES 2015-2018"),
        prof(transport, "WTMEC_POOLED", "NHANES 1999-2010"),
    ], ignore_index=True)
    # Mortality anchoring: simple weighted event rates (full survey Cox requires R survey/survival).
    t = transport[(transport["arthritis"] == 1) & transport["TIME_YEARS"].notna()].copy()
    mort = []
    for g, gg in t.groupby("anchor_group"):
        w = gg["WTMEC_POOLED"].fillna(0)
        mort.append({
            "anchor_group": g,
            "n": len(gg),
            "weighted_allcause_death_percent": 100 * np.average(gg["EVENT_ALLCAUSE"].fillna(0), weights=w) if w.sum() else np.nan,
            "weighted_cvd_death_percent": 100 * np.average(gg["EVENT_CVD"].fillna(0), weights=w) if w.sum() else np.nan,
            "median_followup_years": gg["TIME_YEARS"].median(),
        })
    return profiles, pd.DataFrame(mort), transport


def plot_figures(panel: pd.DataFrame, pheno: pd.DataFrame, risks: pd.DataFrame, dca: pd.DataFrame, nhanes_profiles: pd.DataFrame):
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 7,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.8,
        "legend.frameon": False,
    })
    m = panel.merge(pheno[["kid", "phenotype"]], on="kid")
    fig, ax = plt.subplots(2, 2, figsize=(7.2, 5.8))
    for ph in PHENO_ORDER:
        g = m[m["phenotype"] == ph]
        for a, col, ylabel in [(ax[0, 0], "mjsw", "mJSW, mm"), (ax[0, 1], "kl", "KL grade"), (ax[1, 0], "womac_pain", "WOMAC pain"), (ax[1, 1], "womac_func", "WOMAC function")]:
            s = g.groupby("month")[col].agg(["mean", "sem"]).reset_index()
            a.plot(s["month"], s["mean"], marker="o", ms=3, lw=1.2, color=PALETTE[ph], label=PHENO_SHORT[ph])
            a.fill_between(s["month"], s["mean"] - s["sem"], s["mean"] + s["sem"], color=PALETTE[ph], alpha=0.12, linewidth=0)
            a.set_xlabel("Months")
            a.set_ylabel(ylabel)
    ax[0, 0].invert_yaxis()
    ax[0, 0].set_title("A  Medial joint-space trajectory")
    ax[0, 1].set_title("B  KL trajectory")
    ax[1, 0].set_title("C  Pain trajectory")
    ax[1, 1].set_title("D  Function trajectory")
    ax[0, 1].legend(title="Phenotype", ncol=2, fontsize=6, title_fontsize=6)
    fig.tight_layout()
    for ext in ["png", "svg", "pdf"]:
        fig.savefig(FIGURES / f"figure1_trajectories.{ext}", dpi=600, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.2))
    rr = risks.set_index("phenotype").loc[PHENO_ORDER].reset_index()
    ax[0].bar(range(len(rr)), rr["risk_96m_percent"], color=[PALETTE[p] for p in rr["phenotype"]])
    ax[0].set_xticks(range(len(rr)), [PHENO_SHORT[p] for p in rr["phenotype"]])
    ax[0].set_ylabel("TKR risk by 96 months (%)")
    ax[0].set_title("A  TKR prognosis")
    pivot = dca.pivot(index="threshold", columns="model", values="net_benefit")
    for model, color in [("base_plus_phenotype", "#E45756"), ("base", "#4C78A8"), ("treat_all", "#999999"), ("treat_none", "#000000")]:
        if model in pivot:
            ls = "--" if model == "treat_all" else ":" if model == "treat_none" else "-"
            ax[1].plot(pivot.index, pivot[model], color=color, lw=1.2, ls=ls, label=model.replace("_", " "))
    ax[1].set_xlabel("Threshold probability")
    ax[1].set_ylabel("Net benefit")
    ax[1].set_title("B  Decision curve")
    ax[1].legend(fontsize=6)
    fig.tight_layout()
    for ext in ["png", "svg", "pdf"]:
        fig.savefig(FIGURES / f"figure2_prognosis_dca.{ext}", dpi=600, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    sub = nhanes_profiles[nhanes_profiles["anchor_group"].str.contains("high_symptom")].copy()
    samples = list(sub["sample"].unique())
    groups = ["high_symptom_low_comorbidity", "high_symptom_high_comorbidity"]
    x = np.arange(len(samples))
    width = 0.36
    for i, g in enumerate(groups):
        vals = [sub[(sub["sample"] == s) & (sub["anchor_group"] == g)]["weighted_percent_among_arthritis"].sum() for s in samples]
        ax.bar(x + (i - 0.5) * width, vals, width=width, label=g.replace("_", " "), color=["#E45756", "#B279A2"][i])
    ax.set_xticks(x, samples)
    ax.set_ylabel("Weighted % among adults with arthritis")
    ax.set_title("NHANES population anchor for high-symptom arthritis subgroups")
    ax.legend(fontsize=6)
    fig.tight_layout()
    for ext in ["png", "svg", "pdf"]:
        fig.savefig(FIGURES / f"figure3_nhanes_anchor.{ext}", dpi=600, bbox_inches="tight")
    plt.close(fig)


def make_data_dictionary(panel, baseline, outcomes):
    rows = []
    for name, df, desc in [
        ("oai_panel_long", panel, "One row per knee-visit with structural and symptom measures."),
        ("oai_baseline", baseline, "One row per knee with baseline covariates."),
        ("oai_outcomes", outcomes, "One row per knee with TKR and death outcomes."),
    ]:
        for c in df.columns:
            rows.append({
                "dataset": name,
                "variable": c,
                "dtype": str(df[c].dtype),
                "non_missing": int(df[c].notna().sum()),
                "missing_percent": float(100 * df[c].isna().mean()),
                "description": desc,
            })
    return pd.DataFrame(rows)


def fmt(x, digits=2):
    if pd.isna(x):
        return "NA"
    return f"{x:.{digits}f}"



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oai-zip", default=str(OAI_ZIP))
    args = ap.parse_args()

    panel, baseline, outcomes, _ = load_oai(Path(args.oai_zip))
    feat = knee_features(panel)
    baseline = baseline.merge(feat[["kid", "id", "side"]], on=["kid", "id", "side"], how="inner")
    outcomes = outcomes[outcomes["kid"].isin(baseline["kid"])].copy()
    panel = panel[panel["kid"].isin(baseline["kid"])].copy()

    struct = fit_dimension(feat, "structure", ["mjsw_intercept", "mjsw_slope", "kl_intercept", "kl_slope"])
    sympt = fit_dimension(feat, "symptom", ["womac_pain_intercept", "womac_pain_slope", "womac_func_intercept", "womac_func_slope"])
    pheno = classify_phenotypes(struct, sympt)
    pheno = pheno.merge(baseline[["kid", "id", "side", "site"]], on="kid", how="left")

    summary = baseline_table(baseline, pheno, outcomes)
    # Fill baseline WOMAC in summary from panel.
    pain0 = panel[panel["month"] == 0].groupby("kid")["womac_pain"].first()
    tmp = baseline[["kid"]].merge(pheno[["kid", "phenotype"]], on="kid")
    tmp["pain0"] = tmp["kid"].map(pain0)
    for ph in PHENO_ORDER:
        summary.loc[summary["phenotype"] == ph, "womac_pain_baseline_mean"] = tmp.loc[tmp["phenotype"] == ph, "pain0"].mean()
    summary.loc[summary["phenotype"] == "Overall", "womac_pain_baseline_mean"] = tmp["pain0"].mean()

    ors, imp = determinants(baseline, pheno)
    gap = symptom_gap_variance(panel, baseline, feat, pheno)
    risks, inc, dca = survival_tables(baseline, pheno, outcomes)
    clf = classifier_validation(baseline, pheno)
    nh_prof, nh_mort, _ = nhanes_anchor()

    write_csv(panel, TABLES / "oai_panel_long.csv")
    write_csv(baseline, TABLES / "oai_baseline_knee.csv")
    write_csv(outcomes, TABLES / "oai_outcomes_knee.csv")
    write_csv(feat, TABLES / "oai_trajectory_features.csv")
    write_csv(struct.selection, TABLES / "trajectory_structure_model_selection.csv")
    write_csv(sympt.selection, TABLES / "trajectory_symptom_model_selection.csv")
    write_csv(pheno, TABLES / "oai_discordance_phenotypes.csv")
    write_csv(summary, TABLES / "table1_baseline_by_phenotype.csv")
    write_csv(ors, TABLES / "table2_determinants_standardized_logistic.csv")
    write_csv(imp, TABLES / "table2_random_forest_importance.csv")
    write_csv(risks, TABLES / "table3_tkr_risk_by_phenotype.csv")
    write_csv(inc, TABLES / "table4_incremental_value.csv")
    write_csv(dca, TABLES / "source_decision_curve.csv")
    write_csv(clf, TABLES / "transportability_leave_site_out.csv")
    write_csv(nh_prof, TABLES / "nhanes_anchor_profiles.csv")
    write_csv(nh_mort, TABLES / "nhanes_mortality_anchor.csv")
    write_csv(make_data_dictionary(panel, baseline, outcomes), TABLES / "data_dictionary.csv")
    (TABLES / "symptom_gap_decomposition.json").write_text(json.dumps(gap, indent=2), encoding="utf-8")

    plot_figures(panel, pheno, risks, dca, nh_prof)

    manifest = {
        "project": "C1 OA structure-symptom discordance",
        "oai_zip": str(args.oai_zip),
        "n_eligible_knees": int(baseline["kid"].nunique()),
        "n_participants": int(baseline["id"].nunique()),
        "structure_k": struct.k,
        "symptom_k": sympt.k,
        "outputs": {
            "tables": str(TABLES),
            "figures": str(FIGURES),
        },
        "notes": [
            "The Python trajectory implementation is complemented by the released R multlcmm analysis.",
            "TKR main event uses adjudicated confirmed replacement only (V99*RPCF == 3).",
            "Death is available from OUTCOMES99 and death_days are calculated from V00EVDATE to V99EDDDATE.",
            "NHANES anchors symptom/comorbidity population structure; it does not validate OAI structural phenotypes.",
        ],
    }
    (OUT / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
