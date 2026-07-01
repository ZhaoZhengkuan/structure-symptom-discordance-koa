# Code — structure–symptom discordance in knee osteoarthritis

Analysis pipeline for the manuscript *"Structure–Symptom Discordance as a Longitudinal Phenotype of Knee Osteoarthritis."* This repository contains the **statistical/analysis code only**; figure- and table-rendering utilities are intentionally omitted. Running the pipeline on the source data (see `../data/DATA_DICTIONARY.md`) reproduces the derived datasets and all numeric results.

## Pipeline overview

**Python (`python/`)**
| Script | Purpose |
|---|---|
| `01_main_analysis.py` | Build the analytic cohort from OAI; estimate structural and symptom trajectories; define the 2×2 discordance phenotypes; baseline tables; primary competing-risk/IPCW TKR analysis |
| `02_tierA_statistical_upgrades.py` | Pain-inclusive base model; cross-validated incremental metrics; recalibration; decision-curve inputs; symptom-gap decomposition |
| `03_optimization_runner.py` | Orchestration of the analysis task list |
| `04_mechanism_and_clinical_analysis.py` | MOAKS mediation, FNIH biomarkers, central/widespread pain, post-TKR outcomes, subcohort and sensitivity analyses, unbalanced-refit calibration, 1,000-replicate bootstrap |
| `05_incremental_value_cv.py` | Stand-alone cross-validated AUC/NRI/IDI with cluster bootstrap |
| `06_landmark_analysis.py` | Landmark (0–24-month) re-derivation and post-landmark TKR risk |
| `07_moaks_mediation.py` | Inverse-probability-weighted causal mediation (structure→pain; symptom-dominant excess) |
| `08_post_tkr_outcomes.py` | Post-operative WOMAC pain by pre-operative phenotype, adjusted models |

**R (`R/`)**
| Script | Purpose |
|---|---|
| `00_install_r_packages.R` | Install required R packages |
| `10_lcmm_joint_trajectory.R`, `11_lcmm_joint_composite_trajectory.R` | Latent-class joint trajectory models (corroborating heterogeneity) |
| `20_fine_gray_cmprsk.R`, `21_cause_specific_models.R`, `22_fine_gray_cluster_bootstrap.R` | Competing-risk models (subdistribution and cause-specific) with participant-cluster bootstrap |
| `30_mice_cluster_bootstrap.R` | Multiple imputation with cluster-bootstrap inference |
| `40_joint_model_JMbayes2_template.R` | Shared-random-effects joint longitudinal–survival model (server-side template; optional confirmatory) |
| `41_multlcmm_four_outcome_template.R` | Four-outcome multivariate latent-class trajectory model (server-side template; optional confirmatory) |

## Running

The scripts no longer contain any hard-coded local paths. By default they
assume the following layout, where the data and results folders sit next to
this repository (matching the `../data` reference above):

```
project-root/
├── data/                     # OAI + NHANES source data (you supply this)
├── outputs/                  # created by the scripts
└── structure-symptom-discordance-koa/   # this repository
    ├── python/
    └── R/
```

You can override every path without editing code, via environment variables
(`C1_PROJECT`, `OAI_ZIP`, `C1_DERIVED`, `C1_OPT`, `C1_OUT`) or the per-script
CLI flags (e.g. `--oai-zip`).

1. Obtain OAI and NHANES source data (`../data/DATA_DICTIONARY.md`). Either
   place them under `../data/` as shown above, or set `C1_PROJECT=/path/to/project-root`
   (and, if the OAI zip lives elsewhere, `OAI_ZIP=/path/to/OAICompleteData_ASCII.zip`).
2. Python: `pip install -r requirements.txt`, then run `python/01_…` through `python/08_…` in order.
3. R: `Rscript R/00_install_r_packages.R`, then the `R/` scripts as referenced by the Python orchestration. The two templates (`40_`, `41_`) are heavier server-side confirmatory models and are not required to reproduce the reported results, which rely on the landmark, competing-risk, IPW, and mediation analyses above.

## Dependencies
Python dependencies are pinned in `requirements.txt` (numpy, pandas, scipy, scikit-learn, lifelines, statsmodels, joblib, and related). R dependencies (lcmm, cmprsk, survival, mice, JMbayes2) install via `R/00_install_r_packages.R`.

## Notes
- Inference accommodates within-person clustering (both knees per participant) throughout.
- Random seeds are set for cross-validation and bootstrap reproducibility.
- No source patient-level data are included in this repository.
