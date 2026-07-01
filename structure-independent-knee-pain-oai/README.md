# Structure–Symptom Discordance in Knee Osteoarthritis

Analysis code for a longitudinal study of structure–symptom discordance phenotypes in knee osteoarthritis. The workflow derives knee-level structural and symptom trajectories, defines four discordance phenotypes, evaluates total knee replacement (TKR) prognosis with death as a competing event, and performs prediction, sensitivity, MRI, biomarker, non-knee pain, and postoperative analyses.

## Repository contents

```text
.
├── python/
│   ├── 01_primary_analysis.py
│   ├── 02_statistical_upgrades.py
│   ├── 03_prediction_and_sensitivity.py
│   ├── 04_mechanism_and_clinical.py
│   └── config_example.py
├── R/
│   ├── install_r_packages.R
│   ├── 01_lcmm_composite_trajectory.R
│   ├── 02_fine_gray.R
│   ├── 03_fine_gray_cluster_bootstrap.R
│   ├── 04_cause_specific_models.R
│   └── 05_mice_sensitivity.R
├── requirements.txt
└── .gitignore
```

## Data access

This repository contains code only. It does not contain participant-level data, derived datasets, model outputs, figures, or credentials.

The primary analysis requires controlled-access Osteoarthritis Initiative (OAI) files obtained under the applicable data-use agreement. The population-context analyses use public National Health and Nutrition Examination Survey (NHANES) files. Users are responsible for obtaining the source data and complying with all access and redistribution terms.

Expected local paths default to:

```text
data/OAICompleteData_ASCII.zip
data/nhanes_selected/
outputs/
```

Paths can instead be supplied through environment variables:

```bash
export PROJECT_ROOT="$(pwd)"
export OAI_ZIP="data/OAICompleteData_ASCII.zip"
export NHANES_DIR="data/nhanes_selected"
export OUTPUT_DIR="outputs"
```

`python/config_example.py` shows the same relative-path convention. Do not commit source data or generated outputs.

## Software requirements

- Python 3.10 or later
- R with the packages listed in `R/install_r_packages.R`

Create a Python environment and install the frozen dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
Rscript R/install_r_packages.R
```

## Recommended execution order

Run commands from the repository root.

### 1. Primary extraction and phenotype analysis

```bash
python python/01_primary_analysis.py --oai-zip "$OAI_ZIP"
```

This step extracts the longitudinal OAI panel, estimates knee-level trajectory features, defines the four structure–symptom phenotypes, and writes derived tables and figures locally under `outputs/`.

### 2. Python statistical upgrades

```bash
python python/02_statistical_upgrades.py --bootstrap 200 --imputations 20
python python/03_prediction_and_sensitivity.py
```

These scripts perform participant-cluster bootstrap summaries, imputation sensitivity analyses, cumulative-incidence calculations, repeated cross-validation, calibration, reclassification, IPCW risk estimation, the restricted baseline-to-24-month trajectory sensitivity analysis, and additional robustness checks.

### 3. R longitudinal and competing-risk analyses

```bash
Rscript R/01_lcmm_composite_trajectory.R --kmax 2 --rep 5 --maxiter 120
Rscript R/02_fine_gray.R
Rscript R/03_fine_gray_cluster_bootstrap.R --B 200
Rscript R/04_cause_specific_models.R
Rscript R/05_mice_sensitivity.R --m 20 --B 200
```

The R workflow fits the composite multivariate latent-class mixed model, Fine–Gray and cause-specific models, participant-cluster bootstrap intervals, and the multiple-imputation sensitivity analysis.

### 4. MRI, biomarker, non-knee pain, and postoperative analyses

```bash
python python/04_mechanism_and_clinical.py   --derived outputs/tables   --optimization outputs/optimization   --oai-zip "$OAI_ZIP"   --out outputs/mechanisms   --bootstrap 1000   --small-bootstrap 250
```

The MRI analysis is an inverse-selection-weighted, covariate-adjusted product-of-coefficients association decomposition. It is not a randomized natural-effects analysis. Biomarker analyses are descriptive and exploratory. The non-knee pain count is an indirect proxy rather than a direct quantitative sensory test. Postoperative percentages are calculated across observed postoperative assessments.

## Reproducibility notes

- Fixed random seeds are retained in the scripts.
- The Fine–Gray phenotype confidence intervals use 200 participant-cluster bootstrap resamples.
- Prediction-metric intervals use 1,000 participant-cluster bootstrap resamples.
- Prediction folds in the reported workflow are stratified at the knee level and are not grouped by participant.
- The baseline-to-24-month analysis is a restricted-trajectory sensitivity analysis, not a fully implemented competing-risk landmark analysis.
- The primary phenotype uses observations through 96 months; analyses should not be interpreted as a preoperative clinical prediction rule.
- Generated files belong in `outputs/`, which is excluded by `.gitignore`.

## Privacy and security

Do not commit OAI or NHANES participant-level records, derived knee-level tables, credentials, local environment files, or generated outputs. The included `.gitignore` excludes common data, result, credential, cache, and environment files.

## License and citation

No license is asserted in this code-only release. Add a license only after confirming that it is compatible with the source-data agreements and institutional policy. When citing this repository, use the associated article and the archived repository version or DOI once available.
