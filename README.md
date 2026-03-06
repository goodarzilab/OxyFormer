# OxyFormer

[![Status](https://img.shields.io/badge/status-research-blue)](https://github.com/goodarzilab/OxyFormer)
[![Python](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white)](https://github.com/goodarzilab/OxyFormer)
[![Method](https://img.shields.io/badge/method-Transformer%20%2B%20DML-7c3aed)](https://github.com/goodarzilab/OxyFormer)
[![License](https://img.shields.io/badge/license-Proprietary-red)](LICENSE)
[![Reports](https://img.shields.io/badge/reports-HTML%20%2B%20PDF-111827)](outputs/report)
[![GitHub stars](https://img.shields.io/github/stars/goodarzilab/OxyFormer?style=social)](https://github.com/goodarzilab/OxyFormer)

**OxyFormer** is a causal AI framework for estimating the effects of altitude-derived oxygen exposure on cancer outcomes. It combines **physiologic treatment engineering**, **attention-based tabular transformer pretraining**, **transferred county embeddings**, **orthogonal double machine learning**, **site-level heterogeneity analysis**, and **technical report generation**.

> This repository is a research codebase for county-level causal modeling and representation learning. It is not a clinical decision-support tool and should not be used for patient care.

## Overview

OxyFormer is built around a simple idea: if oxygen availability is a biologically meaningful exposure, then the right modeling stack should do more than fit a single regression. In this repository, the workflow is:

1. Construct a county-level oxygen exposure proxy from elevation and barometric pressure.
2. Fuse county-level public-health, demographic, socioeconomic, and behavioral covariates.
3. Pretrain a tabular transformer to learn reusable county embeddings.
4. Transfer those embeddings into a cross-fitted continuous-treatment DML estimator.
5. Estimate total and mediator-aware direct effects for all-cancer and site-specific endpoints.
6. Generate funder-facing HTML and PDF reports.

## Core Components

- **Treatment engineering**: converts elevation into inspired oxygen proxy and normalized `hypoxia_burden`.
- **Public-data fusion**: merges local files with ACS, PLACES, SVI, and RUCC county covariates.
- **Representation learning**: trains a dual-view masked tabular transformer with auxiliary health heads.
- **Causal estimation**: fits partial-linear DML models with cross-fitted ridge nuisance regressions.
- **Endpoint reuse**: extends the same learned representation across all-cancer and site-level models.
- **Reporting**: exports both an HTML white paper and a LaTeX/PDF technical paper.

## Repository Layout

| Path | Purpose |
| --- | --- |
| `phase1_county_pipeline.py` | Builds the baseline county table and oxygen-derived exposure variables from local input files. |
| `phase2_public_data.py` | Downloads and normalizes county-level public covariates from PLACES, SVI, RUCC, and ACS. |
| `phase25_feature_engineering.py` | Curates a smaller modeling feature set and prepares model matrices. |
| `phase26_foundation_model.py` | Trains the OxyFormer tabular transformer and exports learned county embeddings. |
| `phase3_causal_demo.py` | Fits the main cross-fitted DML models for total and direct effects. |
| `phase4_robustness.py` | Runs seed-sensitivity and leave-one-state-out robustness checks. |
| `plot_phase3_site_forest.py` | Renders lightweight forest plots from site-level effect tables. |
| `build_white_paper_report.py` | Generates the HTML white paper. |
| `build_technical_paper_assets.py` | Generates PDF-ready figures and LaTeX inputs for the technical paper. |
| `report/technical_white_paper.tex` | Main LaTeX source for the technical paper. |
| `causal_ai_blueprint.md` | Long-term roadmap for the broader causal AI program. |

## Data Expectations

### Local root-level input files

Phase 1 expects the following files in the repository root:

- `BYAREA_COUNTY.csv`
- `counties_x_elevation.csv`
- `ACSST5Y2021.S1901-Data.csv`
- `DiabetesPercentage.csv`
- `ObesityAll.csv`

### Public downloads

Phase 2 stores downloaded public data in:

- `data/public/raw`
- `data/public/metadata`

These raw public files are intentionally ignored from Git because they are large and can be regenerated locally.

## Environment

OxyFormer currently runs as a script-based repository rather than a packaged module.

### Minimum requirements

- Python `3.10+`
- `numpy`
- `pandas`
- `requests`
- `matplotlib`
- `torch`
- `pdflatex` if you want to build the PDF technical paper

A minimal install looks like:

```bash
pip install numpy pandas requests matplotlib torch
```

## End-to-End Run Order

Run all commands from the repository root.

### 1) Phase 1: local county table

```bash
python3 phase1_county_pipeline.py --folder . --output-dir outputs/phase1
```

### 2) Phase 2: public covariates

Download public data:

```bash
python3 phase2_public_data.py --root-dir . --acs-year 2024 --download
```

Build the merged public-data table:

```bash
python3 phase2_public_data.py --root-dir . --acs-year 2024 --build
```

### 3) Phase 2.5: feature engineering

```bash
python3 phase25_feature_engineering.py --root-dir .
```

### 4) Phase 2.6: OxyFormer pretraining

```bash
python3 phase26_foundation_model.py --root-dir .
```

This writes learned county embeddings, training history, and model weights into `outputs/phase26`.

### 5) Phase 3: causal models

All-cancer models only:

```bash
python3 phase3_causal_demo.py --root-dir . --use-foundation-embeddings
```

All-cancer + supported site-specific models:

```bash
python3 phase3_causal_demo.py --root-dir . --use-foundation-embeddings --include-site-models
```

### 6) Phase 4: robustness

```bash
python3 phase4_robustness.py --root-dir . --use-foundation-embeddings
```

### 7) Figures and reports

Forest plots:

```bash
python3 plot_phase3_site_forest.py --root-dir .
```

HTML white paper:

```bash
python3 build_white_paper_report.py
```

PDF technical paper assets:

```bash
python3 build_technical_paper_assets.py
```

PDF technical paper:

```bash
cd report
pdflatex -interaction=nonstopmode -output-directory=../outputs/report technical_white_paper.tex
pdflatex -interaction=nonstopmode -output-directory=../outputs/report technical_white_paper.tex
```

## Main Outputs

| Artifact | Path |
| --- | --- |
| Phase 1 summary | `outputs/phase1/phase1_summary.json` |
| Phase 2 summary | `outputs/phase2/phase2_summary.json` |
| Phase 2.5 summary | `outputs/phase25/phase25_summary.json` |
| OxyFormer model summary | `outputs/phase26/phase26_summary.json` |
| Main causal summary | `outputs/phase3/phase3_summary.json` |
| Robustness summary | `outputs/phase4/phase4_summary.json` |
| HTML report | `outputs/report/white_paper.html` |
| PDF technical paper | `outputs/report/technical_white_paper.pdf` |

## Current Modeling Choices

- **Treatment**: `hypoxia_burden = 1 - oxygen_fraction_of_sea_level`
- **Primary outcomes**: `all_cancer_incidence_rate`, `all_cancer_mortality_rate`
- **Representation model**: dual-view masked tabular transformer with auxiliary health targets
- **Causal head**: cross-fitted partial-linear DML
- **Nuisance learners**: ridge regression for treatment and outcome models
- **Geography adjustment**: state dummy variables inferred from county FIPS
- **Site-level extension**: high-support cancer endpoints for incidence and mortality

## Reports

This repository currently produces two report formats:

- **HTML white paper** with left-hand navigation bookmarks at `outputs/report/white_paper.html`
- **LaTeX/PDF technical paper** at `outputs/report/technical_white_paper.pdf`

Both reports are generated from the same underlying model outputs and are intended for technical communication, strategy, and fundraising.

## Git Hygiene

Large raw data files and large generated artifacts are intentionally ignored via `.gitignore` and `.ignore`.

That includes:

- root-level local source files such as `BYAREA_COUNTY.csv` and `BYAREA_COUNTY.TXT`
- downloaded public raw files under `data/public/raw`
- large generated matrices and embeddings under `outputs/phase2`, `outputs/phase25`, and `outputs/phase26`
- Python cache files and LaTeX intermediate files

## License

This repository is released under a **restrictive proprietary license**. See `LICENSE` for details.

## Contact / Use

If you want permission to use, adapt, distribute, or commercialize this codebase, model, or associated artifacts, contact the repository owner(s) for written permission first.
