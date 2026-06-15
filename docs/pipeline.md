# Pipeline

Databricks notebook source (`.py`). Intermediate tables live in **Delta**; manuscript artifacts use the `edi_*` prefix under [`outputs/`](../outputs/) ([Artifacts](artifacts.md)).

**Navigation:** [README](../README.md) · [Methods](methods.md) · [Supplement](methods_supplement.md) · [Artifacts](artifacts.md) · [Releases](releases.md)

---

## Contents

- [Repository layout](#repository-layout)
- [Manuscript run order](#manuscript-run-order-cohort-1)
- [ETL](#etl)
- [Environment](#environment)
- [Verification](#verification)
- [Operational notes](#operational-notes)

---

## Repository layout

```text
data-density/
├── docs/
│   ├── pipeline.md          ← you are here
│   ├── methods.md
│   ├── methods_supplement.md
│   ├── artifacts.md
│   └── releases.md
├── scripts/
│   ├── 00_etl/              OMOP CSV → Delta
│   ├── cohort_1/            manuscript pipeline
│   ├── cohort_2a/           external validation (inference)
│   └── cohort_2b/           external validation (inference)
├── source_data/             gitignored — local OMOP drops
├── outputs/                 edi_* artifacts ([policy](artifacts.md#commit-policy))
└── .env.example
```

Shared library: [`scripts/cohort_1/99_utils.py`](../scripts/cohort_1/99_utils.py) (via `# MAGIC %run ./99_utils`).

---

## Manuscript run order (cohort 1)

```text
OMOP Delta (cohort_1)
    │
    ▼
01_unified_tables ──► unified_daily / unified_yearly
    │
    ▼
02_feature_engineering ──► archetype_features_yearly
    │
    ├── 03a_rules ──► rules_yearly          (RUN_ADAPTIVE=False default)
    └── 03b_gmm ──► gmm_4_yearly + bundle
    │
    ▼
06_residual_analysis ──► outputs/{figures,tables,data}/
    └── 99_report ──► outputs/reports/ + manifest/
```

| Notebook | Delta output | Role |
|----------|--------------|------|
| [`01_unified_tables.py`](../scripts/cohort_1/01_unified_tables.py) | `unified_daily`, `unified_yearly` | Person-date spine, CCI, domain arrays |
| [`02_feature_engineering.py`](../scripts/cohort_1/02_feature_engineering.py) | `archetype_features_yearly` | Utilization features + four domain counts |
| [`03a_rules.py`](../scripts/cohort_1/03a_rules.py) | `rules_yearly` | Seven-archetype rules ([supplement switches](methods_supplement.md#code-switches)) |
| [`03b_gmm.py`](../scripts/cohort_1/03b_gmm.py) | `gmm_4_yearly` + bundle | GMM-4 (`K_VALUES=[4]` default) |
| [`06_residual_analysis.py`](../scripts/cohort_1/06_residual_analysis.py) | — | Residuals, Q1–Q3, `edi_*` writes |
| [`99_report.py`](../scripts/cohort_1/99_report.py) | — | HTML report + manifest (called from `06`) |

### GMM-4 labels

| ID | Label |
|----|-------|
| `0` | Moderate Inpatient |
| `1` | Outpatient Irregular |
| `2` | Outpatient Regular *(Q2/Q3 reference — see [Methods → Q3](methods.md#association-between-disease-burden-and-data-volume))* |
| `3` | High Inpatient |

Label mapping: [`get_gmm4_labels()`](../scripts/cohort_1/99_utils.py) in `99_utils.py`.

---

## ETL

| Notebook | Database |
|----------|----------|
| [`core_etl.py`](../scripts/00_etl/core_etl.py) | `vocab` |
| [`cohort_1_etl.py`](../scripts/00_etl/cohort_1_etl.py) | `cohort_1` |
| [`cohort_2a_etl.py`](../scripts/00_etl/cohort_2a_etl.py) | `cohort_2a` |
| [`cohort_2b_etl.py`](../scripts/00_etl/cohort_2b_etl.py) | `cohort_2b` |

---

## Environment

| Item | Value |
|------|-------|
| Runtime | Databricks + Delta Lake (`spark`, `dbutils` provided) |
| Python stack | PySpark, scikit-learn, NumPy, pandas, SciPy, statsmodels, matplotlib, seaborn, mlflow, cloudpickle, psutil |
| ETL secrets | `SECRET_SCOPE`, `STORAGE_ACCOUNT_KEY` |
| Optional URIs | `GMM_4_MODEL_URI`, `GMM_7_MODEL_URI` (cohort 2a/2b) |
| Seeds | `42` (GMM, stratified sampling) |
| Default bundle | `/dbfs/mnt/models/cohort_1/gmm_4/gmm_bundle.pkl` |

### Supplement sensitivity

| Notebook | Setting |
|----------|---------|
| [`03a_rules.py`](../scripts/cohort_1/03a_rules.py) | `RUN_ADAPTIVE=True` |
| [`03b_gmm.py`](../scripts/cohort_1/03b_gmm.py) | `K_VALUES=[4, 7]` |

Details: [Supplement → S5](methods_supplement.md#s5-alternative-clustering-methods).

---

## Verification

Re-run [`06_residual_analysis.py`](../scripts/cohort_1/06_residual_analysis.py) and confirm every row in [`outputs/manifest/edi_artifact_manifest.csv`](../outputs/manifest/edi_artifact_manifest.csv) exists on disk with non-zero size.

See also: [Artifacts → Manifest](artifacts.md#manifest).

---

## Operational notes

| Topic | Note |
|-------|------|
| Serverless Spark | `configure_spark_optimizations()` skips platform-managed conf keys |
| `%run` on Serverless | `99_report.py` must `%run ./99_utils` for EDI constants |
| Baseline cluster | Q2/Q3 reference = Outpatient Regular (`BASELINE_CLUSTER = 2`) |
| Out of repo | Table 1a, coauthor utilization/residual figures |
| In repo scope | `edi_tbl01b`, Disease Burden figures (`edi_fig01`–`edi_fig04`) |
| `cohort_2b` | `%run` lines need `# MAGIC` prefix |
| Do not commit | OMOP source data, full `outputs/`, `.env`, `*.pkl`, patient-level exports ([`.gitignore`](../.gitignore)) |

---

## See also

- [Methods](methods.md) — statistical definitions for Q1–Q3
- [Artifacts](artifacts.md) — `edi_*` output layout
- [Releases](releases.md) — tagging workflow
