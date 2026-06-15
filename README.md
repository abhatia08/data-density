# EHR Data Density Index (EDI)

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20706796.svg)](https://doi.org/10.5281/zenodo.20706796)
[![Issues](https://img.shields.io/github/issues/abhatia08/data-density)](https://github.com/abhatia08/data-density/issues)

PySpark/Databricks pipeline for the EHR Data Density Index: **GMM-4** utilization archetypes, four-domain density vectors, and **CCI** association analyses on OMOP data.

**Navigation:** [Pipeline](docs/pipeline.md) · [Methods](docs/methods.md) · [Supplement](docs/methods_supplement.md) · [Artifacts](docs/artifacts.md) · [Releases](docs/releases.md)

---

## Authors

Abhishek Bhatia · Tomas McIntee · Sydney Lash · Emily Pfaff

---

## Quick start

1. Copy [`.env.example`](.env.example) → `.env` and configure Databricks secret scope for ETL storage.
2. Run ETL on Databricks:
   ```text
   scripts/00_etl/core_etl.py  →  scripts/00_etl/cohort_1_etl.py
   ```
3. Run the manuscript pipeline ([`scripts/cohort_1/`](scripts/cohort_1/)):
   ```text
   01_unified_tables → 02_feature_engineering → 03a_rules → 03b_gmm → 06_residual_analysis
   ```
   [`06_residual_analysis`](scripts/cohort_1/06_residual_analysis.py) calls [`99_report`](scripts/cohort_1/99_report.py) for the HTML bundle and manifest.

Validation cohorts [`cohort_2a/`](scripts/cohort_2a/) and [`cohort_2b/`](scripts/cohort_2b/) apply pretrained GMM bundles from cohort 1 (**inference only**).

---

## Repository map

```text
data-density/
├── docs/
├── scripts/
│   ├── 00_etl/           OMOP CSV → Delta
│   ├── cohort_1/         manuscript pipeline
│   ├── cohort_2a/        external validation (inference)
│   └── cohort_2b/        external validation (inference)
├── source_data/          gitignored — local OMOP drops
├── outputs/              edi_* artifacts (see policy below)
├── CITATION.cff
├── .zenodo.json
├── LICENSE
├── .env.example
└── .gitignore
```

---

## Documentation

| Doc | Contents |
|-----|----------|
| [**Pipeline**](docs/pipeline.md) | Layout, run order, environment, ops notes |
| [**Methods**](docs/methods.md) | Manuscript methods (LaTeX equations) |
| [**Supplement**](docs/methods_supplement.md) | S1–S7 implementation detail |
| [**Artifacts**](docs/artifacts.md) | `edi_*` naming, layout, manifest, commit policy |
| [**Releases**](docs/releases.md) | Tags, GitHub Releases, [Zenodo DOI](docs/releases.md#zenodo-doi) ([10.5281/zenodo.20706796](https://doi.org/10.5281/zenodo.20706796)) |

---

## Citation

Cite via [`CITATION.cff`](CITATION.cff) or:

```text
Bhatia A, McIntee T, Lash S, Pfaff E. (2026). EHR Data Density Index (EDI) Pipeline
(Version 1.0.0) [Software]. Zenodo. https://doi.org/10.5281/zenodo.20706796
```

Archived at [10.5281/zenodo.20706796](https://doi.org/10.5281/zenodo.20706796) ([record](https://zenodo.org/records/20706796), tag [`v1.0.0`](https://github.com/abhatia08/data-density/releases/tag/v1.0.0)).

---

## Data and commit policy

OMOP tables, model bundles, and patient-level exports stay local (see [`.gitignore`](.gitignore)).

Manuscript figures and aggregate `edi_*` stats under `outputs/` may be committed after a pipeline run — see [Artifacts → Commit policy](docs/artifacts.md#commit-policy).

---

## Issues

Bug reports, reproducibility questions, and feature requests: [**GitHub Issues**](https://github.com/abhatia08/data-density/issues).

Please include your Databricks runtime, cohort, and the notebook step where a failure occurred. Do **not** attach OMOP extracts, `.env` files, or any data containing `person_id`.

---

## License

This project is licensed under the [MIT License](LICENSE) — Copyright © 2026 Abhishek Bhatia, Tomas McIntee, Sydney Lash, Emily Pfaff.
