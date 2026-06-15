## Summary

Code at first manuscript submission (`submission-2026-06-15`).

- GMM-4 utilization archetypes (Outpatient Irregular, Outpatient Regular, Moderate Inpatient, High Inpatient)
- Four-domain density vector and within-cluster residuals
- Q1–Q3 CCI association analyses with `edi_*` artifact manifest

Zenodo DOI for the citable release: [10.5281/zenodo.20706796](https://doi.org/10.5281/zenodo.20706796) (`v1.0.0`).

## Reproduce

1. Configure secrets per [`.env.example`](../../.env.example)
2. Run ETL then [`scripts/cohort_1/`](../../scripts/cohort_1/) notebooks 01 → 02 → 03a → 03b → 06
3. Verify [`outputs/manifest/edi_artifact_manifest.csv`](../../outputs/manifest/edi_artifact_manifest.csv)

## Documentation

- [Releases](../../docs/releases.md)
- [Pipeline](../../docs/pipeline.md)
- [Methods](../../docs/methods.md)
- [Artifacts](../../docs/artifacts.md)
