# Artifacts

Manuscript-facing files use the **`edi_*`** prefix under [`outputs/`](../outputs/).

**Navigation:** [README](../README.md) · [Pipeline](pipeline.md) · [Methods](methods.md) · [Supplement](methods_supplement.md) · [Releases](releases.md)

---

## Contents

- [Naming pattern](#naming-pattern)
- [Layout](#layout)
- [Manifest](#manifest)
- [Manuscript scope](#manuscript-scope)
- [Commit policy](#commit-policy)

---

## Naming pattern

```text
edi_{kind}{order}_{slug}.{ext}
```

| Kind | Extension | Use |
|------|-----------|-----|
| `fig` | `.png` | Figures |
| `tbl` | `.csv` | Tables |
| `data` | `.csv` | Backing stats |
| `report` | `.html` | Internal review bundle (not submitted) |

**Examples:**

| File | Type |
|------|------|
| `edi_fig01_disease_burden.png` | Figure |
| `edi_tbl02_gmm4_profiles.csv` | Table |
| `edi_data03_q2_interaction.csv` | Stats |

---

## Layout

```text
outputs/
├── figures/     edi_fig*.png
├── tables/      edi_tbl*.csv
├── data/        edi_data*.csv
├── reports/     edi_report_*.html
└── manifest/    edi_artifact_manifest.csv
```

Delta tables (`cohort_1.*`) are pipeline intermediates; only paths above are release artifacts.

Pipeline that writes these: [Pipeline → Manuscript run order](pipeline.md#manuscript-run-order-cohort-1)

---

## Manifest

[`outputs/manifest/edi_artifact_manifest.csv`](../outputs/manifest/edi_artifact_manifest.csv) — one row per artifact:

```text
artifact_id,kind,manuscript_location,canonical_path,legacy_path,generating_script,notes
```

| Function | Location |
|----------|----------|
| `write_edi_artifact_manifest()` | [`99_utils.py`](../scripts/cohort_1/99_utils.py) |
| Invoked via | [`99_report.py`](../scripts/cohort_1/99_report.py) |

Scripts write directly to `canonical_path`.

Verification: [Pipeline → Verification](pipeline.md#verification)

---

## Manuscript scope

| Artifact | Role | Methods link |
|----------|------|--------------|
| `edi_fig01`–`edi_fig04` | Disease Burden figures | [Methods → Q1–Q3](methods.md#association-between-disease-burden-and-data-volume) |
| `edi_tbl01b` | Table 1b classification input | [Methods → Clustering](methods.md#cluster-interpretation) |
| `edi_tbl02` | GMM-4 cluster profiles | [Pipeline → GMM-4 labels](pipeline.md#gmm-4-labels) |
| `edi_data01`–`edi_data08` | Q1–Q3 and domain-characterization stats | [Methods → Data volume](methods.md#data-volume-scoring) |

**Not produced here:** Table 1a, utilization/residual distribution figures (coauthors).

---

## Commit policy

| Path | Commit? | Why |
|------|---------|-----|
| `outputs/figures/` | yes | Manuscript PNGs; no row-level data |
| `outputs/data/` | yes | Aggregated Q1–Q3 stats |
| `outputs/tables/edi_tbl02_*.csv` | yes | Cluster-level profiles |
| `outputs/manifest/` | yes | Artifact registry |
| `outputs/tables/edi_tbl01b_*.csv` | **no** | Person-year rows with `person_id` |
| `outputs/reports/` | no | Internal HTML review bundle |

After running [`06_residual_analysis.py`](../scripts/cohort_1/06_residual_analysis.py) on Databricks, copy the allowed paths above into the repo if you want GitHub visitors to see figures without rerunning the pipeline.

See also: [`.gitignore`](../.gitignore) · [Releases → What ships](releases.md#what-ships-in-a-release)

---

## See also

- [Pipeline](pipeline.md) — how artifacts are generated
- [Methods](methods.md) — what Q1–Q3 statistics mean
- [Releases](releases.md) — tagging snapshots
