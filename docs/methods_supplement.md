# Supplementary Methods

**Navigation:** [README](../README.md) · [Pipeline](pipeline.md) · [Methods](methods.md) · [Artifacts](artifacts.md) · [Releases](releases.md)

Main manuscript prose: [Methods](methods.md)

---

## Contents

- [S1 · Unified person-date table](#s1-unified-person-date-table)
- [S2 · Charlson Comorbidity Index](#s2-charlson-comorbidity-index)
- [S3 · Feature engineering](#s3-feature-engineering)
- [S4 · GMM preprocessing and fitting](#s4-gmm-preprocessing-and-fitting)
- [S5 · Alternative clustering methods](#s5-alternative-clustering-methods)
- [S6 · Cross-method concordance](#s6-cross-method-concordance)
- [S7 · Software](#s7-software)

---

## S1. Unified person-date table

Built from seven OMOP domain tables:

| Table | Role |
|-------|------|
| `person` | Demographics |
| `visit_occurrence` | Visit spans |
| `condition_occurrence` | Diagnoses |
| `drug_exposure` | Medications |
| `measurement` | Labs/vitals |
| `procedure_occurrence` | Procedures |
| `observation` | Other clinical facts |

Notebook: [`01_unified_tables.py`](../scripts/cohort_1/01_unified_tables.py) · Main text: [Methods → Unified table](methods.md#unified-person-date-table)

For each domain, we extracted the relevant date column (e.g. `condition_start_date`, `drug_exposure_start_date`) and standardized it to a common `date` field. Concept identifiers were aggregated into arrays per `(person_id, date)`.

The person-date spine included all dates with any clinical activity — formal visit days and days with documented concepts but no recorded visit.

---

## S2. Charlson Comorbidity Index

CCI was calculated at the visit level using ICD mappings from Quan et al. (2005), restricted to EHR problem list conditions (`condition_type_concept_id = 32840`) in the same calendar year as the visit.

| Step | Detail |
|------|--------|
| Code map | ICD-9-CM and ICD-10-CM → 17 CCI categories |
| Weights | Standard Charlson weights `{1, 2, 3, 6}` |
| Duplicates | One row per category when a code maps to multiple groups (Quan et al.) |
| V-codes | ICD-9 V-codes dropped when source vocabulary is ICD-10-CM |
| Person-date | `max(cci)` across visits on that date |

At person-year level, we propagated `max_cci` forward within each patient using an expanding window.

Main text: [Methods → CCI](methods.md#charlson-comorbidity-index)

---

## S3. Feature engineering

Notebook: [`02_feature_engineering.py`](../scripts/cohort_1/02_feature_engineering.py)

### Observation window

Per `(person_id, year)`:

| Field | Definition |
|-------|------------|
| `year_first_date` | Earliest activity date in year |
| `year_last_date` | Latest activity date in year |
| `days_observed` | Distinct active dates |
| `is_full_year` | `days_observed ≥ 300` with activity Feb–Nov |

### Visit regularity

For person-years with ≥2 visits, let $g_i$ denote inter-visit gaps and $\bar{g} = \frac{1}{n}\sum_i g_i$ the mean gap. Irregularity scores:

$$
\text{L1} = \frac{1}{n}\sum_{i=1}^{n}\left|\frac{g_i}{\bar{g}} - 1\right|
$$

$$
\text{L2} = \sqrt{\frac{1}{n}\sum_{i=1}^{n}\left(\frac{g_i}{\bar{g}} - 1\right)^{2}}
$$

Both are **0** for perfectly evenly spaced visits and increase with greater irregularity.

| Rule | Behavior |
|------|----------|
| `< 3` visits | `irregularity_L1`, `irregularity_L2` → `NULL` |
| Flag | `has_valid_regularity = 1` when irregularity is computable |

Main text: [Methods → Feature construction](methods.md#feature-construction)

### Clinical breadth

Total concept occurrences per domain per person-year:

| Column | Domain |
|--------|--------|
| `condition_count` | Conditions |
| `drug_count` | Drugs |
| `procedure_count` | Procedures |
| `measurement_count` | Measurements |

### Prior-year features

Lagged within person: `max_cci`, `visit_count`, `inpatient_visit_count`.

---

## S4. GMM preprocessing and fitting

Notebook: [`03b_gmm.py`](../scripts/cohort_1/03b_gmm.py) · Main text: [Methods → GMM](methods.md#gaussian-mixture-model)

### Clustering feature vector

```text
[visit_count, inpatient_visit_count, hospitalized_days,
 has_valid_regularity, irregularity_L1, irregularity_L2]
```

### Training sample

One random calendar year per person (`seed = 42`).

### Preprocessing pipeline

| Step | Transform |
|------|-----------|
| 1 | Fill `NULL` irregularity with `0.0` |
| 2 | Winsorize count features at 99th percentile |
| 3 | $\tilde{x} = \log(1 + x)$ on winsorized counts |
| 4 | Z-score with `StandardScaler` fit on training sample |
| 5 | Add $\mathcal{N}(0, 10^{-10})$ jitter (seeded) to avoid degenerate covariance |

### Model

```text
sklearn.mixture.GaussianMixture
  n_components = 4
  covariance_type = "tied"
  n_init = 10
  max_iter = 200
  random_state = 42
```

Highest log-likelihood initialization retained.

### Uncertainty

Posterior probabilities $p_k$ per component $k \in \{1,\ldots,K\}$:

$$
H = -\sum_{k=1}^{K} p_k \log p_k \qquad \text{(Shannon entropy)}
$$

| Metric | Definition |
|--------|------------|
| Confidence | $\max_k p_k$ |
| Entropy | $H$ |

---

## S5. Alternative clustering methods

Notebook: [`03a_rules.py`](../scripts/cohort_1/03a_rules.py) · Main text: [Methods → Sensitivity](methods.md#sensitivity-analyses)

### Rule-based (vanilla)

Seven archetypes via priority-ordered thresholds on `inpatient_visit_count`, `visit_count`, and L1 irregularity (threshold `0.5`).

| Visits | Label |
|--------|-------|
| `≤ 1` | Sparse Use *(supplement only; not GMM-4)* |

### Rule-based (adaptive)

Same hierarchy; norm (L1 vs L2) and regularity threshold tuned by separation and stability plateau over `{0.0, 0.05, …, 1.0}`.

### GMM-7

Same preprocessing as GMM-4 with $K = 7$.

### Code switches

| Notebook | Parameter | Effect |
|----------|-----------|--------|
| [`03a_rules.py`](../scripts/cohort_1/03a_rules.py) | `RUN_ADAPTIVE=True` | Adaptive rules |
| [`03b_gmm.py`](../scripts/cohort_1/03b_gmm.py) | `K_VALUES=[4, 7]` | Train GMM-4 and GMM-7 |

Also: [Pipeline → Supplement sensitivity](pipeline.md#supplement-sensitivity)

---

## S6. Cross-method concordance

Pairwise agreement across GMM-4, GMM-7, rules vanilla, and rules adaptive:

| Metric | Use |
|--------|-----|
| Adjusted Rand Index | Cluster alignment |
| Adjusted Mutual Information | Information overlap |
| Cohen's $\kappa$ | Rule-based pairs |
| Confusion matrices | Label mapping |

---

## S7. Software

Python on Databricks (PySpark, Delta Lake):

```text
scikit-learn, NumPy, pandas, SciPy, statsmodels, matplotlib, seaborn
```

Environment details: [Pipeline → Environment](pipeline.md#environment)

---

## See also

- [Methods](methods.md) — manuscript-facing prose
- [Pipeline](pipeline.md) — run order and notebooks
- [Artifacts](artifacts.md) — outputs produced by `06_residual_analysis.py`
