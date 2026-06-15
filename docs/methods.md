# Methods

**Navigation:** [README](../README.md) · [Pipeline](pipeline.md) · [Supplement](methods_supplement.md) · [Artifacts](artifacts.md) · [Releases](releases.md)

---

## Contents

- [Study population](#study-population)
- [Unified person-date table](#unified-person-date-table)
- [Utilization-based clustering](#utilization-based-clustering)
- [Data volume scoring](#data-volume-scoring)

---

## Study population

We received full medical records for **25,000** randomly selected patients from UNC Health, provisioned in the OMOP common data model. Selected patients were ≥18 years of age as of the query date and had at least two encounters with UNC Health between **2018-01-01** and **2024-12-01**; no other selection criteria were applied.

---

## Unified person-date table

To consolidate clinical activity across all OMOP domains, we built a unified person-date table from seven source tables:

| Table | Role |
|-------|------|
| `visit_occurrence` | Visit spans |
| `condition_occurrence` | Diagnoses |
| `drug_exposure` | Medications |
| `measurement` | Labs/vitals |
| `procedure_occurrence` | Procedures |
| `observation` | Other clinical facts |
| `person` | Demographics |

Implementation: [Supplement → S1](methods_supplement.md#s1-unified-person-date-table) · Notebook: [`01_unified_tables.py`](../scripts/cohort_1/01_unified_tables.py)

For each domain, we extracted the relevant date column (e.g. `condition_start_date`, `drug_exposure_start_date`) and standardized it to a common `date` field, then aggregated concept identifiers into arrays per `(person_id, date)` to preserve all clinical information while maintaining a one-row-per-person-date grain.

We exploded multi-day visits into daily records so that each hospitalization day was counted, and flagged inpatient days using OMOP standard visit concept IDs:

| Concept | ID |
|---------|-----|
| Inpatient Visit | `9201` |
| Emergency Room and Inpatient Visit | `262` |

The resulting person-date spine included all dates with any clinical activity — both formal visit days and days with documented clinical concepts but no recorded visit (e.g. standalone laboratory orders or medication refills).

---

## Utilization-based clustering

### Feature construction

We aggregated the daily table to person-year level, yielding features that captured three core dimensions of utilization:

| Dimension | Features |
|-----------|----------|
| Volume | `visit_count` |
| Intensity | `inpatient_visit_count`, `hospitalized_days` |
| Regularity | `irregularity_L1`, `irregularity_L2`, `has_valid_regularity` |

We quantified visit regularity using a continuous irregularity score. For each person-year with at least two visits, we calculated inter-visit gaps $g_i$ and measured how much those gaps deviated from the patient's own average gap $\bar{g}$:

$$
\text{L1} = \frac{1}{n}\sum_{i=1}^{n}\left|\frac{g_i}{\bar{g}} - 1\right| \qquad
\text{L2} = \sqrt{\frac{1}{n}\sum_{i=1}^{n}\left(\frac{g_i}{\bar{g}} - 1\right)^{2}}
$$

Both are zero for perfectly evenly spaced visits and increase with greater irregularity. Person-years with **< 3** visits received `NULL` irregularity scores; `has_valid_regularity` flags whether regularity was computable.

Formulas and field definitions: [Supplement → S3](methods_supplement.md#s3-feature-engineering)

### Gaussian mixture model

We identified utilization archetypes using a **Gaussian Mixture Model (GMM)** with four components, fit to:

```text
visit_count, inpatient_visit_count, hospitalized_days,
has_valid_regularity, irregularity_L1, irregularity_L2
```

GMMs accommodate non-spherical cluster shapes and provide probabilistic assignments. We created a stratified training sample (one randomly selected year per person) to prevent patients with longer records from dominating model fitting, then applied a standard preprocessing pipeline — winsorizing extreme outliers at the 99th percentile, log-transforming skewed count features, and standardizing all features to zero mean and unit variance.

Full preprocessing: [Supplement → S4](methods_supplement.md#s4-gmm-preprocessing-and-fitting) · Notebook: [`03b_gmm.py`](../scripts/cohort_1/03b_gmm.py)

We fit the model with **tied covariance** (shared covariance matrix across components) and **10** random initializations, selecting the solution with the highest log-likelihood.

### Cluster interpretation

The four resulting clusters corresponded to interpretable utilization archetypes:

| Cluster | Pattern |
|---------|---------|
| High Inpatient | Multiple or extended hospitalizations |
| Moderate Inpatient | Occasional inpatient stays |
| Outpatient Regular | Regularly spaced ambulatory encounters |
| Outpatient Irregular | Minimal, irregularly spaced outpatient encounters |

Cluster ID mapping: [Pipeline → GMM-4 labels](pipeline.md#gmm-4-labels)

For each person-year, the model produced posterior probabilities $p_k$ across all four clusters. We used $\max_k p_k$ as a confidence score and Shannon entropy $H = -\sum_k p_k \log p_k$ as an uncertainty measure.

### Charlson Comorbidity Index

We calculated the **Charlson Comorbidity Index (CCI)** for each patient using ICD code mappings from Quan et al. (2005), restricted to EHR problem list conditions occurring in the same calendar year as the associated visit. We mapped ICD-9-CM and ICD-10-CM source codes to the 17 standard CCI comorbidity categories and applied the standard Charlson weighting scheme (weights $\in \{1, 2, 3, 6\}$).

Implementation: [Supplement → S2](methods_supplement.md#s2-charlson-comorbidity-index)

At the person-year level, we propagated `max_cci` forward across years within each patient using an expanding window, because chronic conditions represent permanent diagnoses. This prevented artificially low CCI scores in low-data-density years, which would otherwise create a spurious correlation between data volume and disease burden.

### Sensitivity analyses

We evaluated three alternative clustering approaches:

| Method | Output |
|--------|--------|
| Rule-based (vanilla) | 7 archetypes, fixed thresholds |
| Rule-based (adaptive) | 7 archetypes, tuned thresholds |
| GMM-7 | 7-component mixture |

Details: [Supplement → S5](methods_supplement.md#s5-alternative-clustering-methods)

---

## Data volume scoring

### Domain-level concept counts

To characterize clinical documentation available for each patient-year, we computed data volume from four non-visit OMOP domains: **conditions**, **drugs**, **procedures**, and **measurements**. These counts were independent of the utilization features used for clustering.

For each person-year and domain $d$, we calculated the total number of concept occurrences (non-unique count) $x_{i,d}$. We also computed the sum across all four domains. These metrics vary substantially across utilization clusters.

Output artifacts: [Artifacts → Manuscript scope](artifacts.md#manuscript-scope)

### Within-cluster domain residuals

To assess how a patient's documentation volume deviates from others with similar utilization patterns, we computed within-cluster residuals for each domain count. For person-year $i$ in GMM-4 cluster $k$:

$$
r_{i,d} = x_{i,d} - \bar{x}_{k,d}
$$

Positive $r_{i,d}$ indicates more concept occurrences in domain $d$ than the cluster average; negative values indicate fewer. We also computed unsigned residuals $|r_{i,d}|$. Residuals are in raw count space; because domain counts were not used in clustering, they reflect documentation variation independent of cluster assignment.

Notebook: [`06_residual_analysis.py`](../scripts/cohort_1/06_residual_analysis.py)

### Association between disease burden and data volume

We tested whether disease burden (cumulative maximum CCI) is associated with data volume variation within and across utilization archetypes using four complementary analyses.

| Analysis | Question | Method |
|----------|----------|--------|
| **Q1** | Pooled CCI ↔ residual | Spearman $\rho$ (signed and unsigned) |
| **Q1b** | Stratified CCI ↔ residual | Spearman $\rho$ within each cluster |
| **Q2** | Cluster-specific CCI slopes | OLS with patient-clustered SEs |
| **Q3** | CCI → cluster membership | Multinomial logit |

**Q1b** Bonferroni correction across four domains:

$$
\alpha' = \frac{0.05}{4} = 0.0125
$$

**Q2** interaction model for domain $d$:

$$
r_{i,d} = \beta_{0} + \beta_{1}\,\text{CCI}_i + \sum_{k \neq \text{ref}} \gamma_k \,\mathbb{1}[\text{cluster}_i = k] + \sum_{k \neq \text{ref}} \delta_k \,\text{CCI}_i \cdot \mathbb{1}[\text{cluster}_i = k] + \varepsilon_i
$$

For unsigned outcomes with skewness $> 1.0$, we applied $\log(1 + |r|)$. Standard errors were clustered by `person_id`.

**Q3** multinomial logistic regression with **Outpatient Regular** as reference:

$$
\log \frac{P(Y_i = k)}{P(Y_i = \text{ref})} = \alpha_k + \beta_k \,\text{CCI}_i
$$

Because patients contribute multiple person-years, we used clustered standard errors in OLS models and emphasize effect sizes (Spearman $\rho$, odds ratios) over $p$-values.

Backing stats: `edi_data01`–`edi_data08` ([Artifacts](artifacts.md#manuscript-scope))

### Cluster-level data density characterization

We computed median and interquartile range of domain counts per cluster, tested differences with Kruskal–Wallis tests, and estimated rank-biserial effect sizes comparing each cluster to the full cohort.

---

## See also

- [Supplement](methods_supplement.md) — S1–S7 implementation detail
- [Pipeline](pipeline.md) — notebook run order and environment
- [Artifacts](artifacts.md) — figure and table outputs
