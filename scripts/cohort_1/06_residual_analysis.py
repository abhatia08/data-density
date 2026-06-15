# Databricks notebook source
# MAGIC %md
# MAGIC # 06 - CCI vs Within-Cluster Domain Residuals
# MAGIC
# MAGIC Depends on: `02_feature_engineering`, `03b_gmm`

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Setup and Configuration
# MAGIC
# MAGIC ### 1.1 Import Libraries

# COMMAND ----------

import os
import warnings
from typing import Dict, List

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from pyspark.sql import functions as F, Window
from pyspark.sql.types import StructType, StringType, IntegerType, DoubleType
from scipy.stats import gaussian_kde, skew, spearmanr
import statsmodels.formula.api as smf
from statsmodels.discrete.discrete_model import MNLogit

warnings.filterwarnings("ignore")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.2 Load Shared Utilities

# COMMAND ----------

# MAGIC %run ./99_utils

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.3 Configure Logging and Spark

# COMMAND ----------

VERBOSE = get_verbose(default=True)
gate_prints(VERBOSE)
configure_spark_optimizations()

input_db = "cohort_1"
output_db = "cohort_1"

FIGURE_DIR = EDI_FIGURES_DIR
TABLE_DIR = EDI_TABLES_DIR
DATA_DIR = EDI_DATA_DIR
for _d in (FIGURE_DIR, TABLE_DIR, DATA_DIR):
    os.makedirs(_d, exist_ok=True)
print(f"Manuscript outputs: figures={FIGURE_DIR}, tables={TABLE_DIR}, data={DATA_DIR}")

GMM4_LABELS = get_gmm4_labels()

BASELINE_CLUSTER = 2
BASELINE_LABEL = GMM4_LABELS[BASELINE_CLUSTER]

_RAIN_RNG = np.random.default_rng(0)

_RC_TEXT  = "#1c1c1c"
_RC_MUTED = "#5c5c5c"
_RC_GRID  = "#d4d6d9"
_RC_BG    = "#fafafa"

plt.rcParams.update({
    "figure.facecolor":   "white",
    "axes.facecolor":     _RC_BG,
    "axes.edgecolor":     _RC_GRID,
    "axes.linewidth":     0.8,
    "axes.spines.right":  False,
    "axes.spines.top":    False,
    "axes.grid":          True,
    "axes.grid.axis":     "y",
    "grid.color":         _RC_GRID,
    "grid.linewidth":     0.5,
    "text.color":         _RC_TEXT,
    "axes.labelcolor":    _RC_TEXT,
    "xtick.color":        _RC_MUTED,
    "ytick.color":        _RC_MUTED,
    "xtick.labelsize":    8,
    "ytick.labelsize":    8,
    "axes.labelsize":     9,
    "axes.titlesize":     10,
    "axes.titlepad":      8,
    "font.family":        "sans-serif",
    "font.sans-serif":    ["Source Sans 3", "Helvetica Neue", "Arial", "DejaVu Sans"],
    "legend.frameon":     True,
    "legend.framealpha":  0.9,
    "legend.edgecolor":   _RC_GRID,
    "legend.fontsize":    8,
    "figure.dpi":         150,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.facecolor":  "white",
})

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.4 Load and Join Data

# COMMAND ----------

features = spark.table(f"{input_db}.archetype_features_yearly")
gmm4 = spark.table(f"{input_db}.gmm_4_yearly").select(
    "person_id", "year", "gmm_4_cluster"
)

characterization_data = features.join(gmm4, on=["person_id", "year"], how="inner")
characterization_data.cache()

cci_window = Window.partitionBy("person_id").orderBy("year")
cci_check = characterization_data.withColumn(
    "max_cci_lag", F.lag("max_cci").over(cci_window)
).withColumn(
    "cci_ok", F.when(F.col("max_cci_lag").isNull(), True).otherwise(F.col("max_cci") >= F.col("max_cci_lag"))
)
n_violations = cci_check.filter(~F.col("cci_ok")).count()
if n_violations > 0:
    raise ValueError(f"max_cci must be cumulative; found {n_violations} person-years with decrease")

pdf = characterization_data.select(
    "person_id",
    "year",
    "gmm_4_cluster",
    "max_cci",
    "total_condition_count",
    "total_drug_count",
    "total_procedure_count",
    "total_measurement_count",
).toPandas()

pdf = pdf.dropna(subset=["gmm_4_cluster"]).copy()
pdf["gmm_4_cluster"] = pdf["gmm_4_cluster"].astype(int)

# Validate GMM4_LABELS cover all clusters in data
actual_clusters = set(pdf["gmm_4_cluster"].unique())
label_clusters = set(GMM4_LABELS.keys())
if not actual_clusters <= label_clusters:
    raise ValueError(f"Clusters {actual_clusters - label_clusters} missing from GMM4_LABELS")
if BASELINE_CLUSTER not in actual_clusters:
    raise ValueError(f"BASELINE_CLUSTER {BASELINE_CLUSTER} not in data")

print(f"Loaded {len(pdf):,} person-years across {pdf['gmm_4_cluster'].nunique()} clusters")
print(f"Unique patients: {pdf['person_id'].nunique():,}")
print(pdf["gmm_4_cluster"].value_counts().sort_index().rename(GMM4_LABELS))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Compute Within-Cluster Residuals

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.1 Compute Domain Residuals
# MAGIC
# MAGIC Within-cluster mean-centered residuals for OMOP domain counts
# MAGIC (total conditions, drugs, procedures, measurements — non-distinct occurrences).
# MAGIC These features are **not** part of the clustering solution, so their residuals
# MAGIC capture how a patient's clinical documentation deviates from others in
# MAGIC the same utilization archetype -- independent of what drove cluster assignment.
# MAGIC
# MAGIC Computed in raw count space (no scaling), then mean-centered per cluster.
# MAGIC Positive = patient has more total concept occurrences in that domain than the cluster average.

# COMMAND ----------

pdf, domain_resid_cols = compute_cluster_mean_centered_residuals(
    pdf, "gmm_4_cluster", get_domain_breadth_features(), prefix="domain_resid"
)
pdf["cluster_label"] = pdf["gmm_4_cluster"].map(GMM4_LABELS)

print("\nDomain residuals (mean by cluster -- should be ~0 by construction):")
domain_means = pdf.groupby("cluster_label")[domain_resid_cols].mean().round(4)
display(spark.createDataFrame(domain_means.reset_index()))

print("\nDomain residuals (std by cluster -- captures within-cluster spread):")
domain_stds = pdf.groupby("cluster_label")[domain_resid_cols].std().round(4)
display(spark.createDataFrame(domain_stds.reset_index()))

unsigned_domain_resid_cols = []
for col in domain_resid_cols:
    unsigned_col = col.replace("domain_resid_", "domain_abs_resid_")
    pdf[unsigned_col] = pdf[col].abs()
    unsigned_domain_resid_cols.append(unsigned_col)

print("\nUnsigned domain residuals (mean by cluster):")
unsigned_means = pdf.groupby("cluster_label")[unsigned_domain_resid_cols].mean().round(4)
display(spark.createDataFrame(unsigned_means.reset_index()))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.2 Save Residuals to Delta

# COMMAND ----------

output_cols = ["person_id", "year", "gmm_4_cluster", "cluster_label", "max_cci"]
output_cols += domain_resid_cols
output_cols += unsigned_domain_resid_cols

# Explicit schema for residuals table
residuals_schema = (
    StructType()
    .add("person_id", StringType(), False)
    .add("year", IntegerType(), False)
    .add("gmm_4_cluster", IntegerType(), False)
    .add("cluster_label", StringType(), True)
    .add("max_cci", IntegerType(), True)
)
for col_name in domain_resid_cols + unsigned_domain_resid_cols:
    residuals_schema = residuals_schema.add(col_name, DoubleType(), True)

residuals_spark = spark.createDataFrame(pdf[output_cols], schema=residuals_schema)

output_table = f"{output_db}.gmm4_residuals_yearly"
residuals_spark.write.format("delta").mode("overwrite").option("overwriteSchema", "true") \
    .saveAsTable(output_table)

try:
    spark.sql(f"ALTER TABLE {output_table} SET TBLPROPERTIES ('comment' = 'Per-patient-year domain residuals from GMM-4 assignment')")
except Exception as comment_err:
    warn(f"Skipping table comment update: {comment_err}")

print(f"Saved {output_table} ({len(pdf):,} rows)")
display(residuals_spark.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Figure Helpers (Shared Across Q1-Q3 Visuals)

# COMMAND ----------

analysis_df = pdf.dropna(subset=["max_cci"] + domain_resid_cols).copy()
analysis_df["max_cci"] = analysis_df["max_cci"].astype(int)

# Outcome columns: signed and unsigned domain residuals
outcome_cols = domain_resid_cols + unsigned_domain_resid_cols

OUTCOME_LABELS: Dict[str, str] = {}
for _domain_col, _domain_label in get_domain_label_map().items():
    _short = _domain_col.replace("total_", "").replace("_count", "")
    _base = _domain_label.lower()
    OUTCOME_LABELS[f"domain_resid_{_short}"] = f"within-cluster {_base} total-count residual (signed)"
    OUTCOME_LABELS[f"domain_abs_resid_{_short}"] = f"within-cluster |{_base} total-count| deviation (unsigned)"


def outcome_label(outcome_name: str) -> str:
    return OUTCOME_LABELS.get(outcome_name, outcome_name)




def _raincloud_panel(
    ax: plt.Axes,
    data_by_group: List[pd.DataFrame],
    positions: List[int],
    colors: List[str],
    col: str,
    rain_max: int = 400,
    clip_percentiles: tuple = (0.5, 99.5),
) -> None:
    """Raincloud per group: half-violin (KDE) on the left, narrow box on the
    right, and jittered 'rain' dots further right. Matches the echo-oif
    pub-toolkit raincloud template, adapted to use per-cluster Wong colors.
    Rain is subsampled to `rain_max` points to keep large-N panels readable."""
    for pos, grp, color in zip(positions, data_by_group, colors):
        vals = grp[col].dropna().to_numpy()
        if len(vals) == 0:
            continue

        # Half-violin (cloud) on the LEFT of the tick
        try:
            kde = gaussian_kde(vals)
            lo, hi = np.percentile(vals, list(clip_percentiles))
            if hi > lo:
                y_range = np.linspace(lo, hi, 200)
                density = kde(y_range)
                peak = density.max()
                if peak > 0:
                    density = density / peak * 0.34
                    ax.fill_betweenx(
                        y_range, pos - density, pos,
                        facecolor=color, alpha=0.55,
                        edgecolor=_RC_MUTED, linewidth=0.5, zorder=2,
                    )
        except (np.linalg.LinAlgError, ValueError):
            pass  # degenerate distribution (all identical values etc.)

        # Narrow box on the RIGHT of the tick -- median + IQR + Tukey whiskers
        bp = ax.boxplot(
            [vals], positions=[pos + 0.12], widths=0.09,
            patch_artist=True, showfliers=False, manage_ticks=False,
        )
        for patch in bp["boxes"]:
            patch.set_facecolor("white")
            patch.set_edgecolor(color)
            patch.set_linewidth(1.1)
        for median in bp["medians"]:
            median.set_color(_RC_TEXT)
            median.set_linewidth(1.6)
        for element in ("whiskers", "caps"):
            for line in bp[element]:
                line.set_color(color)
                line.set_linewidth(0.9)

        # Jittered rain to the RIGHT of the box
        n_rain = min(len(vals), rain_max)
        if len(vals) > n_rain:
            rain_vals = _RAIN_RNG.choice(vals, size=n_rain, replace=False)
        else:
            rain_vals = vals
        jitter = _RAIN_RNG.uniform(0.22, 0.36, size=n_rain)
        ax.scatter(
            pos + jitter, rain_vals,
            s=5, alpha=0.32, color=color,
            edgecolors="none", zorder=1,
        )


def _save_fig(fig: plt.Figure, name: str) -> None:
    path = os.path.join(FIGURE_DIR, name)
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"  Saved -> {path}")
    display(fig)
    plt.close(fig)


def _save_table(df: pd.DataFrame, name: str, *, data: bool = False) -> None:
    base = DATA_DIR if data else TABLE_DIR
    path = os.path.join(base, name)
    df.to_csv(path, index=False)
    print(f"  Saved -> {path}")


def _plot_violin_grid(
    col_list: List[str],
    label_list: List[str],
    cluster_dfs: List[pd.DataFrame],
    y_label: str,
    suptitle: str,
    fname: str,
    y_log: bool = False,
    y_log_linthresh: float = 1.0,
) -> None:
    """3x2 raincloud grid with shared legend. Reused for abs-resid and
    domain-count panels. `y_log=True` applies a symlog y-scale (linear below
    `y_log_linthresh`, log above) -- safe for non-negative data containing
    exact zeros, which ordinary log scales cannot show."""
    n_panels = len(col_list)
    nrows, ncols = 3, 2
    fig, axes_2d = plt.subplots(nrows, ncols, figsize=(10, 12), sharey=False)
    axes_flat = axes_2d.ravel()
    for idx, (ax, col, dlabel) in enumerate(zip(axes_flat, col_list, label_list)):
        _raincloud_panel(ax, cluster_dfs, _xpos, _cluster_colors_ordered, col)
        ax.set_xticks(_xpos)
        ax.set_xticklabels(_cluster_xtick_labels, fontsize=7)
        for tick_lbl, cname in zip(ax.get_xticklabels(), CLUSTER_ORDER):
            if cname == BASELINE_LABEL:
                tick_lbl.set_fontweight("bold")
        ax.set_ylabel(y_label, fontsize=8)
        ax.set_title(dlabel, fontsize=9, fontweight="semibold")
        ax.set_xlabel("")
        ax.set_xlim(min(_xpos) - 0.55, max(_xpos) + 0.55)
        if y_log:
            ax.set_yscale("symlog", linthresh=y_log_linthresh)
            ax.axhline(y_log_linthresh, color=_RC_GRID, lw=0.5,
                       linestyle=":", alpha=0.8, zorder=0)
    for ax in axes_flat[n_panels:]:
        ax.set_visible(False)
    _handles = [
        mpatches.Patch(facecolor=CLUSTER_COLORS[lbl], label=lbl, alpha=0.8)
        for lbl in CLUSTER_ORDER
    ]
    fig.legend(
        handles=_handles, loc="lower right", bbox_to_anchor=(0.95, 0.05),
        ncol=2, fontsize=8, framealpha=0.9,
    )
    fig.suptitle(suptitle, fontsize=11, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    _save_fig(fig, fname)


def _run_q1_pooled(
    df: pd.DataFrame, outcomes: List[str]
) -> Dict[str, dict]:
    """Q1 -- Pooled Spearman rho for each outcome vs max_cci. Returns {outcome: stats_dict}."""
    results = {}
    print("Q1 -- Pooled Spearman's Rho (residual vs CCI)")
    for oc in outcomes:
        subset = df.dropna(subset=[oc]).copy()
        rho, pval = spearmanr(subset[oc], subset["max_cci"])
        n_obs = len(subset)
        n_patients = subset["person_id"].nunique()
        results[oc] = {"rho": rho, "p": pval, "n": n_obs, "n_patients": n_patients}
        print(f"\n  {outcome_label(oc)}: rho={rho:.4f}, n={n_obs:,} ({n_patients:,} patients), "
              f"{rho_interp(rho)}")
    return results


def _run_q1b_stratified(
    df: pd.DataFrame, outcomes: List[str], bonferroni_alpha: float
) -> Dict[str, pd.DataFrame]:
    """Q1b -- Stratified Spearman rho within each cluster. Returns {outcome: DataFrame}."""
    results = {}
    print(f"Q1b -- Stratified Spearman's Rho (Bonferroni alpha = {bonferroni_alpha})")
    for oc in outcomes:
        rows = []
        for cluster_id in sorted(df["gmm_4_cluster"].unique()):
            subset = df[df["gmm_4_cluster"] == cluster_id].dropna(subset=[oc, "max_cci"])
            rho, p = spearmanr(subset[oc], subset["max_cci"])
            rows.append({
                "outcome": oc,
                "cluster_id": cluster_id,
                "cluster_label": GMM4_LABELS.get(cluster_id, str(cluster_id)),
                "is_reference": cluster_id == BASELINE_CLUSTER,
                "n_person_years": len(subset),
                "n_patients": subset["person_id"].nunique(),
                "spearman_rho": rho,
                "p_value": p,
                "significant_bonferroni": p < bonferroni_alpha,
            })
        strat_df = pd.DataFrame(rows)
        results[oc] = strat_df

        print(f"\n  Outcome: {outcome_label(oc)}")
        strat_display = strat_df.drop(columns=["outcome"]).copy()
        strat_display["p_value"] = strat_display["p_value"].apply(
            lambda x: f"{x:.4e}" if x < 0.001 else f"{x:.4f}"
        )
        display(spark.createDataFrame(strat_display))
    return results


def _build_rb_pivot(rb_rows: List[dict]) -> pd.DataFrame:
    """Pivot rank-biserial rows into Domain x Cluster matrix, ordered by CLUSTER_ORDER."""
    _rb_df = pd.DataFrame(rb_rows)
    return (
        _rb_df.pivot(index="Domain", columns="Cluster", values="Rank_Biserial")
        .reindex(columns=[c for c in CLUSTER_ORDER if c in _rb_df["Cluster"].unique()])
    )


# Build ordered list of per-cluster DataFrames used in violin helpers
_cluster_dfs_plot = [
    analysis_df[analysis_df["cluster_label"] == lbl]
    for lbl in CLUSTER_ORDER
]
_cluster_colors_ordered = [CLUSTER_COLORS[lbl] for lbl in CLUSTER_ORDER]


def _fmt_cluster_tick(lbl: str) -> str:
    """Wrap cluster name across lines; mark the baseline as '(reference)'."""
    base = lbl.replace(" ", "\n")
    if lbl == BASELINE_LABEL:
        return f"{base}\n(reference)"
    return base


_cluster_xtick_labels = [_fmt_cluster_tick(lbl) for lbl in CLUSTER_ORDER]
_xpos = list(range(len(CLUSTER_ORDER)))

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## 4. Q1 -- Is Higher CCI Associated With Residual Variation?
# MAGIC
# MAGIC **Q1 (Pooled Spearman)**: Correlation between disease burden and each residual outcome,
# MAGIC pooled across all clusters. All outcomes are cluster-adjusted by construction.
# MAGIC
# MAGIC **Q1b (Stratified Spearman)**: Same correlation within each cluster -- Simpson's paradox check.
# MAGIC If pooled rho is near zero but stratified rhos are non-zero with opposite signs,
# MAGIC the relationship depends on patient archetype.
# MAGIC
# MAGIC **Caveat**: Multiple person-years per patient inflate significance. Focus on rho, not p-value.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4.1 Q1 -- Pooled Spearman's Rho

# COMMAND ----------

q1_results = _run_q1_pooled(analysis_df, outcome_cols)

_save_table(
    pd.DataFrame([
        {"outcome": k, "n_person_years": v["n"], "n_patients": v["n_patients"],
         "spearman_rho": round(v["rho"], 4), "p_value": v["p"]}
        for k, v in q1_results.items()
    ]),
    EDI_DATA01_Q1,
    data=True,
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4.2 Q1b -- Stratified Spearman's Rho (Exploratory)
# MAGIC
# MAGIC Bonferroni correction applied: alpha = 0.05 / 4 = 0.0125

# COMMAND ----------

BONFERRONI_ALPHA = 0.05 / len(GMM4_LABELS)

q1b_results = _run_q1b_stratified(analysis_df, outcome_cols, BONFERRONI_ALPHA)

_save_table(
    pd.concat(q1b_results.values(), ignore_index=True),
    EDI_DATA02_Q1B,
    data=True,
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4.3 Q1 Visuals -- Within-Cluster Absolute Deviation in Domain Counts by Archetype
# MAGIC
# MAGIC Wider violins = greater within-cluster heterogeneity for that domain.

# COMMAND ----------

# unsigned_domain_resid_cols are produced from get_domain_breadth_features() in
# compute_cluster_mean_centered_residuals and preserve the same order.
_abs_domain_label_list = [get_domain_label_map()[c] for c in get_domain_breadth_features()]

_plot_violin_grid(
    col_list=unsigned_domain_resid_cols,
    label_list=_abs_domain_label_list,
    cluster_dfs=_cluster_dfs_plot,
    y_label="|deviation| (symlog scale, linear below 1)",
    suptitle=(
        "Within-Cluster Absolute Deviation in Domain Counts by Utilization Archetype\n"
        "(raincloud: KDE + box + jittered points; y-axis symlog to expose tails)"
    ),
    fname=EDI_FIG02_ABS_RESID,
    y_log=True,
)

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## 5. Q2 -- Does the CCI-Residual Slope Differ Across Clusters?
# MAGIC
# MAGIC **OLS with interaction**: `residual ~ max_cci * C(gmm_4_cluster)`
# MAGIC
# MAGIC For unsigned (absolute deviation) outcomes, a `log1p` transform may be used when skew is high.
# MAGIC Signed domain residuals stay on their original centered scale so the coefficient remains
# MAGIC interpretable as "more vs less than the cluster average."
# MAGIC
# MAGIC Uses **clustered standard errors** by `person_id` to account for non-independence
# MAGIC of repeated person-years from the same patient.
# MAGIC
# MAGIC This section also includes the **domain count characterization** (median/IQR, Kruskal-Wallis,
# MAGIC rank-biserial) because those analyses directly inform the Q2 interpretation:
# MAGIC do clusters differ in clinical data density, and does the CCI slope interact with that density?

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.1 OLS Interaction Models

# COMMAND ----------

# Store Q2 results per outcome: {outcome_col: {"model": ..., "interaction_terms": [...], "outcome_label": ...}}
q2_results = {}

if BASELINE_CLUSTER not in set(analysis_df["gmm_4_cluster"].dropna().astype(int).unique()):
    raise ValueError(
        f"BASELINE_CLUSTER {BASELINE_CLUSTER} not present in analysis data. "
        f"Available clusters: {sorted(analysis_df['gmm_4_cluster'].dropna().astype(int).unique().tolist())}"
    )

print("Q2 -- OLS: residual ~ max_cci * C(gmm_4_cluster)")
print(f"  Reference = Cluster {BASELINE_CLUSTER}: {GMM4_LABELS[BASELINE_CLUSTER]}")
print(f"  Clustered SE by person_id ({analysis_df['person_id'].nunique():,} clusters)")

SKEW_THRESHOLD = 1.0  # local constant for OLS transform decision
for outcome_col in outcome_cols:
    ols_vals = analysis_df[outcome_col].dropna()
    ols_skewness = skew(ols_vals)
    ols_col = f"_ols_{outcome_col}"
    is_signed = ols_vals.min() < 0
    ols_df = analysis_df[["person_id", "max_cci", "gmm_4_cluster", outcome_col]].copy()
    print(f"\n  --- Outcome: {outcome_label(outcome_col)} (skewness={ols_skewness:.3f}) ---")

    if abs(ols_skewness) > SKEW_THRESHOLD and not is_signed:
        ols_df[ols_col] = np.log1p(ols_df[outcome_col])
        ols_outcome_label = f"log1p({outcome_col})"
        print(f"  Skewness > {SKEW_THRESHOLD} -- applying log1p transform")
    else:
        ols_df[ols_col] = ols_df[outcome_col]
        ols_outcome_label = outcome_col
        if is_signed:
            print("  Outcome is signed and mean-centered; keeping original scale for interpretation")

    ols_model = smf.ols(
        f"{ols_col} ~ max_cci * C(gmm_4_cluster, Treatment(reference={BASELINE_CLUSTER}))",
        data=ols_df,
    ).fit(
        cov_type="cluster",
        cov_kwds={"groups": ols_df["person_id"]},
    )
    print(f"  R^2={ols_model.rsquared:.4f}, nobs={int(ols_model.nobs):,}")

    interaction_terms = [t for t in ols_model.pvalues.index if "max_cci:C(" in t]
    if interaction_terms:
        print("  Interaction term p-values (does CCI slope differ from baseline cluster?):")
        for term in interaction_terms:
            coef = ols_model.params[term]
            pval = ols_model.pvalues[term]
            print(f"    {term}: coef={coef:.4f}, p={pval:.4e} {sig_label(pval)}")

    q2_results[outcome_col] = {
        "model": ols_model,
        "interaction_terms": interaction_terms,
        "outcome_label": ols_outcome_label,
    }

_q2_interaction_rows = []
for oc, q2_res in q2_results.items():
    for term in q2_res.get("interaction_terms", []):
        _q2_interaction_rows.append({
            "outcome": oc,
            "outcome_label": q2_res["outcome_label"],
            "term": term,
            "coef": round(q2_res["model"].params[term], 4),
            "p_value": q2_res["model"].pvalues[term],
        })
if _q2_interaction_rows:
    _save_table(pd.DataFrame(_q2_interaction_rows), EDI_DATA03_Q2, data=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.2 Domain Count Characterization by Cluster
# MAGIC
# MAGIC If a researcher restricted their study to a single utilization archetype, how would their
# MAGIC available clinical data differ from the full cohort? For each of the four OMOP domains
# MAGIC (conditions, drugs, procedures, measurements), we compare the within-year
# MAGIC total concept occurrence counts across clusters using:
# MAGIC - **Median / IQR** per cluster to show the distribution
# MAGIC - **Kruskal-Wallis test** to confirm counts differ across clusters
# MAGIC - **Rank-biserial correlation** (effect size) between each cluster and the overall cohort

# COMMAND ----------

from scipy.stats import kruskal

DOMAIN_COLS = get_domain_breadth_features()
DOMAIN_LABELS = get_domain_label_map()

domain_df = analysis_df.copy()
for col in DOMAIN_COLS:
    domain_df[col] = domain_df[col].fillna(0)

# COMMAND ----------

# MAGIC %md
# MAGIC #### 5.2a Per-Cluster Median / IQR Summary
# MAGIC
# MAGIC Compare medians within each domain row; large spreads confirm archetypes inhabit different data-density environments.

# COMMAND ----------

summary_stats = []
for col in DOMAIN_COLS:
    for cluster_id, label in sorted(GMM4_LABELS.items()):
        subset = domain_df[domain_df["gmm_4_cluster"] == cluster_id][col]
        q25, q50, q75 = subset.quantile([0.25, 0.50, 0.75]).values
        summary_stats.append(
            {
                "Domain": DOMAIN_LABELS[col],
                "Cluster": label,
                "N": len(subset),
                "Median": round(q50, 1),
                "Q1": round(q25, 1),
                "Q3": round(q75, 1),
                "IQR": round(q75 - q25, 1),
            }
        )
    # Full-cohort row
    all_vals = domain_df[col]
    q25, q50, q75 = all_vals.quantile([0.25, 0.50, 0.75]).values
    summary_stats.append(
        {
            "Domain": DOMAIN_LABELS[col],
            "Cluster": "Full Cohort",
            "N": len(all_vals),
            "Median": round(q50, 1),
            "Q1": round(q25, 1),
            "Q3": round(q75, 1),
            "IQR": round(q75 - q25, 1),
        }
    )

domain_summary_pdf = pd.DataFrame(summary_stats)
print("Domain Count Summary (Median [IQR]) by Cluster")
display(spark.createDataFrame(domain_summary_pdf))
_save_table(domain_summary_pdf, EDI_DATA05_DOMAIN_SUMMARY, data=True)

# COMMAND ----------

# MAGIC %md
# MAGIC #### 5.2b Kruskal-Wallis Tests and Rank-Biserial Effect Sizes
# MAGIC
# MAGIC KW confirms distributional differences; rank-biserial (-1 to +1) shows direction and magnitude per cluster vs. cohort.

# COMMAND ----------

kw_rows = []
for col in DOMAIN_COLS:
    groups = [
        domain_df[domain_df["gmm_4_cluster"] == c][col].values
        for c in sorted(GMM4_LABELS.keys())
    ]
    non_empty = [g for g in groups if len(g) > 0]
    if len(non_empty) < 2:
        warn(f"Kruskal-Wallis skipped for {col}: fewer than 2 non-empty groups")
        kw_rows.append({
            "Domain": DOMAIN_LABELS[col],
            "KW_Statistic": float("nan"),
            "P_Value": "---",
            "Sig": "n/a",
        })
        continue
    stat, p = kruskal(*non_empty)
    kw_rows.append(
        {
            "Domain": DOMAIN_LABELS[col],
            "KW_Statistic": round(stat, 2),
            "P_Value": f"{p:.4e}" if p < 0.001 else f"{p:.4f}",
            "Sig": sig_label(p),
        }
    )

_kw_pdf = pd.DataFrame(kw_rows)
_save_table(_kw_pdf, EDI_DATA06_DOMAIN_KW, data=True)

# Per-cluster rank-biserial vs full cohort
rb_rows = []
for col in DOMAIN_COLS:
    ref = domain_df[col].values
    for cluster_id, label in sorted(GMM4_LABELS.items()):
        grp = domain_df[domain_df["gmm_4_cluster"] == cluster_id][col].values
        rb = rank_biserial(grp, ref)
        rb_rows.append(
            {
                "Domain": DOMAIN_LABELS[col],
                "Cluster": label,
                "Rank_Biserial": round(rb, 3),
                "Direction": "higher than cohort" if rb > 0.1 else ("lower than cohort" if rb < -0.1 else "similar to cohort"),
            }
        )

_rb_pdf = pd.DataFrame(rb_rows)
print("\nRank-Biserial Effect Size: each cluster vs full cohort")
display(spark.createDataFrame(_rb_pdf))
_save_table(_rb_pdf, EDI_DATA07_DOMAIN_RB, data=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.3 Q2 Visuals -- Domain Count Violins
# MAGIC
# MAGIC Raw total concept occurrence counts per year by archetype -- shows the absolute data-density landscape each cluster inhabits.

# COMMAND ----------

_cluster_dfs_domain = [
    domain_df[domain_df["cluster_label"] == lbl]
    for lbl in CLUSTER_ORDER
]
_domain_label_list = [DOMAIN_LABELS[c] for c in DOMAIN_COLS]

_plot_violin_grid(
    col_list=DOMAIN_COLS,
    label_list=_domain_label_list,
    cluster_dfs=_cluster_dfs_domain,
    y_label="Total concept occurrences / year (symlog scale, linear below 1)",
    suptitle=(
        "OMOP Domain Counts (Total Concepts/Year) by Utilization Archetype\n"
        "(raincloud: KDE + box + jittered points; y-axis symlog to expose tails)"
    ),
    fname=EDI_FIG04_DOMAIN_BREADTH,
    y_log=True,
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.4 Q2 Visual -- Rank-Biserial Heatmap
# MAGIC
# MAGIC Red = cluster ranks above cohort; blue = below. Cells near 0 = representative; near +/-1 = strong selection effect.

# COMMAND ----------

_rb_pivot = _build_rb_pivot(rb_rows)

fig, ax = plt.subplots(figsize=(8, 4.5))

_n_rows, _n_cols = _rb_pivot.shape
_cell_w, _cell_h = 1.0, 1.0

for i in range(_n_rows):
    for j in range(_n_cols):
        val = _rb_pivot.values[i, j]
        # Diverging color: positive -> warm coral/red, negative -> steel blue/navy
        if val >= 0:
            intensity = min(abs(val), 1.0)
            r = 0.95 - 0.25 * intensity
            g = 0.55 - 0.35 * intensity
            b = 0.40 - 0.25 * intensity
        else:
            intensity = min(abs(val), 1.0)
            r = 0.30 - 0.15 * intensity
            g = 0.50 - 0.15 * intensity
            b = 0.72 + 0.10 * intensity
        cell = plt.Rectangle(
            (j * _cell_w, (_n_rows - 1 - i) * _cell_h),
            _cell_w, _cell_h,
            facecolor=(r, g, b), edgecolor="white", linewidth=2.5,
        )
        ax.add_patch(cell)
        txt_color = "white" if abs(val) > 0.50 else _RC_TEXT
        ax.text(
            j * _cell_w + _cell_w / 2,
            (_n_rows - 1 - i) * _cell_h + _cell_h / 2,
            f"{val:.2f}", ha="center", va="center",
            fontsize=11, fontweight="bold", color=txt_color,
        )

ax.set_xlim(0, _n_cols * _cell_w)
ax.set_ylim(0, _n_rows * _cell_h)
ax.set_xticks([j * _cell_w + _cell_w / 2 for j in range(_n_cols)])
ax.set_xticklabels(
    [_fmt_cluster_tick(c) for c in _rb_pivot.columns],
    fontsize=9, fontweight="semibold",
)
# Bold the baseline (reference) column header
for _tlbl, _col_name in zip(ax.get_xticklabels(), _rb_pivot.columns):
    if _col_name == BASELINE_LABEL:
        _tlbl.set_fontweight("bold")
ax.set_yticks([(_n_rows - 1 - i) * _cell_h + _cell_h / 2 for i in range(_n_rows)])
ax.set_yticklabels(_rb_pivot.index, fontsize=9.5)
ax.xaxis.set_tick_params(length=0)
ax.yaxis.set_tick_params(length=0)
ax.set_aspect("equal")
for spine in ax.spines.values():
    spine.set_visible(False)
ax.set_facecolor("white")
ax.grid(False)

# Colorbar via ScalarMappable
_sm = cm.ScalarMappable(cmap="RdBu_r", norm=mcolors.Normalize(vmin=-1, vmax=1))
_sm.set_array([])
_cbar = fig.colorbar(_sm, ax=ax, shrink=0.75, pad=0.04, aspect=25)
_cbar.set_label("Rank-biserial", fontsize=9)
_cbar.ax.tick_params(labelsize=8)
_cbar.outline.set_visible(False)

ax.set_title(
    "Rank-Biserial Effect Size: Each Cluster vs. Full Cohort\n"
    "(-1 = cluster lower than cohort; +1 = cluster higher)",
    fontsize=10.5, fontweight="bold", pad=14,
)

fig.tight_layout()
_save_fig(fig, EDI_FIG03_RANK_BISERIAL)

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## 6. Q3 -- Does Disease Burden Predict Cluster Assignment?
# MAGIC
# MAGIC **Multinomial Logistic Regression**: `gmm_4_cluster ~ max_cci`
# MAGIC
# MAGIC `statsmodels.MNLogit` uses the lowest-valued category as the baseline.
# MAGIC We recode cluster IDs so that `BASELINE_CLUSTER` (Outpatient Regular = 2) becomes 0,
# MAGIC ensuring it is used as the baseline reference.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 6.1 Multinomial Logit

# COMMAND ----------

# Collapse to the last person-year per patient for MNLogit.
_last_idx = analysis_df.groupby("person_id")["year"].idxmax()
mnlogit_df = (
    analysis_df.loc[_last_idx, ["person_id", "year", "gmm_4_cluster", "max_cci"]]
    .dropna()
    .copy()
)
print(f"Q3 sample: {len(mnlogit_df):,} unique patients (collapsed from {len(analysis_df):,} person-years)")

sorted_clusters = sorted(mnlogit_df["gmm_4_cluster"].unique())
reordered = [BASELINE_CLUSTER] + [c for c in sorted_clusters if c != BASELINE_CLUSTER]
cluster_to_baseline_zero = {orig: new for new, orig in enumerate(reordered)}
baseline_zero_to_cluster = {new: orig for orig, new in cluster_to_baseline_zero.items()}

mnlogit_df["cluster_recoded"] = mnlogit_df["gmm_4_cluster"].map(cluster_to_baseline_zero)

y = mnlogit_df["cluster_recoded"].astype(int)
X = pd.DataFrame({"const": 1.0, "max_cci": mnlogit_df["max_cci"].astype(float)})

mnlogit_model = MNLogit(y, X).fit(disp=False, maxiter=200)
if not mnlogit_model.mle_retvals["converged"]:
    warnings.warn("MNLogit did not converge; interpret results with caution")

print(f"Q3 -- Multinomial Logistic Regression: cluster ~ max_cci")
print(f"Baseline: Cluster {BASELINE_CLUSTER} ({BASELINE_LABEL})")
print(mnlogit_model.summary())

print("\nOdds Ratios for max_cci (exp(coef)) per cluster vs. baseline:")
mnlogit_or_rows = []
n_nonbaseline = mnlogit_model.params.shape[1]
for j in range(n_nonbaseline):
    recoded_id = j + 1
    orig_id = baseline_zero_to_cluster[recoded_id]
    coef = mnlogit_model.params.iloc[1, j]
    pval = mnlogit_model.pvalues.iloc[1, j]
    odds_ratio = np.exp(coef)
    mnlogit_or_rows.append(
        {
            "vs_cluster": orig_id,
            "vs_label": GMM4_LABELS.get(orig_id, str(orig_id)),
            "odds_ratio": round(odds_ratio, 4),
            "coef": round(coef, 4),
            "p_value": pval,
            "sig": sig_label(pval),
            "is_reference": False,
        }
    )
    print(f"  Cluster {orig_id} ({GMM4_LABELS.get(orig_id)}): OR={odds_ratio:.4f}, p={pval:.4e} {sig_label(pval)}")

try:
    for j, _or_row in enumerate(mnlogit_or_rows):
        _se = float(mnlogit_model.bse.iloc[1, j])
        _c  = _or_row["coef"]
        _or_row["ci_lo_95"] = round(float(np.exp(_c - 1.96 * _se)), 4)
        _or_row["ci_hi_95"] = round(float(np.exp(_c + 1.96 * _se)), 4)
    print("\n  95% CI added to OR rows via Wald method")
except (IndexError, KeyError, AttributeError) as _ci_err:
    print(f"  Warning: 95% CI extraction failed ({_ci_err}); CI set to NaN")
    for _or_row in mnlogit_or_rows:
        _or_row.setdefault("ci_lo_95", float("nan"))
        _or_row.setdefault("ci_hi_95", float("nan"))

_baseline_row = {
    "vs_cluster":   BASELINE_CLUSTER,
    "vs_label":     BASELINE_LABEL,
    "odds_ratio":   1.0,
    "coef":         0.0,
    "p_value":      float("nan"),
    "sig":          "ref",
    "ci_lo_95":     float("nan"),
    "ci_hi_95":     float("nan"),
    "is_reference": True,
}
mnlogit_or_rows = [_baseline_row] + mnlogit_or_rows

_save_table(pd.DataFrame(mnlogit_or_rows), EDI_DATA04_Q3, data=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 6.2 Q3 Visual -- Forest Plot (Odds Ratios with 95% CI)
# MAGIC
# MAGIC OR per 1-unit CCI increase vs. Outpatient Regular baseline. Points right of OR=1 = higher CCI predicts membership.

# COMMAND ----------

_or_plot = sorted(mnlogit_or_rows, key=lambda r: r["odds_ratio"])
_n_or = len(_or_plot)

fig, ax = plt.subplots(figsize=(8, max(3.5, _n_or * 1.1)))
ax.set_facecolor("white")

_REF_COLOR = "#666666"

for idx, row in enumerate(_or_plot):
    is_ref = bool(row.get("is_reference", False))
    color  = _REF_COLOR if is_ref else CLUSTER_COLORS.get(row["vs_label"], "#555555")
    ci_lo  = row.get("ci_lo_95", float("nan"))
    ci_hi  = row.get("ci_hi_95", float("nan"))

    if idx % 2 == 0:
        ax.axhspan(idx - 0.4, idx + 0.4, color=_RC_BG, zorder=0)

    if is_ref:
        ax.scatter(row["odds_ratio"], idx, s=120, marker="D", color="white",
                   edgecolors=_REF_COLOR, linewidths=1.8, zorder=4)
        ax.text(row["odds_ratio"] * 1.08, idx, "OR 1.00  (reference)",
                va="center", ha="left", fontsize=9, color=_REF_COLOR,
                fontweight="medium", fontstyle="italic")
        continue

    if pd.notna(ci_lo) and pd.notna(ci_hi):
        ax.plot([ci_lo, ci_hi], [idx, idx], lw=3.0, color=color,
                solid_capstyle="round", alpha=0.80, zorder=2)
        for cx in [ci_lo, ci_hi]:
            ax.vlines(cx, idx - 0.15, idx + 0.15, lw=1.8, color=color, alpha=0.80, zorder=2)

    ax.scatter(row["odds_ratio"], idx, s=100, color=color,
               edgecolors="white", linewidths=1.6, zorder=4)

    _ci_str = ""
    if pd.notna(ci_lo) and pd.notna(ci_hi):
        _ci_str = f"  [{ci_lo:.2f}-{ci_hi:.2f}]"
    _x_label = (ci_hi if pd.notna(ci_hi) else row["odds_ratio"]) * 1.08
    ax.text(_x_label, idx, f"OR {row['odds_ratio']:.2f}{_ci_str}  {row['sig']}",
            va="center", ha="left", fontsize=9, color=_RC_TEXT, fontweight="medium")

ax.axvline(1.0, color=_RC_MUTED, lw=1.0, linestyle="--", alpha=0.6, zorder=1)
ax.text(1.0, _n_or - 0.5, "OR = 1\n(no effect)", ha="center", va="bottom",
        fontsize=7.5, color=_RC_MUTED, fontstyle="italic")

ax.set_xscale("log")
ax.set_yticks(range(_n_or))
_ytick_labels = [
    f"{r['vs_label']}\n(reference)" if r.get("is_reference") else r["vs_label"]
    for r in _or_plot
]
ax.set_yticklabels(_ytick_labels, fontsize=10, fontweight="semibold")
for _tlbl, _row in zip(ax.get_yticklabels(), _or_plot):
    if _row.get("is_reference"):
        _tlbl.set_fontweight("bold")
        _tlbl.set_color(_REF_COLOR)
ax.set_xlabel(f"Odds Ratio (log scale) per 1-unit CCI increase vs. {BASELINE_LABEL}", fontsize=9.5)
ax.set_title(
    f"Q3: Does CCI Predict Cluster Assignment?\n"
    f"Multinomial Logit  |  n = {len(mnlogit_df):,} patients  |  baseline = {BASELINE_LABEL}",
    fontsize=10.5, fontweight="bold", pad=12,
)
ax.grid(axis="x", color=_RC_GRID, linewidth=0.4, alpha=0.6)
ax.grid(axis="y", visible=False)
ax.spines["left"].set_visible(False)
ax.spines["bottom"].set_color(_RC_GRID)
ax.tick_params(axis="y", length=0)
ax.set_ylim(-0.6, _n_or - 0.4)

fig.tight_layout()
_save_fig(fig, EDI_FIG01_Q3_FOREST)

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## 7. Consolidated Summary Table

# COMMAND ----------

summary_rows = []

for outcome_col in outcome_cols:
    # Q1 -- Pooled Spearman
    q1_row = q1_results.get(outcome_col, {})
    summary_rows.append(
        {
            "Outcome": outcome_col,
            "Question": f"Q1: Is higher CCI associated with {outcome_label(outcome_col)}? (pooled)",
            "Test": "Pooled Spearman's rho",
            "N": q1_row.get("n", 0),
            "Statistic": f"rho = {q1_row.get('rho', float('nan')):.4f}",
            "P_Value": fmt_p(q1_row.get("p", float("nan"))),
            "Sig": sig_label(q1_row.get("p", 1.0)),
            "Interpretation": rho_interp(q1_row.get("rho", 0.0)),
        }
    )

    # Q1b -- Stratified Spearman (one row per cluster)
    for _, row in q1b_results.get(outcome_col, pd.DataFrame()).iterrows():
        pval = row["p_value"]
        summary_rows.append(
            {
                "Outcome": outcome_col,
                "Question": f"Q1b: Within {row['cluster_label']} -- CCI vs {outcome_label(outcome_col)}",
                "Test": "Stratified Spearman's rho",
                "N": int(row["n_person_years"]),
                "Statistic": f"rho = {row['spearman_rho']:.4f}",
                "P_Value": fmt_p(pval),
                "Sig": sig_label(pval),
                "Interpretation": rho_interp(row["spearman_rho"]),
            }
        )

    # Q2 -- OLS interaction (one row per interaction term)
    q2_row = q2_results.get(outcome_col, {})
    for term in q2_row.get("interaction_terms", []):
        coef = q2_row["model"].params[term]
        pval = q2_row["model"].pvalues[term]
        summary_rows.append(
            {
                "Outcome": outcome_col,
                "Question": f"Q2: Does the CCI slope differ by cluster for {outcome_label(outcome_col)}?",
                "Test": f"OLS interaction (clustered SE) | outcome={q2_row['outcome_label']}",
                "N": int(q2_row["model"].nobs),
                "Statistic": f"coef = {coef:.4f}",
                "P_Value": fmt_p(pval),
                "Sig": sig_label(pval),
                "Interpretation": "slope differs from baseline" if pval < 0.05 else "slope not significantly different",
            }
        )

# Q3 -- Multinomial logit (one row per cluster vs baseline; not residual-specific)
for row in mnlogit_or_rows:
    pval = row["p_value"]
    _ci_lo = row.get("ci_lo_95", float("nan"))
    _ci_hi = row.get("ci_hi_95", float("nan"))
    _ci_str = (
        f" [{_ci_lo:.2f}-{_ci_hi:.2f}]"
        if pd.notna(_ci_lo) and pd.notna(_ci_hi)
        else ""
    )
    summary_rows.append(
        {
            "Outcome": "cluster_assignment",
            # Include the target cluster name so each Q3 row is self-describing
            "Question": f"Q3: Does CCI predict {row['vs_label']} assignment? (vs {GMM4_LABELS[BASELINE_CLUSTER]})",
            "Test": "Multinomial Logit (OR for max_cci)",
            "N": len(mnlogit_df),
            "Statistic": f"OR = {row['odds_ratio']:.4f}{_ci_str}",
            "P_Value": fmt_p(pval),
            "Sig": sig_label(pval),
            "Interpretation": (
                f"Higher CCI {'increases' if row['odds_ratio'] > 1 else 'decreases'} odds of "
                f"{row['vs_label']} vs {GMM4_LABELS[BASELINE_CLUSTER]}"
            ),
        }
    )

summary_pdf = pd.DataFrame(summary_rows)

print("=" * 80)
print("SUMMARY: CCI vs Within-Cluster Residuals Analysis")
print("=" * 80)
display(spark.createDataFrame(summary_pdf))
_save_table(summary_pdf, EDI_DATA08_SUMMARY, data=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## 7.5 Auxiliary Manuscript Tables
# MAGIC
# MAGIC Per-patient-year classification for Table 1b (above/below cluster mean on
# MAGIC average cross-domain residual) and GMM-4 cluster profiles dumped from Delta.

# COMMAND ----------

_t1b = analysis_df[["person_id", "year", "gmm_4_cluster", "cluster_label", "max_cci"] + domain_resid_cols].copy()
_t1b["avg_cross_domain_residual"] = _t1b[domain_resid_cols].mean(axis=1)
_t1b["above_below_cluster_mean"] = np.where(_t1b["avg_cross_domain_residual"] >= 0, "above", "below")
_t1b["is_reference_cluster"] = _t1b["gmm_4_cluster"].astype(int) == BASELINE_CLUSTER
_save_table(
    _t1b[["person_id", "year", "gmm_4_cluster", "cluster_label", "is_reference_cluster", "max_cci",
          "avg_cross_domain_residual", "above_below_cluster_mean"]],
    EDI_TBL01B_CLASSIFICATION,
)

# COMMAND ----------

_cluster_profiles_pdf = spark.table(f"{input_db}.gmm_4_cluster_profiles").toPandas()
_cluster_key_col = "gmm_4_cluster" if "gmm_4_cluster" in _cluster_profiles_pdf.columns else "cluster_key"
_cluster_ids_int = _cluster_profiles_pdf[_cluster_key_col].astype(int)
_cluster_profiles_pdf.insert(1, "cluster_label", _cluster_ids_int.map(GMM4_LABELS))
_cluster_profiles_pdf["is_reference"] = _cluster_ids_int == BASELINE_CLUSTER
_save_table(_cluster_profiles_pdf, EDI_TBL02_CLUSTER_PROFILES)

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## 8. Generate Report

# COMMAND ----------

# MAGIC %run ./99_report

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Cleanup

# COMMAND ----------

characterization_data.unpersist()
