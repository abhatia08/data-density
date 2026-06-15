# Databricks notebook source
# MAGIC %md
# MAGIC **Purpose**: Shared helpers for cohort_1 notebooks

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Setup and Configuration

# COMMAND ----------

# Standard library
import csv
import math
import os
import warnings
from datetime import datetime
from contextlib import contextmanager
from typing import List, Literal, Optional, Dict, Any, Tuple

# Type alias used throughout to constrain the irregularity penalty parameter.
# Valid values: 'L1' (mean absolute deviation), 'L2' (RMS deviation), 'BOTH'.
IrregularityPenalty = Literal["L1", "L2", "BOTH"]
MethodName = Literal["rules_vanilla", "rules_adaptive", "gmm_4", "gmm_7"]

# Scientific computing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from cycler import cycler

# Machine learning
from sklearn.metrics import (
    silhouette_samples, 
    davies_bouldin_score, 
    adjusted_rand_score,
    cohen_kappa_score
)
from scipy.stats import mstats

# PySpark
from pyspark.sql import functions as F, SparkSession, DataFrame, Window
from pyspark.sql.types import (
    StructType, StructField, StringType, DoubleType, 
    IntegerType, LongType, BooleanType, ArrayType
)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# COMMAND ----------

def configure_spark_optimizations():
    # Skip platform-managed keys on Serverless (CONFIG_NOT_AVAILABLE).
    spark = SparkSession.builder.getOrCreate()

    tuning = [
        # Adaptive Query Execution (already on by default in DBR 13+/Serverless)
        ("spark.sql.adaptive.enabled", "true"),
        ("spark.sql.adaptive.coalescePartitions.enabled", "true"),
        ("spark.sql.adaptive.skewJoin.enabled", "true"),
        # Delta Lake optimizations
        ("spark.databricks.delta.optimizeWrite.enabled", "true"),
        ("spark.databricks.delta.autoCompact.enabled", "true"),
        # Arrow optimization for Pandas conversions
        ("spark.sql.execution.arrow.pyspark.enabled", "true"),
        ("spark.sql.execution.arrow.maxRecordsPerBatch", "10000"),
        # Broadcast join optimization (10MB threshold)
        ("spark.sql.autoBroadcastJoinThreshold", "10485760"),
    ]
    for key, value in tuning:
        try:
            spark.conf.set(key, value)
        except Exception:
            pass

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Display Helpers

# COMMAND ----------

_QUIET = os.environ.get("UTILS_QUIET", "1") in ("1", "true", "True")

def section(title: str) -> None:
    # Print section header in notebook style
    if _QUIET:
        return
    print(f"\n## {title}")

def info(message: str) -> None:
    # Print lightweight info message
    if _QUIET:
        return
    print(f"· {message}")

def success(message: str) -> None:
    # Print lightweight success message with checkmark
    if _QUIET:
        return
    print(f"✓ {message}")

def warn(message: str) -> None:
    # Print lightweight warning message with exclamation
    if _QUIET:
        return
    print(f"! {message}")

def kv(items: dict, pad: int | None = None) -> None:
    # Print aligned key:value pairs
    if _QUIET or not items:
        return
    pad = pad or max(len(str(k)) for k in items.keys())
    for k, v in items.items():
        print(f"{str(k).rjust(pad)}: {v}")


def sig_label(p: float) -> str:
    """APA-style significance codes: *** p<0.001, ** p<0.01, * p<0.05, ns otherwise."""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def rho_interp(rho: float) -> str:
    """Interpret Spearman rho magnitude and direction (negligible/weak/moderate/strong)."""
    mag = abs(rho)
    direction = "positive" if rho > 0 else "negative"
    if mag < 0.1:
        return "negligible"
    elif mag < 0.3:
        return f"weak {direction}"
    elif mag < 0.5:
        return f"moderate {direction}"
    else:
        return f"strong {direction}"


def fmt_p(p: float) -> str:
    """APA-style p-value formatter: scientific notation for p < 0.001, 4 dp otherwise. NaN → em-dash."""
    if math.isnan(p):
        return "—"
    return f"{p:.3e}" if p < 0.001 else f"{p:.4f}"


# Colorblind-safe 7-color qualitative palette (Wong 2011 / Okabe-Ito extended).
# Used across all scripts; call get_palette(k) to get the right subset for k groups.
STYLE_PALETTE = [
    "#0072B2",  # Blue         – primary series
    "#D55E00",  # Orange       – emphasis / significant
    "#009E73",  # Green        – third group
    "#CC79A7",  # Pink         – fourth group
    "#56B4E9",  # Light blue   – control / baseline
    "#E69F00",  # Gold         – treatment / intervention
    "#999999",  # Gray         – non-significant / background
]

# Recommended hand-picked subsets per group count (from style_reference.html)
_PALETTE_SUBSETS: Dict[int, List[str]] = {
    2: ["#D55E00", "#56B4E9"],
    3: ["#0072B2", "#D55E00", "#009E73"],
    4: ["#56B4E9", "#D55E00", "#009E73", "#E69F00"],
}


def get_palette(k: int) -> List[str]:
    """Return the style-reference-approved color subset for k groups.

    Uses curated hand-picked subsets for k ∈ {2, 3, 4}; the full 7-color
    palette for k ≤ 7; and viridis interpolation for k > 7.
    """
    import matplotlib.pyplot as _plt
    if k <= 1:
        return STYLE_PALETTE[:1]
    if k in _PALETTE_SUBSETS:
        return _PALETTE_SUBSETS[k]
    if k <= 7:
        return STYLE_PALETTE[:k]
    return [_plt.cm.viridis(i / (k - 1)) for i in range(k)]


def apply_exploratory_plot_style() -> None:
    """Apply shared matplotlib/seaborn style for cohort_1 exploratory figures.

    Call once after %run ./99_utils so 04/05 (and any notebook using the same
    look) get consistent typography, spines, grid, and STYLE_PALETTE color cycle.
    """
    sns.set_style("ticks")
    plt.rcParams.update({
        "figure.dpi": 300,
        "font.family": "sans-serif",
        "font.sans-serif": ["Source Sans 3", "Helvetica Neue", "Arial", "DejaVu Sans"],
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "axes.labelweight": "bold",
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.alpha": 0.25,
        "grid.color": "#9aa0a6",
        "legend.frameon": False,
        "legend.fontsize": 9,
        "axes.prop_cycle": cycler(color=STYLE_PALETTE),
    })


# COMMAND ----------

def set_utils_verbose(verbose: bool) -> None:
    # Toggle helper verbosity
    global _QUIET
    _QUIET = not bool(verbose)

def get_verbose(default: bool = True) -> bool:
    # Read global pipeline verbosity from env, fallback to provided default
    val = os.environ.get("PIPELINE_VERBOSE")
    if val is None:
        return bool(default)
    return str(val) in ("1", "true", "True", "yes", "YES")

def gate_prints(verbose: bool) -> None:
    # Optionally silence built-in print for cleaner runs (does not affect display())
    try:
        import builtins as _b
    except Exception:
        return
    set_utils_verbose(verbose)
    if verbose:
        if hasattr(gate_prints, "_orig_print"):
            _b.print = getattr(gate_prints, "_orig_print")
    else:
        if not hasattr(gate_prints, "_orig_print"):
            setattr(gate_prints, "_orig_print", _b.print)
        def _noop_print(*args, **kwargs):
            return None
        _b.print = _noop_print


@contextmanager
def quiet_prints():
    """Temporarily silence print statements and restore reliably."""
    import builtins as _b
    _orig_print = _b.print
    _b.print = lambda *args, **kwargs: None
    try:
        yield
    finally:
        _b.print = _orig_print

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Cache Management

# COMMAND ----------

def check_cache_exists(required_tables=None, output_db="cohort_1", verbose=True):
    # Check if cached results exist for the requested tables
    try:
        spark = SparkSession.builder.getOrCreate()

        if required_tables is None:
            required_tables = [
                "rules_yearly",
                "rules_adaptive_yearly",
                "gmm_4_yearly",
                "gmm_7_yearly"
            ]

        existing = []
        missing = []

        for table in required_tables:
            full_name = table if "." in table else f"{output_db}.{table}"
            db_part, tbl_part = full_name.split(".", 1)
            if table_exists(db_part, tbl_part):
                existing.append(full_name)
            else:
                missing.append(full_name)

        if verbose and missing:
            warn(f"Missing {len(missing)} tables")
            if not _QUIET:
                kv({"missing": ", ".join(missing)})

        return len(missing) == 0
    except Exception as exc:
        if verbose:
            warn(f"Cache check failed: {exc}")
        return False

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Cluster Profiling & Quality

# COMMAND ----------

def create_cluster_profile_table(
    data_spark: DataFrame,
    method_name: str,
    cluster_col: str,
    output_db: str = "cohort_1",
) -> DataFrame:
    # Generate comprehensive cluster profiles with numeric feature summaries, CCI composition, and rule-based archetype composition
    # Ensure cluster key is string to avoid type conflicts
    clustered = (
        data_spark
        .filter(F.col(cluster_col).isNotNull())
        .withColumn('cluster_key', F.col(cluster_col).cast('string'))
    )
    
    # Numeric features for profiling — canonical list from get_profile_features()
    numeric_features = get_profile_features()
    
    # Numeric aggregations: mean, min, max
    agg_exprs = [F.count('*').alias('n')]
    for feat in numeric_features:
        agg_exprs.extend([
            F.mean(feat).alias(f'{feat}_mean'),
            F.min(feat).alias(f'{feat}_min'),
            F.max(feat).alias(f'{feat}_max')
        ])
    
    profiles = clustered.groupBy('cluster_key').agg(*agg_exprs).orderBy('cluster_key')
    
    # CCI composition (% None/Low/Moderate/High)
    cci_binned = clustered.withColumn('cci_category',
        F.when(F.col('max_cci') == 0, 'None')
         .when((F.col('max_cci') >= 1) & (F.col('max_cci') <= 2), 'Low')
         .when((F.col('max_cci') >= 3) & (F.col('max_cci') <= 4), 'Moderate')
         .otherwise('High')
    )
    
    cci_pct = cci_binned.groupBy('cluster_key', 'cci_category').count()
    cci_total = cci_binned.groupBy('cluster_key').count().withColumnRenamed('count', 'total')
    cci_result = cci_pct.join(cci_total, 'cluster_key') \
        .withColumn('percentage', (F.col('count') / F.col('total')) * 100)
    cci_pivot = cci_result.groupBy('cluster_key').pivot('cci_category', ['None', 'Low', 'Moderate', 'High']) \
        .agg(F.first('percentage'))
    
    # Archetype composition (join with rule-based archetypes)
    spark = SparkSession.builder.getOrCreate()
    try:
        rules = spark.table(f"{output_db}.rules_yearly") \
            .select('person_id', 'year', F.col('archetype').alias('rule_archetype'))
        
        arch_joined = clustered.join(rules, ['person_id', 'year'], 'left') \
            .filter(F.col('rule_archetype').isNotNull())
        
        arch_pct = arch_joined.groupBy('cluster_key', 'rule_archetype').count() \
            .join(arch_joined.groupBy('cluster_key').count().withColumnRenamed('count', 'total'), 'cluster_key') \
            .withColumn('percentage', (F.col('count') / F.col('total')) * 100)
        
        arch_concat = arch_pct.groupBy('cluster_key').agg(
            F.concat_ws('; ', F.collect_list(
                F.concat(F.col('rule_archetype'), F.lit(' '),
                        F.round(F.col('percentage'), 1).cast('string'), F.lit('%'))
            )).alias('archetype_composition')
        )
    except Exception:
        # If rule-based archetypes not available, create empty archetype composition
        arch_concat = profiles.select('cluster_key') \
            .withColumn('archetype_composition', F.lit(None).cast('string'))
    
    # Join all components
    result = profiles.join(cci_pivot, 'cluster_key', 'left') \
        .join(arch_concat, 'cluster_key', 'left') \
        .withColumn('method', F.lit(method_name))
    
    # Rename cluster_key back to original cluster_col for consistency
    result = result.withColumnRenamed('cluster_key', cluster_col)
    
    return result

# COMMAND ----------

def find_cluster_medoids(data_pdf, method_name, cluster_col, feature_cols):
    # Find representative patient (medoid) closest to each cluster centroid
    medoids = []
    
    for cluster in data_pdf[cluster_col].unique():
        if pd.isna(cluster) or cluster == -1:
            continue
        
        cluster_data = data_pdf[data_pdf[cluster_col] == cluster]
        
        if len(cluster_data) == 0:
            continue
        
        # Calculate centroid (mean of all features)
        centroid = cluster_data[feature_cols].mean().values
        
        # Find patient closest to centroid (Euclidean distance)
        distances = np.linalg.norm(cluster_data[feature_cols].values - centroid, axis=1)
        medoid_idx = distances.argmin()
        medoid_row = cluster_data.iloc[medoid_idx]
        
        medoids.append({
            'method': method_name,
            'cluster': cluster,
            'person_id': int(medoid_row['person_id']),
            'year': int(medoid_row['year']),
            'distance_to_centroid': float(distances[medoid_idx])
        })
    
    return pd.DataFrame(medoids)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4.1 Cluster Quality Metrics

# COMMAND ----------

def calculate_separation_metrics(
    data_pdf: pd.DataFrame,
    cluster_col: str,
    feature_cols: List[str],
) -> Tuple[Optional[pd.DataFrame], Optional[float]]:
    # Calculate cluster separation metrics (silhouette scores, Davies-Bouldin index)
    # Filter to valid clusters only
    valid_data = data_pdf[data_pdf[cluster_col].notna() & (data_pdf[cluster_col] != -1)]
    
    if len(valid_data) < 2:
        return None, None
    
    X = valid_data[feature_cols].values
    labels = valid_data[cluster_col].values
    
    # Need at least 2 clusters for separation metrics
    if len(np.unique(labels)) < 2:
        return None, None
    
    # Per-sample silhouette scores
    sil_samples = silhouette_samples(X, labels)
    
    # Aggregate by cluster
    results = []
    for cluster in np.unique(labels):
        cluster_mask = labels == cluster
        cluster_sil = sil_samples[cluster_mask].mean()
        results.append({
            'cluster': cluster,
            'silhouette_mean': float(cluster_sil),
            'cluster_size': int(cluster_mask.sum())
        })
    
    # Overall Davies-Bouldin index (lower = better separation)
    db_index = davies_bouldin_score(X, labels)
    
    return pd.DataFrame(results), float(db_index)

# COMMAND ----------

def calculate_feature_importance(
    data_spark: DataFrame,
    method_name: str,
    cluster_col: str,
    feature_cols: List[str]
) -> pd.DataFrame:
    # Calculate feature importance using Cohen's d effect size for each cluster versus the rest

    safe_cols = [F.coalesce(F.col(c), F.lit(0.0)) for c in feature_cols]

    # Global aggregates: count, sum, and sum of squares per feature
    agg_expressions = [F.count('*').alias('n_total')]
    for col_name, col_expr in zip(feature_cols, safe_cols):
        agg_expressions.append(F.sum(col_expr).alias(f'{col_name}_sum'))
        agg_expressions.append(F.sum(col_expr * col_expr).alias(f'{col_name}_sum_sq'))

    global_stats = data_spark.agg(*agg_expressions).collect()[0]

    # Cluster-level aggregates
    cluster_agg_exprs = [F.count('*').alias('n')]
    for col_name, col_expr in zip(feature_cols, safe_cols):
        cluster_agg_exprs.append(F.sum(col_expr).alias(f'{col_name}_sum'))
        cluster_agg_exprs.append(F.sum(col_expr * col_expr).alias(f'{col_name}_sum_sq'))

    cluster_stats = (
        data_spark
        .filter(F.col(cluster_col).isNotNull())
        .withColumn('cluster_key', F.col(cluster_col).cast('string'))
        .groupBy('cluster_key')
        .agg(*cluster_agg_exprs)
    )

    n_total = int(global_stats['n_total'])
    importance_records = []

    for row in cluster_stats.collect():
        cluster = row['cluster_key']
        n_cluster = int(row['n'])
        n_other = n_total - n_cluster

        for feat in feature_cols:
            # Cluster statistics
            cluster_sum = float(row[f'{feat}_sum'])
            cluster_sum_sq = float(row[f'{feat}_sum_sq'])
            cluster_mean = cluster_sum / n_cluster if n_cluster else 0.0
            if n_cluster > 1:
                cluster_var = (cluster_sum_sq - (cluster_sum ** 2) / n_cluster) / (n_cluster - 1)
                cluster_var = max(cluster_var, 0.0)
            else:
                cluster_var = 0.0

            # Complement statistics
            total_sum = float(global_stats[f'{feat}_sum'])
            total_sum_sq = float(global_stats[f'{feat}_sum_sq'])
            other_sum = total_sum - cluster_sum
            other_sum_sq = total_sum_sq - cluster_sum_sq
            other_mean = other_sum / n_other if n_other else 0.0
            if n_other > 1:
                other_var = (other_sum_sq - (other_sum ** 2) / n_other) / (n_other - 1)
                other_var = max(other_var, 0.0)
            else:
                other_var = 0.0

            # Pooled standard deviation across cluster vs rest
            pooled_std = 0.0
            cohens_d = 0.0
            if n_cluster > 1 and n_other > 1:
                pooled_n = n_cluster + n_other - 2
                pooled_var = ((n_cluster - 1) * cluster_var + (n_other - 1) * other_var) / pooled_n
                pooled_var = max(pooled_var, 0.0)
                if pooled_var > 0:
                    pooled_std = float(np.sqrt(pooled_var))
                    cohens_d = abs((cluster_mean - other_mean) / pooled_std)

            importance_records.append({
                'method': method_name,
                'cluster': cluster,
                'feature': feat,
                'cohens_d': float(cohens_d),
                'cluster_mean': float(cluster_mean),
                'cluster_std': float(np.sqrt(cluster_var)) if cluster_var > 0 else 0.0,
                'global_mean': float(total_sum / n_total) if n_total else 0.0,
                'global_std': 0.0,  # populated in the second pass below (per-feature global variance)
                'pooled_std': float(pooled_std),
                'importance_composite': float(cohens_d),
                'rest_mean': float(other_mean),
                'rest_std': float(np.sqrt(other_var)) if other_var > 0 else 0.0
            })

    # Populate global_std for completeness (uses overall sample variance)
    for record in importance_records:
        feat = record['feature']
        total_sum = float(global_stats[f'{feat}_sum'])
        total_sum_sq = float(global_stats[f'{feat}_sum_sq'])
        if n_total > 1:
            global_var = (total_sum_sq - (total_sum ** 2) / n_total) / (n_total - 1)
            global_var = max(global_var, 0.0)
            record['global_std'] = float(np.sqrt(global_var))

    return pd.DataFrame(importance_records)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Rules-Based Classification

# COMMAND ----------

def classify_archetype_logic(
    inpatient_visit_count: int,
    visit_count: int,
    irregularity_score: Optional[float],
    irregularity_threshold: float = 0.5,
    return_reason: bool = False,
):
    """Classify a patient-year into one of 7 utilization archetypes.

    Preconditions
    -------------
    - visit_count includes ALL visits (inpatient + outpatient); inpatient_visit_count
      is a strict subset of visit_count.
    - irregularity_score is None when the patient has <3 visits (insufficient data
      to compute inter-visit regularity).

    Returns
    -------
    str or (str, str)
        Archetype label, or (label, reason) tuple when return_reason=True.
    """
    # Handle null irregularity_score
    if irregularity_score is None:
        irregularity_score = 1.0  # Default to high irregularity if missing

    # Priority 1: Multiple Complex Episodes (≥2 IP visits)
    if inpatient_visit_count >= 2:
        archetype = 'Multiple Complex Episodes'
        reason = f'IP visits >= 2 (n={inpatient_visit_count})'
        return (archetype, reason) if return_reason else archetype

    # Priority 2: Sporadic Complex Episodes (1 IP visit)
    if inpatient_visit_count == 1:
        archetype = 'Sporadic Complex Episodes'
        reason = '1 IP visit'
        return (archetype, reason) if return_reason else archetype

    # Priority 3: Sparse Use (≤1 visit)
    if visit_count <= 1:
        archetype = 'Sparse Use'
        reason = 'Single visit or no visits'
        return (archetype, reason) if return_reason else archetype

    # Priority 4: Frequent visits (≥4) - check regularity
    if visit_count >= 4:
        if irregularity_score < irregularity_threshold:  # LOW irregularity = regular
            archetype = 'Regular Frequent'
            reason = f'≥4 visits + regular (score={irregularity_score:.2f})'
        else:
            archetype = 'Irregular Frequent'
            reason = f'≥4 visits + irregular (score={irregularity_score:.2f})'
        return (archetype, reason) if return_reason else archetype

    # Priority 5: Infrequent visits (2-3) - check regularity
    if irregularity_score < irregularity_threshold:  # LOW irregularity = regular
        archetype = 'Regular Infrequent'
        reason = f'2-3 visits + regular (score={irregularity_score:.2f})'
    else:
        archetype = 'Irregular Infrequent'
        reason = f'2-3 visits + irregular (score={irregularity_score:.2f})'
    return (archetype, reason) if return_reason else archetype


def build_archetype_schema() -> StructType:
    """Return the canonical schema for rules-based archetype classification outputs."""
    return StructType([
        StructField("archetype", StringType(), False),
        StructField("reason", StringType(), False),
    ])


def make_classify_archetype_udf(irregularity_threshold: float = 0.5):
    """
    Factory to create a Spark UDF that wraps `classify_archetype_logic`.

    Returns a UDF that takes (inpatient_visit_count, visit_count, irregularity_score)
    and returns struct<archetype:string, reason:string>.
    """
    archetype_schema = build_archetype_schema()

    @F.udf(archetype_schema)
    def _udf(inpatient_visit_count, visit_count, irregularity_score):
        return classify_archetype_logic(
            inpatient_visit_count,
            visit_count,
            irregularity_score,
            irregularity_threshold=irregularity_threshold,
            return_reason=True,
        )

    return _udf


def add_longitudinal_tracking(df: DataFrame, archetype_col: str = "archetype") -> DataFrame:
    """Add prior-year archetype, change flag, segment ID, and tenure columns.

    Parameters
    ----------
    df : DataFrame
        Spark DataFrame containing person_id, year, and the archetype column.
    archetype_col : str
        Name of the column holding archetype/cluster labels.

    Returns
    -------
    DataFrame with four additional columns:
        prior_year_archetype_rules, archetype_changed, segment_id, years_in_current_archetype
    """
    person_window = Window.partitionBy("person_id").orderBy("year")
    seg_win = Window.partitionBy("person_id").orderBy("year").rowsBetween(Window.unboundedPreceding, 0)

    return (
        df
        .withColumn("prior_year_archetype_rules", F.lag(archetype_col, 1).over(person_window))
        .withColumn(
            "archetype_changed",
            F.when(
                F.col("prior_year_archetype_rules").isNotNull()
                & (F.col(archetype_col) != F.col("prior_year_archetype_rules")),
                1,
            ).otherwise(0),
        )
        .withColumn("segment_id", F.sum("archetype_changed").over(seg_win))
        .withColumn(
            "years_in_current_archetype",
            F.row_number().over(Window.partitionBy("person_id", "segment_id").orderBy("year")),
        )
    )

# COMMAND ----------

def table_exists(db_name, table_name):
    # Check if a Delta table exists in the given database
    try:
        spark = SparkSession.builder.getOrCreate()
        full_name = f"{db_name}.{table_name}"
        # Use a lighter check than count() - just try to get schema
        spark.table(full_name).schema
        return True
    except Exception as e:
        # More informative error handling for debugging
        err_str = str(e)
        if "TABLE_OR_VIEW_NOT_FOUND" in err_str or "Table or view not found" in err_str:
            return False
        # Log other errors for visibility
        import sys
        print(f"Warning: Error checking table {db_name}.{table_name}: {e}", file=sys.stderr)
        return False

# COMMAND ----------

def clear_downstream_tables(
    db_name: str,
    core_tables: List[str],
    use_widget: bool = True,
    verbose: bool = True,
    clear_downstream: bool = False
) -> bool:
    # Clear downstream delta tables, keeping only core tables
    # Get user preference via widget if requested
    if use_widget:
        try:
            dbutils.widgets.dropdown(
                "clear_downstream", 
                "false", 
                ["true", "false"], 
                "Clear downstream delta tables (keeps core + cohort tables only)?"
            )
            clear_downstream = dbutils.widgets.get("clear_downstream") == "true"
        except Exception:
            # If widgets not available (e.g., in non-Databricks environment), skip
            if verbose:
                info("Widgets not available, skipping downstream table cleanup")
            return False
    
    if not clear_downstream:
        if verbose:
            info("Skipping downstream table cleanup (clear_downstream = false)")
        return False
    
    # Get all tables in the database
    spark = SparkSession.builder.getOrCreate()
    tables_df = spark.sql(f"SHOW TABLES IN {db_name}")
    all_tables = [row.tableName for row in tables_df.collect()]
    
    # Identify tables to keep (core tables, case-insensitive)
    tables_to_keep = {tbl.lower() for tbl in core_tables}
    
    # Find tables to delete (everything except core tables)
    tables_to_delete = [tbl for tbl in all_tables if tbl.lower() not in tables_to_keep]
    
    if tables_to_delete:
        if verbose:
            info(f"Found {len(tables_to_delete)} downstream tables to delete:")
            for tbl in sorted(tables_to_delete):
                info(f"  - {tbl}")
        else:
            print(f"Found {len(tables_to_delete)} downstream tables to delete:")
            for tbl in sorted(tables_to_delete):
                print(f"  - {tbl}")
        
        # Delete each downstream table
        deleted_count = 0
        for tbl in tables_to_delete:
            try:
                spark.sql(f"DROP TABLE IF EXISTS `{db_name}`.`{tbl}`")
                deleted_count += 1
            except Exception as e:
                if verbose:
                    warn(f"Failed to delete {db_name}.{tbl}: {str(e)}")
                else:
                    print(f"WARNING: Failed to delete {db_name}.{tbl}: {str(e)}")
        
        if verbose:
            success(f"Cleared {deleted_count} downstream tables. Kept {len(tables_to_keep)} core tables.")
        else:
            print(f"SUCCESS: Cleared {deleted_count} downstream tables. Kept {len(tables_to_keep)} core tables.")
        return True
    else:
        if verbose:
            info("No downstream tables found to delete. All tables are core tables.")
        else:
            print("No downstream tables found to delete. All tables are core tables.")
        return False

# COMMAND ----------

def count_no_match(arr_col):
    # Count the number of elements in an array column that equal 0 (no matching concept cases)
    if isinstance(arr_col, str):
        return F.size(F.expr(f"filter({arr_col}, x -> x = 0)"))
    else:
        # If it's already a Column, convert to string representation
        col_name = arr_col._jc.toString()
        return F.size(F.expr(f"filter({col_name}, x -> x = 0)"))

# COMMAND ----------

def load_features_with_regularity(input_db, penalty='L1', threshold=None):
    # Load archetype_features_yearly and add irregularity_score column
    spark = SparkSession.builder.getOrCreate()
    features = spark.table(f"{input_db}.archetype_features_yearly")

    if penalty == 'BOTH':
        # When using both, don't create irregularity_score column
        # The individual irregularity_l1 and irregularity_l2 columns will be used directly
        pass
    else:
        # Create a single irregularity_score column based on the selected penalty
        features = features.withColumn(
            "irregularity_score",
            F.when(F.lit(penalty) == "L2", F.col("irregularity_l2")).otherwise(F.col("irregularity_l1"))
        )
    
    # Only add is_irregular if threshold is provided (for rule-based methods)
    if threshold is not None:
        features = features.withColumn(
            "is_irregular",
            (F.col("irregularity_score") > F.lit(threshold)).cast("boolean")
        )

    return features

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Feature Engineering & Validation
# MAGIC
# MAGIC ### 6.1 Feature Loading and Regularity

# COMMAND ----------

def compute_regularity(
    gaps: Optional[List[float]],
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Compute irregularity scores from inter-visit gaps.

    Returns (irregularity_l1, irregularity_l2, central_diff).
    All three elements are None when gaps is None or too short (<= 1 interval).
    """
    if gaps is None or len(gaps) <= 1:
        return (None, None, None)

    gaps = np.array(gaps)
    mean_gap = np.mean(gaps)
    if mean_gap <= 1e-10:
        return (None, None, None)

    mean_variation_l1 = np.mean(np.abs((gaps / mean_gap) - 1))
    mean_square_variation_l2 = np.sqrt(np.mean((gaps / mean_gap - 1) ** 2))

    irregularity_l1 = mean_variation_l1
    irregularity_l2 = mean_square_variation_l2

    return (float(irregularity_l1), float(irregularity_l2), float(mean_gap))


def build_regularity_schema():
    """Return the schema used by the regularity UDF."""
    from pyspark.sql.types import StructType, StructField, DoubleType

    return StructType([
        StructField("irregularity_l1", DoubleType(), True),
        StructField("irregularity_l2", DoubleType(), True),
        StructField("central_diff", DoubleType(), True),
    ])


def make_compute_regularity_udf():
    """Return a Spark UDF for regularity metrics."""
    schema = build_regularity_schema()
    expected_names = ("irregularity_l1", "irregularity_l2", "central_diff")
    actual_names = tuple(field.name for field in schema.fields)
    if actual_names != expected_names:
        raise ValueError(
            f"Regularity schema mismatch. Expected {expected_names}, got {actual_names}."
        )
    return F.udf(compute_regularity, schema)

from dataclasses import dataclass


@dataclass
class FeaturePipelineState:
    """Fitted artifacts from prepare_feature_matrix.

    Bundle the scaler and winsorization caps so that callers can pass the
    fitted state explicitly when applying the same pipeline to new data,
    rather than threading two separate arguments through every call site.
    """
    scaler: Any
    winsor_caps: Dict[str, float]


def prepare_feature_matrix(
    features_df: pd.DataFrame,
    features_for_clustering: List[str],
    irregularity_penalty: IrregularityPenalty,
    winsorize_features: List[str] = None,
    apply_log_transform: bool = True,
    fit_scaler: bool = True,
    scaler: Optional[Any] = None,
    winsor_caps: Optional[Dict[str, float]] = None,
    percentile: float = 0.99,
    seed: int = 42
) -> Tuple[np.ndarray, Any, Dict[str, float]]:
    # Unified preprocessing pipeline: null filling, winsorization, log transformation, and scaling
    from sklearn.preprocessing import StandardScaler

    if irregularity_penalty not in ("L1", "L2", "BOTH"):
        raise ValueError(
            f"irregularity_penalty must be 'L1', 'L2', or 'BOTH'; got '{irregularity_penalty}'"
        )

    if winsorize_features is None:
        winsorize_features = ['visit_count', 'inpatient_visit_count', 'hospitalized_days']
    
    # Step 1: Fill nulls in irregularity features (<3 visits → 0.0; has_valid_regularity flag carries signal)
    # This is consistent with rules-based philosophy: no imputation, explicit handling of insufficient data.
    if irregularity_penalty == 'BOTH':
        features_df = features_df.copy()
        features_df['irregularity_l1'] = features_df['irregularity_l1'].fillna(0.0)
        features_df['irregularity_l2'] = features_df['irregularity_l2'].fillna(0.0)
    else:
        features_df = features_df.copy()
        irregularity_col = 'irregularity_l1' if irregularity_penalty == 'L1' else 'irregularity_l2'
        if irregularity_col in features_df.columns:
            features_df[irregularity_col] = features_df[irregularity_col].fillna(0.0)
        if 'irregularity_score' in features_df.columns:
            features_df['irregularity_score'] = features_df['irregularity_score'].fillna(0.0)
    
    # Step 2: Extract feature matrix and handle NaN/Inf
    feature_matrix = np.nan_to_num(
        features_df[features_for_clustering].values,
        nan=0.0, posinf=0.0, neginf=0.0
    )
    
    # Step 3: Winsorization (compute caps if not provided)
    if winsor_caps is None:
        winsor_caps = {}
        for i, feat in enumerate(features_for_clustering):
            if feat in winsorize_features:
                col_vals = feature_matrix[:, i]
                cap = float(np.nanpercentile(col_vals[np.isfinite(col_vals)], percentile * 100))
                winsor_caps[feat] = cap
    
    # Apply winsorization caps
    for i, feat in enumerate(features_for_clustering):
        if feat in winsorize_features and feat in winsor_caps:
            cap = winsor_caps[feat]
            if np.isfinite(cap):
                feature_matrix[:, i] = np.minimum(feature_matrix[:, i], cap)
    
    # Step 4: Log transformation
    if apply_log_transform:
        feature_matrix_transformed = feature_matrix.copy()
        for i, feat in enumerate(features_for_clustering):
            if feat in winsorize_features:
                feature_matrix_transformed[:, i] = np.log1p(feature_matrix_transformed[:, i])
    else:
        feature_matrix_transformed = feature_matrix
    
    # Step 5: Scaling
    if fit_scaler or scaler is None:
        scaler = StandardScaler()
        feature_matrix_scaled = scaler.fit_transform(feature_matrix_transformed)
    else:
        feature_matrix_scaled = scaler.transform(feature_matrix_transformed)
    
    # Add small jitter to prevent identical points (seeded, ~1e-10)
    np.random.seed(seed)
    feature_matrix_scaled += np.random.normal(0, 1e-10, feature_matrix_scaled.shape)
    
    return feature_matrix_scaled, scaler, winsor_caps

# COMMAND ----------

# MAGIC %md
# MAGIC ### 6.2 Data Validation Framework

# COMMAND ----------

def validate_clustering_input(
    feature_matrix: np.ndarray,
    n_samples: int = None,
    n_features: int = None,
    method_name: str = "clustering"
) -> Dict[str, Any]:
    # Validate feature matrix before clustering
    validation_results = {
        'n_samples': feature_matrix.shape[0],
        'n_features': feature_matrix.shape[1],
        'has_nan': np.isnan(feature_matrix).any(),
        'has_inf': np.isinf(feature_matrix).any(),
        'all_zero': (feature_matrix == 0).all(axis=1).any(),
        'constant_features': []
    }
    
    # Check for constant features (zero variance)
    for i in range(feature_matrix.shape[1]):
        if np.std(feature_matrix[:, i]) < 1e-10:
            validation_results['constant_features'].append(i)
    
    # Validate dimensions
    if n_samples is not None and feature_matrix.shape[0] != n_samples:
        raise ValueError(
            f"[{method_name}] Feature matrix has {feature_matrix.shape[0]} samples, "
            f"expected {n_samples}"
        )
    
    if n_features is not None and feature_matrix.shape[1] != n_features:
        raise ValueError(
            f"[{method_name}] Feature matrix has {feature_matrix.shape[1]} features, "
            f"expected {n_features}"
        )
    
    # Check for NaN/Inf (should be handled by preprocessing, but verify)
    
    # Check minimum sample size
    if feature_matrix.shape[0] < 10:
        raise ValueError(
            f"[{method_name}] Insufficient samples: {feature_matrix.shape[0]} "
            f"(minimum 10 required)"
        )
    
    return validation_results

def validate_clustering_results(
    labels: np.ndarray,
    method_name: str,
    min_cluster_size: int = 10,
    max_outlier_rate: float = 0.6,
    expected_n_clusters: int = None,
    allow_outliers: bool = True
) -> Dict[str, Any]:
    # Validate clustering results before saving
    n_samples = len(labels)
    unique_labels = np.unique(labels)
    n_outliers = np.sum(labels == -1) if allow_outliers else 0
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
    outlier_rate = n_outliers / n_samples if n_samples > 0 else 0.0
    
    validation_results = {
        'n_samples': n_samples,
        'n_clusters': n_clusters,
        'n_outliers': n_outliers,
        'outlier_rate': outlier_rate,
        'cluster_sizes': {},
        'warnings': [],
        'errors': []
    }
    
    # Check cluster sizes
    for label in unique_labels:
        if label == -1:
            continue
        cluster_size = np.sum(labels == label)
        validation_results['cluster_sizes'][int(label)] = int(cluster_size)
        
        if cluster_size < min_cluster_size:
            validation_results['warnings'].append(
                f"Cluster {label} has only {cluster_size} samples (minimum: {min_cluster_size})"
            )
    
    # Check number of clusters
    if n_clusters == 0:
        validation_results['errors'].append("No clusters found (all samples are outliers)")
    
    if expected_n_clusters is not None and n_clusters != expected_n_clusters:
        validation_results['warnings'].append(
            f"Found {n_clusters} clusters, expected {expected_n_clusters}"
        )
    
    # Check outlier rate
    if allow_outliers and outlier_rate > max_outlier_rate:
        validation_results['errors'].append(
            f"Outlier rate {outlier_rate:.1%} exceeds maximum {max_outlier_rate:.1%}"
        )
    
    if not allow_outliers and n_outliers > 0:
        validation_results['errors'].append(
            f"Found {n_outliers} outliers but method does not allow outliers"
        )
    
    # Check for label gaps (e.g., clusters 0, 1, 3 but missing 2)
    if n_clusters > 0:
        non_outlier_labels = [l for l in unique_labels if l != -1]
        if len(non_outlier_labels) > 0:
            min_label = min(non_outlier_labels)
            max_label = max(non_outlier_labels)
            expected_labels = set(range(min_label, max_label + 1))
            actual_labels = set(non_outlier_labels)
            missing_labels = expected_labels - actual_labels
            if missing_labels:
                validation_results['warnings'].append(
                    f"Missing cluster labels: {sorted(missing_labels)}"
                )
    
    # Raise errors if critical issues found
    if validation_results['errors']:
        error_msg = f"[{method_name}] Clustering validation failed:\n" + "\n".join(validation_results['errors'])
        raise ValueError(error_msg)
    
    return validation_results

# COMMAND ----------

# MAGIC %md
# MAGIC ### 6.3 Stratified Sampling

# COMMAND ----------

def handle_clustering_error(
    method_name: str,
    error: Exception,
    context: str = "",
    raise_error: bool = True
) -> None:
    # Standardized error handling for clustering operations
    import traceback
    
    error_type = type(error).__name__
    error_msg = str(error)
    
    enhanced_msg = f"[{method_name}] Error during {context if context else 'clustering'}:\n"
    enhanced_msg += f"  Type: {error_type}\n"
    enhanced_msg += f"  Message: {error_msg}"
    
    if VERBOSE:
        enhanced_msg += f"\n\nFull traceback:\n{traceback.format_exc()}"
    
    warn(enhanced_msg)
    
    if raise_error:
        raise type(error)(enhanced_msg) from error

def validate_before_clustering(
    features: DataFrame,
    features_for_clustering: List[str],
    irregularity_penalty: IrregularityPenalty,
    method_name: str,
    min_samples: int = 10
) -> bool:
    # Pre-flight validation checks before starting clustering
    try:
        # Check required features exist
        missing_features = [f for f in features_for_clustering if f not in features.columns]
        if missing_features:
            raise ValueError(
                f"[{method_name}] Missing required features: {missing_features}"
            )

        # Check sample size
        n_samples = features.count()
        if n_samples < min_samples:
            raise ValueError(
                f"[{method_name}] Insufficient samples: {n_samples} "
                f"(minimum {min_samples} required)"
            )
        
        # Check for required irregularity columns based on penalty
        if irregularity_penalty == 'BOTH':
            required_irregularity = ['irregularity_l1', 'irregularity_l2']
        elif irregularity_penalty == 'L1':
            required_irregularity = ['irregularity_l1']
        else:
            required_irregularity = ['irregularity_l2']
        
        missing_irregularity = [f for f in required_irregularity if f not in features.columns]
        if missing_irregularity:
            warn(
                f"[{method_name}] Missing irregularity columns: {missing_irregularity}. "
                f"Will fill with 0.0 during preprocessing."
            )
        
        info(f"[{method_name}] Pre-flight validation passed: {n_samples:,} samples, {len(features_for_clustering)} features")
        return True
        
    except Exception as e:
        handle_clustering_error(method_name, e, "pre-flight validation", raise_error=True)
        return False

# COMMAND ----------

# MAGIC %md
# MAGIC ### 6.4 Standardized Error Handling

# COMMAND ----------

def create_stratified_training_sample(features_spark, seed=42):
    # Create stratified training sample: one random year per person
    from pyspark.sql.window import Window

    person_window = Window.partitionBy("person_id").orderBy(F.rand(seed=seed))
    training_sample_ids = features_spark.select('person_id', 'year').distinct().withColumn(
        'row_num', F.row_number().over(person_window)
    ).filter(F.col('row_num') == 1).drop('row_num')

    # Add training_data flag
    training_sample_ids = training_sample_ids.withColumn('training_data', F.lit(1))

    return training_sample_ids

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Clustering Core Functions
# MAGIC
# MAGIC ### 7.1 Standard Cluster Analysis Pipeline

# COMMAND ----------

def run_standard_cluster_analysis(data_spark, method_name, cluster_col, feature_cols, output_db, using_cached=False):
    # Run all standard cluster analysis steps: profiles, medoids, separation, importance
    results = {}

    if using_cached:
        print(f"Loading cached analysis results for {method_name}")
        spark = SparkSession.builder.getOrCreate()
        try:
            results['profiles'] = spark.table(f"{output_db}.{method_name}_cluster_profiles")
            results['medoids'] = spark.table(f"{output_db}.{method_name}_medoids").toPandas()
            results['separation'] = spark.table(f"{output_db}.cluster_separation_metrics") \
                .filter(F.col('method') == method_name).toPandas()
            results['importance'] = spark.table(f"{output_db}.feature_importance") \
                .filter(F.col('method') == method_name).toPandas()
            print(f"Loaded cached {method_name} analysis")
            return results
        except Exception as e:
            print(f"Could not load cached results: {e}. Computing from scratch...")

    # 1. Cluster profiles
    section(f"{method_name}: cluster profiles")
    profile_table = create_cluster_profile_table(
        data_spark=data_spark,
        method_name=method_name,
        cluster_col=cluster_col,
        output_db=output_db
    )
    profile_table.write.format("delta").mode("overwrite") \
        .option("overwriteSchema", "true") \
        .saveAsTable(f"{output_db}.{method_name}_cluster_profiles")
    success(f"saved {output_db}.{method_name}_cluster_profiles")
    results['profiles'] = profile_table

    # 2. Medoids
    section(f"{method_name}: medoids")
    features_pdf = data_spark.select(['person_id', 'year', cluster_col] + feature_cols).toPandas()
    medoids_df = find_cluster_medoids(
        data_pdf=features_pdf,
        method_name=method_name,
        cluster_col=cluster_col,
        feature_cols=feature_cols
    )
    spark = SparkSession.builder.getOrCreate()
    spark.createDataFrame(medoids_df).write.format("delta").mode("overwrite") \
        .option("overwriteSchema", "true") \
        .saveAsTable(f"{output_db}.{method_name}_medoids")
    success(f"saved {output_db}.{method_name}_medoids ({len(medoids_df)})")
    results['medoids'] = medoids_df

    # 3. Separation metrics
    section(f"{method_name}: separation metrics")
    separation_df, db_index = calculate_separation_metrics(
        data_pdf=features_pdf,
        cluster_col=cluster_col,
        feature_cols=feature_cols
    )
    if separation_df is not None:
        # Delete-then-insert keyed on method: repeated runs replace rather than
        # accumulate duplicates for the same method.
        sep_table = f"{output_db}.cluster_separation_metrics"
        if spark.catalog.tableExists(sep_table):
            spark.sql(f"DELETE FROM {sep_table} WHERE method = '{method_name}'")
        spark.createDataFrame(separation_df) \
            .withColumn('cluster', F.col('cluster').cast('string')) \
            .withColumn('method', F.lit(method_name)) \
            .withColumn('davies_bouldin_index', F.lit(db_index)) \
            .write.format("delta").mode("append") \
            .option("mergeSchema", "true") \
            .saveAsTable(sep_table)
        success(f"saved separation metrics (DB={db_index:.3f})")
        results['separation'] = separation_df

    # 4. Feature importance
    section(f"{method_name}: feature importance")
    importance_df = calculate_feature_importance(
        data_spark=data_spark,
        method_name=method_name,
        cluster_col=cluster_col,
        feature_cols=feature_cols
    )
    # Delete-then-insert keyed on method (same rationale as separation metrics).
    imp_table = f"{output_db}.feature_importance"
    if spark.catalog.tableExists(imp_table):
        spark.sql(f"DELETE FROM {imp_table} WHERE method = '{method_name}'")
    spark.createDataFrame(importance_df) \
        .withColumn('cluster', F.col('cluster').cast('string')) \
        .write.format("delta").mode("append") \
        .option("mergeSchema", "true") \
        .saveAsTable(imp_table)
    success(f"saved feature importance ({len(importance_df)})")
    results['importance'] = importance_df

    success(f"{method_name}: analysis complete")
    return results

# COMMAND ----------

# MAGIC %md
# MAGIC ### 7.2 Cluster Label Export

# COMMAND ----------

def export_cluster_labels(data_spark, model_id, cluster_col, confidence_col, output_db,
                         entropy_col=None, is_outlier_col=None, cluster_label_col=None):
    # Export cluster labels to standardized cluster_labels_yearly table
    from datetime import datetime

    # For rule-based method, cluster is actually archetype string
    is_rules = (model_id == 'rules' or model_id == 'rules_adaptive')

    # Build select expression
    select_expr = [
        F.col('person_id'),
        F.col('year'),
        F.lit(model_id).alias('model_id')
    ]

    # Cluster number (None for rules since it uses labels)
    if is_rules:
        select_expr.append(F.lit(None).cast('int').alias('cluster'))
        select_expr.append(F.col(cluster_col).alias('cluster_label'))
    else:
        select_expr.append(F.col(cluster_col).cast('int').alias('cluster'))
        if cluster_label_col and cluster_label_col in data_spark.columns:
            select_expr.append(F.col(cluster_label_col).alias('cluster_label'))
        else:
            # Generate label from cluster number
            select_expr.append(
                F.when(F.col(cluster_col) == -1, F.lit('Outlier'))
                 .otherwise(F.concat(F.lit("Cluster "), F.col(cluster_col).cast("string")))
                 .alias('cluster_label')
            )

    # Confidence
    select_expr.append(F.col(confidence_col).cast('double').alias('confidence'))

    # Entropy (optional)
    if entropy_col and entropy_col in data_spark.columns:
        select_expr.append(F.col(entropy_col).cast('double').alias('entropy'))
    else:
        select_expr.append(F.lit(None).cast('double').alias('entropy'))

    # Outlier flag
    if is_outlier_col and is_outlier_col in data_spark.columns:
        select_expr.append(F.col(is_outlier_col).cast('int').alias('is_outlier'))
    else:
        # Infer from cluster == -1 for non-rules methods
        if not is_rules:
            select_expr.append(F.when(F.col(cluster_col) == -1, F.lit(1)).otherwise(F.lit(0)).alias('is_outlier'))
        else:
            select_expr.append(F.lit(0).alias('is_outlier'))

    # Timestamp
    select_expr.append(F.lit(datetime.now().isoformat()).alias('run_timestamp'))

    # Export
    export_df = data_spark.select(*select_expr)
    export_df.write.format('delta').mode('append').saveAsTable(f"{output_db}.cluster_labels_yearly")

    success(f"appended {model_id} labels to {output_db}.cluster_labels_yearly")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Cross-Method Analysis

# COMMAND ----------

def create_cluster_stats_table(df: pd.DataFrame, cluster_col: str, feature_cols: List[str]) -> pd.DataFrame:
    # Create cluster statistics with mean (IQR) format for display
    # Filter to valid clusters (exclude outliers)
    valid_df = df[
        (df[cluster_col].notna()) &
        (df[cluster_col] != -1)
    ].copy()
    
    if len(valid_df) == 0:
        return None
    
    # Compute statistics for each cluster
    cluster_stats = []
    for cluster_id in sorted(valid_df[cluster_col].unique()):
        cluster_data = valid_df[valid_df[cluster_col] == cluster_id]
        
        stats = {'cluster': cluster_id, 'n': len(cluster_data)}
        
        for col in feature_cols:
            if col in cluster_data.columns:
                vals = cluster_data[col].dropna()
                if len(vals) > 0:
                    mean_val = vals.mean()
                    q1 = vals.quantile(0.25)
                    q3 = vals.quantile(0.75)
                    # Format as "mean (Q1–Q3)"
                    stats[col] = f"{mean_val:.2f} ({q1:.2f}–{q3:.2f})"
                else:
                    stats[col] = "N/A"
        
        cluster_stats.append(stats)
    
    return pd.DataFrame(cluster_stats)

# COMMAND ----------

def analyze_cluster_composition(
    df: pd.DataFrame, 
    unsupervised_col: str, 
    reference_col: str, 
    dominant_threshold: float = 50.0
) -> Tuple[
    Optional[pd.DataFrame],
    Optional[pd.DataFrame],
    Optional[Dict[Any, str]],
    Optional[Dict[Any, Optional[str]]],
]:
    # Analyze how unsupervised clusters map to reference (e.g., rules-based) archetypes
    # Filter to valid data (exclude outliers and missing values)
    valid_df = df[
        df[reference_col].notna() & 
        df[unsupervised_col].notna() &
        (df[unsupervised_col] != -1)
    ].copy()
    
    if len(valid_df) == 0:
        return None, None, None, None
    
    # Create composition matrix (percentages)
    composition = pd.crosstab(
        valid_df[unsupervised_col],
        valid_df[reference_col],
        normalize='index'
    ) * 100
    
    # Create count matrix for reference
    composition_counts = pd.crosstab(
        valid_df[unsupervised_col],
        valid_df[reference_col]
    )
    
    # Build improved relabeling: use dominant archetype (>threshold) or "Mixed"
    relabeling = {}
    dominant_map = {}
    
    for cluster_id in composition_counts.index:
        if cluster_id not in composition.index:
            relabeling[cluster_id] = "Unknown"
            dominant_map[cluster_id] = None
            continue
            
        comp_row = composition.loc[cluster_id]
        # Find dominant archetype (>threshold)
        dominant = comp_row[comp_row >= dominant_threshold]
        
        if len(dominant) > 0:
            # Use dominant archetype
            dominant_arch = dominant.idxmax()
            relabeling[cluster_id] = dominant_arch
            dominant_map[cluster_id] = dominant_arch
        else:
            # No clear dominant - use top 2 if they sum to >threshold, otherwise "Mixed"
            top2 = comp_row.nlargest(2)
            if len(top2) > 0 and top2.sum() >= dominant_threshold:
                # Top 2 together are dominant
                relabeling[cluster_id] = " + ".join(top2.index.tolist())
                dominant_map[cluster_id] = None
            else:
                # Truly mixed
                relabeling[cluster_id] = "Mixed"
                dominant_map[cluster_id] = None
    
    return composition, composition_counts, relabeling, dominant_map

# COMMAND ----------

def calculate_pairwise_concordance(
    df: pd.DataFrame, 
    method1: str, 
    method2: str, 
    use_relabeled: bool = True
) -> Tuple[float, float, float]:
    # Calculate concordance metrics between two clustering methods
    # Determine column names
    def get_col(method):
        if method.startswith('rules'):
            return f'{method}_cluster'
        # For GMM methods, prefer label → relabeled → raw cluster
        for suffix in ['_label', '_relabeled', '_cluster']:
            col = f'{method}{suffix}'
            if col in df.columns:
                return col
        return f'{method}_cluster'
    
    if use_relabeled:
        col1 = get_col(method1)
        col2 = get_col(method2)
    else:
        col1 = f'{method1}_cluster'
        col2 = f'{method2}_cluster'
    
    if col1 not in df.columns or col2 not in df.columns:
        return np.nan, np.nan, np.nan
    
    valid_df = df[
        df[col1].notna() & df[col2].notna() &
        (df[col1] != 'Unknown') & (df[col2] != 'Unknown')
    ]
    
    if len(valid_df) < 10:
        return np.nan, np.nan, np.nan
    
    ari = adjusted_rand_score(valid_df[col1], valid_df[col2])
    from sklearn.metrics import adjusted_mutual_info_score
    ami = adjusted_mutual_info_score(valid_df[col1], valid_df[col2])
    agreement = (valid_df[col1] == valid_df[col2]).mean()
    
    return ari, ami, agreement

# COMMAND ----------

def get_method_cluster_column(method: str, df_columns: List[str]) -> Optional[str]:
    # Get the appropriate cluster column for a method, preferring human-readable labels
    if method.startswith('rules'):
        col = f'{method}_cluster'
        return col if col in df_columns else None
    
    # For GMM methods, prefer display label → relabeled → raw cluster
    for suffix in ['_label', '_relabeled', '_cluster']:
        col = f'{method}{suffix}'
        if col in df_columns:
            return col
    return None

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Method Configuration & Helpers
# MAGIC
# MAGIC ### 9.1 Method Configuration Dictionary

# COMMAND ----------

# Standard method configuration used across analysis
METHOD_CONFIG = {
    'rules_vanilla': {
        'table': 'rules_yearly',
        'cluster_col': 'archetype',
        'confidence_col': 'archetype_confidence',
        'medoids_table': 'rules_medoids'
    },
    'rules_adaptive': {
        'table': 'rules_adaptive_yearly',
        'cluster_col': 'archetype',
        'confidence_col': 'archetype_confidence',
        'medoids_table': 'rules_adaptive_medoids'
    },
    'gmm_4': {
        'table': 'gmm_4_yearly',
        'cluster_col': 'gmm_4_cluster',
        'confidence_col': 'gmm_4_confidence',
        'medoids_table': 'gmm_4_medoids'
    },
    'gmm_7': {
        'table': 'gmm_7_yearly',
        'cluster_col': 'gmm_7_cluster',
        'confidence_col': 'gmm_7_confidence',
        'medoids_table': 'gmm_7_medoids'
    }
}

def get_method_config(method: MethodName | str) -> Dict[str, str]:
    # Get configuration for a clustering method
    return METHOD_CONFIG.get(method, {})

# COMMAND ----------

def summarize_range(x: pd.Series) -> str:
    # Format range as min-max
    if len(x) == 0:
        return "N/A"
    return f"{x.min():.0f}–{x.max():.0f}"

def create_cluster_summary(
    df: pd.DataFrame, 
    cluster_col: str, 
    feature_cols: Optional[List[str]] = None
) -> Optional[pd.DataFrame]:
    # Create comprehensive cluster summary with means, medians, and ranges
    if feature_cols is None:
        feature_cols = get_profile_features()
    
    # Use available columns
    summary_cols = [c for c in feature_cols if c in df.columns]
    
    # Filter to valid clusters (exclude outliers)
    valid_df = df[
        (df[cluster_col].notna()) &
        (df[cluster_col] != -1)
    ].copy()
    
    if len(valid_df) == 0:
        return None
    
    # Compute statistics for each cluster
    cluster_stats = []
    for cluster_id in sorted(valid_df[cluster_col].unique()):
        cluster_data = valid_df[valid_df[cluster_col] == cluster_id]
        
        stats = {'cluster': cluster_id, 'n': len(cluster_data)}
        
        for col in summary_cols:
            if col in cluster_data.columns:
                stats[f'{col}_mean'] = cluster_data[col].mean()
                stats[f'{col}_median'] = cluster_data[col].median()
                stats[f'{col}_range'] = summarize_range(cluster_data[col])
        
        cluster_stats.append(stats)
    
    return pd.DataFrame(cluster_stats)

# COMMAND ----------

def build_flow_based_labels(
    composition_df: pd.DataFrame,
    rules_to_gmm: pd.DataFrame,
    fallback_labels: Dict,
    min_contrib: float = 5.0
) -> Dict:
    # Build flow-based labels showing percentage of rules archetypes flowing into each GMM cluster
    flow_label_map = {}
    
    for cluster_id in composition_df.index:
        if cluster_id not in rules_to_gmm.columns:
            flow_label_map[cluster_id] = fallback_labels.get(cluster_id, "Unknown")
            continue

        col_flow = rules_to_gmm[cluster_id]
        contrib = col_flow[col_flow >= min_contrib].sort_values(ascending=False)

        if len(contrib) == 0:
            flow_label_map[cluster_id] = fallback_labels.get(cluster_id, "Mixed")
        else:
            pieces = [f"{pct:.0f}% {arch}" for arch, pct in contrib.items()]
            flow_label_map[cluster_id] = " + ".join(pieces)
    
    return flow_label_map

# COMMAND ----------

def apply_relabeling_to_dataframe(
    df: pd.DataFrame,
    relabeling_maps: Dict[str, Dict]
) -> pd.DataFrame:
    # Apply relabeling maps to create relabeled and display label columns
    df = df.copy()
    
    for method, relabel_map in relabeling_maps.items():
        orig_col = f'{method}_cluster'
        
        if orig_col not in df.columns:
            continue
        
        # Clean label (no cluster ID prefix)
        new_col = f'{method}_relabeled'
        df[new_col] = df[orig_col].map(relabel_map).fillna('Unknown')
        
        # Display label: "<cluster_id>: <archetype_label>"
        disp_col = f'{method}_label'
        display_map = {cid: f"{cid}: {lbl}" for cid, lbl in relabel_map.items()}
        df[disp_col] = df[orig_col].map(display_map).fillna('Unknown')
    
    return df

# COMMAND ----------

def build_gmm_splitting_table(
    df: pd.DataFrame,
    gmm4_col: str,
    gmm7_col: str,
    relabeling_maps: Dict[str, Dict],
    threshold: float = 15.0
) -> pd.DataFrame:
    # Build table showing how GMM-4 clusters split into GMM-7 clusters
    gmm4_to_7 = pd.crosstab(df[gmm4_col], df[gmm7_col], normalize='index') * 100
    
    split_rows = []
    gmm4_labels = relabeling_maps.get('gmm_4', {})
    
    for gmm4_cluster in sorted(gmm4_to_7.index):
        row = gmm4_to_7.loc[gmm4_cluster]
        major_gmm7 = row[row >= threshold].sort_values(ascending=False)
        
        gmm4_label = gmm4_labels.get(gmm4_cluster, "N/A")
        
        if len(major_gmm7) > 1:
            splits_to = ", ".join([f"GMM7-{c} ({v:.0f}%)" for c, v in major_gmm7.items()])
            split_type = "Splits"
        elif len(major_gmm7) == 1:
            splits_to = f"GMM7-{major_gmm7.index[0]} ({major_gmm7.iloc[0]:.0f}%)"
            split_type = "Maps to"
        else:
            splits_to = "Distributed"
            split_type = "Diffuse"
        
        split_rows.append({
            'gmm_4_cluster': gmm4_cluster,
            'gmm_4_rules_label': gmm4_label,
            'split_type': split_type,
            'maps_to_gmm_7': splits_to
        })
    
    return pd.DataFrame(split_rows)

# COMMAND ----------

def build_concordance_matrices(
    df: pd.DataFrame,
    methods: List[str],
    use_relabeled: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Build concordance matrices for all method pairs
    ari_matrix = pd.DataFrame(index=methods, columns=methods)
    agreement_matrix = pd.DataFrame(index=methods, columns=methods)
    
    for m1 in methods:
        for m2 in methods:
            if m1 == m2:
                ari_matrix.loc[m1, m2] = 1.0
                agreement_matrix.loc[m1, m2] = 1.0
            else:
                ari, _, agree = calculate_pairwise_concordance(df, m1, m2, use_relabeled=use_relabeled)
                ari_matrix.loc[m1, m2] = ari
                agreement_matrix.loc[m1, m2] = agree
    
    return ari_matrix, agreement_matrix

# COMMAND ----------

def create_confusion_matrix_with_summary(
    df: pd.DataFrame,
    method1: str,
    method2: str,
    spark_session=None
) -> Optional[Dict]:
    # Create and display confusion matrix between two methods
    col1 = get_method_cluster_column(method1, df.columns.tolist())
    col2 = get_method_cluster_column(method2, df.columns.tolist())

    if col1 is None or col2 is None or col1 not in df.columns or col2 not in df.columns:
        return None

    valid_df = df[
        df[col1].notna() & df[col2].notna() &
        (df[col1] != 'Unknown') & (df[col2] != 'Unknown')
    ].copy()

    if len(valid_df) == 0:
        return None

    # Create confusion matrix (row-normalized)
    conf_matrix = pd.crosstab(valid_df[col1], valid_df[col2], normalize='index') * 100

    # Calculate overall agreement
    valid_df['match'] = valid_df[col1] == valid_df[col2]
    overall_agreement = valid_df['match'].mean() * 100
    
    # Display if spark_session provided
    if spark_session:
        print(f"\n{method1.upper()} vs {method2.upper()}")
        print(f"\nConfusion Matrix (% of {method1} cluster → {method2} cluster):")
        spark_session.createDataFrame(conf_matrix.round(1).reset_index()).show()
    
    return {
        'method1': method1,
        'method2': method2,
        'n_person_years': len(valid_df),
        'overall_agreement_pct': round(overall_agreement, 1),
        'confusion_matrix': conf_matrix
    }

# COMMAND ----------

def create_model_scorecard(
    df: pd.DataFrame,
    methods: List[str],
    cluster_summaries: Dict[str, pd.DataFrame],
    composition_results: Dict[str, pd.DataFrame],
    concordance_matrix: pd.DataFrame,
    agreement_matrix: pd.DataFrame,
    input_db: str,
    spark_session=None
) -> pd.DataFrame:
    """Create a model scorecard for each method.

    Guaranteed columns:
    - method
    - method_type

    Additional metric columns are conditionally populated and use np.nan
    when unavailable (for example, if summary/composition inputs are missing).
    """
    # Create comprehensive model scorecard with metrics for each method
    scorecard_rows = []
    
    # Load Davies-Bouldin index if available
    db_index_map = {}
    if spark_session:
        try:
            sep_tbl = spark_session.table(f"{input_db}.cluster_separation_metrics")
            sep_pdf = sep_tbl.select("method", "davies_bouldin_index").dropDuplicates(["method", "davies_bouldin_index"]).toPandas()
            for _, r in sep_pdf.iterrows():
                db_index_map[r["method"]] = float(r["davies_bouldin_index"])
        except Exception:
            pass
    
    total_person_years = len(df)
    
    for method in methods:
        row = {"method": method}
        row["method_type"] = "rules" if method.startswith("rules") else "gmm"
        
        cluster_col = f"{method}_cluster"
        if cluster_col not in df.columns:
            scorecard_rows.append(row)
            continue
        
        # Assignment / coverage
        valid_mask = df[cluster_col].notna()
        outlier_mask = valid_mask & (df[cluster_col] == -1)
        inlier_mask = valid_mask & (df[cluster_col] != -1)
        
        inliers = int(inlier_mask.sum())
        outliers = int(outlier_mask.sum())
        
        row["person_years_inliers"] = inliers
        row["pct_person_years_inliers"] = (100.0 * inliers / total_person_years) if total_person_years else np.nan
        row["pct_person_years_outliers"] = (100.0 * outliers / total_person_years) if total_person_years else 0.0
        
        # Cluster size statistics
        if method in cluster_summaries:
            summary = cluster_summaries[method]
            sizes = summary["n"]
            row["n_clusters"] = int(len(summary))
            row["median_cluster_size"] = float(sizes.median())
            row["min_cluster_size"] = int(sizes.min())
            row["max_cluster_size"] = int(sizes.max())
            mean_size = float(sizes.mean())
            row["cv_cluster_size"] = float(sizes.std() / mean_size) if mean_size > 0 else np.nan
        else:
            row["n_clusters"] = np.nan
            row["median_cluster_size"] = np.nan
            row["min_cluster_size"] = np.nan
            row["max_cluster_size"] = np.nan
            row["cv_cluster_size"] = np.nan
        
        # Purity vs rules_vanilla
        if method in composition_results:
            comp = composition_results[method]
            max_purity = comp.max(axis=1)
            row["mean_cluster_purity_vs_rules"] = float(max_purity.mean())
            row["median_cluster_purity_vs_rules"] = float(max_purity.median())
            row["n_pure_clusters_vs_rules"] = int((max_purity >= 70.0).sum())
            row["pct_pure_clusters_vs_rules"] = float(100.0 * (max_purity >= 70.0).mean())
        elif method == "rules_vanilla":
            n_clust = int(row.get("n_clusters", np.nan)) if not pd.isna(row.get("n_clusters")) else np.nan
            row["mean_cluster_purity_vs_rules"] = 100.0
            row["median_cluster_purity_vs_rules"] = 100.0
            row["n_pure_clusters_vs_rules"] = n_clust
            row["pct_pure_clusters_vs_rules"] = 100.0
        else:
            row["mean_cluster_purity_vs_rules"] = np.nan
            row["median_cluster_purity_vs_rules"] = np.nan
            row["n_pure_clusters_vs_rules"] = np.nan
            row["pct_pure_clusters_vs_rules"] = np.nan
        
        # Agreement vs reference methods
        for ref in ["rules_vanilla", "gmm_4", "gmm_7"]:
            ari_col = f"ARI_vs_{ref}"
            agree_col = f"Agree_vs_{ref}_pct"
            if (ref in concordance_matrix.columns) and (method in concordance_matrix.index):
                ari_val = concordance_matrix.loc[method, ref]
                agree_val = agreement_matrix.loc[method, ref]
                row[ari_col] = float(ari_val) if not pd.isna(ari_val) else np.nan
                row[agree_col] = float(agree_val * 100.0) if not pd.isna(agree_val) else np.nan
            else:
                row[ari_col] = np.nan
                row[agree_col] = np.nan
        
        # Patient-level agreement
        if "agreement_relabeled" in df.columns:
            mask = inlier_mask & df["agreement_relabeled"].notna()
            vals = df.loc[mask, "agreement_relabeled"]
            if len(vals) > 0:
                row["mean_patient_agreement"] = float(vals.mean())
                row["pct_patients_agreement_ge_0_75"] = float((vals >= 0.75).mean() * 100.0)
            else:
                row["mean_patient_agreement"] = np.nan
                row["pct_patients_agreement_ge_0_75"] = np.nan
        else:
            row["mean_patient_agreement"] = np.nan
            row["pct_patients_agreement_ge_0_75"] = np.nan
        
        # Davies-Bouldin index
        db_key = "rules" if method == "rules_vanilla" else method
        row["davies_bouldin_index"] = float(db_index_map.get(db_key, np.nan))
        
        scorecard_rows.append(row)
    
    return pd.DataFrame(scorecard_rows)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 9.2 Medoid Loading and Labeling

# COMMAND ----------

def load_medoids_with_labels(
    methods: List[str],
    input_db: str,
    relabeling_maps: Optional[Dict[str, Dict]] = None
) -> pd.DataFrame:
    # Load medoids from all methods and apply human-readable labels
    frames = []
    
    for m in methods:
        config = get_method_config(m)
        if not config:
            continue
        
        medoids_table = config.get('medoids_table')
        if not medoids_table:
            continue
        
        table_name = f"{input_db}.{medoids_table}"
        if not table_exists(input_db, medoids_table):
            continue
        
        cluster_col = config.get('cluster_col')

        # Load medoids table
        sdf = spark.table(table_name)
        available_cols = set(sdf.columns)

        # find_cluster_medoids saves as 'cluster', but config may specify 'archetype' etc.
        # Check for actual column in table, fallback to 'cluster' if config column missing
        actual_cluster_col = None
        if cluster_col and cluster_col in available_cols:
            actual_cluster_col = cluster_col
        elif 'cluster' in available_cols:
            actual_cluster_col = 'cluster'

        select_cols = ['person_id', 'year']
        if actual_cluster_col:
            select_cols.append(actual_cluster_col)
        
        sdf = sdf.select(*select_cols).dropna(subset=['person_id', 'year']).dropDuplicates(['person_id', 'year'])
        pdf = sdf.toPandas()
        pdf['method'] = m

        # Store raw cluster ID and apply labels
        if actual_cluster_col and actual_cluster_col in pdf.columns:
            pdf['cluster_id'] = pdf[actual_cluster_col]

            if relabeling_maps and m in relabeling_maps:
                # Try mapping as-is first, then with int conversion (handles type mismatches)
                remap = relabeling_maps[m]
                cluster_vals = pdf[actual_cluster_col]
                labels = cluster_vals.map(remap)
                if labels.isna().all() and len(cluster_vals.dropna()) > 0:
                    try:
                        labels = cluster_vals.astype(int).map(remap)
                    except (ValueError, TypeError):
                        pass
                pdf['cluster_label'] = labels.fillna(cluster_vals.astype(str))
            else:
                pdf['cluster_label'] = pdf[actual_cluster_col].astype(str)
        else:
            pdf['cluster_id'] = 'Unknown'
            pdf['cluster_label'] = 'Unknown'
        
        frames.append(pdf)
    
    if frames:
        return pd.concat(frames, ignore_index=True)
    return pd.DataFrame(columns=['person_id', 'year', 'method', 'cluster_id', 'cluster_label'])

# COMMAND ----------

def get_core_clustering_features(use_continuous_irregularity: bool = True, irregularity_penalty: IrregularityPenalty = 'L1', include_regularity_flag: bool = True) -> List[str]:
    # Return the canonical list of core clustering features shared across methods
    base_features = ['visit_count', 'inpatient_visit_count', 'hospitalized_days']

    # Add regularity flag for GMM clustering (allows sparse patients to cluster together)
    if include_regularity_flag and use_continuous_irregularity:
        base_features.append('has_valid_regularity')

    if irregularity_penalty == 'BOTH':
        # Include both L1 and L2 irregularity as separate features
        irregularity_features = ['irregularity_l1', 'irregularity_l2']
    elif use_continuous_irregularity:
        # Use the single irregularity_score column (which will be L1 or L2 based on penalty)
        irregularity_features = ['irregularity_score']
    else:
        # Binary version for rule-based methods
        irregularity_features = ['is_irregular']

    return base_features + irregularity_features

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Feature Sets & DataFrame Utilities
# MAGIC
# MAGIC ### 10.1 Core Feature Definitions

# COMMAND ----------

def get_profile_features() -> List[str]:
    # Return the canonical list of profiling features used in cluster profile tables
    return [
        'visit_count', 'inpatient_visit_count', 'hospitalized_days',
        'max_cci',
        'total_condition_count', 'total_drug_count', 'total_procedure_count',
        'total_measurement_count',
    ]


def get_domain_breadth_features() -> List[str]:
    """Return the four OMOP domain breadth count column names (within-year total concept occurrences)."""
    return [
        'total_condition_count',
        'total_drug_count',
        'total_procedure_count',
        'total_measurement_count',
    ]


def get_domain_label_map() -> Dict[str, str]:
    """Return display labels for each domain breadth column."""
    return {
        'total_condition_count': 'Conditions',
        'total_drug_count': 'Drugs',
        'total_procedure_count': 'Procedures',
        'total_measurement_count': 'Measurements',
    }


def get_gmm4_labels() -> Dict[int, str]:
    """GMM-4 cluster labels aligned with the first-submission manuscript."""
    return {
        0: "Moderate Inpatient",
        1: "Outpatient Irregular",
        2: "Outpatient Regular",
        3: "High Inpatient",
    }


# GMM-4 display order and colors (Wong 2011 colorblind-safe), low → high acuity
CLUSTER_ORDER: List[str] = [
    "Outpatient Irregular",
    "Outpatient Regular",
    "Moderate Inpatient",
    "High Inpatient",
]
CLUSTER_COLORS: Dict[str, str] = {
    "Outpatient Irregular": "#56B4E9",
    "Outpatient Regular":   "#009E73",
    "Moderate Inpatient":   "#E69F00",
    "High Inpatient":       "#D55E00",
}


# Manuscript output layout (edi_* prefix under outputs/)
EDI_OUTPUT_ROOT = "outputs"
EDI_FIGURES_DIR = f"{EDI_OUTPUT_ROOT}/figures"
EDI_TABLES_DIR = f"{EDI_OUTPUT_ROOT}/tables"
EDI_DATA_DIR = f"{EDI_OUTPUT_ROOT}/data"
EDI_REPORTS_DIR = f"{EDI_OUTPUT_ROOT}/reports"
EDI_MANIFEST_PATH = f"{EDI_OUTPUT_ROOT}/manifest/edi_artifact_manifest.csv"

EDI_FIG01_Q3_FOREST = "edi_fig01_q3_forest_odds_ratios.png"
EDI_FIG02_ABS_RESID = "edi_fig02_domain_abs_residual_violins.png"
EDI_FIG03_RANK_BISERIAL = "edi_fig03_rank_biserial_heatmap.png"
EDI_FIG04_DOMAIN_BREADTH = "edi_fig04_domain_breadth_violins.png"

EDI_TBL01B_CLASSIFICATION = "edi_tbl01b_patient_classification.csv"
EDI_TBL02_CLUSTER_PROFILES = "edi_tbl02_gmm4_cluster_profiles.csv"

EDI_DATA01_Q1 = "edi_data01_q1_pooled_spearman.csv"
EDI_DATA02_Q1B = "edi_data02_q1b_stratified_spearman.csv"
EDI_DATA03_Q2 = "edi_data03_q2_ols_interaction.csv"
EDI_DATA04_Q3 = "edi_data04_q3_multinomial_or.csv"
EDI_DATA05_DOMAIN_SUMMARY = "edi_data05_domain_count_summary.csv"
EDI_DATA06_DOMAIN_KW = "edi_data06_domain_kruskal_wallis.csv"
EDI_DATA07_DOMAIN_RB = "edi_data07_domain_rank_biserial.csv"
EDI_DATA08_SUMMARY = "edi_data08_summary_all_questions.csv"

EDI_REPORT_HTML = "edi_report_residual_analysis.html"

EDI_INDIVIDUAL_RECORD_TABLES = {EDI_TBL01B_CLASSIFICATION}


def get_edi_artifact_manifest_rows() -> List[Dict[str, str]]:
    """Registry rows for outputs/manifest/edi_artifact_manifest.csv."""
    rows = [
        {
            "artifact_id": "edi_fig01_q3_forest_odds_ratios",
            "kind": "fig",
            "manuscript_location": "Disease Burden — multinomial logit forest (vs Outpatient Regular)",
            "canonical_path": f"{EDI_FIGURES_DIR}/{EDI_FIG01_Q3_FOREST}",
            "legacy_path": "",
            "generating_script": "scripts/cohort_1/06_residual_analysis.py",
            "notes": "Q3 odds ratios with 95% Wald CI",
        },
        {
            "artifact_id": "edi_fig02_domain_abs_residual_violins",
            "kind": "fig",
            "manuscript_location": "Disease Burden — within-cluster absolute domain deviation",
            "canonical_path": f"{EDI_FIGURES_DIR}/{EDI_FIG02_ABS_RESID}",
            "legacy_path": "",
            "generating_script": "scripts/cohort_1/06_residual_analysis.py",
            "notes": "Q1 unsigned residuals by GMM-4 archetype",
        },
        {
            "artifact_id": "edi_fig03_rank_biserial_heatmap",
            "kind": "fig",
            "manuscript_location": "Disease Burden — rank-biserial vs full cohort",
            "canonical_path": f"{EDI_FIGURES_DIR}/{EDI_FIG03_RANK_BISERIAL}",
            "legacy_path": "",
            "generating_script": "scripts/cohort_1/06_residual_analysis.py",
            "notes": "Cluster-level data density characterization",
        },
        {
            "artifact_id": "edi_fig04_domain_breadth_violins",
            "kind": "fig",
            "manuscript_location": "Disease Burden — total domain concept counts by archetype",
            "canonical_path": f"{EDI_FIGURES_DIR}/{EDI_FIG04_DOMAIN_BREADTH}",
            "legacy_path": "",
            "generating_script": "scripts/cohort_1/06_residual_analysis.py",
            "notes": "Four OMOP domains (conditions, drugs, procedures, measurements)",
        },
        {
            "artifact_id": "edi_tbl01b_patient_classification",
            "kind": "tbl",
            "manuscript_location": "Table 1b — cross-domain residual grouping (2022 slice assembled by coauthor)",
            "canonical_path": f"{EDI_TABLES_DIR}/{EDI_TBL01B_CLASSIFICATION}",
            "legacy_path": "",
            "generating_script": "scripts/cohort_1/06_residual_analysis.py",
            "notes": "Per patient-year avg signed residual vs cluster mean; individual-level rows",
        },
        {
            "artifact_id": "edi_tbl02_gmm4_cluster_profiles",
            "kind": "tbl",
            "manuscript_location": "Supporting — GMM-4 utilization feature profiles",
            "canonical_path": f"{EDI_TABLES_DIR}/{EDI_TBL02_CLUSTER_PROFILES}",
            "legacy_path": "",
            "generating_script": "scripts/cohort_1/06_residual_analysis.py",
            "notes": "Exported from cohort_1.gmm_4_cluster_profiles Delta table",
        },
    ]
    data_artifacts = [
        (EDI_DATA01_Q1, "Q1 pooled Spearman (CCI vs domain residuals)"),
        (EDI_DATA02_Q1B, "Q1b stratified Spearman by cluster"),
        (EDI_DATA03_Q2, "Q2 OLS interaction terms (clustered SE)"),
        (EDI_DATA04_Q3, "Q3 multinomial logit odds ratios"),
        (EDI_DATA05_DOMAIN_SUMMARY, "Domain count median/IQR by cluster"),
        (EDI_DATA06_DOMAIN_KW, "Domain count Kruskal-Wallis by cluster"),
        (EDI_DATA07_DOMAIN_RB, "Domain count rank-biserial vs cohort"),
        (EDI_DATA08_SUMMARY, "Consolidated Q1–Q3 summary table"),
    ]
    for fname, note in data_artifacts:
        stem = fname.rsplit(".", 1)[0]
        rows.append({
            "artifact_id": stem,
            "kind": "data",
            "manuscript_location": "Backing data / supplement",
            "canonical_path": f"{EDI_DATA_DIR}/{fname}",
            "legacy_path": "",
            "generating_script": "scripts/cohort_1/06_residual_analysis.py",
            "notes": note,
        })
    rows.append({
        "artifact_id": "edi_report_residual_analysis",
        "kind": "report",
        "manuscript_location": "Internal HTML report (not submitted)",
        "canonical_path": f"{EDI_REPORTS_DIR}/{EDI_REPORT_HTML}",
        "legacy_path": "",
        "generating_script": "scripts/cohort_1/99_report.py",
        "notes": "Self-contained review bundle; figures embedded base64",
    })
    return rows


def write_edi_artifact_manifest(manifest_path: str | None = None) -> str:
    """Write edi_artifact_manifest.csv from the registry."""
    from pathlib import Path

    path = Path(manifest_path or EDI_MANIFEST_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = get_edi_artifact_manifest_rows()
    fieldnames = [
        "artifact_id", "kind", "manuscript_location", "canonical_path",
        "legacy_path", "generating_script", "notes",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return str(path)


def reorder_gmm4_for_display(df: pd.DataFrame, index_is_labels: bool = True) -> pd.DataFrame:
    """Reorder dataframe index by CLUSTER_ORDER (GMM-4 display order)."""
    if len(df) == 0:
        return df
    g4 = get_gmm4_labels()
    if index_is_labels:
        ordered = [lbl for lbl in CLUSTER_ORDER if lbl in df.index]
    else:
        ordered = [
            cid for lbl in CLUSTER_ORDER
            for cid in df.index
            if g4.get(cid) == lbl
        ]
    ordered += [x for x in df.index if x not in ordered]
    return df.reindex(ordered)


def get_gmm7_labels() -> Dict[int, str]:
    """Return human-readable labels for GMM-7 cluster IDs (cohort_1 archetypes)."""
    return {
        0: "Outpatient-Only – High Volume, Irregular Visits",
        1: "Sparse Use",
        2: "Moderate Inpatient – Few Admissions, Short Stays",
        3: "High Inpatient – Few Admissions, Prolonged Stays",
        4: "Outpatient-Only – Low Volume, Regular Visits",
        5: "High Inpatient – Many Admissions, High Volume",
        6: "Moderate Inpatient – Many Admissions, Short Stays",
    }


def get_feature_display_label_map() -> Dict[str, str]:
    """Return display labels for clustering features (stats tables, figures)."""
    return {
        'visit_count': 'Visit Count',
        'inpatient_visit_count': 'Inpatient Visits',
        'hospitalized_days': 'Hospitalized Days',
        'irregularity_l1': 'Irregularity (L1)',
        'irregularity_penalty': 'Irregularity (Penalty)',
        'max_cci': 'Max CCI',
        'total_condition_count': 'Total Conditions',
        'total_drug_count': 'Total Drugs',
        'total_procedure_count': 'Total Procedures',
        'total_measurement_count': 'Total Measurements',
    }


def get_core_figure_features() -> List[str]:
    """Return the four core features used in Figure 1 (cluster composition)."""
    return ['visit_count', 'inpatient_visit_count', 'hospitalized_days', 'irregularity_l1']


def prepare_medoid_timeline_data(
    vo: DataFrame,
    ud: DataFrame,
    medoid_keys: pd.DataFrame,
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Build daily and inpatient DataFrames for medoid patient timelines.

    Parameters
    ----------
    vo : Spark DataFrame
        visit_occurrence with person_id, start_date, visit_concept_name, visit_concept_id,
        visit_start_ts, visit_end_ts
    ud : Spark DataFrame
        unified_daily with person_id, date, visit_count, year
    medoid_keys : pd.DataFrame
        Columns: person_id, year, cluster_id, cluster_label

    Returns
    -------
    combined_daily : pd.DataFrame or None
        Daily visit data with cluster_id, cluster_label, y_label, day_of_year
    combined_ip : pd.DataFrame or None
        Inpatient stay data with y_label, start_doy, end_doy
    """
    daily_frames, ip_frames = [], []
    for _, row in medoid_keys.iterrows():
        pid, yr = int(row['person_id']), int(row['year'])
        cluster_id, cluster_label = row['cluster_id'], row['cluster_label']
        y_label = f"{cluster_label} | {pid}"

        daily_spark = ud.filter((F.col("person_id") == pid) & (F.col("year") == yr))
        visit_day = vo.filter((F.col("person_id") == pid) & (F.year("start_date") == yr)).groupBy(
            "person_id", F.col("start_date").alias("date")
        ).agg(F.first(F.trim(F.coalesce(F.col("visit_concept_name"), F.lit("Unknown")))).alias("visit_concept_name"))

        daily_pdf = daily_spark.join(visit_day, on=["person_id", "date"], how="left").toPandas()
        if len(daily_pdf) > 0:
            daily_pdf = daily_pdf.copy()
            daily_pdf['cluster_id'] = cluster_id
            daily_pdf['cluster_label'] = cluster_label
            daily_pdf['y_label'] = y_label
            daily_pdf['day_of_year'] = pd.to_datetime(daily_pdf['date']).dt.dayofyear
            daily_frames.append(daily_pdf)

        ip_spark = vo.filter(
            (F.col("person_id") == pid) & (F.year("start_date") == yr) & (F.col("visit_concept_id").isin([9201, 262]))
        ).select("person_id", F.col("visit_start_ts").alias("start_ts"), F.coalesce(F.col("visit_end_ts"), F.col("visit_start_ts")).alias("end_ts"))
        ip_pdf = ip_spark.toPandas()
        if len(ip_pdf) > 0:
            ip_pdf = ip_pdf.copy()
            ip_pdf['y_label'] = y_label
            ip_pdf['start_ts'] = pd.to_datetime(ip_pdf['start_ts'])
            ip_pdf['end_ts'] = pd.to_datetime(ip_pdf['end_ts'])
            ip_pdf['start_doy'] = ip_pdf['start_ts'].dt.dayofyear
            ip_pdf['end_doy'] = ip_pdf['end_ts'].dt.dayofyear
            ip_frames.append(ip_pdf)

    combined_daily = pd.concat(daily_frames, ignore_index=True) if daily_frames else None
    combined_ip = pd.concat(ip_frames, ignore_index=True) if ip_frames else None
    return combined_daily, combined_ip


def compute_cluster_mean_centered_residuals(
    df: pd.DataFrame,
    group_col: str,
    value_cols: List[str],
    prefix: str = 'domain_resid',
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Compute within-group mean-centered residuals for the given value columns.
    Returns (df with new residual columns added, list of residual column names).
    Missing values in value_cols are filled with 0 before centering.
    """
    df = df.copy()
    df[value_cols] = df[value_cols].fillna(0)
    means = df.groupby(group_col)[value_cols].transform('mean')
    resid_cols = []
    for col in value_cols:
        short = col.replace('total_', '').replace('_count', '')
        resid_col = f'{prefix}_{short}'
        df[resid_col] = df[col] - means[col]
        resid_cols.append(resid_col)
    return df, resid_cols


def rank_biserial(group_vals: np.ndarray, reference_vals: np.ndarray) -> float:
    """
    Rank-biserial correlation between a group subset and a reference sample.
    Ranges from -1 (group consistently lower) to +1 (group consistently higher).
    """
    from scipy.stats import mannwhitneyu
    n1, n2 = len(group_vals), len(reference_vals)
    if n1 < 2 or n2 < 2:
        return float('nan')
    # Guard: when both arrays are constant, mannwhitneyu may raise ValueError in some
    # scipy versions (zero-variance input). Return 0.0 when identical, nan when different.
    if np.all(group_vals == group_vals[0]) and np.all(reference_vals == reference_vals[0]):
        return 0.0 if group_vals[0] == reference_vals[0] else float('nan')
    u_stat, _ = mannwhitneyu(group_vals, reference_vals, alternative='two-sided')
    return 1 - (2 * u_stat) / (n1 * n2)


# COMMAND ----------

def fill_na_zero(data_spark: DataFrame, cols: List[str]) -> DataFrame:
    # Fill nulls with zeros for all provided columns in a single pass
    return data_spark.fillna({c: 0 for c in cols})

# COMMAND ----------

def spark_to_pandas_sampled(df: DataFrame, max_rows: int = 1_000_000, seed: int = 42) -> pd.DataFrame:
    # Convert a Spark DataFrame to Pandas with a size guard (samples if exceeds max_rows)
    count = df.count()
    if count > max_rows:
        frac = max_rows / float(count)
        return df.sample(withReplacement=False, fraction=frac, seed=seed).toPandas()
    return df.toPandas()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 10.2 Temporal Aggregation Helpers (Unified Tables)

# COMMAND ----------

def aggregate_temporal_unified(df, temporal_col):
    """
    Aggregate the unified daily table into a temporal grain (week/month/year).
    """
    rolled = df.groupBy("person_id", temporal_col).agg(
        F.first("gender").alias("gender"),
        F.first("race").alias("race"),
        F.first("ethnicity").alias("ethnicity"),
        F.min("birth_date").alias("birth_date"),
        F.sum("visit_count").alias("visit_count"),
        F.flatten(F.collect_list("visit_occurrence_ids")).alias("visit_occurrence_ids"),
        F.flatten(F.collect_list("condition_concept_ids")).alias("condition_concept_ids"),
        F.flatten(F.collect_list("condition_source_codes")).alias("condition_source_codes"),
        F.flatten(F.collect_list("condition_vocab_ids")).alias("condition_vocab_ids"),
        F.flatten(F.collect_list("drug_concept_ids")).alias("drug_concept_ids"),
        F.flatten(F.collect_list("measurement_concept_ids")).alias("measurement_concept_ids"),
        F.flatten(F.collect_list("procedure_concept_ids")).alias("procedure_concept_ids"),
        F.flatten(F.collect_list("observation_concept_ids")).alias("observation_concept_ids"),
        F.sum("condition_count").alias("condition_count"),
        F.sum("drug_count").alias("drug_count"),
        F.sum("measurement_count").alias("measurement_count"),
        F.sum("procedure_count").alias("procedure_count"),
        F.sum("observation_count").alias("observation_count"),
        F.sum("total_data_points").alias("total_data_points"),
        F.sum("condition_nomatch_ct").alias("condition_nomatch_ct"),
        F.sum("drug_nomatch_ct").alias("drug_nomatch_ct"),
        F.sum("measurement_nomatch_ct").alias("measurement_nomatch_ct"),
        F.sum("procedure_nomatch_ct").alias("procedure_nomatch_ct"),
        F.sum("observation_nomatch_ct").alias("observation_nomatch_ct"),
        F.sum("hospitalized_flag").alias("hospitalized_days"),
        F.max("hospitalized_flag").alias("any_hospitalized_flag"),
        F.count("date").alias("days_in_period"),
        F.sum(F.when(F.col("visit_count") > 0, 1).otherwise(0)).alias("active_days"),
        F.countDistinct("date").alias("unique_dates_with_data"),
        F.sum(F.when(F.col("visit_count") > 0, 1).otherwise(0)).alias("days_with_visits"),
        F.sum("has_clinical_concepts").alias("days_with_clinical_concepts"),
        F.collect_list(F.when(F.col("visit_count") > 0, F.col("date"))).alias("dates_with_visits"),
        F.collect_list(F.when(F.col("has_clinical_concepts") == 1, F.col("date"))).alias("dates_with_clinical_concepts"),
        F.min("date").alias("period_start_date"),
        F.max("date").alias("period_end_date"),
        F.min("age_at_day").alias("age_at_start"),
        F.max("cci_total").alias("cci_total"),
        F.max("cci_myocardial_infarction").alias("cci_myocardial_infarction"),
        F.max("cci_congestive_heart_failure").alias("cci_congestive_heart_failure"),
        F.max("cci_peripheral_vascular").alias("cci_peripheral_vascular"),
        F.max("cci_cerebrovascular").alias("cci_cerebrovascular"),
        F.max("cci_dementia").alias("cci_dementia"),
        F.max("cci_chronic_pulmonary").alias("cci_chronic_pulmonary"),
        F.max("cci_connective_tissue").alias("cci_connective_tissue"),
        F.max("cci_peptic_ulcer").alias("cci_peptic_ulcer"),
        F.max("cci_mild_liver").alias("cci_mild_liver"),
        F.max("cci_diabetes_uncomplicated").alias("cci_diabetes_uncomplicated"),
        F.max("cci_diabetes_complicated").alias("cci_diabetes_complicated"),
        F.max("cci_paralysis").alias("cci_paralysis"),
        F.max("cci_renal").alias("cci_renal"),
        F.max("cci_cancer").alias("cci_cancer"),
        F.max("cci_severe_liver").alias("cci_severe_liver"),
        F.max("cci_metastatic_cancer").alias("cci_metastatic_cancer"),
        F.max("cci_hiv_aids").alias("cci_hiv_aids"),
    )

    rolled = (
        rolled
        .withColumn("condition_unique_ct", F.size(F.array_distinct(F.col("condition_concept_ids"))))
        .withColumn("drug_unique_ct", F.size(F.array_distinct(F.col("drug_concept_ids"))))
        .withColumn("measurement_unique_ct", F.size(F.array_distinct(F.col("measurement_concept_ids"))))
        .withColumn("procedure_unique_ct", F.size(F.array_distinct(F.col("procedure_concept_ids"))))
        .withColumn("observation_unique_ct", F.size(F.array_distinct(F.col("observation_concept_ids"))))
        .withColumn(
            "total_unique_concepts",
            F.col("condition_unique_ct")
            + F.col("drug_unique_ct")
            + F.col("measurement_unique_ct")
            + F.col("procedure_unique_ct")
            + F.col("observation_unique_ct")
        )
        .withColumn(
            "total_nomatch_concepts",
            F.col("condition_nomatch_ct")
            + F.col("drug_nomatch_ct")
            + F.col("measurement_nomatch_ct")
            + F.col("procedure_nomatch_ct")
            + F.col("observation_nomatch_ct")
        )
    )
    return rolled.withColumnRenamed(temporal_col, "period")


def finalize_unified_temporal(df):
    """
    Finalize schema/column order for unified weekly/monthly/yearly tables.
    """
    return (
        df.withColumnRenamed("age_at_start", "age")
        .select(
            "person_id",
            "period",
            "period_start_date",
            "period_end_date",
            "birth_date",
            "age",
            "gender",
            "race",
            "ethnicity",
            "days_in_period",
            "active_days",
            "unique_dates_with_data",
            "days_with_visits",
            "days_with_clinical_concepts",
            "dates_with_visits",
            "dates_with_clinical_concepts",
            "visit_count",
            "hospitalized_days",
            "total_data_points",
            "total_unique_concepts",
            "total_nomatch_concepts",
            "cci_total",
            "cci_myocardial_infarction",
            "cci_congestive_heart_failure",
            "cci_peripheral_vascular",
            "cci_cerebrovascular",
            "cci_dementia",
            "cci_chronic_pulmonary",
            "cci_connective_tissue",
            "cci_peptic_ulcer",
            "cci_mild_liver",
            "cci_diabetes_uncomplicated",
            "cci_diabetes_complicated",
            "cci_paralysis",
            "cci_renal",
            "cci_cancer",
            "cci_severe_liver",
            "cci_metastatic_cancer",
            "cci_hiv_aids",
            "condition_concept_ids",
            "condition_source_codes",
            "condition_vocab_ids",
            "condition_count",
            "condition_unique_ct",
            "condition_nomatch_ct",
            "drug_concept_ids",
            "drug_count",
            "drug_unique_ct",
            "drug_nomatch_ct",
            "measurement_concept_ids",
            "measurement_count",
            "measurement_unique_ct",
            "measurement_nomatch_ct",
            "procedure_concept_ids",
            "procedure_count",
            "procedure_unique_ct",
            "procedure_nomatch_ct",
            "observation_concept_ids",
            "observation_count",
            "observation_unique_ct",
            "observation_nomatch_ct",
        )
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## 11. Adaptive Tuning Helpers
# MAGIC
# MAGIC Functions for automatic penalty and threshold selection in rules-based classification.

# COMMAND ----------

def empirical_penalty_characterization(
    features_spark,
    sample_size: int = None,
    show_plots: bool = True,
):
    """
    Comprehensive empirical comparison of L1 vs L2 irregularity penalties.

    Analyzes distribution characteristics, outlier sensitivity, and correlation
    to recommend the best penalty for the dataset.

    Parameters
    ----------
    features_spark : DataFrame
        Spark DataFrame with columns: person_id, year, visit_count, inpatient_visit_count,
        irregularity_l1, irregularity_l2
    sample_size : int, optional
        If provided, sample down to this many rows for memory efficiency
    show_plots : bool
        Whether to display diagnostic plots

    Returns
    -------
    dict with keys:
        recommendation: 'L1' or 'L2'
        confidence: 'HIGH', 'MODERATE', or 'LOW'
        agreement: float (average binary agreement across thresholds)
        correlation: float (Pearson correlation between L1 and L2)
        criteria: dict of individual criterion results
        threshold_stability_winner: str or None
    """
    spark = SparkSession.builder.getOrCreate()

    print("AUTOMATIC PENALTY SELECTION")

    reg_data_spark = features_spark.select(
        'person_id', 'year', 'visit_count', 'inpatient_visit_count',
        'irregularity_l1', 'irregularity_l2'
    ).filter(
        F.col('irregularity_l1').isNotNull() & F.col('irregularity_l2').isNotNull()
    )

    if sample_size is not None:
        total_count = reg_data_spark.count()
        if total_count > sample_size:
            sample_fraction = sample_size / total_count
            reg_data_spark = reg_data_spark.sample(withReplacement=False, fraction=sample_fraction, seed=42)
            print(f"Sampled {sample_size:,} from {total_count:,} rows for analysis")

    reg_data = reg_data_spark.toPandas()
    print(f"\nAnalyzing {len(reg_data):,} person-years with valid irregularity scores\n")

    # 1. Distribution characteristics
    print("1. DISTRIBUTION CHARACTERISTICS")
    print("-" * 70)

    stats_table = pd.DataFrame({
        'L1': reg_data['irregularity_l1'].describe(),
        'L2': reg_data['irregularity_l2'].describe()
    }).T
    print(stats_table.round(3))

    l1_cv = reg_data['irregularity_l1'].std() / reg_data['irregularity_l1'].mean()
    l2_cv = reg_data['irregularity_l2'].std() / reg_data['irregularity_l2'].mean()

    print(f"\nCoefficient of Variation:")
    print(f"  L1: {l1_cv:.3f}")
    print(f"  L2: {l2_cv:.3f}")
    print(f"  → {'L1' if l1_cv < l2_cv else 'L2'} has lower relative variance (more robust)")

    # 2. Outlier sensitivity
    print("\n2. OUTLIER SENSITIVITY")
    print("-" * 70)

    high_visit = reg_data[reg_data['visit_count'] >= reg_data['visit_count'].quantile(0.9)]

    l1_high_std, l2_high_std = 0.0, 0.0
    if len(high_visit) > 0:
        l1_high_std = high_visit['irregularity_l1'].std()
        l2_high_std = high_visit['irregularity_l2'].std()
        print(f"In high-visit cases (top 10%, n={len(high_visit):,}):")
        print(f"  L1 std: {l1_high_std:.3f}")
        print(f"  L2 std: {l2_high_std:.3f}")
        print(f"  → {'L1' if l1_high_std < l2_high_std else 'L2'} is more stable in complex cases")

    # 3. Correlation and agreement
    print("\n3. CORRELATION & AGREEMENT")
    print("-" * 70)

    corr_pearson = reg_data['irregularity_l1'].corr(reg_data['irregularity_l2'])
    corr_spearman = reg_data['irregularity_l1'].corr(reg_data['irregularity_l2'], method='spearman')

    print(f"Pearson correlation: {corr_pearson:.3f}")
    print(f"Spearman correlation: {corr_spearman:.3f}")

    thresholds_test = [0.3, 0.4, 0.5, 0.6, 0.7]
    agreements = []
    for thresh in thresholds_test:
        agree = ((reg_data['irregularity_l1'] > thresh) == (reg_data['irregularity_l2'] > thresh)).mean()
        agreements.append(agree)

    avg_agreement = np.mean(agreements)

    print(f"\nBinary classification agreement across thresholds:")
    for thresh, agree in zip(thresholds_test, agreements):
        print(f"  Threshold {thresh}: {agree:.1%}")
    print(f"  Average: {avg_agreement:.1%}")

    # 4. Visualization
    if show_plots:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        axes[0, 0].scatter(reg_data['irregularity_l1'], reg_data['irregularity_l2'], alpha=0.2, s=3, c='steelblue')
        axes[0, 0].plot([0, 1], [0, 1], 'r--', alpha=0.5, label='y=x')
        axes[0, 0].set_xlabel('L1 Irregularity')
        axes[0, 0].set_ylabel('L2 Irregularity')
        axes[0, 0].set_title(f'L1 vs L2 (r={corr_pearson:.3f})')
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)

        axes[0, 1].hist(reg_data['irregularity_l1'], bins=50, alpha=0.5, label='L1', density=True)
        axes[0, 1].hist(reg_data['irregularity_l2'], bins=50, alpha=0.5, label='L2', density=True)
        axes[0, 1].set_xlabel('Irregularity Score')
        axes[0, 1].set_ylabel('Density')
        axes[0, 1].set_title('Score Distributions')
        axes[0, 1].legend()
        axes[0, 1].grid(alpha=0.3)

        reg_data['diff'] = reg_data['irregularity_l1'] - reg_data['irregularity_l2']
        axes[1, 0].hist(reg_data['diff'], bins=50, edgecolor='black', alpha=0.7, color='purple')
        axes[1, 0].axvline(0, color='r', linestyle='--')
        axes[1, 0].set_xlabel('L1 - L2')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].set_title('Difference Distribution')
        axes[1, 0].grid(alpha=0.3)

        axes[1, 1].plot(thresholds_test, agreements, marker='o', linewidth=2)
        axes[1, 1].set_xlabel('Threshold')
        axes[1, 1].set_ylabel('Agreement Rate')
        axes[1, 1].set_title('L1/L2 Agreement vs Threshold')
        axes[1, 1].grid(alpha=0.3)
        axes[1, 1].set_ylim([min(agreements) - 0.05, 1.0])

        plt.tight_layout()
        plt.show()

    # 5. Decision criteria
    print("\nDECISION CRITERIA (EMPIRICAL)")

    criteria_scores = {}
    criteria_scores['robustness'] = 'L1' if l1_cv < l2_cv else 'L2'
    criteria_scores['agreement_sufficient'] = avg_agreement > 0.95

    if len(high_visit) > 0:
        criteria_scores['outlier_handling'] = 'L1' if l1_high_std < l2_high_std else 'L2'
    else:
        criteria_scores['outlier_handling'] = None

    print("\nEmpirical criteria evaluation:")
    print(f"  1. Robustness (CV): {criteria_scores['robustness']}")
    print(f"  2. High agreement (>95%): {'Yes' if criteria_scores['agreement_sufficient'] else 'No'}")
    if criteria_scores['outlier_handling'] is not None:
        print(f"  3. Outlier handling: {criteria_scores['outlier_handling']}")
    else:
        print(f"  3. Outlier handling: Insufficient data (tie)")

    # Final recommendation
    print("\nRECOMMENDATION")

    scored_criteria = {k: v for k, v in criteria_scores.items() if k != 'agreement_sufficient' and v is not None}
    l1_wins = sum(1 for v in scored_criteria.values() if v == 'L1')
    total_criteria = len(scored_criteria)

    is_split_decision = (
        total_criteria == 2 and
        l1_wins == 1 and
        criteria_scores['robustness'] != criteria_scores['outlier_handling']
    )

    threshold_stability_winner = None

    # Tiebreaker logic for split decisions
    if is_split_decision:
        print("\nSPLIT DECISION DETECTED - applying threshold stability tiebreaker")
        test_thresholds = np.arange(0.2, 0.8, 0.05)

        def _measure_stability(penalty_col):
            classify_test = reg_data.copy()
            results = []
            for t in test_thresholds:
                n_regular = ((classify_test[penalty_col] > t) & (classify_test['visit_count'] >= 2)).sum()
                pct_regular = n_regular / len(classify_test) * 100.0
                results.append(pct_regular)
            derivatives = np.abs(np.gradient(results, test_thresholds))
            smoothed = pd.Series(derivatives).rolling(window=3, center=True, min_periods=1).mean().values
            threshold_low = np.percentile(smoothed, 20)
            in_plateau = smoothed <= threshold_low
            max_width, current_width = 0, 0
            for is_plat in in_plateau:
                if is_plat:
                    current_width += 1
                    max_width = max(max_width, current_width)
                else:
                    current_width = 0
            plateau_width = max_width * 0.05
            avg_derivative = np.mean(smoothed)
            return plateau_width, avg_derivative

        l1_plateau, l1_deriv = _measure_stability('irregularity_l1')
        l2_plateau, l2_deriv = _measure_stability('irregularity_l2')

        if abs(l1_plateau - l2_plateau) > 0.05:
            threshold_stability_winner = 'L1' if l1_plateau > l2_plateau else 'L2'
        else:
            threshold_stability_winner = 'L1' if l1_deriv < l2_deriv else 'L2'

        criteria_scores['threshold_stability'] = threshold_stability_winner

    # Determine final recommendation
    if criteria_scores['agreement_sufficient'] and not is_split_decision:
        recommendation = criteria_scores['robustness']
        confidence = 'HIGH'
    elif is_split_decision and threshold_stability_winner is not None:
        l1_total = sum([1 for k, v in criteria_scores.items()
                       if k in ['robustness', 'outlier_handling', 'threshold_stability'] and v == 'L1'])
        recommendation = 'L1' if l1_total >= 2 else 'L2'
        confidence = 'HIGH'
    elif l1_wins == total_criteria:
        recommendation = 'L1'
        confidence = 'HIGH'
    elif l1_wins > total_criteria / 2:
        recommendation = 'L1'
        confidence = 'MODERATE'
    elif l1_wins == 0 and total_criteria > 0:
        recommendation = 'L2'
        confidence = 'MODERATE'
    else:
        recommendation = criteria_scores.get('robustness', 'L1')
        confidence = 'MODERATE'

    print(f"\nSelected penalty: {recommendation} (confidence: {confidence})")

    return {
        'recommendation': recommendation,              # 'L1' or 'L2'
        'confidence': confidence,                      # 'HIGH', 'MODERATE', or 'LOW'
        'agreement': avg_agreement,                    # float, average binary agreement across thresholds
        'correlation': corr_pearson,                   # float, Pearson r between L1 and L2 classification
        'criteria': criteria_scores,                   # dict of individual criterion results
        'threshold_stability_winner': threshold_stability_winner,  # 'L1', 'L2', or None
    }

# COMMAND ----------

def find_optimal_threshold(
    features_spark,
    penalty: str = 'L1',
    threshold_range=None,
    threshold_min: float = 0.0,
    threshold_max: float = 1.0,
    threshold_step: float = 0.05,
    show_plots: bool = True,
    use_sampling: bool = True,
):
    """
    Find optimal regularity threshold by identifying stability plateau.

    Uses a two-stage approach: (1) search on stratified sample, (2) validate on full data.

    Parameters
    ----------
    features_spark : DataFrame
        Spark DataFrame with columns: inpatient_visit_count, visit_count, irregularity_l1/l2
    penalty : str
        'L1' or 'L2' - which irregularity column to use
    threshold_range : array-like, optional
        Explicit range of thresholds to test. If None, derived from data quantiles.
    threshold_min, threshold_max, threshold_step : float
        Used when threshold_range is None and quantile derivation fails
    show_plots : bool
        Whether to display diagnostic plots
    use_sampling : bool
        If True, use stratified sample for search then validate on full data

    Returns
    -------
    dict with keys:
        optimal_threshold: float
        confidence: 'HIGH', 'MODERATE', or 'LOW'
        regular_pct, irregular_pct: float
        stability_min_derivative, stability_plateau_width: float
        js_divergence_sample_full: float
    """
    spark = SparkSession.builder.getOrCreate()

    print(f"AUTOMATIC THRESHOLD SELECTION (using {penalty} penalty)")

    irregularity_col = 'irregularity_l1' if penalty == 'L1' else 'irregularity_l2'

    # Stage 1: Stratified sampling for threshold search
    if use_sampling:
        print("\nSTAGE 1: Threshold search on stratified sample")
        print("-" * 70)
        window = Window.partitionBy('person_id').orderBy(F.rand(seed=42))
        features_sample = features_spark.withColumn('rn', F.row_number().over(window)) \
            .filter(F.col('rn') == 1).drop('rn')
        features_sample.cache()
        sample_count = features_sample.count()
        full_count = features_spark.count()
        features_for_search = features_sample
        print(f"Using {sample_count:,} unique persons (from {full_count:,} total rows)")
    else:
        features_for_search = features_spark

    # Data-driven threshold domain
    if threshold_range is None:
        try:
            thresh_subset = features_for_search.where(
                (F.col('inpatient_visit_count') == 0) &
                (F.col('visit_count') >= 2) &
                (F.col(irregularity_col).isNotNull())
            )
            qs = thresh_subset.selectExpr(
                f"percentile_approx({irregularity_col}, array(0.01, 0.99), 10000) as qs"
            ).collect()[0]['qs']
            q_min, q_max = float(qs[0]), float(qs[1])
            q_min = max(0.0, q_min)
            q_max = min(1.0, q_max)
            if not np.isfinite(q_min) or not np.isfinite(q_max) or q_min >= q_max:
                q_min, q_max = threshold_min, threshold_max
        except Exception:
            q_min, q_max = threshold_min, threshold_max
        step = max(threshold_step, 1e-3)
        threshold_range = np.round(np.arange(q_min, q_max + step/2.0, step), 6)

    # Compute distributions across thresholds
    thresholds_pdf = pd.DataFrame({'threshold': threshold_range})
    thresholds_sdf = spark.createDataFrame(thresholds_pdf)

    base = features_for_search.select(
        F.col('inpatient_visit_count').alias('ip'),
        F.col('visit_count').alias('vc'),
        F.col(irregularity_col).alias('reg')
    )
    joined = base.crossJoin(F.broadcast(thresholds_sdf))

    def _sum(cond):
        return F.sum(F.when(cond, F.lit(1)).otherwise(F.lit(0)))

    dist_agg = joined.groupBy('threshold').agg(
        _sum(F.col('ip') >= 2).alias('Multiple Complex Episodes'),
        _sum(F.col('ip') == 1).alias('Sporadic Complex Episodes'),
        _sum((F.col('ip') == 0) & (F.col('vc') <= 1)).alias('Sparse Use'),
        _sum((F.col('ip') == 0) & (F.col('vc') >= 4) & F.col('reg').isNotNull() & (F.col('reg') < F.col('threshold'))).alias('Regular Frequent'),
        _sum((F.col('ip') == 0) & (F.col('vc') >= 4) & (F.col('reg').isNull() | (F.col('reg') >= F.col('threshold')))).alias('Irregular Frequent'),
        _sum((F.col('ip') == 0) & (F.col('vc').between(2,3)) & F.col('reg').isNotNull() & (F.col('reg') < F.col('threshold'))).alias('Regular Infrequent'),
        _sum((F.col('ip') == 0) & (F.col('vc').between(2,3)) & (F.col('reg').isNull() | (F.col('reg') >= F.col('threshold')))).alias('Irregular Infrequent'),
        F.count(F.lit(1)).alias('total')
    ).orderBy('threshold')

    dist_pdf = dist_agg.toPandas()
    archetypes_cols = [
        'Multiple Complex Episodes', 'Sporadic Complex Episodes', 'Sparse Use',
        'Regular Frequent', 'Irregular Frequent', 'Regular Infrequent', 'Irregular Infrequent'
    ]
    for c in archetypes_cols:
        dist_pdf[f'{c}_pct'] = dist_pdf[c] / dist_pdf['total'] * 100.0

    records = []
    for _, r in dist_pdf.iterrows():
        for c in archetypes_cols:
            records.append({
                'threshold': float(r['threshold']),
                'archetype': c,
                'count': int(r[c]),
                'pct': float(r[f'{c}_pct'])
            })
    results_df = pd.DataFrame(records)

    if use_sampling:
        features_sample.unpersist()

    # Analyze stability
    print("\n1. THRESHOLD STABILITY ANALYSIS")
    print("-" * 70)

    regular_archetypes = ['Regular Frequent', 'Regular Infrequent', 'Irregular Frequent', 'Irregular Infrequent']
    regular_subset = results_df[results_df['archetype'].isin(regular_archetypes)].copy()
    regular_pivot = regular_subset.pivot(index='threshold', columns='archetype', values='pct').fillna(0.0).sort_index()
    thr_vals = regular_pivot.index.values

    for a in regular_archetypes:
        regular_pivot[f'{a}_deriv'] = np.gradient(regular_pivot[a].values, thr_vals) if len(thr_vals) > 1 else np.zeros(len(thr_vals))

    deriv_cols = [f'{a}_deriv' for a in regular_archetypes]
    stability_curve = pd.DataFrame({
        'threshold': thr_vals,
        'total_derivative': np.abs(regular_pivot[deriv_cols]).sum(axis=1).values
    })

    window_size = min(5, max(3, int(round(len(stability_curve) * 0.1))))
    stability_curve['smoothed_derivative'] = stability_curve['total_derivative'].rolling(window=window_size, center=True, min_periods=1).mean()
    min_val = stability_curve['smoothed_derivative'].min()
    tol = max(0.05, np.quantile(stability_curve['smoothed_derivative'], 0.1))
    within = stability_curve['smoothed_derivative'] <= (min_val + tol)

    best_start, best_end, cur_start = 0, 0, None
    for i, flag in enumerate(within.values):
        if flag and cur_start is None:
            cur_start = i
        if (not flag or i == len(within.values) - 1) and cur_start is not None:
            cur_end = i if not flag else i
            if (cur_end - cur_start) > (best_end - best_start):
                best_start, best_end = cur_start, cur_end
            cur_start = None

    plateau_slice = stability_curve.iloc[best_start:best_end+1]
    optimal_threshold = float(plateau_slice['threshold'].iloc[len(plateau_slice)//2])

    print(f"\nMost stable threshold (plateau center): {optimal_threshold:.2f}")
    print(f"Derivative min: {min_val:.2f}, Plateau width (steps): {best_end - best_start + 1}")

    # Stage 2: Validate on full dataset
    sample_dist = None
    full_dist = None

    if use_sampling:
        print("\nSTAGE 2: Validation on full dataset")
        print("-" * 70)

        def classify_at_threshold(ip, v, r, threshold):
            if ip >= 2:
                return 'Multiple Complex Episodes'
            if ip == 1:
                return 'Sporadic Complex Episodes'
            if v <= 1:
                return 'Sparse Use'
            if r is None:
                return 'Irregular Frequent' if v >= 4 else 'Irregular Infrequent'
            if v >= 4:
                return 'Regular Frequent' if r < threshold else 'Irregular Frequent'
            return 'Regular Infrequent' if r < threshold else 'Irregular Infrequent'

        classify_udf_full = F.udf(
            lambda ip, v, r: classify_at_threshold(ip, v, r, optimal_threshold),
            StringType()
        )

        classified_full = features_spark.withColumn(
            'archetype_temp',
            classify_udf_full(F.col('inpatient_visit_count'), F.col('visit_count'), F.col(irregularity_col))
        )

        dist_full_df = classified_full.groupBy('archetype_temp').count().orderBy('archetype_temp').toPandas()
        dist_full_df['pct'] = dist_full_df['count'] / dist_full_df['count'].sum() * 100

        sample_dist = results_df[np.isclose(results_df['threshold'], optimal_threshold)].set_index('archetype')['pct']
        full_dist = dist_full_df.set_index('archetype_temp')['pct']

        regular_pct = dist_full_df[dist_full_df['archetype_temp'].isin(['Regular Frequent', 'Regular Infrequent'])]['pct'].sum()
        irregular_pct = dist_full_df[dist_full_df['archetype_temp'].isin(['Irregular Frequent', 'Irregular Infrequent'])]['pct'].sum()
    else:
        threshold_dist = results_df[np.isclose(results_df['threshold'], optimal_threshold)]
        regular_pct = threshold_dist[threshold_dist['archetype'].isin(['Regular Frequent', 'Regular Infrequent'])]['pct'].sum()
        irregular_pct = threshold_dist[threshold_dist['archetype'].isin(['Irregular Frequent', 'Irregular Infrequent'])]['pct'].sum()

    print(f"\nAt threshold={optimal_threshold:.2f}:")
    print(f"  Regular: {regular_pct:.1f}%")
    print(f"  Irregular: {irregular_pct:.1f}%")

    # Visualization
    if show_plots:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        pivot = results_df.pivot(index='threshold', columns='archetype', values='pct').fillna(0)
        pivot.plot(kind='area', stacked=True, ax=axes[0, 0], alpha=0.7)
        axes[0, 0].axvline(optimal_threshold, color='red', linestyle='--', linewidth=2)
        axes[0, 0].set_xlabel('Threshold')
        axes[0, 0].set_ylabel('% of Population')
        axes[0, 0].set_title('Archetype Distribution vs Threshold')
        axes[0, 0].legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8)
        axes[0, 0].grid(alpha=0.3)

        for archetype in regular_archetypes:
            subset = regular_subset[regular_subset['archetype'] == archetype]
            axes[0, 1].plot(subset['threshold'], subset['pct'], marker='o', label=archetype, linewidth=2)
        axes[0, 1].axvline(optimal_threshold, color='red', linestyle='--', linewidth=2)
        axes[0, 1].set_xlabel('Threshold')
        axes[0, 1].set_ylabel('% of Population')
        axes[0, 1].set_title('Regularity-Dependent Archetypes')
        axes[0, 1].legend(fontsize=8)
        axes[0, 1].grid(alpha=0.3)

        axes[1, 0].plot(stability_curve['threshold'], stability_curve['total_derivative'], marker='o', linewidth=2)
        axes[1, 0].axvline(optimal_threshold, color='red', linestyle='--', linewidth=2)
        axes[1, 0].set_xlabel('Threshold')
        axes[1, 0].set_ylabel('Total Rate of Change')
        axes[1, 0].set_title('Stability Curve (Lower = More Stable)')
        axes[1, 0].grid(alpha=0.3)

        split_data = []
        for t in threshold_range:
            t_dist = results_df[results_df['threshold'] == t]
            t_reg = t_dist[t_dist['archetype'].isin(['Regular Frequent', 'Regular Infrequent'])]['pct'].sum()
            t_irreg = t_dist[t_dist['archetype'].isin(['Irregular Frequent', 'Irregular Infrequent'])]['pct'].sum()
            split_data.append({'threshold': t, 'Regular': t_reg, 'Irregular': t_irreg})
        split_df = pd.DataFrame(split_data)
        axes[1, 1].plot(split_df['threshold'], split_df['Regular'], marker='o', label='Regular', linewidth=2)
        axes[1, 1].plot(split_df['threshold'], split_df['Irregular'], marker='o', label='Irregular', linewidth=2)
        axes[1, 1].axvline(optimal_threshold, color='red', linestyle='--', linewidth=2)
        axes[1, 1].axhline(50, color='gray', linestyle=':', alpha=0.5)
        axes[1, 1].set_xlabel('Threshold')
        axes[1, 1].set_ylabel('% of Population')
        axes[1, 1].set_title('Regular vs Irregular Split')
        axes[1, 1].legend()
        axes[1, 1].grid(alpha=0.3)

        plt.tight_layout()
        plt.show()

    # Assess confidence
    def _js_divergence(p, q):
        p = np.asarray(p, dtype=float)
        q = np.asarray(q, dtype=float)
        p = p / (p.sum() + 1e-12)
        q = q / (q.sum() + 1e-12)
        m = 0.5 * (p + q)
        def _kl(a, b):
            a = np.clip(a, 1e-12, 1.0)
            b = np.clip(b, 1e-12, 1.0)
            return np.sum(a * np.log(a / b))
        return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)

    jsd = 0.0
    if sample_dist is not None and full_dist is not None:
        cats = sorted(set(sample_dist.index) | set(full_dist.index))
        s = np.array([sample_dist.get(c, 0.0) for c in cats])
        f = np.array([full_dist.get(c, 0.0) for c in cats])
        jsd = float(_js_divergence(s, f))

    plateau_width_steps = int(best_end - best_start + 1)
    plateau_width = float(plateau_width_steps * threshold_step)

    if plateau_width_steps >= 3 and jsd < 0.02:
        confidence = 'HIGH'
    elif plateau_width_steps >= 2 and jsd < 0.05:
        confidence = 'MODERATE'
    else:
        confidence = 'LOW'

    print(f"\nSelected threshold: {optimal_threshold:.2f} (confidence: {confidence})")

    return {
        'optimal_threshold': optimal_threshold,
        'confidence': confidence,
        'regular_pct': regular_pct,
        'irregular_pct': irregular_pct,
        'stability_min_derivative': float(min_val),
        'stability_plateau_width': float(plateau_width),
        'js_divergence_sample_full': float(jsd),
        'threshold_grid_min': float(threshold_range.min()) if len(threshold_range) > 0 else float('nan'),
        'threshold_grid_max': float(threshold_range.max()) if len(threshold_range) > 0 else float('nan'),
        'threshold_grid_step': float(threshold_step)
    }

# COMMAND ----------

# MAGIC %md
# MAGIC ## 12. Visualization Helpers
# MAGIC
# MAGIC Timeline plotting and color palette generation for patient journey visualizations

# COMMAND ----------

def build_global_palette_from_spark(table_name: str, label_col: str = "visit_concept_name", palette_name: str = "tab10") -> Dict[str, Any]:
    # Build a global color palette from visit concepts in a Spark table
    try:
        import seaborn as sns
        has_sns = True
    except Exception:
        has_sns = False

    spark = SparkSession.builder.getOrCreate()
    all_labels = (
        spark.table(table_name)
        .select(label_col)
        .distinct()
        .rdd.flatMap(lambda x: x)
        .collect()
    )
    all_labels = sorted([lbl if lbl is not None else "Unknown" for lbl in all_labels])

    if has_sns:
        color_list = sns.color_palette(palette_name, n_colors=len(all_labels))
    else:
        color_list = [f"C{i % 10}" for i in range(len(all_labels))]

    return {lbl: color_list[i % len(color_list)] for i, lbl in enumerate(all_labels)}

# COMMAND ----------

def plot_patient_timelines_exploded(
    daily_df: pd.DataFrame,
    years: Optional[List[int]] = None,
    page: int = 1,
    page_size: int = 20,
    palette: Optional[Dict[str, Any]] = None,
    inpatient_df: Optional[pd.DataFrame] = None,
    concat_col: str = "visit_concepts_concat",
    delim: str = ';',
    lane_height: float = 0.8,
    year_pad: float = 0.0,
    rect_color: str = '#F2F2F2',
    edge_color: str = '#7A7A7A',
    dot_size: float = 12,
    jitter: float = 0.12,
    figsize: tuple = (14, 10),
    person_col: str = 'person_id',
    date_col: str = 'date',
    death_col: str = 'deathdate'
):
    # Plot patient timelines showing daily visits with optional inpatient episodes and death markers
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from matplotlib.ticker import FixedLocator, FixedFormatter
    from matplotlib.lines import Line2D

    # Copy & basic normalization
    d = daily_df.copy()
    d[date_col] = pd.to_datetime(d[date_col])
    if "year" not in d.columns:
        d["year"] = d[date_col].dt.year

    # Explode concatenated concepts into one-per-row
    if concat_col in d.columns and "visit_concept_name" not in d.columns:
        tokens = (
            d[concat_col]
            .fillna("")
            .astype(str)
            .str.split(r"\s*"+delim+r"\s*")
            .apply(lambda L: [t for t in L if t != ""])
        )
        d = d.drop(columns=[concat_col]).assign(concept_list=tokens)
        d = d.explode('concept_list', ignore_index=True).rename(
            columns={"concept_list": "visit_concept_name"}
        )
    elif "visit_concept_name" not in d.columns:
        d["visit_concept_name"] = "Unknown"

    # Choose years to display
    if years is None:
        years = list(range(int(d["year"].min()), int(d["year"].max()) + 1))
    years = list(years)
    year_to_idx = {y: i for i, y in enumerate(years)}
    n_years = len(years)

    # Helper: map timestamp to continuous x-scale
    def _ts_to_x(ts_series):
        ts = pd.to_datetime(ts_series)
        yrs = ts.dt.year
        s0 = pd.to_datetime(yrs.astype(str) + "-01-01")
        s1 = pd.to_datetime((yrs + 1).astype(str) + "-01-01")
        frac = (ts - s0) / (s1 - s0)
        return yrs.map(year_to_idx).astype(float).values + frac.values

    # Restrict to display years & map date -> x
    d = d[d["year"].isin(years)].copy()
    d["x"] = _ts_to_x(d[date_col])

    # Paginate patients
    people = (
        d[[person_col]]
        .drop_duplicates()
        .sort_values(person_col)
        .reset_index(drop=True)
    )
    total = len(people)
    start_i = (page - 1) * page_size
    end_i = min(start_i + page_size, total)
    people_page = people.iloc[start_i:end_i, 0].tolist()

    dd = d[d[person_col].isin(people_page)].copy()
    y_map = {pid: i for i, pid in enumerate(people_page)}
    dd["y"] = dd[person_col].map(y_map).astype(float)
    dd["y_jit"] = dd["y"] + ((np.random.rand(len(dd)) - 0.5) * 2 * jitter if jitter else 0)

    # Colors: use global palette if provided
    label_col = "visit_concept_name"
    label_series = dd[label_col].fillna("Unknown")
    if palette is None:
        try:
            import seaborn as sns
            color_list = sns.color_palette("tab10", n_colors=len(label_series.unique()))
        except (ImportError, ValueError):
            color_list = [f"C{i % 10}" for i in range(len(label_series.unique()))]
        labels = sorted(label_series.unique())
        palette = {lbl: color_list[i % len(color_list)] for i, lbl in enumerate(labels)}

    dot_colors = label_series.map(palette).fillna("#555555")

    # Create plot
    fig, ax = plt.subplots(figsize=figsize)

    # Year rectangles per lane (background swim lanes)
    for pid, y in y_map.items():
        y0 = y - lane_height / 2.0
        for yi in range(n_years):
            ax.add_patch(
                Rectangle(
                    (yi, y0),
                    1 - 2 * year_pad,
                    lane_height,
                    facecolor=rect_color,
                    edgecolor=edge_color,
                    linewidth=1.0,
                    zorder=1,
                )
            )

    # Scatter dots for visits
    ax.scatter(dd["x"], dd["y_jit"], s=dot_size, c=dot_colors, zorder=5, linewidths=0)

    # Death marks (one per person if available)
    if death_col in d.columns and d[death_col].notna().any():
        death_df = (
            d.dropna(subset=[death_col])[[person_col, death_col]]
            .drop_duplicates([person_col])
            .copy()
        )
        death_df[death_col] = pd.to_datetime(death_df[death_col])
        death_df = death_df[death_df[person_col].isin(people_page)]
        if not death_df.empty:
            death_df["x"] = _ts_to_x(death_df[death_col])
            death_df["y"] = death_df[person_col].map(y_map).astype(float)
            ax.scatter(death_df["x"], death_df["y"], marker="x", color="red", s=80, zorder=6)

    # Inpatient spans (optional horizontal bars)
    if inpatient_df is not None and not inpatient_df.empty:
        ip = inpatient_df.copy()
        ip["start_ts"] = pd.to_datetime(ip["start_ts"])
        ip["end_ts"] = pd.to_datetime(ip["end_ts"])
        ip["end_ts"] = ip["end_ts"].fillna(ip["start_ts"] + pd.Timedelta(days=1))
        ip = ip[ip[person_col].isin(people_page)]
        if not ip.empty:
            ip["x0"] = _ts_to_x(ip["start_ts"])
            ip["x1"] = _ts_to_x(ip["end_ts"])
            ip["y"] = ip[person_col].map(y_map).astype(float)

            # Merge overlapping spans per person for clean bars
            merged = []
            for pid, grp in ip.sort_values([person_col, "x0", "x1"]).groupby(person_col):
                cur = None
                for _, r in grp.iterrows():
                    if cur is None or r["x0"] > cur["x1"]:
                        if cur is not None:
                            merged.append(cur)
                        cur = {"person_id": pid, "y": r["y"], "x0": r["x0"], "x1": r["x1"]}
                    else:
                        cur["x1"] = max(cur["x1"], r["x1"])
                if cur is not None:
                    merged.append(cur)

            # Draw magenta horizontal lines for inpatient episodes
            for r in merged:
                ax.hlines(r["y"] + 0.25, r["x0"], r["x1"], colors="#FF00FF", linewidth=5.0, alpha=0.9, zorder=4)

    # Axes & labels
    ax.set_ylim(-0.8, len(people_page) - 0.2)
    ax.set_xlim(0, n_years)
    ax.set_yticks(list(range(len(people_page))))
    ax.set_yticklabels(people_page)
    ax.xaxis.set_major_locator(FixedLocator(np.arange(n_years) + 0.5))
    ax.xaxis.set_major_formatter(FixedFormatter([str(y) for y in years]))
    ax.set_xlabel('Year')
    ax.set_ylabel('Person ID')
    ax.set_title(
        f"Patient Timelines ({years[0]}–{years[-1]})  "
        f"[Patients {start_i+1}–{end_i} of {total}]"
    )

    plt.tight_layout()

    # Legend (only categories present on this page)
    present = sorted(label_series.unique())
    legend_elements = [
        Line2D([0], [0], marker="o", color="w",
               label=lbl, markerfacecolor=palette.get(lbl, "#555555"), markersize=8)
        for lbl in present
    ]
    if death_col in d.columns and d[death_col].notna().any():
        legend_elements.append(
            Line2D([0], [0], marker="x", color="red", label="Death",
                   linestyle='None', markersize=8)
        )
    if inpatient_df is not None and not inpatient_df.empty:
        legend_elements.append(Line2D([0], [0], color="#FF00FF", lw=3.0, label="Inpatient stay"))

    if legend_elements:
        ax.legend(handles=legend_elements, title="Visit Concept",
                  bbox_to_anchor=(1.05, 1), loc="upper left")

    return fig, ax

# COMMAND ----------

def plot_medoid_timelines(
    daily_df: pd.DataFrame,
    inpatient_df: Optional[pd.DataFrame] = None,
    method_name: str = "Method",
    palette: Optional[Dict[str, Any]] = None,
    figsize: tuple = (14, 8),
    dot_size: float = 25,
    jitter: float = 0.08
):
    # Plot medoid timelines with day-of-year x-axis (Jan-Dec) and cluster|person_id on y-axis
    from matplotlib.patches import Rectangle
    from matplotlib.ticker import FixedLocator, FixedFormatter
    from matplotlib.lines import Line2D

    # Order lanes by human-readable cluster label, then person_id (embedded in y_label)
    if 'cluster_label' in daily_df.columns:
        label_order = (
            daily_df[['cluster_label', 'y_label']]
            .drop_duplicates()
            .sort_values(['cluster_label', 'y_label'])['y_label']
            .tolist()
        )
    else:
        # Fallback to cluster_id ordering if labels are unavailable
        label_order = (
            daily_df[['cluster_id', 'y_label']]
            .drop_duplicates()
            .sort_values('cluster_id')['y_label']
            .tolist()
        )
    y_map = {lbl: i for i, lbl in enumerate(label_order)}
    n_lanes = len(label_order)

    fig, ax = plt.subplots(figsize=figsize)
    if n_lanes == 0:
        ax.set_title(f"{method_name.upper()} - No medoid data available")
        return fig, ax

    d = daily_df.copy()
    d['y'] = d['y_label'].map(y_map).astype(float)
    d['y_jit'] = d['y'] + ((np.random.rand(len(d)) - 0.5) * 2 * jitter if jitter else 0)

    if 'visit_concept_name' not in d.columns:
        d['visit_concept_name'] = 'Unknown'
    label_series = d['visit_concept_name'].fillna('Unknown')

    labels = sorted(label_series.unique())
    if palette is None:
        palette = {lbl: sns.color_palette("tab10", len(labels))[i % 10] for i, lbl in enumerate(labels)}

    # Assign distinct marker shapes per visit concept to improve distinguishability
    marker_cycle = ['o', 's', '^', 'D', 'P', 'X', 'v', '<', '>']
    marker_map = {lbl: marker_cycle[i % len(marker_cycle)] for i, lbl in enumerate(labels)}

    # Lane backgrounds
    for y in range(n_lanes):
        ax.add_patch(Rectangle((1, y - 0.4), 365, 0.8, facecolor='#F2F2F2', edgecolor='#7A7A7A', lw=0.5, zorder=1))

    # Scatter visits by concept with distinct colors and marker shapes
    for lbl in labels:
        mask = (label_series == lbl)
        if not mask.any():
            continue
        ax.scatter(
            d.loc[mask, 'day_of_year'],
            d.loc[mask, 'y_jit'],
            s=dot_size,
            c=[palette.get(lbl, '#555555')],
            marker=marker_map.get(lbl, 'o'),
            zorder=5,
            linewidths=0,
        )

    # Inpatient bars
    if inpatient_df is not None and len(inpatient_df) > 0:
        ip = inpatient_df.copy()
        ip['y'] = ip['y_label'].map(y_map)
        ip = ip.dropna(subset=['y'])
        for _, r in ip.iterrows():
            ax.hlines(r['y'] + 0.25, r['start_doy'], r['end_doy'], colors='#FF00FF', lw=5.0, alpha=0.9, zorder=4)

    ax.set_xlim(1, 366)
    ax.set_ylim(-0.6, n_lanes - 0.4)
    ax.set_yticks(list(range(n_lanes)))
    ax.set_yticklabels(label_order, fontsize=9)
    ax.set_ylabel('Cluster | Person ID')

    month_starts = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
    ax.xaxis.set_major_locator(FixedLocator(month_starts))
    ax.xaxis.set_major_formatter(FixedFormatter(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']))
    ax.set_xlabel('Month (within training year)')
    ax.set_title(f'{method_name.upper()} - Medoid Patient Timelines', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')

    # Legend
    present = sorted(label_series.unique())[:15]
    handles = [
        Line2D(
            [0], [0],
            marker=marker_map.get(lbl, 'o'),
            color='w',
            label=lbl,
            markerfacecolor=palette.get(lbl, '#555555'),
            markersize=8,
            markeredgecolor='none',
        )
        for lbl in present
    ]
    if inpatient_df is not None and len(inpatient_df) > 0:
        handles.append(Line2D([0], [0], color='#FF00FF', lw=3.0, label='Inpatient stay'))
    if handles:
        ax.legend(handles=handles, title='Visit Concept', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=7)

    plt.tight_layout()
    return fig, ax

# COMMAND ----------

# MAGIC %md
# MAGIC ## 13. GMM Training Helpers
# MAGIC
# MAGIC Functions for training and predicting with Gaussian Mixture Models.

# COMMAND ----------

def train_and_predict_gmm(
    k_value: int,
    model_id: str,
    feature_matrix_scaled: np.ndarray,
    feature_matrix_all_scaled: np.ndarray,
    clustering_pdf_all: pd.DataFrame,
    features,
    output_db: str,
    scaler=None,
    winsor_caps=None,
    covariance_type: str = "full",
    n_init: int = 5,
    max_iter: int = 200,
    random_state: int = 42,
    using_cached: bool = False,
    save_bundle: bool = True,
    bundle_dir: str = "/dbfs/mnt/models/cohort_1",
):
    """
    Train GMM model and predict clusters for all person-years.

    Parameters
    ----------
    k_value : int
        Number of clusters
    model_id : str
        Identifier for this model (e.g., "gmm_4", "gmm_7")
    feature_matrix_scaled : ndarray
        Scaled feature matrix for training
    feature_matrix_all_scaled : ndarray
        Scaled feature matrix for all data (prediction)
    clustering_pdf_all : DataFrame
        Pandas DataFrame with person_id, year, training_data columns
    features : DataFrame
        Spark DataFrame with full feature set
    output_db : str
        Database name for output table
    scaler : optional
        Fitted scaler to save in bundle
    winsor_caps : optional
        Winsorization caps to save in bundle
    covariance_type, n_init, max_iter, random_state : GMM hyperparameters
    using_cached : bool
        If True, try to load from cached table first
    save_bundle : bool
        If True, save model bundle to DBFS for reuse
    bundle_dir : str
        Directory for model bundles

    Returns
    -------
    gmm_output : Spark DataFrame
        Full output with cluster assignments
    results_pdf : Pandas DataFrame
        Results with person_id, year, cluster, confidence, entropy
    gmm_model : GaussianMixture or None
        Trained model (None if loaded from cache)
    """
    from sklearn.mixture import GaussianMixture
    from scipy.stats import entropy
    import cloudpickle

    spark = SparkSession.builder.getOrCreate()

    output_table = f"{model_id}_yearly"
    cluster_col = f"{model_id}_cluster"
    confidence_col = f"{model_id}_confidence"
    entropy_col = f"{model_id}_entropy"
    label_col = f"{model_id}_cluster_label"
    prior_col = f"prior_year_{model_id}_cluster"

    # Check for cached results
    if using_cached:
        try:
            gmm_output = spark.table(f"{output_db}.{output_table}")
            required_cols = {"person_id", "year", cluster_col, confidence_col, entropy_col}
            if (required_cols - set(gmm_output.columns)):
                raise ValueError(f"Cached {output_table} missing required columns")

            if "training_data" not in gmm_output.columns:
                gmm_output = gmm_output.withColumn("training_data", F.lit(0))

            results_pdf = gmm_output.select(
                "person_id", "year", "training_data", cluster_col, confidence_col, entropy_col
            ).toPandas()
            info(f"Loaded cached {model_id.upper()} clustering")
            return gmm_output, results_pdf, None
        except Exception as cache_err:
            using_cached = False
            # If cache load failed, we need training/prediction matrices available.
            if feature_matrix_scaled is None or feature_matrix_all_scaled is None or clustering_pdf_all is None:
                raise ValueError(
                    f"Failed to load cached {output_db}.{output_table} ({cache_err}) and no in-memory matrices "
                    f"were provided to retrain {model_id}. Set RETRAIN_MODEL=True to recompute matrices."
                ) from cache_err

    # Train model
    info(f"Training GMM with K={k_value}...")
    gmm_model = GaussianMixture(
        n_components=k_value,
        covariance_type=covariance_type,
        n_init=n_init,
        max_iter=max_iter,
        random_state=random_state,
        verbose=1
    )

    gmm_model.fit(feature_matrix_scaled)
    info(f"{model_id.upper()} trained: converged={gmm_model.converged_}, BIC={gmm_model.bic(feature_matrix_scaled):.2f}")

    # Predict on full dataset
    info(f"Predicting clusters for all person-years...")
    cluster_labels_all = gmm_model.predict(feature_matrix_all_scaled)
    cluster_probs_all = gmm_model.predict_proba(feature_matrix_all_scaled)
    cluster_confidence_all = cluster_probs_all.max(axis=1)
    cluster_entropy_all = [entropy(p + 1e-10) for p in cluster_probs_all]
    info(f"Prediction complete for {len(cluster_labels_all):,} person-years")

    # Validate clustering results
    try:
        validation_results = validate_clustering_results(
            labels=cluster_labels_all,
            method_name=model_id,
            min_cluster_size=10,
            max_outlier_rate=0.0,
            expected_n_clusters=k_value,
            allow_outliers=False
        )
        info(f"Validation passed: {validation_results['n_clusters']} clusters, "
             f"cluster sizes: {min(validation_results['cluster_sizes'].values())}-{max(validation_results['cluster_sizes'].values())}")
    except Exception as e:
        handle_clustering_error(model_id, e, "result validation", raise_error=False)

    # Create results DataFrame
    results_pdf = pd.DataFrame({
        'person_id': clustering_pdf_all['person_id'],
        'year': clustering_pdf_all['year'],
        'training_data': clustering_pdf_all['training_data'],
        cluster_col: cluster_labels_all,
        confidence_col: cluster_confidence_all,
        entropy_col: cluster_entropy_all
    })

    # Join back to features and save
    person_window = Window.partitionBy("person_id").orderBy("year")

    gmm_output = features.join(
        spark.createDataFrame(results_pdf).select('person_id', 'year', cluster_col, confidence_col, entropy_col),
        on=['person_id', 'year'],
        how='left'
    ).withColumn(
        label_col,
        F.concat(F.lit("Cluster "), F.col(cluster_col).cast("string"))
    ).withColumn(
        prior_col,
        F.lag(cluster_col, 1).over(person_window)
    )

    gmm_output.write.format("delta").mode("overwrite").option("overwriteSchema", "true") \
        .saveAsTable(f"{output_db}.{output_table}")
    spark.sql(f"OPTIMIZE {output_db}.{output_table} ZORDER BY (person_id)")
    info(f"Saved {output_db}.{output_table}")

    # Save model bundle for reuse
    if save_bundle and scaler is not None:
        bundle = {
            "model": gmm_model,
            "scaler": scaler,
            "winsor_caps": winsor_caps,
        }
        model_bundle_dir = f"{bundle_dir}/{model_id}"
        bundle_path = f"{model_bundle_dir}/gmm_bundle.pkl"
        try:
            # Ensure directory exists (dbutils.fs.mkdirs for DBFS)
            import os
            os.makedirs(model_bundle_dir, exist_ok=True)
            with open(bundle_path, "wb") as f:
                cloudpickle.dump(bundle, f)
            info(f"Saved {model_id.upper()} bundle to {bundle_path}")
        except Exception as e:
            warn(f"Could not save bundle: {e}")

    return gmm_output, results_pdf, gmm_model
