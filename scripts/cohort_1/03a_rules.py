# Databricks notebook source
# MAGIC %md
# MAGIC **Purpose**: Rule-based utilization archetypes (vanilla default; optional adaptive)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Setup and Configuration

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.1 Import Libraries

# COMMAND ----------

from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, cohen_kappa_score

from pyspark.sql import functions as F
from pyspark.sql.window import Window

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.2 Define Configuration Parameters

# COMMAND ----------

# Vanilla settings
VANILLA_PENALTY = "L1"
VANILLA_THRESHOLD = 0.5

# Run control — adaptive rules are supplement-only; keep False for manuscript pipeline
RUN_ADAPTIVE = False

# Adaptive settings (ignored if RUN_ADAPTIVE=False)
AUTO_TUNE_PENALTY = True      # Set to True to automatically select L1 vs L2
AUTO_TUNE_THRESHOLD = True    # Set to True to automatically find optimal threshold
MANUAL_PENALTY = "L1"         # Fallback if AUTO_TUNE_PENALTY=False
MANUAL_THRESHOLD = 0.5        # Fallback if AUTO_TUNE_THRESHOLD=False

# Advanced tuning configuration (adaptive only)
THRESHOLD_SEARCH_MIN = 0.0    # Minimum threshold to test
THRESHOLD_SEARCH_MAX = 1.0    # Maximum threshold to test
THRESHOLD_SEARCH_STEP = 0.05  # Step size for threshold search
SHOW_TUNING_PLOTS = True      # Display diagnostic plots during tuning
PENALTY_SAMPLE_SIZE = None    # Max rows for penalty analysis (None = all data)
USE_THRESHOLD_SAMPLING = True # Two-stage: sample for search, validate on full

# Model caching configuration
RETRAIN_MODEL = False  # Use cached results when available; set True to force re-computation
CLEAR_CACHE = False    # Keep cached results; set True to clear before running

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.3 Load Shared Utilities

# COMMAND ----------

# MAGIC %run ./99_utils

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.4 Configure Logging and Optimization

# COMMAND ----------

VERBOSE = get_verbose(default=True)
gate_prints(VERBOSE)
configure_spark_optimizations()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.5 Cache Management

# COMMAND ----------

output_db = "cohort_1"

# Cache tables (vanilla + adaptive)
CACHE_TABLES = [
    f"{output_db}.rules_yearly",
    f"{output_db}.rules_cluster_profiles",
    f"{output_db}.rules_medoids",
    f"{output_db}.rules_adaptive_yearly",
    f"{output_db}.rules_adaptive_cluster_profiles",
    f"{output_db}.rules_adaptive_medoids",
]

if CLEAR_CACHE:
    spark.sql(f"DROP TABLE IF EXISTS {output_db}.cluster_separation_metrics")
    spark.sql(f"DROP TABLE IF EXISTS {output_db}.feature_importance")
    info("Dropped shared tables for schema reset")
    for table in CACHE_TABLES:
        spark.sql(f"DROP TABLE IF EXISTS {table}")

cache_exists = check_cache_exists(CACHE_TABLES) if CACHE_TABLES else False
USING_CACHED_RESULTS = cache_exists and not RETRAIN_MODEL

if USING_CACHED_RESULTS:
    info("Using cached results")
else:
    info("Computing from scratch" if not cache_exists else "Retraining model")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Load Features

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.1 Load Feature Table

# COMMAND ----------

input_db = "cohort_1"
input_table = "archetype_features_yearly"

features = spark.table(f"{input_db}.{input_table}")
features.cache()
row_count = features.count()
info(f"Loaded {row_count:,} person-years from {input_db}.{input_table}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.2 Display Feature Distribution Summary

# COMMAND ----------

features.select(
    'inpatient_visit_count', 'visit_count', 'irregularity_l1', 'irregularity_l2'
).summary("count", "mean", "min", "25%", "50%", "75%", "max").show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Vanilla Rules Classification
# MAGIC
# MAGIC Fixed threshold (0.5) and L1 penalty - deterministic and reproducible.

# COMMAND ----------

if not USING_CACHED_RESULTS:
    info("Running VANILLA rules classification...")

    # Add irregularity score and is_regular flag using vanilla settings
    features_vanilla = features.withColumn(
        "irregularity_score",
        F.when(F.lit(VANILLA_PENALTY) == "L2", F.col("irregularity_l2")).otherwise(F.col("irregularity_l1"))
    ).withColumn(
        "is_regular",
        (F.col("irregularity_score") < F.lit(VANILLA_THRESHOLD)).cast("boolean")
    )

    # Apply classification using UDF factory from utils
    classify_udf = make_classify_archetype_udf(VANILLA_THRESHOLD)

    features_vanilla = features_vanilla.withColumn(
        "archetype_result",
        classify_udf(
            F.col("inpatient_visit_count"),
            F.col("visit_count"),
            F.col("irregularity_score")
        )
    ).withColumn(
        "archetype",
        F.col("archetype_result.archetype")
    ).withColumn(
        "archetype_reason",
        F.col("archetype_result.reason")
    ).drop("archetype_result")

    # Add longitudinal tracking (prior-year archetype, change flag, segment tenure)
    features_vanilla = add_longitudinal_tracking(features_vanilla).withColumn(
        "archetype_confidence",
        F.lit(1.0).cast("double")
    )

    # Save to Delta
    output_table_vanilla = "rules_yearly"
    features_vanilla.write \
        .format("delta") \
        .mode("overwrite") \
        .option("overwriteSchema", "true") \
        .saveAsTable(f"{output_db}.{output_table_vanilla}")
    spark.sql(f"OPTIMIZE {output_db}.{output_table_vanilla} ZORDER BY (person_id)")
    info(f"Saved vanilla classification to {output_db}.{output_table_vanilla}")

else:
    features_vanilla = spark.table(f"{output_db}.rules_yearly")
    info("Loaded cached vanilla results")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Adaptive Rules Classification
# MAGIC
# MAGIC Auto-tuned threshold and penalty via stability plateau analysis.
# MAGIC
# MAGIC **Skipped if `RUN_ADAPTIVE=False`**

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4.1 Run Automatic Penalty Selection

# COMMAND ----------

if RUN_ADAPTIVE and not USING_CACHED_RESULTS:
    info("Running ADAPTIVE rules classification...")

    # Automatic penalty selection
    if AUTO_TUNE_PENALTY:
        penalty_analysis = empirical_penalty_characterization(
            features,
            sample_size=PENALTY_SAMPLE_SIZE,
            show_plots=SHOW_TUNING_PLOTS
        )
        SELECTED_PENALTY = penalty_analysis['recommendation']
        PENALTY_CONFIDENCE = penalty_analysis['confidence']
        info(f"AUTO-SELECTED PENALTY: {SELECTED_PENALTY} (confidence: {PENALTY_CONFIDENCE})")
    else:
        SELECTED_PENALTY = MANUAL_PENALTY
        PENALTY_CONFIDENCE = 'MANUAL'

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4.2 Run Automatic Threshold Selection

# COMMAND ----------

if RUN_ADAPTIVE and not USING_CACHED_RESULTS:
    # Automatic threshold selection
    if AUTO_TUNE_THRESHOLD:
        threshold_analysis = find_optimal_threshold(
            features,
            penalty=SELECTED_PENALTY,
            threshold_range=None,
            threshold_min=THRESHOLD_SEARCH_MIN,
            threshold_max=THRESHOLD_SEARCH_MAX,
            threshold_step=THRESHOLD_SEARCH_STEP,
            show_plots=SHOW_TUNING_PLOTS,
            use_sampling=USE_THRESHOLD_SAMPLING
        )
        SELECTED_THRESHOLD = threshold_analysis['optimal_threshold']
        THRESHOLD_CONFIDENCE = threshold_analysis['confidence']
        info(f"AUTO-SELECTED THRESHOLD: {SELECTED_THRESHOLD:.2f} (confidence: {THRESHOLD_CONFIDENCE})")
    else:
        SELECTED_THRESHOLD = MANUAL_THRESHOLD
        THRESHOLD_CONFIDENCE = 'MANUAL'

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4.3 Save Selected Parameters

# COMMAND ----------

if RUN_ADAPTIVE and not USING_CACHED_RESULTS:
    params_data = [{
        'run_timestamp': datetime.now().isoformat(),
        'penalty': SELECTED_PENALTY,
        'penalty_source': 'AUTO' if AUTO_TUNE_PENALTY else 'MANUAL',
        'penalty_confidence': PENALTY_CONFIDENCE,
        'threshold': float(SELECTED_THRESHOLD),
        'threshold_source': 'AUTO' if AUTO_TUNE_THRESHOLD else 'MANUAL',
        'threshold_confidence': THRESHOLD_CONFIDENCE,
        'stability_min_derivative': float(threshold_analysis.get('stability_min_derivative', np.nan)) if AUTO_TUNE_THRESHOLD else None,
        'stability_plateau_width': float(threshold_analysis.get('stability_plateau_width', np.nan)) if AUTO_TUNE_THRESHOLD else None,
    }]

    params_df = spark.createDataFrame(params_data)

    try:
        spark.table(f"{output_db}.rules_parameters")
        write_mode = 'append'
    except Exception:
        write_mode = 'overwrite'
        info(f"Creating new parameters table: {output_db}.rules_parameters")

    params_df.write.format('delta').mode(write_mode).option("mergeSchema", "true").saveAsTable(f"{output_db}.rules_parameters")
    info(f"Saved parameters: Penalty={SELECTED_PENALTY}, Threshold={SELECTED_THRESHOLD:.2f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4.4 Apply Adaptive Classification

# COMMAND ----------

if RUN_ADAPTIVE and not USING_CACHED_RESULTS:
    # Add irregularity score using selected penalty
    features_adaptive = features.withColumn(
        "irregularity_score",
        F.when(F.lit(SELECTED_PENALTY) == "L2", F.col("irregularity_l2")).otherwise(F.col("irregularity_l1"))
    ).withColumn(
        "is_regular",
        (F.col("irregularity_score") < F.lit(SELECTED_THRESHOLD)).cast("boolean")
    )

    # Null-irregularity audit (threshold-relevant subset)
    try:
        relevant = features_adaptive.where((F.col('inpatient_visit_count') == 0) & (F.col('visit_count') >= 2))
        null_count = relevant.where(F.col('irregularity_score').isNull()).count()
        total_relevant = relevant.count()
        null_rate = (null_count / total_relevant) if total_relevant > 0 else 0.0
        info(f"Null irregularity within threshold-relevant subset: {null_count:,}/{total_relevant:,} ({null_rate*100:.2f}%)")
    except Exception:
        pass

    # Apply classification using UDF factory from utils
    classify_udf_adaptive = make_classify_archetype_udf(SELECTED_THRESHOLD)

    features_adaptive = features_adaptive.withColumn(
        "archetype_result",
        classify_udf_adaptive(
            F.col("inpatient_visit_count"),
            F.col("visit_count"),
            F.col("irregularity_score")
        )
    ).withColumn(
        "archetype",
        F.col("archetype_result.archetype")
    ).withColumn(
        "archetype_reason",
        F.col("archetype_result.reason")
    ).drop("archetype_result")

    # Confidence mapping:
    # - IP-based and Sparse rules → confidence = 1.0
    # - Regularity-based rules → confidence increases with distance from threshold (scaled by IQR)
    # - Null regularity in regularity-based strata → low confidence (0.2)
    try:
        freq_q = features_adaptive.where(
            (F.col('inpatient_visit_count') == 0) & (F.col('visit_count') >= 4) & F.col('irregularity_score').isNotNull()
        ).selectExpr("percentile_approx(irregularity_score, array(0.25, 0.75), 10000) as q").collect()
        inf_q = features_adaptive.where(
            (F.col('inpatient_visit_count') == 0) & (F.col('visit_count').between(2,3)) & F.col('irregularity_score').isNotNull()
        ).selectExpr("percentile_approx(irregularity_score, array(0.25, 0.75), 10000) as q").collect()
        freq_iqr = float(freq_q[0]['q'][1] - freq_q[0]['q'][0]) if freq_q and freq_q[0]['q'] is not None else 0.0
        inf_iqr = float(inf_q[0]['q'][1] - inf_q[0]['q'][0]) if inf_q and inf_q[0]['q'] is not None else 0.0
    except Exception:
        freq_iqr, inf_iqr = 0.0, 0.0

    freq_scale = max(freq_iqr, 0.10)
    inf_scale = max(inf_iqr, 0.10)

    # Flatten confidence: deterministic rules → 1.0; null irregularity → low (0.2);
    # otherwise scale linearly by distance from threshold, capped at 1.0.
    is_deterministic = (F.col("inpatient_visit_count") >= 1) | (F.col("visit_count") <= 1)
    is_freq = F.col("visit_count") >= 4
    conf_scale = F.when(is_freq, F.lit(freq_scale)).otherwise(F.lit(inf_scale))
    distance_confidence = F.least(
        F.lit(1.0),
        F.abs(F.col("irregularity_score") - F.lit(SELECTED_THRESHOLD)) / conf_scale,
    )
    features_adaptive = features_adaptive.withColumn(
        "archetype_confidence",
        F.when(is_deterministic, F.lit(1.0))
         .when(F.col("irregularity_score").isNull(), F.lit(0.2))
         .otherwise(distance_confidence)
    )

    # Add longitudinal tracking (prior-year archetype, change flag, segment tenure)
    features_adaptive = add_longitudinal_tracking(features_adaptive)

    # Save to Delta
    output_table_adaptive = "rules_adaptive_yearly"
    features_adaptive.write \
        .format("delta") \
        .mode("overwrite") \
        .option("overwriteSchema", "true") \
        .saveAsTable(f"{output_db}.{output_table_adaptive}")
    spark.sql(f"OPTIMIZE {output_db}.{output_table_adaptive} ZORDER BY (person_id)")
    info(f"Saved adaptive classification to {output_db}.{output_table_adaptive}")

elif RUN_ADAPTIVE:
    features_adaptive = spark.table(f"{output_db}.rules_adaptive_yearly")
    info("Loaded cached adaptive results")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Summary Statistics

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.1 Vanilla Distribution

# COMMAND ----------

# --- Vanilla Distribution ---
vanilla_results = spark.table(f"{output_db}.rules_yearly") if USING_CACHED_RESULTS else features_vanilla
archetype_dist_vanilla = vanilla_results.groupBy("archetype").agg(
    F.count("*").alias("person_years"),
    F.countDistinct("person_id").alias("unique_persons")
).orderBy(F.desc("person_years"))
print("VANILLA ARCHETYPE DISTRIBUTION:")
display(archetype_dist_vanilla)

transition_summary_vanilla = vanilla_results.filter(
    F.col("prior_year_archetype_rules").isNotNull()
).agg(
    F.count("*").alias("total_person_years_with_prior"),
    F.sum("archetype_changed").alias("total_changes"),
    (F.sum("archetype_changed") / F.count("*") * 100).alias("pct_changed")
)
display(transition_summary_vanilla)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.2 Adaptive Distribution

# COMMAND ----------

if RUN_ADAPTIVE:
    adaptive_results = spark.table(f"{output_db}.rules_adaptive_yearly") if USING_CACHED_RESULTS else features_adaptive
    archetype_dist_adaptive = adaptive_results.groupBy("archetype").agg(
        F.count("*").alias("person_years"),
        F.countDistinct("person_id").alias("unique_persons")
    ).orderBy(F.desc("person_years"))
    print("ADAPTIVE ARCHETYPE DISTRIBUTION:")
    display(archetype_dist_adaptive)

    transition_summary_adaptive = adaptive_results.filter(
        F.col("prior_year_archetype_rules").isNotNull()
    ).agg(
        F.count("*").alias("total_person_years_with_prior"),
        F.sum("archetype_changed").alias("total_changes"),
        (F.sum("archetype_changed") / F.count("*") * 100).alias("pct_changed")
    )
    display(transition_summary_adaptive)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Cluster Profiling

# COMMAND ----------

feature_cols = ['visit_count', 'inpatient_visit_count', 'hospitalized_days']

# COMMAND ----------

# MAGIC %md
# MAGIC ### 6.1 Vanilla Cluster Profiles

# COMMAND ----------

# --- Vanilla Cluster Profiles ---
vanilla_results = spark.table(f"{output_db}.rules_yearly") if USING_CACHED_RESULTS else features_vanilla

profile_table_vanilla = create_cluster_profile_table(
    data_spark=vanilla_results,
    method_name='rules',
    cluster_col='archetype'
)
spark.sql(f"DROP TABLE IF EXISTS {output_db}.rules_cluster_profiles")
profile_table_vanilla.write.format("delta").mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable(f"{output_db}.rules_cluster_profiles")

features_pdf = spark_to_pandas_sampled(
    vanilla_results.select(['person_id', 'year', 'archetype'] + feature_cols),
    max_rows=1_000_000,
    seed=42
)
separation_df, db_index = calculate_separation_metrics(
    data_pdf=features_pdf,
    cluster_col='archetype',
    feature_cols=feature_cols
)
if separation_df is not None:
    spark.createDataFrame(separation_df) \
        .withColumn('method', F.lit('rules')) \
        .withColumn('davies_bouldin_index', F.lit(db_index)) \
        .write.format("delta").mode("append") \
        .saveAsTable(f"{output_db}.cluster_separation_metrics")

# Feature importance (Cohen's d) — append to shared table
importance_df = calculate_feature_importance(
    data_spark=vanilla_results,
    method_name='rules',
    cluster_col='archetype',
    feature_cols=feature_cols
)
spark.createDataFrame(importance_df) \
    .withColumn('cluster', F.col('cluster').cast('string')) \
    .write.format("delta").mode("append") \
    .option("mergeSchema", "true") \
    .saveAsTable(f"{output_db}.feature_importance")

# Cluster medoids
medoids_df = find_cluster_medoids(
    data_pdf=features_pdf,
    method_name='rules',
    cluster_col='archetype',
    feature_cols=feature_cols
)
spark.createDataFrame(medoids_df).write.format("delta").mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable(f"{output_db}.rules_medoids")

# COMMAND ----------

#
# Additional comparison/profiling/export/summary sections:
# When RUN_ADAPTIVE=False, skip all adaptive-specific comparisons and profiling.
if RUN_ADAPTIVE:
    features_classified = spark.table(f"{output_db}.rules_adaptive_yearly") if USING_CACHED_RESULTS else features_adaptive
    output_table = "rules_adaptive_yearly"
else:
    # Skip all remaining adaptive sections when RUN_ADAPTIVE=False
    print("RUN_ADAPTIVE=False - Skipping adaptive comparison, profiling, and export sections.")
    dbutils.notebook.exit("Completed vanilla rules only (RUN_ADAPTIVE=False)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Adaptive vs Vanilla Comprehensive Comparison
# MAGIC
# MAGIC Systematic evaluation across four dimensions: coverage, stability, separation, and cross-method consistency
# MAGIC
# MAGIC **Skipped if `RUN_ADAPTIVE=False`**

# COMMAND ----------

if RUN_ADAPTIVE:
    print("ADAPTIVE VS VANILLA: COMPREHENSIVE COMPARISON")

    # Load data
    vanilla_rules = spark.table(f"{output_db}.rules_yearly").select(
        'person_id', 'year', F.col('archetype').alias('archetype_vanilla')
    )
    adaptive_df = features_classified.select(
        'person_id', 'year', F.col('archetype').alias('archetype_adaptive'),
        'visit_count', 'inpatient_visit_count', 'hospitalized_days',
        'max_cci', 'total_condition_count', 'total_drug_count', 'total_procedure_count',
        'total_measurement_count', 'irregularity_score'
    )
    base_join = adaptive_df.join(vanilla_rules, ['person_id', 'year'], 'inner')
    n_total = base_join.count()
    print(f"Comparing {n_total:,} person-years\n")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 7.1 Coverage Analysis: Vanilla → Adaptive Mapping

# COMMAND ----------

if RUN_ADAPTIVE:
    print("1. COVERAGE ANALYSIS")
    print("-" * 70)

    # Compute confusion matrix
    counts_df = base_join.groupBy('archetype_vanilla', 'archetype_adaptive').count().toPandas()
    confusion_matrix = counts_df.pivot(index='archetype_vanilla', columns='archetype_adaptive', values='count').fillna(0)

    # Normalize by vanilla (row-wise) for purity
    confusion_pct = confusion_matrix.div(confusion_matrix.sum(axis=1), axis=0) * 100

    # Overall agreement
    agreement = (base_join.filter(F.col('archetype_vanilla') == F.col('archetype_adaptive')).count() / n_total) * 100
    kappa_data = spark_to_pandas_sampled(
        base_join.select('archetype_vanilla', 'archetype_adaptive'),
        max_rows=500_000,
        seed=42
    )
    kappa = cohen_kappa_score(kappa_data['archetype_vanilla'], kappa_data['archetype_adaptive'])

    print(f"Overall agreement: {agreement:.1f}%")
    print(f"Cohen's kappa: {kappa:.3f}")
    print(f"\nPurity by vanilla archetype (% mapped to dominant adaptive archetype):")
    for arch in confusion_pct.index:
        max_purity = confusion_pct.loc[arch].max()
        dominant = confusion_pct.loc[arch].idxmax()
        print(f"  {arch:<30} → {dominant:<30} ({max_purity:.1f}%)")

    # Visualization: Confusion matrix heatmap
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Absolute counts
    sns.heatmap(confusion_matrix, annot=True, fmt='.0f', cmap='Blues', ax=axes[0],
                cbar_kws={'label': 'Count'})
    axes[0].set_title(f'Vanilla → Adaptive Mapping (Counts)\nAgreement: {agreement:.1f}%, Kappa: {kappa:.3f}')
    axes[0].set_xlabel('Adaptive Archetype')
    axes[0].set_ylabel('Vanilla Archetype')
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].tick_params(axis='y', rotation=0)

    # Percentage (row-normalized)
    sns.heatmap(confusion_pct, annot=True, fmt='.1f', cmap='RdYlGn', vmin=0, vmax=100, ax=axes[1],
                cbar_kws={'label': '% of Vanilla'})
    axes[1].set_title('Vanilla → Adaptive Mapping (% within Vanilla)')
    axes[1].set_xlabel('Adaptive Archetype')
    axes[1].set_ylabel('Vanilla Archetype')
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].tick_params(axis='y', rotation=0)

    plt.tight_layout()
    plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 7.2 Temporal Stability Analysis

# COMMAND ----------

if RUN_ADAPTIVE:
        print("\n2. TEMPORAL STABILITY ANALYSIS")
    print("-" * 70)

    person_window = Window.partitionBy("person_id").orderBy("year")

    # Vanilla transitions
    vanilla_with_prior = vanilla_rules.withColumn(
        'prior_vanilla', F.lag('archetype_vanilla', 1).over(person_window)
    )
    vanilla_transitions = vanilla_with_prior.filter(F.col('prior_vanilla').isNotNull())
    vanilla_transition_pct = (vanilla_transitions.filter(
        F.col('archetype_vanilla') != F.col('prior_vanilla')
    ).count() / vanilla_transitions.count()) * 100

    # Adaptive transitions
    adaptive_with_prior = adaptive_df.withColumn(
        'prior_adaptive', F.lag('archetype_adaptive', 1).over(person_window)
    )
    adaptive_transitions = adaptive_with_prior.filter(F.col('prior_adaptive').isNotNull())
    adaptive_transition_pct = (adaptive_transitions.filter(
        F.col('archetype_adaptive') != F.col('prior_adaptive')
    ).count() / adaptive_transitions.count()) * 100

    # Per-archetype stability
    vanilla_arch_stability = vanilla_transitions.groupBy('prior_vanilla').agg(
        (F.sum(F.when(F.col('archetype_vanilla') == F.col('prior_vanilla'), 1).otherwise(0)) /
         F.count('*') * 100).alias('stability_pct')
    ).toPandas().rename(columns={'prior_vanilla': 'archetype', 'stability_pct': 'vanilla_stability'})

    adaptive_arch_stability = adaptive_transitions.groupBy('prior_adaptive').agg(
        (F.sum(F.when(F.col('archetype_adaptive') == F.col('prior_adaptive'), 1).otherwise(0)) /
         F.count('*') * 100).alias('stability_pct')
    ).toPandas().rename(columns={'prior_adaptive': 'archetype', 'stability_pct': 'adaptive_stability'})

    stability_comparison = vanilla_arch_stability.merge(adaptive_arch_stability, on='archetype', how='outer').fillna(0)
    stability_comparison['improvement'] = stability_comparison['adaptive_stability'] - stability_comparison['vanilla_stability']

    print(f"Overall transition rate:")
    print(f"  Vanilla:  {vanilla_transition_pct:.2f}% (lower is more stable)")
    print(f"  Adaptive: {adaptive_transition_pct:.2f}%")
    print(f"  Improvement: {vanilla_transition_pct - adaptive_transition_pct:+.2f} percentage points")

    print(f"\nPer-archetype stability (% staying in same archetype year-over-year):")
    display(spark.createDataFrame(stability_comparison))

    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Overall comparison
    methods = ['Vanilla', 'Adaptive']
    transition_rates = [vanilla_transition_pct, adaptive_transition_pct]
    colors = ['#d62728' if t > 30 else '#2ca02c' for t in transition_rates]
    axes[0].bar(methods, transition_rates, color=colors, alpha=0.7, edgecolor='black')
    axes[0].axhline(30, color='gray', linestyle='--', alpha=0.5, label='30% threshold')
    axes[0].set_ylabel('Transition Rate (%)')
    axes[0].set_title('Year-over-Year Transition Rate\n(Lower = More Stable)')
    axes[0].set_ylim([0, max(transition_rates) * 1.2])
    for i, v in enumerate(transition_rates):
        axes[0].text(i, v + 1, f'{v:.1f}%', ha='center', fontweight='bold')
    axes[0].legend()

    # Per-archetype comparison
    x = np.arange(len(stability_comparison))
    width = 0.35
    axes[1].barh(x - width/2, stability_comparison['vanilla_stability'], width, label='Vanilla', alpha=0.7)
    axes[1].barh(x + width/2, stability_comparison['adaptive_stability'], width, label='Adaptive', alpha=0.7)
    axes[1].set_yticks(x)
    axes[1].set_yticklabels(stability_comparison['archetype'], fontsize=8)
    axes[1].set_xlabel('Stability (%)')
    axes[1].set_title('Per-Archetype Stability\n(Higher = More Stable)')
    axes[1].legend()
    axes[1].grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 7.3 Feature Separation Analysis

# COMMAND ----------

if RUN_ADAPTIVE:
    print("\n3. FEATURE SEPARATION ANALYSIS")
    print("-" * 70)

    feature_cols_sep = get_profile_features()

    # Calculate feature importance for both
    vanilla_for_importance = vanilla_rules.join(
        features_classified.drop('archetype'), ['person_id', 'year'], 'left'
    ).withColumnRenamed('archetype_vanilla', 'cluster')

    adaptive_for_importance = adaptive_df.withColumnRenamed('archetype_adaptive', 'cluster')

    vanilla_imp = calculate_feature_importance(
        data_spark=vanilla_for_importance,
        method_name='rules',
        cluster_col='cluster',
        feature_cols=feature_cols_sep
    )
    adaptive_imp = calculate_feature_importance(
        data_spark=adaptive_for_importance,
        method_name='rules_adaptive',
        cluster_col='cluster',
        feature_cols=feature_cols_sep
    )

    # Convert to DataFrames and aggregate
    vanilla_imp_df = pd.DataFrame(vanilla_imp)
    adaptive_imp_df = pd.DataFrame(adaptive_imp)

    # Mean Cohen's d per feature (across all cluster pairs)
    vanilla_sep_by_feat = vanilla_imp_df.groupby('feature')['cohens_d'].apply(lambda x: np.abs(x).mean()).reset_index()
    vanilla_sep_by_feat.columns = ['feature', 'vanilla_cohens_d']

    adaptive_sep_by_feat = adaptive_imp_df.groupby('feature')['cohens_d'].apply(lambda x: np.abs(x).mean()).reset_index()
    adaptive_sep_by_feat.columns = ['feature', 'adaptive_cohens_d']

    separation_comparison = vanilla_sep_by_feat.merge(adaptive_sep_by_feat, on='feature')
    separation_comparison['improvement'] = separation_comparison['adaptive_cohens_d'] - separation_comparison['vanilla_cohens_d']
    separation_comparison = separation_comparison.sort_values('improvement', ascending=False)

    # Overall separation
    overall_vanilla = vanilla_sep_by_feat['vanilla_cohens_d'].mean()
    overall_adaptive = adaptive_sep_by_feat['adaptive_cohens_d'].mean()

    print(f"Mean Cohen's d (across all features and cluster pairs):")
    print(f"  Vanilla:  {overall_vanilla:.3f}")
    print(f"  Adaptive: {overall_adaptive:.3f}")
    print(f"  Improvement: {overall_adaptive - overall_vanilla:+.3f} ({((overall_adaptive/overall_vanilla - 1)*100):+.1f}%)")

    print(f"\nPer-feature separation (mean |Cohen's d|):")
    display(spark.createDataFrame(separation_comparison))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 7.4 Cross-Method Consistency

# COMMAND ----------

if RUN_ADAPTIVE:
    print("\n4. CROSS-METHOD CONSISTENCY")
    print("-" * 70)

    # Compare to data-driven clustering methods if available
    methods_to_compare = []
    for m in ['gmm_4', 'gmm_7']:
        try:
            spark.table(f"{output_db}.{m}_yearly")
            methods_to_compare.append(m)
        except Exception:
            pass

    if len(methods_to_compare) > 0:
        print(f"Comparing consistency with methods: {methods_to_compare}")

        consistency_results = []

        def enc(s):
            return pd.factorize(s)[0]

        for m in methods_to_compare:
            m_df = spark.table(f"{output_db}.{m}_yearly").select('person_id', 'year', F.col(f'{m}_cluster').alias('m_cluster'))

            ja = adaptive_df.join(m_df, ['person_id', 'year'], 'inner').toPandas()
            jv = vanilla_rules.join(m_df, ['person_id', 'year'], 'inner').toPandas()

            if len(ja) > 0 and len(jv) > 0:
                ari_a = adjusted_rand_score(enc(ja['archetype_adaptive']), ja['m_cluster'])
                ari_v = adjusted_rand_score(enc(jv['archetype_vanilla']), jv['m_cluster'])
                ami_a = adjusted_mutual_info_score(enc(ja['archetype_adaptive']), ja['m_cluster'])
                ami_v = adjusted_mutual_info_score(enc(jv['archetype_vanilla']), jv['m_cluster'])

                consistency_results.append({
                    'method': m,
                    'ARI_vanilla': ari_v,
                    'ARI_adaptive': ari_a,
                    'ARI_delta': ari_a - ari_v,
                    'AMI_vanilla': ami_v,
                    'AMI_adaptive': ami_a,
                    'AMI_delta': ami_a - ami_v
                })

        if len(consistency_results) > 0:
            consistency_df = pd.DataFrame(consistency_results)
            print("\nConsistency comparison (higher = more consistent with data-driven method):")
            display(spark.createDataFrame(consistency_df))

            # Visualization
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))

            x = np.arange(len(consistency_df))
            width = 0.35

            axes[0].bar(x - width/2, consistency_df['ARI_vanilla'], width, label='Vanilla', alpha=0.7)
            axes[0].bar(x + width/2, consistency_df['ARI_adaptive'], width, label='Adaptive', alpha=0.7)
            axes[0].set_xticks(x)
            axes[0].set_xticklabels(consistency_df['method'])
            axes[0].set_ylabel('Adjusted Rand Index')
            axes[0].set_title('ARI vs Data-Driven Clusters')
            axes[0].legend()
            axes[0].grid(axis='y', alpha=0.3)

            axes[1].bar(x - width/2, consistency_df['AMI_vanilla'], width, label='Vanilla', alpha=0.7)
            axes[1].bar(x + width/2, consistency_df['AMI_adaptive'], width, label='Adaptive', alpha=0.7)
            axes[1].set_xticks(x)
            axes[1].set_xticklabels(consistency_df['method'])
            axes[1].set_ylabel('Adjusted Mutual Info')
            axes[1].set_title('AMI vs Data-Driven Clusters')
            axes[1].legend()
            axes[1].grid(axis='y', alpha=0.3)

            plt.tight_layout()
            plt.show()
        else:
            print("No overlap found for consistency comparison")
    else:
        print("No data-driven clustering methods available for comparison")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 7.5 Unified Scorecard

# COMMAND ----------

print("\n5. UNIFIED SCORECARD")
print("-" * 70)

scorecard = []
scorecard.append({'dimension': 'Coverage (Agreement)', 'vanilla': agreement, 'adaptive': agreement, 'winner': 'Tie'})
scorecard.append({'dimension': 'Stability (Transition Rate, lower=better)', 'vanilla': vanilla_transition_pct, 'adaptive': adaptive_transition_pct, 'winner': 'Adaptive' if adaptive_transition_pct < vanilla_transition_pct else 'Vanilla'})
scorecard.append({'dimension': 'Separation (Mean |Cohen d|, higher=better)', 'vanilla': overall_vanilla, 'adaptive': overall_adaptive, 'winner': 'Adaptive' if overall_adaptive > overall_vanilla else 'Vanilla'})

adaptive_wins = sum(1 for r in scorecard if r['winner'] == 'Adaptive')
print(f"Adaptive wins {adaptive_wins}/{len(scorecard)} dimensions")
display(spark.createDataFrame(pd.DataFrame(scorecard)))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Comprehensive Cluster Profiling (Adaptive)

# COMMAND ----------

# Define feature columns for profiling (subset for rules method)
feature_cols = ['visit_count', 'inpatient_visit_count', 'hospitalized_days']

profile_table = create_cluster_profile_table(
    data_spark=features_classified,
    method_name='rules_adaptive',
    cluster_col='archetype'
)
spark.sql(f"DROP TABLE IF EXISTS {output_db}.rules_adaptive_cluster_profiles")
profile_table.write.format("delta").mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable(f"{output_db}.rules_adaptive_cluster_profiles")
print(f"Saved {output_db}.rules_adaptive_cluster_profiles")

features_pdf = spark_to_pandas_sampled(
    features_classified.select(['person_id', 'year', 'archetype'] + feature_cols),
    max_rows=1_000_000,
    seed=42
)
medoids_df = find_cluster_medoids(
    data_pdf=features_pdf,
    method_name='rules',
    cluster_col='archetype',
    feature_cols=feature_cols
)
spark.createDataFrame(medoids_df).write.format("delta").mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable(f"{output_db}.rules_adaptive_medoids")
print(f"Saved {output_db}.rules_adaptive_medoids ({len(medoids_df)} medoids)")

separation_df, db_index = calculate_separation_metrics(
    data_pdf=features_pdf,
    cluster_col='archetype',
    feature_cols=feature_cols
)
if separation_df is not None:
    spark.createDataFrame(separation_df) \
        .withColumn('method', F.lit('rules_adaptive')) \
        .withColumn('davies_bouldin_index', F.lit(db_index)) \
        .write.format("delta").mode("append") \
        .saveAsTable(f"{output_db}.cluster_separation_metrics")
    print(f"Saved separation metrics (DB Index: {db_index:.3f})")

importance_df = calculate_feature_importance(
    data_spark=features_classified,
    method_name='rules_adaptive',
    cluster_col='archetype',
    feature_cols=feature_cols
)
spark.createDataFrame(importance_df) \
    .withColumn('cluster', F.col('cluster').cast('string')) \
    .write.format("delta").mode("append") \
    .option("mergeSchema", "true") \
    .saveAsTable(f"{output_db}.feature_importance")
print(f"Saved feature importance ({len(importance_df)} records)")

print("Rule-based clustering complete")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Standardized Label Export (for unified yearly tables)
# MAGIC
# MAGIC ### 9.1 Export to Cluster Labels Table

# COMMAND ----------

if not USING_CACHED_RESULTS:
    # Vanilla export
    vanilla_results = spark.table(f"{output_db}.rules_yearly") if USING_CACHED_RESULTS else features_vanilla
    vanilla_results.select(
        F.col('person_id'),
        F.col('year'),
        F.lit('rules').alias('model_id'),
        F.lit(None).cast('int').alias('cluster'),
        F.col('archetype').alias('cluster_label'),
        F.lit(None).cast('double').alias('confidence'),
        F.lit(None).cast('double').alias('entropy'),
        F.lit(0).alias('is_outlier'),
        F.lit(datetime.now().isoformat()).alias('run_timestamp')
    ).write.format('delta').mode('append').saveAsTable(f"{output_db}.cluster_labels_yearly")

    # Adaptive export
    features_classified.select(
        F.col('person_id'),
        F.col('year'),
        F.lit('rules_adaptive').alias('model_id'),
        F.lit(None).cast('int').alias('cluster'),
        F.col('archetype').alias('cluster_label'),
        F.col('archetype_confidence').cast('double').alias('confidence'),
        F.lit(None).cast('double').alias('entropy'),
        F.lit(0).alias('is_outlier'),
        F.lit(datetime.now().isoformat()).alias('run_timestamp')
    ).write.format('delta').mode('append').saveAsTable(f"{output_db}.cluster_labels_yearly")
    print(f"Appended rules labels to cluster_labels_yearly")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Cleanup Cached DataFrames
# MAGIC
# MAGIC ### 10.1 Unpersist Cached DataFrames

# COMMAND ----------

features.unpersist()
print("Cache cleanup completed")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 11. Execution Summary
# MAGIC
# MAGIC ### 11.1 Display Configuration and Output Summary

# COMMAND ----------

try:
    print("\nCONFIGURATION SUMMARY")
    print(f"Penalty: {SELECTED_PENALTY} ({PENALTY_CONFIDENCE})")
    print(f"Threshold: {SELECTED_THRESHOLD:.2f} ({THRESHOLD_CONFIDENCE})")
except Exception:
    pass

print("\nOutput tables:")
print(f"  {output_db}.rules_yearly")
print(f"  {output_db}.rules_cluster_profiles")
print(f"  {output_db}.rules_medoids")
print(f"  {output_db}.rules_adaptive_yearly")
print(f"  {output_db}.rules_adaptive_cluster_profiles")
print(f"  {output_db}.rules_adaptive_medoids")
print(f"  {output_db}.cluster_separation_metrics")
print(f"  {output_db}.feature_importance")
print(f"  {output_db}.cluster_labels_yearly")
print(f"  {output_db}.rules_parameters")

info("Rules classification complete")