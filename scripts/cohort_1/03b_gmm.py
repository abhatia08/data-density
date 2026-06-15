# Databricks notebook source
# MAGIC %md
# MAGIC **Purpose**: GMM clustering (default K=4; set K_VALUES=[4,7] for supplement)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Setup and Configuration

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.1 Import Libraries

# COMMAND ----------

import gc

import numpy as np
import pandas as pd
import psutil

from pyspark.sql import functions as F
from pyspark.sql.window import Window

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.2 Define Configuration Parameters

# COMMAND ----------

K_VALUES = [4]  # Manuscript primary model (GMM-4). Add 7 for supplement sensitivity.

# Irregularity penalty configuration
IRREGULARITY_PENALTY = "BOTH"  # 'L1', 'L2', or 'BOTH'

# Model caching configuration
RETRAIN_MODEL = False  # Use cached results when available; set True to force re-computation
CLEAR_CACHE = False    # Keep cached results; set True to clear before running

SAVE_MODEL_BUNDLES = True
BUNDLE_DIR = "/dbfs/mnt/models/cohort_1"

# GMM hyperparameters
COVARIANCE_TYPE = 'tied'
N_INIT = 10
MAX_ITER = 200
RANDOM_STATE = 42

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
person_window = Window.partitionBy("person_id").orderBy("year")

# Cache tables per K
CACHE_TABLES_BY_K = {}
for k in K_VALUES:
    mid = f"gmm_{k}"
    CACHE_TABLES_BY_K[k] = [
        f"{output_db}.{mid}_yearly",
        f"{output_db}.{mid}_cluster_profiles",
        f"{output_db}.{mid}_medoids",
    ]

ALL_CACHE_TABLES = [t for ks in CACHE_TABLES_BY_K.values() for t in ks]

if CLEAR_CACHE:
    for table in ALL_CACHE_TABLES:
        try:
            spark.sql(f"DROP TABLE IF EXISTS {table}")
        except Exception as exc:
            warn(f"Could not drop {table}: {exc}")
    for shared_table in ["cluster_separation_metrics", "feature_importance"]:
        try:
            spark.sql(f"DROP TABLE IF EXISTS {output_db}.{shared_table}")
        except Exception as exc:
            warn(f"Could not drop {output_db}.{shared_table}: {exc}")

# Determine cache usage per K
USING_CACHED_BY_K = {}
for k in K_VALUES:
    cache_exists_k = check_cache_exists(CACHE_TABLES_BY_K[k])
    USING_CACHED_BY_K[k] = cache_exists_k and not RETRAIN_MODEL

if all(USING_CACHED_BY_K.values()):
    info("Using cached results (all K values)")
else:
    info("Computing from scratch (one or more K values need training)" if not RETRAIN_MODEL else "Retraining model")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Load Features

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.1 Load Feature Table

# COMMAND ----------

input_db = "cohort_1"

features = load_features_with_regularity(input_db, IRREGULARITY_PENALTY)
features.cache()
row_count = features.count()
info(f"Loaded {row_count:,} person-years from {input_db}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Training Sample

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.1 Create Stratified Sample

# COMMAND ----------

features_for_clustering = get_core_clustering_features(
    use_continuous_irregularity=True,
    irregularity_penalty=IRREGULARITY_PENALTY
)

ANY_NEEDS_TRAINING = any(not USING_CACHED_BY_K[k] for k in K_VALUES)

# Create stratified random sample (one year per person) only if we will train at least one model
if ANY_NEEDS_TRAINING:
    training_sample_ids = create_stratified_training_sample(features, seed=RANDOM_STATE)

    # Join training flag back to features
    features = features.join(
        training_sample_ids,
        ['person_id', 'year'],
        'left'
    ).fillna({'training_data': 0})

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Preprocessing

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4.1 Prepare Training Feature Matrix

# COMMAND ----------

if ANY_NEEDS_TRAINING:
    # Load training sample
    clustering_data_sample = features.filter(F.col("training_data") == 1).select(
        'person_id', 'year', *features_for_clustering
    )
    clustering_pdf = clustering_data_sample.toPandas()

    # Validate before preprocessing
    validate_before_clustering(
        features=clustering_data_sample,
        features_for_clustering=features_for_clustering,
        irregularity_penalty=IRREGULARITY_PENALTY,
        method_name='gmm',
        min_samples=10
    )

    # Prepare feature matrix - scaler and caps computed ONCE and reused
    feature_matrix_scaled, scaler, WINSOR_CAPS = prepare_feature_matrix(
        features_df=clustering_pdf,
        features_for_clustering=features_for_clustering,
        irregularity_penalty=IRREGULARITY_PENALTY,
        winsorize_features=['visit_count', 'inpatient_visit_count', 'hospitalized_days'],
        apply_log_transform=True,
        fit_scaler=True,
        scaler=None,
        winsor_caps=None,
        percentile=0.99,
        seed=RANDOM_STATE
    )

    # Validate feature matrix
    validation_results = validate_clustering_input(
        feature_matrix_scaled,
        n_samples=len(clustering_pdf),
        n_features=len(features_for_clustering),
        method_name='gmm'
    )

    info(f"Training feature matrix: {feature_matrix_scaled.shape}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4.2 Prepare Full Dataset for Prediction

# COMMAND ----------

if ANY_NEEDS_TRAINING:
    # Memory check
    est_memory_gb = (row_count * len(features_for_clustering) * 8) / (1024**3) * 3
    available_memory_gb = psutil.virtual_memory().available / (1024**3)
    info(f"Memory check: {row_count:,} rows, ~{est_memory_gb:.2f} GB needed, {available_memory_gb:.2f} GB available")

    if est_memory_gb > available_memory_gb * 0.8:
        raise MemoryError(
            f"Insufficient memory: ~{est_memory_gb:.1f} GB needed, {available_memory_gb:.1f} GB available. "
            f"Reduce dataset size or run on larger driver."
        )

    # Load full dataset
    clustering_data_all = features.select('person_id', 'year', 'training_data', *features_for_clustering)
    clustering_pdf_all = clustering_data_all.toPandas()
    info(f"Loaded {len(clustering_pdf_all):,} person-years to driver")

    # Apply same preprocessing (reuse scaler and caps)
    feature_matrix_all_scaled, _, _ = prepare_feature_matrix(
        features_df=clustering_pdf_all,
        features_for_clustering=features_for_clustering,
        irregularity_penalty=IRREGULARITY_PENALTY,
        winsorize_features=['visit_count', 'inpatient_visit_count', 'hospitalized_days'],
        apply_log_transform=True,
        fit_scaler=False,  # Use existing scaler
        scaler=scaler,
        winsor_caps=WINSOR_CAPS,
        percentile=0.99,
        seed=RANDOM_STATE
    )
else:
    clustering_pdf_all = None
    feature_matrix_all_scaled = None
    feature_matrix_scaled = None
    scaler = None
    WINSOR_CAPS = None

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Model/Parameter Selection

# COMMAND ----------

# Model parameters
np.random.seed(RANDOM_STATE)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Cluster All Years
# MAGIC
# MAGIC Train each K on the stratified training sample and predict clusters for all person-years.

# COMMAND ----------

# Define feature columns for profiling (extended set beyond clustering features)
feature_cols = get_profile_features()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Cluster Analysis
# MAGIC
# MAGIC Standard cluster profiling, medoids, separation, and feature importance.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Export Results
# MAGIC
# MAGIC ### 8.1 Export to Standardized Label Table

# COMMAND ----------

for k in K_VALUES:
    model_id = f"gmm_{k}"
    USING_CACHED_RESULTS = USING_CACHED_BY_K[k]
    section(f"Training GMM with K={k}")

    # Use the shared train_and_predict_gmm function from utils
    gmm_output, results_pdf, gmm_model = train_and_predict_gmm(
        k_value=k,
        model_id=model_id,
        feature_matrix_scaled=feature_matrix_scaled if not USING_CACHED_RESULTS else None,
        feature_matrix_all_scaled=feature_matrix_all_scaled,
        clustering_pdf_all=clustering_pdf_all,
        features=features,
        output_db=output_db,
        scaler=scaler,
        winsor_caps=WINSOR_CAPS,
        covariance_type=COVARIANCE_TYPE,
        n_init=N_INIT,
        max_iter=MAX_ITER,
        random_state=RANDOM_STATE,
        using_cached=USING_CACHED_RESULTS,
        save_bundle=SAVE_MODEL_BUNDLES,
        bundle_dir=BUNDLE_DIR,
    )

    cluster_col = f"{model_id}_cluster"
    confidence_col = f"{model_id}_confidence"
    entropy_col = f"{model_id}_entropy"

    section(f"Running cluster analysis for K={k}")

    analysis_results = run_standard_cluster_analysis(
        data_spark=gmm_output,
        method_name=model_id,
        cluster_col=cluster_col,
        feature_cols=feature_cols,
        output_db=output_db,
        using_cached=USING_CACHED_RESULTS
    )

    # Save analysis results
    if not USING_CACHED_RESULTS:
        profile_table = analysis_results.get('profiles')
        if profile_table is not None:
            profile_table.write.format("delta").mode("overwrite").option("overwriteSchema", "true") \
                .saveAsTable(f"{output_db}.{model_id}_cluster_profiles")

        medoids_df = analysis_results.get('medoids')
        if medoids_df is not None and len(medoids_df) > 0:
            spark.createDataFrame(medoids_df).write.format("delta").mode("overwrite").option("overwriteSchema", "true") \
                .saveAsTable(f"{output_db}.{model_id}_medoids")

    if not USING_CACHED_RESULTS:
        export_cluster_labels(
            data_spark=gmm_output,
            model_id=model_id,
            cluster_col=cluster_col,
            confidence_col=confidence_col,
            entropy_col=entropy_col,
            output_db=output_db
        )

    info(f"Completed {model_id.upper()}")

    # Clean up memory between K values
    gc.collect()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Cleanup
# MAGIC
# MAGIC ### 9.1 Unpersist Cached DataFrames

# COMMAND ----------

try:
    features.unpersist()
except Exception:
    pass

# COMMAND ----------

