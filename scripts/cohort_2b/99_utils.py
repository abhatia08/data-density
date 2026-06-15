# Databricks notebook source
# MAGIC %md
# MAGIC **Purpose**: Cohort 2b utilities (thin wrapper over cohort_1 utilities). Provides shared helpers plus pretrained GMM model loading for inference-only reuse.

# COMMAND ----------

# Reuse the complete cohort_1 utility set
# MAGIC %run ../cohort_1/99_utils

# COMMAND ----------

# Cohort-specific defaults
DEFAULT_INPUT_DB = "cohort_2b"
DEFAULT_OUTPUT_DB = "cohort_2b"

# COMMAND ----------

import os
import mlflow
import cloudpickle
import pandas as pd
from scipy.stats import entropy
from pyspark.sql import functions as F
from pyspark.sql import SparkSession
from pyspark.sql.window import Window


def get_gmm_model_uri(model_id: str, default_env: str | None = None) -> str:
    """Resolve DBFS or MLflow URI for a pretrained GMM bundle saved by cohort_1."""
    key = default_env or f"{model_id.upper()}_MODEL_URI"
    uri = os.environ.get(key)
    if uri:
        return uri

    default_path = f"/dbfs/mnt/models/cohort_1/{model_id}"
    if os.path.exists(default_path):
        return default_path

    raise ValueError(
        f"Missing pretrained model bundle for {model_id}. "
        f"Set env var {key}, or run cohort_1 GMM training first "
        f"(expected bundle at {default_path}/gmm_bundle.pkl)."
    )


def load_gmm_bundle(model_id: str, artifact_name: str = "gmm_bundle.pkl"):
    """
    Load pretrained GMM bundle (model + scaler + winsor caps) saved by cohort_1.
    The bundle should be a pickled dict with keys: model, scaler, winsor_caps.
    """
    uri = get_gmm_model_uri(model_id)
    local_path = mlflow.artifacts.download_artifacts(uri)
    bundle_path = os.path.join(local_path, artifact_name)
    if not os.path.exists(bundle_path):
        raise FileNotFoundError(f"Expected bundle at {bundle_path} (from {uri})")
    with open(bundle_path, "rb") as f:
        bundle = cloudpickle.load(f)
    model = bundle.get("model") or bundle.get("gmm_model")
    scaler = bundle.get("scaler")
    winsor_caps = bundle.get("winsor_caps")
    if model is None:
        raise ValueError("Bundle missing 'model' entry")
    return model, scaler, winsor_caps


def prepare_features_for_gmm(features_pdf, features_for_clustering, scaler, winsor_caps, irregularity_penalty="BOTH"):
    """
    Apply the exact preprocessing used in cohort_1 training to new cohort features.
    """
    feature_matrix_scaled, _, _ = prepare_feature_matrix(
        features_df=features_pdf,
        features_for_clustering=features_for_clustering,
        irregularity_penalty=irregularity_penalty,
        winsorize_features=['visit_count', 'inpatient_visit_count', 'hospitalized_days'],
        apply_log_transform=True,
        fit_scaler=False,
        scaler=scaler,
        winsor_caps=winsor_caps,
        percentile=0.99,
        seed=42
    )
    return feature_matrix_scaled


def predict_with_gmm(
    model_id: str,
    bundle_artifact: str,
    features_spark,
    features_for_clustering: list[str],
    irregularity_penalty: str = "BOTH",
    spark_session=None,
):
    """Predict clusters using a pretrained cohort_1 GMM bundle."""
    spark_session = spark_session or SparkSession.builder.getOrCreate()

    model, scaler, winsor_caps = load_gmm_bundle(model_id, artifact_name=bundle_artifact)

    features_pdf = features_spark.select(["person_id", "year"] + features_for_clustering).toPandas()
    feature_matrix_scaled = prepare_features_for_gmm(
        features_pdf=features_pdf,
        features_for_clustering=features_for_clustering,
        scaler=scaler,
        winsor_caps=winsor_caps,
        irregularity_penalty=irregularity_penalty,
    )

    cluster_labels = model.predict(feature_matrix_scaled)
    cluster_probs = model.predict_proba(feature_matrix_scaled)
    cluster_conf = cluster_probs.max(axis=1)
    cluster_entropy = [entropy(p + 1e-10) for p in cluster_probs]

    pred_pdf = pd.DataFrame({
        "person_id": features_pdf["person_id"],
        "year": features_pdf["year"],
        f"{model_id}_cluster": cluster_labels,
        f"{model_id}_confidence": cluster_conf,
        f"{model_id}_entropy": cluster_entropy,
        "training_data": 0,
    })
    pred_spark = spark_session.createDataFrame(pred_pdf)

    pw = Window.partitionBy("person_id").orderBy("year")
    pred_spark = pred_spark.withColumn(
        f"prior_year_{model_id}_cluster",
        F.lag(F.col(f"{model_id}_cluster"), 1).over(pw),
    )

    return pred_spark

