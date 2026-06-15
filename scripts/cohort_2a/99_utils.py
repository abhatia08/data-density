# Databricks notebook source
# MAGIC %md
# MAGIC **Purpose**: Cohort 2a utilities. Provides shared helpers plus pretrained GMM model loading for inference-only reuse.

# COMMAND ----------

# MAGIC %run ../cohort_1/99_utils

# COMMAND ----------

# Cohort-specific defaults
DEFAULT_INPUT_DB = "cohort_2a"
DEFAULT_OUTPUT_DB = "cohort_2a"

# COMMAND ----------

import os
import mlflow
import cloudpickle
import pandas as pd

# COMMAND ----------

# Analysis helpers (shared across cohort notebooks)
from pyspark.sql import functions as F
from pyspark.sql import SparkSession
from pyspark.sql.window import Window
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler
from pyspark.ml.regression import LinearRegression
from scipy.stats import f as f_dist
from scipy.stats import entropy
from scipy.stats import t as t_dist


def get_gmm_model_uri(model_id: str, default_env: str | None = None) -> str:
    """
    Resolve the DBFS path for a pretrained GMM model bundle.

    Priority:
    1. Environment variable (e.g., GMM_4_MODEL_URI)
    2. Default DBFS path: /dbfs/mnt/models/cohort_1/{model_id}/

    The bundle is saved by cohort_1's GMM training scripts.
    """
    key = default_env or f"{model_id.upper()}_MODEL_URI"
    uri = os.environ.get(key)
    if uri:
        return uri

    # Default path where cohort_1 saves the model bundles
    default_path = f"/dbfs/mnt/models/cohort_1/{model_id}"
    if os.path.exists(default_path):
        return default_path

    raise ValueError(
        f"Missing pretrained model bundle for {model_id}. "
        f"Either set env var {key}, or ensure cohort_1's GMM training has run "
        f"(expected bundle at {default_path}/gmm_bundle.pkl)."
    )


def load_gmm_bundle(model_id: str, artifact_name: str = "gmm_bundle.pkl"):
    """
    Load pretrained GMM bundle (model + scaler + winsor caps) saved by cohort_1.
    The bundle should be a pickled dict with keys: model, scaler, winsor_caps.

    Handles both:
    - Local DBFS paths (e.g., /dbfs/mnt/models/cohort_1/gmm_4/)
    - MLflow artifact URIs (e.g., runs:/abc123/model)
    """
    uri = get_gmm_model_uri(model_id)

    # Check if it's a local path or an MLflow URI
    if os.path.isdir(uri):
        # Local DBFS path
        local_path = uri
    else:
        # MLflow artifact URI - download to local
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
    """
    Predict clusters for a cohort using a pretrained cohort_1 GMM bundle.

    Returns a Spark DataFrame with:
      person_id, year, {model_id}_cluster, {model_id}_confidence, {model_id}_entropy, training_data,
      prior_year_{model_id}_cluster
    """
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


def spark_map_expr(mapping: dict):
    """
    Create a Spark SQL map expression from a Python dict.

    Parameters
    ----------
    mapping : dict
        Keys/values to embed as literals into a Spark map().
    """
    # create_map expects alternating key/value expressions
    return F.create_map(*[x for k, v in mapping.items() for x in (F.lit(k), F.lit(v))])


def anova_pvalue_unadjusted(df, cluster_col: str, value_col: str):
    """
    One-way ANOVA p-value computed from per-cluster aggregates.
    H0: all cluster means are equal.
    """
    d = df.select(cluster_col, value_col).where(
        F.col(cluster_col).isNotNull() & F.col(value_col).isNotNull()
    )

    rows = (
        d.groupBy(cluster_col)
        .agg(
            F.count("*").cast("double").alias("n"),
            F.mean(value_col).cast("double").alias("mean"),
            F.var_samp(value_col).cast("double").alias("var"),
        )
        .collect()
    )

    k = len(rows)
    if k < 2:
        return None

    n_total = sum(r["n"] for r in rows)
    if n_total <= k:
        return None

    overall_mean = sum(r["n"] * r["mean"] for r in rows) / n_total
    ss_between = sum(r["n"] * (r["mean"] - overall_mean) ** 2 for r in rows)

    ss_within = 0.0
    for r in rows:
        if r["n"] > 1 and r["var"] is not None:
            ss_within += (r["n"] - 1.0) * r["var"]

    df_between = float(k - 1)
    df_within = float(n_total - k)
    if df_within <= 0 or ss_within <= 0:
        return None

    ms_between = ss_between / df_between
    ms_within = ss_within / df_within
    if ms_within == 0:
        return None

    f_stat = ms_between / ms_within
    return float(f_dist.sf(f_stat, df_between, df_within))


def _fit_rss(df, label_col: str, feature_cols: list[str]):
    """
    Fit OLS via Spark MLlib LinearRegression and return (rss, n, p_params).
    p_params counts intercept + coefficients.
    """
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    train = assembler.transform(df).select(F.col(label_col).alias("label"), "features")

    lr = LinearRegression(
        labelCol="label",
        featuresCol="features",
        fitIntercept=True,
        regParam=0.0,
        elasticNetParam=0.0,
        maxIter=50,
        solver="normal",
    )
    model = lr.fit(train)
    summary = model.summary

    n = int(summary.numInstances)
    rss = float(summary.meanSquaredError) * float(n)
    p_params = int(model.coefficients.size) + 1  # + intercept
    return rss, n, p_params


def pvalue_adjusted_for_cci(
    df,
    cluster_col: str,
    value_col: str,
    cci_col: str = "prior_year_max_cci",
):
    """
    Partial F-test for cluster effect controlling for prior-year CCI:
      full:    y ~ cci + cci_missing + cluster_dummies
      reduced: y ~ cci + cci_missing
    Returns p-value for the *overall* cluster effect.
    """
    if cci_col not in df.columns:
        return None

    base = (
        df.select(cluster_col, value_col, cci_col)
        .where(F.col(cluster_col).isNotNull() & F.col(value_col).isNotNull())
        .withColumn("cci_missing", F.col(cci_col).isNull().cast("double"))
        .withColumn("cci_value", F.coalesce(F.col(cci_col).cast("double"), F.lit(0.0)))
    )

    # Build one-hot encoding for the cluster column
    indexer = StringIndexer(
        inputCol=cluster_col,
        outputCol="cluster_idx",
        handleInvalid="keep",
    )
    encoder = OneHotEncoder(
        inputCols=["cluster_idx"],
        outputCols=["cluster_ohe"],
        dropLast=True,
        handleInvalid="keep",
    )

    # Fit reduced model (no cluster)
    rss_reduced, n_red, p_red = _fit_rss(base, value_col, ["cci_value", "cci_missing"])

    # Fit full model (with cluster) - fit indexer/encoder once and reuse
    indexed = indexer.fit(base).transform(base)
    full_df = encoder.fit(indexed).transform(indexed)
    rss_full, n_full, p_full = _fit_rss(full_df, value_col, ["cci_value", "cci_missing", "cluster_ohe"])

    if n_red != n_full:
        return None

    df1 = float(p_full - p_red)
    df2 = float(n_full - p_full)
    if df1 <= 0 or df2 <= 0:
        return None

    f_stat = ((rss_reduced - rss_full) / df1) / (rss_full / df2)
    if f_stat < 0:
        return None

    return float(f_dist.sf(f_stat, df1, df2))


def build_transposed_means_with_pvalues(
    df,
    cluster_label_col: str,
    metric_specs,
    pivot_order=None,
    adjusted_cci_col: str = "prior_year_max_cci",
    decimals: int = 1,
):
    """
    Build a transposed table of per-cluster means with per-metric p-values.

    Output shape:
    - Rows: Metric label
    - Columns: cluster labels (pivot)
    - Plus p-value columns:
      - p_value_unadjusted (one-way ANOVA across clusters)
      - p_value_adjusted_for_prior_year_max_cci (partial F-test controlling for CCI)
    """
    metric_specs = [(lbl, col) for (lbl, col) in metric_specs if col in df.columns]
    if len(metric_specs) == 0:
        return None

    # Means by cluster (wide)
    agg_exprs = [F.round(F.mean(F.col(col)), decimals).alias(col) for _, col in metric_specs]
    wide = df.groupBy(cluster_label_col).agg(*agg_exprs)

    # Transpose to Metric x Cluster
    stack_parts = []
    for metric_label, col in metric_specs:
        stack_parts.append(f"'{metric_label}'")
        stack_parts.append(col)
    stack_expr = f"stack({len(metric_specs)}, {', '.join(stack_parts)}) as (Metric, value)"

    long = wide.select(F.col(cluster_label_col).alias("cluster_label"), F.expr(stack_expr))
    pivoted = (
        long.groupBy("Metric")
        .pivot("cluster_label", pivot_order)
        .agg(F.first("value"))
    )

    # p-values per metric (formatted as <0.001, <0.01, <0.05, or rounded value)
    def format_pvalue(p):
        if p is None:
            return None
        if p < 0.001:
            return "<0.001"
        elif p < 0.01:
            return "<0.01"
        elif p < 0.05:
            return "<0.05"
        else:
            return f"{p:.2f}"

    p_rows = []
    for metric_label, value_col in metric_specs:
        p_unadj = anova_pvalue_unadjusted(df, cluster_label_col, value_col)
        p_adj = pvalue_adjusted_for_cci(df, cluster_label_col, value_col, cci_col=adjusted_cci_col)
        p_rows.append((metric_label, format_pvalue(p_unadj), format_pvalue(p_adj)))

    spark_session = df.sparkSession
    p_df = spark_session.createDataFrame(
        p_rows,
        schema=["Metric", "p_value_unadjusted", "p_value_adjusted"],
    )
    out = pivoted.join(p_df, on="Metric", how="left")

    # Preserve caller-provided metric order (metric_specs), rather than alphabetical
    metric_order = [lbl for (lbl, _) in metric_specs]
    order_arr = F.array(*[F.lit(x) for x in metric_order])
    out = out.withColumn("_metric_order", F.array_position(order_arr, F.col("Metric")))
    out = out.orderBy(F.col("_metric_order").asc(), F.col("Metric").asc()).drop("_metric_order")
    return out


def pvalue_cohort_diff_unadjusted(
    df,
    cohort_col: str,
    value_col: str,
    cohort_a: str = "cohort_1",
    cohort_b: str = "cohort_2a",
):
    """
    Welch two-sample t-test p-value for difference in means between two cohorts.
    H0: mean_a == mean_b
    """
    d = df.select(cohort_col, value_col).where(
        F.col(cohort_col).isNotNull() & F.col(value_col).isNotNull()
    )

    rows = (
        d.groupBy(cohort_col)
        .agg(
            F.count("*").cast("double").alias("n"),
            F.mean(value_col).cast("double").alias("mean"),
            F.var_samp(value_col).cast("double").alias("var"),
        )
        .collect()
    )
    stats = {r[cohort_col]: r for r in rows}
    if cohort_a not in stats or cohort_b not in stats:
        return None

    n1, m1, v1 = float(stats[cohort_a]["n"]), float(stats[cohort_a]["mean"]), stats[cohort_a]["var"]
    n2, m2, v2 = float(stats[cohort_b]["n"]), float(stats[cohort_b]["mean"]), stats[cohort_b]["var"]
    if n1 < 2 or n2 < 2 or v1 is None or v2 is None:
        return None

    se2 = (v1 / n1) + (v2 / n2)
    if se2 <= 0:
        return None

    t_stat = (m1 - m2) / (se2 ** 0.5)
    # Satterthwaite degrees of freedom
    df_num = se2 ** 2
    df_den = ((v1 / n1) ** 2) / (n1 - 1.0) + ((v2 / n2) ** 2) / (n2 - 1.0)
    if df_den <= 0:
        return None
    dof = df_num / df_den

    return float(2.0 * t_dist.sf(abs(t_stat), dof))


def pvalue_cohort_diff_adjusted(
    df,
    cohort_col: str,
    value_col: str,
    cci_col: str = "prior_year_max_cci",
    cohort_a: str = "cohort_1",
    cohort_b: str = "cohort_2a",
    adjust_for_days_observed: bool = False,
    days_observed_col: str = "days_observed",
    adjust_for_year: bool = False,
    year_col: str = "year",
    y_transform: str = "none",
):
    """
    Partial F-test for cohort effect controlling for covariates:
      reduced: y ~ covariates
      full:    y ~ covariates + cohort_ind

    Supported covariates:
    - prior-year CCI (cci_value + cci_missing)
    - days_observed (log1p) if available/enabled
    - year (numeric) if available/enabled

    y_transform:
    - "none": use raw value_col
    - "log1p": use log(1 + value_col)

    cohort_ind == 1 for cohort_b, 0 for cohort_a.
    """
    if cci_col not in df.columns:
        return None

    select_cols = ["person_id", "year", cohort_col, value_col, cci_col]
    if adjust_for_days_observed and days_observed_col in df.columns:
        select_cols.append(days_observed_col)
    if adjust_for_year and year_col in df.columns and year_col not in select_cols:
        select_cols.append(year_col)

    base = (
        df.select(*select_cols)
        .where(F.col(cohort_col).isNotNull() & F.col(value_col).isNotNull())
        .where(F.col(cohort_col).isin([cohort_a, cohort_b]))
        .withColumn("cci_missing", F.col(cci_col).isNull().cast("double"))
        .withColumn("cci_value", F.coalesce(F.col(cci_col).cast("double"), F.lit(0.0)))
        .withColumn("cohort_ind", (F.col(cohort_col) == F.lit(cohort_b)).cast("double"))
    )

    # Add optional covariates
    feature_cols = ["cci_value", "cci_missing"]
    if adjust_for_days_observed and days_observed_col in df.columns:
        base = base.withColumn("log_days_observed", F.log1p(F.col(days_observed_col).cast("double")))
        feature_cols.append("log_days_observed")

    if adjust_for_year and year_col in df.columns:
        base = base.withColumn("year_value", F.col(year_col).cast("double"))
        feature_cols.append("year_value")

    # Optional transform on outcome
    if y_transform == "log1p":
        base = base.withColumn("_y", F.log1p(F.col(value_col).cast("double")))
        y_col = "_y"
    else:
        y_col = value_col

    rss_reduced, n_red, p_red = _fit_rss(base, y_col, feature_cols)
    rss_full, n_full, p_full = _fit_rss(base, y_col, feature_cols + ["cohort_ind"])

    if n_red != n_full:
        return None

    df1 = float(p_full - p_red)
    df2 = float(n_full - p_full)
    if df1 <= 0 or df2 <= 0:
        return None

    f_stat = ((rss_reduced - rss_full) / df1) / (rss_full / df2)
    if f_stat < 0:
        return None

    return float(f_dist.sf(f_stat, df1, df2))



def build_within_cluster_cohort_comparison(
    df,
    cohort_col: str,
    cluster_label_col: str,
    metric_specs,
    cohort_a: str = "cohort_1",
    cohort_b: str = "cohort_2a",
    adjusted_cci_col: str = "prior_year_max_cci",
    cluster_order=None,
    adjust_for_days_observed: bool = False,
    adjust_for_year: bool = False,
    adjusted_y_transform: str = "none",
):
    """
    Within each cluster label, compare cohort_a vs cohort_b for each metric.

    OPTIMIZED: Computes all statistics in a single Spark pass, then calculates
    p-values in Python to avoid thousands of small Spark jobs.

    Output columns:
    - Metric
    - cluster_label
    - mean_{cohort_a}, mean_{cohort_b}
    - diff_mean (mean_b - mean_a)
    - p_value_unadjusted (Welch t-test)
    - p_value_adjusted_for_prior_year_max_cci (Welch t-test after residualizing the metric
      against prior-year max CCI across both cohorts within-cluster; approximates ANCOVA)
    """
    metric_specs = [(lbl, col) for (lbl, col) in metric_specs if col in df.columns]
    if len(metric_specs) == 0:
        return None

    value_cols = [col for _, col in metric_specs]

    d = df.where(
        F.col(cluster_label_col).isNotNull()
        & F.col(cohort_col).isin([cohort_a, cohort_b])
    )

    # NOTE: adjust_for_days_observed / adjust_for_year are intentionally ignored here to keep
    # this optimized single-pass implementation. We only adjust for prior-year max CCI.

    x = F.col(adjusted_cci_col).cast("double")

    # Compute all statistics in ONE Spark aggregation per (cluster, cohort):
    # - raw: n/mean/var for display (diff_mean)
    # - y_t: n/mean/var for unadjusted p (optionally log1p)
    # - adjusted: sums needed to residualize y_t against CCI (x) without per-metric regressions
    agg_exprs = []
    for _, col in metric_specs:
        y_raw = F.col(col).cast("double")
        y_t = F.log1p(y_raw) if adjusted_y_transform == "log1p" else y_raw

        # Display stats (raw)
        agg_exprs.extend([
            F.count(F.when(y_raw.isNotNull(), 1)).alias(f"{col}_n"),
            F.mean(y_raw).alias(f"{col}_mean"),
            F.var_samp(y_raw).alias(f"{col}_var"),
        ])

        # Unadjusted p-value stats (transformed, if requested)
        agg_exprs.extend([
            F.count(F.when(y_t.isNotNull(), 1)).alias(f"{col}__t_n"),
            F.mean(y_t).alias(f"{col}__t_mean"),
            F.var_samp(y_t).alias(f"{col}__t_var"),
        ])

        # Adjusted p-value aggregates (only rows with both y_t and x present)
        cond = y_t.isNotNull() & x.isNotNull()
        y_t_c = F.when(cond, y_t)
        x_c = F.when(cond, x)
        agg_exprs.extend([
            F.count(F.when(cond, 1)).alias(f"{col}__adj_n"),
            F.sum(y_t_c).alias(f"{col}__sum_y"),
            F.sum(y_t_c * y_t_c).alias(f"{col}__sum_y2"),
            F.sum(x_c).alias(f"{col}__sum_x"),
            F.sum(x_c * x_c).alias(f"{col}__sum_x2"),
            F.sum(x_c * y_t_c).alias(f"{col}__sum_xy"),
        ])

    stats_df = d.groupBy(cluster_label_col, cohort_col).agg(*agg_exprs)
    stats_pdf = stats_df.toPandas()

    # Build results in Python (no more Spark jobs)
    p_rows = []
    for cluster_val in stats_pdf[cluster_label_col].unique():
        cluster_stats = stats_pdf[stats_pdf[cluster_label_col] == cluster_val]
        stats_a = cluster_stats[cluster_stats[cohort_col] == cohort_a]
        stats_b = cluster_stats[cluster_stats[cohort_col] == cohort_b]

        for metric_label, col in metric_specs:
            n_a = int(stats_a[f"{col}_n"].values[0]) if len(stats_a) > 0 else 0
            n_b = int(stats_b[f"{col}_n"].values[0]) if len(stats_b) > 0 else 0
            m_a = float(stats_a[f"{col}_mean"].values[0]) if len(stats_a) > 0 and n_a > 0 else None
            m_b = float(stats_b[f"{col}_mean"].values[0]) if len(stats_b) > 0 and n_b > 0 else None
            # Unadjusted p uses transformed stats (if requested)
            tn_a = int(stats_a[f"{col}__t_n"].values[0]) if len(stats_a) > 0 else 0
            tn_b = int(stats_b[f"{col}__t_n"].values[0]) if len(stats_b) > 0 else 0
            tm_a = float(stats_a[f"{col}__t_mean"].values[0]) if len(stats_a) > 0 and tn_a > 0 else None
            tm_b = float(stats_b[f"{col}__t_mean"].values[0]) if len(stats_b) > 0 and tn_b > 0 else None
            tv_a = stats_a[f"{col}__t_var"].values[0] if len(stats_a) > 0 and tn_a > 1 else None
            tv_b = stats_b[f"{col}__t_var"].values[0] if len(stats_b) > 0 and tn_b > 1 else None

            diff_mean = (m_b - m_a) if m_a is not None and m_b is not None else None

            # Welch t-test (unadjusted p-value)
            p_unadj = None
            if tn_a >= 2 and tn_b >= 2 and tv_a is not None and tv_b is not None and tv_a > 0 and tv_b > 0:
                se2 = (tv_a / tn_a) + (tv_b / tn_b)
                if se2 > 0:
                    t_stat = (tm_a - tm_b) / (se2 ** 0.5)
                    df_num = se2 ** 2
                    df_den = ((tv_a / tn_a) ** 2) / (tn_a - 1) + ((tv_b / tn_b) ** 2) / (tn_b - 1)
                    if df_den > 0:
                        dof = df_num / df_den
                        p_unadj = float(2.0 * t_dist.sf(abs(t_stat), dof))

            # Adjusted p-value: residualize y (optionally log1p) against CCI using pooled
            # (both cohorts) within-cluster linear regression, then Welch t-test on residual means.
            p_adj = None
            try:
                an_a = int(stats_a[f"{col}__adj_n"].values[0]) if len(stats_a) > 0 else 0
                an_b = int(stats_b[f"{col}__adj_n"].values[0]) if len(stats_b) > 0 else 0
                if an_a >= 2 and an_b >= 2:
                    sum_y_a = float(stats_a[f"{col}__sum_y"].values[0])
                    sum_y2_a = float(stats_a[f"{col}__sum_y2"].values[0])
                    sum_x_a = float(stats_a[f"{col}__sum_x"].values[0])
                    sum_x2_a = float(stats_a[f"{col}__sum_x2"].values[0])
                    sum_xy_a = float(stats_a[f"{col}__sum_xy"].values[0])

                    sum_y_b = float(stats_b[f"{col}__sum_y"].values[0])
                    sum_y2_b = float(stats_b[f"{col}__sum_y2"].values[0])
                    sum_x_b = float(stats_b[f"{col}__sum_x"].values[0])
                    sum_x2_b = float(stats_b[f"{col}__sum_x2"].values[0])
                    sum_xy_b = float(stats_b[f"{col}__sum_xy"].values[0])

                    n_all = an_a + an_b
                    sum_y = sum_y_a + sum_y_b
                    sum_x = sum_x_a + sum_x_b
                    sum_y2 = sum_y2_a + sum_y2_b
                    sum_x2 = sum_x2_a + sum_x2_b
                    sum_xy = sum_xy_a + sum_xy_b

                    xbar = sum_x / n_all
                    ybar = sum_y / n_all
                    sxx = sum_x2 - (n_all * xbar * xbar)
                    if sxx > 0:
                        sxy = sum_xy - (n_all * xbar * ybar)
                        b_hat = sxy / sxx
                        a_hat = ybar - (b_hat * xbar)

                        # cohort-specific residual means
                        mean_y_a = sum_y_a / an_a
                        mean_x_a = sum_x_a / an_a
                        mean_y_b = sum_y_b / an_b
                        mean_x_b = sum_x_b / an_b
                        mean_res_a = mean_y_a - (a_hat + b_hat * mean_x_a)
                        mean_res_b = mean_y_b - (a_hat + b_hat * mean_x_b)

                        # cohort-specific residual variances from SSE using only aggregates
                        def _sse(n, sy, sy2, sx, sx2, sxy):
                            return (
                                sy2
                                + (a_hat * a_hat) * n
                                + (b_hat * b_hat) * sx2
                                - 2.0 * a_hat * sy
                                - 2.0 * b_hat * sxy
                                + 2.0 * a_hat * b_hat * sx
                            )

                        sse_a = _sse(an_a, sum_y_a, sum_y2_a, sum_x_a, sum_x2_a, sum_xy_a)
                        sse_b = _sse(an_b, sum_y_b, sum_y2_b, sum_x_b, sum_x2_b, sum_xy_b)

                        # Numerical guard
                        if sse_a >= 0 and sse_b >= 0:
                            var_res_a = sse_a / (an_a - 1)
                            var_res_b = sse_b / (an_b - 1)
                            if var_res_a > 0 and var_res_b > 0:
                                se2_adj = (var_res_a / an_a) + (var_res_b / an_b)
                                if se2_adj > 0:
                                    t_stat_adj = (mean_res_a - mean_res_b) / (se2_adj ** 0.5)
                                    df_num = se2_adj ** 2
                                    df_den = ((var_res_a / an_a) ** 2) / (an_a - 1) + ((var_res_b / an_b) ** 2) / (an_b - 1)
                                    if df_den > 0:
                                        dof_adj = df_num / df_den
                                        p_adj = float(2.0 * t_dist.sf(abs(t_stat_adj), dof_adj))
            except Exception:
                p_adj = None

            p_rows.append({
                "Metric": metric_label,
                "cluster_label": cluster_val,
                f"mean_{cohort_a}": m_a,
                f"mean_{cohort_b}": m_b,
                "diff_mean": diff_mean,
                f"n_{cohort_a}": n_a if n_a > 0 else None,
                f"n_{cohort_b}": n_b if n_b > 0 else None,
                "p_value_unadjusted": p_unadj,
                "p_value_adjusted_for_prior_year_max_cci": p_adj,
            })

    result_pdf = pd.DataFrame(p_rows)

    # Preserve caller-provided metric order
    metric_order = [lbl for (lbl, _) in metric_specs]
    result_pdf["_metric_order"] = result_pdf["Metric"].apply(
        lambda x: metric_order.index(x) if x in metric_order else 999
    )

    # Optional cluster ordering
    if cluster_order is not None:
        result_pdf["_cluster_order"] = result_pdf["cluster_label"].apply(
            lambda x: cluster_order.index(x) if x in cluster_order else 999
        )
        result_pdf = result_pdf.sort_values(["_cluster_order", "_metric_order"]).drop(columns=["_cluster_order"])
    else:
        result_pdf = result_pdf.sort_values(["cluster_label", "_metric_order"])

    result_pdf = result_pdf.drop(columns=["_metric_order"])

    spark_session = df.sparkSession
    return spark_session.createDataFrame(result_pdf)


def cluster_cohort_counts(
    df,
    cohort_col: str,
    cluster_label_col: str,
    cohort_order=None,
    cluster_order=None,
):
    """
    Simple publication-friendly cluster sizes by cohort (n rows per cluster per cohort).
    """
    cohort_order = cohort_order or ["cohort_1", "cohort_2a"]
    d = df.where(F.col(cluster_label_col).isNotNull() & F.col(cohort_col).isNotNull())
    counts = d.groupBy(cluster_label_col, cohort_col).agg(F.count("*").cast("long").alias("n"))
    wide = (
        counts.select(
            F.col(cluster_label_col).alias("cluster_label"),
            F.col(cohort_col).alias("cohort"),
            F.col("n"),
        )
        .groupBy("cluster_label")
        .pivot("cohort", cohort_order)
        .agg(F.first("n"))
    )

    if cluster_order is not None:
        cl_arr = F.array(*[F.lit(x) for x in cluster_order])
        wide = wide.withColumn("_cluster_order", F.array_position(cl_arr, F.col("cluster_label")))
        wide = wide.orderBy(F.col("_cluster_order").asc(), F.col("cluster_label").asc()).drop("_cluster_order")
    else:
        wide = wide.orderBy("cluster_label")

    return wide


def pivot_metric_by_cluster(
    df_long,
    value_col: str,
    cluster_order=None,
    metric_order=None,
    formatted_col: str | None = None,
):
    """
    Pivot a long within-cluster table into a Metric x Cluster wide table.

    Parameters
    ----------
    df_long : Spark DataFrame
        Must include: Metric, cluster_label, and value_col.
    value_col : str
        Column to pivot into cluster columns.
    cluster_order : list[str] | None
        Optional explicit cluster column order (passed to pivot()).
    metric_order : list[str] | None
        Optional explicit metric row order.
    formatted_col : str | None
        If provided, pivot this column name instead of value_col (used for string-formatted output).
    """
    vcol = formatted_col or value_col
    base = df_long.select("Metric", "cluster_label", F.col(vcol).alias("value"))
    wide = base.groupBy("Metric").pivot("cluster_label", cluster_order).agg(F.first("value"))

    if metric_order is not None:
        metric_arr = F.array(*[F.lit(x) for x in metric_order])
        wide = wide.withColumn("_metric_order", F.array_position(metric_arr, F.col("Metric")))
        wide = wide.orderBy(F.col("_metric_order").asc(), F.col("Metric").asc()).drop("_metric_order")
    else:
        wide = wide.orderBy("Metric")

    return wide


def with_formatted_diff(df_long, diff_col: str = "diff_mean", out_col: str = "diff_fmt", decimals: int = 1):
    """
    Add a string-formatted diff column (e.g., +12.3, -4.0) for publication-like tables.
    """
    fmt = f"%+.{int(decimals)}f"
    return df_long.withColumn(out_col, F.when(F.col(diff_col).isNull(), F.lit(None)).otherwise(F.format_string(fmt, F.col(diff_col))))


def with_formatted_pvalue_stars(
    df_long,
    p_col: str,
    out_col: str = "p_fmt",
    stars: bool = True,
):
    """
    Add a string-formatted p-value column, optionally with significance stars.
    """
    p = F.col(p_col)
    p_fmt = F.when(p.isNull(), F.lit(None)).otherwise(F.format_string("%.2e", p))

    if not stars:
        return df_long.withColumn(out_col, p_fmt)

    p_stars = (
        F.when(p.isNull(), F.lit(""))
        .when(p < F.lit(0.001), F.lit("***"))
        .when(p < F.lit(0.01), F.lit("**"))
        .when(p < F.lit(0.05), F.lit("*"))
        .otherwise(F.lit(""))
    )
    return df_long.withColumn(out_col, F.concat(p_fmt, p_stars))


def with_pub_cell_diff_p(
    df_long,
    diff_col: str = "diff_mean",
    p_col: str = "p_value_adjusted_for_prior_year_max_cci",
    out_col: str = "cell",
    diff_decimals: int = 1,
    stars: bool = True,
):
    """
    Create a single publication-style cell string combining effect size + p-value.

    Example cell: "+12.3 (p=1.20e-04**)"
    """
    tmp = with_formatted_diff(df_long, diff_col=diff_col, out_col="_diff_fmt", decimals=diff_decimals)
    tmp = with_formatted_pvalue_stars(tmp, p_col=p_col, out_col="_p_fmt", stars=stars)
    return tmp.withColumn(
        out_col,
        F.when(F.col("_diff_fmt").isNull() | F.col("_p_fmt").isNull(), F.lit(None)).otherwise(
            F.concat(F.col("_diff_fmt"), F.lit(" (p="), F.col("_p_fmt"), F.lit(")"))
        ),
    ).drop("_diff_fmt", "_p_fmt")


def build_pub_cohort_comparison_matrix(
    df_long,
    cluster_order=None,
    metric_order=None,
    diff_col: str = "diff_mean",
    p_col: str = "p_value_adjusted_for_prior_year_max_cci",
    diff_decimals: int = 1,
    stars: bool = True,
):
    """
    Build a single wide table (Metric x Cluster) with cells like:
      Δ (p-value)
    """
    formatted = with_pub_cell_diff_p(
        df_long,
        diff_col=diff_col,
        p_col=p_col,
        out_col="cell",
        diff_decimals=diff_decimals,
        stars=stars,
    )
    return pivot_metric_by_cluster(
        df_long=formatted,
        value_col="cell",
        cluster_order=cluster_order,
        metric_order=metric_order,
        formatted_col="cell",
    )


def build_html_cohort_comparison_table(
    df_long,
    cluster_counts_df,
    cluster_order=None,
    metric_order=None,
    diff_col: str = "diff_mean",
    p_col: str = "p_value_unadjusted",
    p_adj_col: str = "p_value_adjusted_for_prior_year_max_cci",
    diff_decimals: int = 1,
    title: str = "Cohort 1 vs Cohort 2a Comparison",
):
    """
    Build a styled HTML table for cohort comparisons.

    Layout:
    - Columns: clusters (header includes n1/n2)
    - Rows: metrics
    - Cells: Δ value (green if +, red if -) with p and p_adj below

    Returns HTML string suitable for displayHTML(...) in Databricks.
    """
    # Collect data to driver and convert to dicts for easier access
    rows = [row.asDict() for row in df_long.collect()]
    counts_rows = [row.asDict() for row in cluster_counts_df.collect()]

    # Build cluster -> (n1, n2) lookup
    cluster_n = {}
    for r in counts_rows:
        cluster_n[r["cluster_label"]] = (r.get("cohort_1", 0) or 0, r.get("cohort_2a", 0) or 0)

    # Determine cluster order
    if cluster_order is None:
        cluster_order = sorted(cluster_n.keys())

    # Determine metric order
    if metric_order is None:
        metric_order = sorted(set(r["Metric"] for r in rows))

    # Build lookup: (metric, cluster) -> (diff, p, p_adj)
    data = {}
    for r in rows:
        key = (r["Metric"], r["cluster_label"])
        data[key] = (r.get(diff_col), r.get(p_col), r.get(p_adj_col))

    # Helper: format diff with color (green if +, red if -)
    def fmt_diff(val):
        if val is None:
            return '<span style="color:#999;">—</span>'
        # Round first to check for -0.0 edge case
        rounded = round(val, diff_decimals)
        if rounded == 0:
            # Avoid displaying "-0.0" or "+0.0" - just show "0.0" in neutral gray
            return '<span style="color:#111 !important; font-weight:700;">0.0</span>'
        color = "#166534" if rounded > 0 else "#b91c1c"
        sign = "+" if rounded > 0 else ""
        return f'<span style="color:{color} !important; font-weight:700;">{sign}{rounded:.{diff_decimals}f}</span>'

    # Helper: format p-value with threshold display
    def fmt_p_threshold(val):
        """Format p-value: show value if >0.05, else show ≤threshold."""
        if val is None:
            return "—"
        if val <= 0.001:
            return "≤0.001"
        elif val <= 0.01:
            return "≤0.01"
        elif val <= 0.05:
            return "≤0.05"
        else:
            return f"{val:.2f}"

    def _is_sig(val, alpha: float = 0.05) -> bool:
        try:
            return (val is not None) and float(val) <= alpha
        except Exception:
            return False

    def _cell_style(diff_val, p_val, p_adj_val) -> str:
        """
        Return inline CSS for a cell to emphasize significance:
        - Strong highlight if p_adj is significant (CCI-adjusted)
        - Lighter highlight if only unadjusted p is significant
        - Neutral if not significant / not substantive
        """
        rounded_diff = round(diff_val, diff_decimals) if diff_val is not None else None
        if rounded_diff == 0 or diff_val is None:
            return "border:1px solid #d1d5db; padding:10px 12px; text-align:center; background:#fff;"

        # Significance tiers
        adj_sig = _is_sig(p_adj_val, 0.05)
        unadj_sig = _is_sig(p_val, 0.05)

        # Border accent matches sign
        accent = "#166534" if rounded_diff > 0 else "#b91c1c"

        # Use border emphasis (not background tint) for consistent readability in Databricks
        if adj_sig:
            return (
                f"border:1px solid #d1d5db; border-left:5px solid {accent}; "
                f"padding:10px 12px; text-align:center; background:#fff;"
            )
        if unadj_sig:
            return (
                f"border:1px solid #d1d5db; border-left:3px solid {accent}; "
                f"padding:10px 12px; text-align:center; background:#fff;"
            )

        return "border:1px solid #d1d5db; padding:10px 12px; text-align:center; background:#fff;"

    # Helper: build cell with diff and two p-values
    def fmt_cell(diff_val, p_val, p_adj_val):
        diff_html = fmt_diff(diff_val)
        # Check if rounded diff is zero - if so, p-values are not practically meaningful
        rounded_diff = round(diff_val, diff_decimals) if diff_val is not None else None
        if rounded_diff == 0:
            # Gray out p-values with explanation
            p_html = '<span class="ns" style="color:#111 !important;">n.s.</span>'
        else:
            p_str = fmt_p_threshold(p_val)
            p_adj_str = fmt_p_threshold(p_adj_val)
            # Emphasize significance via text tone (background handled at <td> level)
            adj_sig = _is_sig(p_adj_val, 0.05)
            unadj_sig = _is_sig(p_val, 0.05)
            weight = "600" if adj_sig else ("500" if unadj_sig else "400")
            p_html = (
                f'<span class="ptext" style="color:#111 !important; font-weight:{weight};">'
                f'p={p_str}, p<sub>adj</sub>={p_adj_str}'
                f'</span>'
            )
        return f'{diff_html}<br>{p_html}'

    # Build HTML
    html = []
    html.append(
        """
<style>
  /* Databricks-safe: force HIGH-CONTRAST table regardless of theme */
  .cohort-compare-wrap { font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial; }
  .cohort-compare-title { margin: 0 0 10px 0; font-weight: 800; letter-spacing: 0.2px; color:#111; }

  .cohort-compare {
    border-collapse: separate;
    border-spacing: 0;
    width: 100%;
    background: #fff;
    color: #111;
    font-size: 13px;
    border: 1px solid #d1d5db;
    border-radius: 10px;
    overflow: hidden;
  }

  .cohort-compare thead th {
    position: sticky;
    top: 0;
    background: #f3f4f6;
    z-index: 2;
    border-bottom: 1px solid #d1d5db;
    color:#111;
  }

  .cohort-compare th, .cohort-compare td { border-right: 1px solid #d1d5db; vertical-align: middle; color:#111; }
  .cohort-compare tr > *:last-child { border-right: none; }

  .cohort-compare th { padding: 10px 12px; text-align: center; font-weight: 800; line-height: 1.2; }
  .cohort-compare th.metric { text-align: left; min-width: 220px; }
  .cohort-compare th.cluster { max-width: 200px; white-space: normal; word-break: break-word; }

  .cohort-compare td { padding: 10px 12px; text-align: center; background:#fff; color:#111; }
  .cohort-compare tbody tr:hover td { background:#fafafa; }

  .cohort-compare .counts { font-size: 12px; font-weight: 600; color:#111; opacity: 0.75; }
  .cohort-compare .ptext { font-size: 12px; color:#111; }
  .cohort-compare .ns { font-style: italic; font-size: 12px; color:#111; opacity:0.6; }

  .cohort-compare-note { margin: 8px 0 0 0; font-size: 12px; color:#111; opacity:0.75; line-height: 1.35; }
  .legend-chip { padding: 1px 6px; border-radius: 999px; border: 1px solid #d1d5db; background:#fff; }
</style>
"""
    )
    html.append(f'<div class="cohort-compare-wrap"><div class="cohort-compare-title">{title}</div>')
    html.append('<table class="cohort-compare">')

    # Header row
    html.append('<thead><tr>')
    html.append('<th class="metric">Metric</th>')
    for cl in cluster_order:
        n1, n2 = cluster_n.get(cl, (0, 0))
        html.append(
            f'<th class="cluster">{cl}<br><span class="counts">n₁={n1:,} · n₂={n2:,}</span></th>'
        )
    html.append('</tr></thead>')

    # Body rows (all white background)
    html.append('<tbody>')
    for metric in metric_order:
        html.append('<tr>')
        html.append(f'<td style="text-align:left; font-weight:700; border-top:1px solid #d1d5db;">{metric}</td>')
        for cl in cluster_order:
            diff_val, p_val, p_adj_val = data.get((metric, cl), (None, None, None))
            cell_html = fmt_cell(diff_val, p_val, p_adj_val)
            td_style = _cell_style(diff_val, p_val, p_adj_val)
            # Ensure a top border for row separation (inline style already sets border-left etc.)
            html.append(f'<td style="{td_style} border-top:1px solid #d1d5db;">{cell_html}</td>')
        html.append('</tr>')
    html.append('</tbody>')

    html.append('</table>')
    html.append('<div class="cohort-compare-note">')
    html.append('Δ = mean(Cohort 2a) − mean(Cohort 1). p = unadjusted, p<sub>adj</sub> = adjusted for prior-year max CCI.<br>')
    html.append('<span class="ns">n.s.</span> = not substantive (Δ rounds to 0; p-values omitted).<br>')
    html.append('<span class="legend-chip"><b>thick border</b></span> = p<sub>adj</sub>≤0.05; '
                '<span class="legend-chip"><b>thin border</b></span> = p≤0.05 only. Border color matches Δ direction.')
    html.append('</div></div>')

    return "\n".join(html)

