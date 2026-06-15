# Databricks notebook source
# MAGIC %md
# MAGIC **Purpose**: Apply rules-based and pretrained GMM (K=4, K=7) clustering to cohort_2b person-year features (no retraining; reuse cohort_1 models).

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Setup

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.window import Window

# COMMAND ----------

# MAGIC %run ./99_utils

# COMMAND ----------

VERBOSE = get_verbose(default=True)
gate_prints(VERBOSE)
configure_spark_optimizations()

input_db = "cohort_2b"
output_db = "cohort_2b"

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Load Features

# COMMAND ----------

features = spark.table(f"{input_db}.archetype_features_yearly")
features.cache()
features.count()

features_for_clustering = get_core_clustering_features(use_continuous_irregularity=True, irregularity_penalty="BOTH")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Rules-Based Classification (vanilla)

# COMMAND ----------

IRREGULARITY_THRESHOLD = 0.5
person_window = Window.partitionBy("person_id").orderBy("year")

features_rules = features.withColumn(
    "irregularity_score",
    F.col("irregularity_l1")
).withColumn(
    "is_regular",
    (F.col("irregularity_score") < F.lit(IRREGULARITY_THRESHOLD)).cast("boolean")
)

classify_udf = make_classify_archetype_udf(irregularity_threshold=IRREGULARITY_THRESHOLD)

rules_yearly = features_rules.withColumn(
    "classification_result",
    classify_udf(
        F.col("inpatient_visit_count"),
        F.col("visit_count"),
        F.col("irregularity_score")
    )
).withColumn(
    "archetype",
    F.col("classification_result.archetype")
).withColumn(
    "archetype_reason",
    F.col("classification_result.reason")
).drop("classification_result")

rules_yearly = rules_yearly.withColumn(
    "prior_year_archetype_rules",
    F.lag("archetype", 1).over(person_window)
).withColumn(
    "archetype_changed",
    F.when(
        (F.col("prior_year_archetype_rules").isNotNull()) &
        (F.col("archetype") != F.col("prior_year_archetype_rules")),
        1
    ).otherwise(0)
).withColumn(
    "segment_id",
    F.sum(F.col("archetype_changed")).over(
        Window.partitionBy("person_id").orderBy("year").rowsBetween(Window.unboundedPreceding, 0)
    )
).withColumn(
    "years_in_current_archetype",
    F.row_number().over(
        Window.partitionBy("person_id", "segment_id").orderBy("year")
    )
).withColumn(
    "archetype_confidence",
    F.lit(1.0).cast("double")
)

rules_yearly.write.format("delta").mode("overwrite").option("overwriteSchema", "true") \
    .saveAsTable(f"{output_db}.rules_yearly")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Apply GMM-4 and GMM-7 (pretrained bundles)

# COMMAND ----------

GMM4_MODEL_ID = "gmm_4"
GMM7_MODEL_ID = "gmm_7"
BUNDLE_NAME = "gmm_bundle.pkl"

gmm4_preds = predict_with_gmm(
    model_id=GMM4_MODEL_ID,
    bundle_artifact=BUNDLE_NAME,
    features_spark=features,
    features_for_clustering=features_for_clustering,
    irregularity_penalty="BOTH",
    spark_session=spark,
)
gmm7_preds = predict_with_gmm(
    model_id=GMM7_MODEL_ID,
    bundle_artifact=BUNDLE_NAME,
    features_spark=features,
    features_for_clustering=features_for_clustering,
    irregularity_penalty="BOTH",
    spark_session=spark,
)

gmm4_output = (
    features.join(gmm4_preds, ["person_id", "year"], "left")
    .withColumn("gmm_4_cluster_label", F.concat(F.lit("Cluster "), F.col("gmm_4_cluster").cast("string")))
)
gmm7_output = (
    features.join(gmm7_preds, ["person_id", "year"], "left")
    .withColumn("gmm_7_cluster_label", F.concat(F.lit("Cluster "), F.col("gmm_7_cluster").cast("string")))
)

gmm4_output.write.format("delta").mode("overwrite").option("overwriteSchema", "true") \
    .saveAsTable(f"{output_db}.gmm_4_yearly")
gmm7_output.write.format("delta").mode("overwrite").option("overwriteSchema", "true") \
    .saveAsTable(f"{output_db}.gmm_7_yearly")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Cleanup

# COMMAND ----------

features.unpersist()
