# Databricks notebook source
# MAGIC %md
# MAGIC **Purpose**: Cohort 2a analysis - consolidate cluster assignments with features and run comparative “counts of stuff” summaries.

# COMMAND ----------

import warnings
from pyspark.sql import functions as F

# COMMAND ----------

# MAGIC %run ./99_utils

# COMMAND ----------

warnings.filterwarnings("ignore")
VERBOSE = get_verbose(default=True)
gate_prints(VERBOSE)
configure_spark_optimizations()

input_db = "cohort_2a"

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Load data

# COMMAND ----------

features = spark.table(f"{input_db}.archetype_features_yearly")
unified_yearly = spark.table(f"{input_db}.unified_yearly")
rules = spark.table(f"{input_db}.rules_yearly").withColumnRenamed("archetype", "rules_vanilla_cluster")
gmm4 = spark.table(f"{input_db}.gmm_4_yearly")
gmm7 = spark.table(f"{input_db}.gmm_7_yearly")

# Join features with unified_yearly to get concept count columns for analysis
concept_cols = [
    "total_data_points", "total_unique_concepts",
    "condition_count", "condition_unique_ct",
    "drug_count", "drug_unique_ct",
    "procedure_count", "procedure_unique_ct",
    "measurement_count", "measurement_unique_ct",
    "observation_count", "observation_unique_ct",
]
features = features.join(
    unified_yearly.select("person_id", F.col("period").cast("int").alias("year"), *concept_cols),
    on=["person_id", "year"],
    how="left",
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Consolidated label table

# COMMAND ----------

characterization_data = (
    features
    .join(rules.select("person_id", "year", "rules_vanilla_cluster"), ["person_id", "year"], "left")
    .join(gmm4.select("person_id", "year", "gmm_4_cluster", "gmm_4_confidence"), ["person_id", "year"], "left")
    .join(gmm7.select("person_id", "year", "gmm_7_cluster", "gmm_7_confidence"), ["person_id", "year"], "left")
)

characterization_data.cache()
characterization_data.count()
characterization_data.write.format("delta").mode("overwrite").option("overwriteSchema", "true") \
    .saveAsTable(f"{input_db}.yearly_archetypes_allmethods")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Comparative “counts of stuff” by cluster (means + p-values)
# MAGIC
# MAGIC For each clustering method (rules → gmm_4 → gmm_7), this produces a transposed table:
# MAGIC - Rows = metrics
# MAGIC - Columns = cluster labels
# MAGIC - Final columns:
# MAGIC   - `p_value_unadjusted`: one-way ANOVA across clusters
# MAGIC   - `p_value_adjusted_for_prior_year_max_cci`: ANCOVA-style partial F-test controlling for `prior_year_max_cci`

# COMMAND ----------

GMM4_LABELS = {
    0: "Moderate Inpatient",
    1: "Sparse Use",
    2: "Outpatient-Only",
    3: "High Inpatient",
}

GMM7_LABELS = {
    0: "Outpatient-Only – High Volume, Irregular Visits",
    1: "Sparse Use",
    2: "Moderate Inpatient – Few Admissions, Short Stays",
    3: "High Inpatient – Few Admissions, Prolonged Stays",
    4: "Outpatient-Only – Low Volume, Regular Visits",
    5: "High Inpatient – Many Admissions, High Volume",
    6: "Moderate Inpatient – Many Admissions, Short Stays",
}

METRICS = [
    # All concepts (all domains)
    ("All concepts (Non-unique)", "total_data_points"),
    ("All concepts (Unique)", "total_unique_concepts"),

    # Domain breakdown (non-unique then unique)
    ("Condition concepts (Non-unique)", "condition_count"),
    ("Condition concepts (Unique)", "condition_unique_ct"),
    ("Drug concepts (Non-unique)", "drug_count"),
    ("Drug concepts (Unique)", "drug_unique_ct"),
    ("Procedure concepts (Non-unique)", "procedure_count"),
    ("Procedure concepts (Unique)", "procedure_unique_ct"),
    ("Measurement concepts (Non-unique)", "measurement_count"),
    ("Measurement concepts (Unique)", "measurement_unique_ct"),
    ("Observation concepts (Non-unique)", "observation_count"),
    ("Observation concepts (Unique)", "observation_unique_ct"),
]

# Cluster orderings: low → high utilization
RULES_CLUSTER_ORDER = [
    "Sparse Use",
    "Regular Infrequent",
    "Irregular Infrequent",
    "Regular Frequent",
    "Irregular Frequent",
    "Multiple Complex Episodes",
    "Sporadic Complex Episodes",
]

# Rules
rules_table = build_transposed_means_with_pvalues(
    df=characterization_data,
    cluster_label_col="rules_vanilla_cluster",
    metric_specs=METRICS,
    pivot_order=RULES_CLUSTER_ORDER,
)
if rules_table is not None:
    display(rules_table)

# GMM-4 (named)
gmm4_label_expr = F.coalesce(
    spark_map_expr(GMM4_LABELS).getItem(F.col("gmm_4_cluster")),
    F.concat(F.lit("Cluster "), F.col("gmm_4_cluster").cast("string")),
)
gmm4_named = characterization_data.withColumn("gmm_4_named", gmm4_label_expr)
gmm4_pivot_order = ["Sparse Use", "Outpatient-Only", "Moderate Inpatient", "High Inpatient"]
gmm4_table = build_transposed_means_with_pvalues(
    df=gmm4_named,
    cluster_label_col="gmm_4_named",
    metric_specs=METRICS,
    pivot_order=gmm4_pivot_order,
)
if gmm4_table is not None:
    display(gmm4_table)

# GMM-7 (named) - ordered low → high utilization
gmm7_label_expr = F.coalesce(
    spark_map_expr(GMM7_LABELS).getItem(F.col("gmm_7_cluster")),
    F.concat(F.lit("Cluster "), F.col("gmm_7_cluster").cast("string")),
)
gmm7_named = characterization_data.withColumn("gmm_7_named", gmm7_label_expr)
gmm7_pivot_order = [
    "Sparse Use",
    "Outpatient-Only – Low Volume, Regular Visits",
    "Outpatient-Only – High Volume, Irregular Visits",
    "Moderate Inpatient – Few Admissions, Short Stays",
    "Moderate Inpatient – Many Admissions, Short Stays",
    "High Inpatient – Few Admissions, Prolonged Stays",
    "High Inpatient – Many Admissions, High Volume",
]
gmm7_table = build_transposed_means_with_pvalues(
    df=gmm7_named,
    cluster_label_col="gmm_7_named",
    metric_specs=METRICS,
    pivot_order=gmm7_pivot_order,
)
if gmm7_table is not None:
    display(gmm7_table)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Cross-cohort comparison: Cohort 1 vs Cohort 2a (within the same cluster)
# MAGIC
# MAGIC For each clustering method, and **within each cluster**, compare Cohort 1 vs Cohort 2a on the same metrics.
# MAGIC
# MAGIC Output columns:
# MAGIC - `mean_cohort_1`, `mean_cohort_2a`, `diff_mean`
# MAGIC - `p_value_unadjusted` (Welch t-test)
# MAGIC - `p_value_adjusted_for_prior_year_max_cci` (partial F-test controlling for prior-year CCI)

# COMMAND ----------

cohort1_db = "cohort_1"

features_c1 = spark.table(f"{cohort1_db}.archetype_features_yearly")
unified_yearly_c1 = spark.table(f"{cohort1_db}.unified_yearly")
rules_c1 = spark.table(f"{cohort1_db}.rules_yearly").withColumnRenamed("archetype", "rules_vanilla_cluster")
gmm4_c1 = spark.table(f"{cohort1_db}.gmm_4_yearly")
gmm7_c1 = spark.table(f"{cohort1_db}.gmm_7_yearly")

# Join cohort_1 features with unified_yearly to get concept count columns
features_c1 = features_c1.join(
    unified_yearly_c1.select("person_id", F.col("period").cast("int").alias("year"), *concept_cols),
    on=["person_id", "year"],
    how="left",
)

char_c1 = (
    features_c1
    .join(rules_c1.select("person_id", "year", "rules_vanilla_cluster"), ["person_id", "year"], "left")
    .join(gmm4_c1.select("person_id", "year", "gmm_4_cluster", "gmm_4_confidence"), ["person_id", "year"], "left")
    .join(gmm7_c1.select("person_id", "year", "gmm_7_cluster", "gmm_7_confidence"), ["person_id", "year"], "left")
    .withColumn("cohort", F.lit("cohort_1"))
)

char_c2a = characterization_data.withColumn("cohort", F.lit("cohort_2a"))

combined = char_c1.unionByName(char_c2a, allowMissingColumns=True)

# Rules: within-cluster cohort comparison
metric_order = [lbl for (lbl, _) in METRICS]

rules_cohort_compare = build_within_cluster_cohort_comparison(
    df=combined,
    cohort_col="cohort",
    cluster_label_col="rules_vanilla_cluster",
    metric_specs=METRICS,
    cluster_order=RULES_CLUSTER_ORDER,
    adjust_for_days_observed=True,
    adjust_for_year=False,
    adjusted_y_transform="log1p",
)
rules_sizes = cluster_cohort_counts(
    df=combined,
    cohort_col="cohort",
    cluster_label_col="rules_vanilla_cluster",
    cluster_order=RULES_CLUSTER_ORDER,
)
rules_html = build_html_cohort_comparison_table(
    df_long=rules_cohort_compare,
    cluster_counts_df=rules_sizes,
    cluster_order=RULES_CLUSTER_ORDER,
    metric_order=metric_order,
    title="Rules-based Clustering: Cohort 1 vs Cohort 2a",
)
displayHTML(rules_html)

# GMM-4 (named): within-cluster cohort comparison
gmm4_combined = combined.withColumn("gmm_4_named", gmm4_label_expr)
gmm4_cohort_compare = build_within_cluster_cohort_comparison(
    df=gmm4_combined,
    cohort_col="cohort",
    cluster_label_col="gmm_4_named",
    metric_specs=METRICS,
    cluster_order=gmm4_pivot_order,
    adjust_for_days_observed=True,
    adjust_for_year=False,
    adjusted_y_transform="log1p",
)
gmm4_sizes = cluster_cohort_counts(
    df=gmm4_combined,
    cohort_col="cohort",
    cluster_label_col="gmm_4_named",
    cluster_order=gmm4_pivot_order,
)
gmm4_html = build_html_cohort_comparison_table(
    df_long=gmm4_cohort_compare,
    cluster_counts_df=gmm4_sizes,
    cluster_order=gmm4_pivot_order,
    metric_order=metric_order,
    title="GMM-4 Clustering: Cohort 1 vs Cohort 2a",
)
displayHTML(gmm4_html)

# GMM-7 (named): within-cluster cohort comparison
gmm7_combined = combined.withColumn("gmm_7_named", gmm7_label_expr)
gmm7_cohort_compare = build_within_cluster_cohort_comparison(
    df=gmm7_combined,
    cohort_col="cohort",
    cluster_label_col="gmm_7_named",
    metric_specs=METRICS,
    cluster_order=gmm7_pivot_order,
    adjust_for_days_observed=True,
    adjust_for_year=False,
    adjusted_y_transform="log1p",
)
gmm7_sizes = cluster_cohort_counts(
    df=gmm7_combined,
    cohort_col="cohort",
    cluster_label_col="gmm_7_named",
    cluster_order=gmm7_pivot_order,
)
gmm7_html = build_html_cohort_comparison_table(
    df_long=gmm7_cohort_compare,
    cluster_counts_df=gmm7_sizes,
    cluster_order=gmm7_pivot_order,
    metric_order=metric_order,
    title="GMM-7 Clustering: Cohort 1 vs Cohort 2a",
)
displayHTML(gmm7_html)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Cleanup

# COMMAND ----------

characterization_data.unpersist()

