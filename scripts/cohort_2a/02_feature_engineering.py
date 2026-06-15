# Databricks notebook source
# MAGIC %md
# MAGIC **Purpose**: Person-year features for cohort_2a (cohort_1 spec)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Setup and Configuration

# COMMAND ----------

# Scientific computing
import numpy as np

# PySpark
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, LongType

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.2 Load Shared Utilities

# COMMAND ----------

# MAGIC %run ./99_utils

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.3 Configure Logging and Optimization

# COMMAND ----------

VERBOSE = get_verbose(default=True)
gate_prints(VERBOSE)
configure_spark_optimizations()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Load and Validate Input Data
# MAGIC
# MAGIC ### 2.1 Load Unified Daily Table

# COMMAND ----------

# Load from Delta store
input_db = "cohort_2a"
unified_daily = spark.table(f"{input_db}.unified_daily")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.2 Select Required Columns

# COMMAND ----------

# Select required columns
unified_daily = unified_daily.select(
    F.col("person_id"),
    F.col("date").cast("date"),
    F.coalesce(F.col("visit_count"), F.lit(1)).cast("int").alias("visit_count"),
    F.col("visit_occurrence_ids"),
    F.coalesce(F.col("hospitalized_flag"), F.lit(0)).cast("int").alias("hospitalized_flag"),
    F.coalesce(F.col("total_data_points"), F.lit(1)).cast("int").alias("total_data_points"),
    # Clinical concept arrays for breadth calculation
    F.col("condition_concept_ids"),
    F.col("drug_concept_ids"),
    F.col("procedure_concept_ids"),
    # CCI columns for stratification
    F.coalesce(F.col("cci_total"), F.lit(0)).cast("int").alias("cci_total"),
    F.coalesce(F.col("cci_myocardial_infarction"), F.lit(0)).cast("int").alias("cci_myocardial_infarction"),
    F.coalesce(F.col("cci_congestive_heart_failure"), F.lit(0)).cast("int").alias("cci_congestive_heart_failure"),
    F.coalesce(F.col("cci_diabetes_uncomplicated"), F.lit(0)).cast("int").alias("cci_diabetes_uncomplicated"),
    F.coalesce(F.col("cci_diabetes_complicated"), F.lit(0)).cast("int").alias("cci_diabetes_complicated"),
    F.coalesce(F.col("cci_renal"), F.lit(0)).cast("int").alias("cci_renal"),
    F.coalesce(F.col("cci_cancer"), F.lit(0)).cast("int").alias("cci_cancer"),
    F.coalesce(F.col("cci_metastatic_cancer"), F.lit(0)).cast("int").alias("cci_metastatic_cancer")
).withColumn("year", F.year("date"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Cache Data for Performance

# COMMAND ----------

unified_daily = unified_daily.repartition("person_id").cache()
unified_daily.count()  # Materialize cache

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Observation Window Features

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4.1 Calculate Year-Level Observation Periods and Days Observed

# COMMAND ----------

observation_window = unified_daily.groupBy("person_id", "year").agg(
    F.min("date").alias("year_first_date"),
    F.max("date").alias("year_last_date"),
    F.countDistinct("date").alias("days_observed")
)


# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Core Feature: Visit Volume

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.1 Aggregate Total Visits per Person-Year

# COMMAND ----------

# Total visits per person-year (sum of daily visit_count)
visit_frequency = unified_daily.groupBy("person_id", "year").agg(
    F.sum("visit_count").alias("visit_count")
)


# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Core Feature: Inpatient Complexity

# COMMAND ----------

# MAGIC %md
# MAGIC ### 6.1 Count Distinct Inpatient Visits and Hospitalized Days

# COMMAND ----------

# Hospitalization features
hospitalization = unified_daily.groupBy("person_id", "year").agg(
    F.size(
        F.array_distinct(
            F.flatten(
                F.collect_list(
                    F.when(F.col("hospitalized_flag") == 1, F.col("visit_occurrence_ids"))
                )
            )
        )
    ).alias("inpatient_visit_count"),
    F.countDistinct(
        F.when(F.col("hospitalized_flag") == 1, F.col("date"))
    ).alias("hospitalized_days")
).fillna(0, subset=["inpatient_visit_count", "hospitalized_days"])


# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Feature: Regularity

# COMMAND ----------

# Define patient-year-date window for lag operations (within each year)
patient_date_window = Window.partitionBy("person_id", "year").orderBy("date")

# Filter to active days and calculate gaps
active_days = unified_daily.filter(F.col("visit_count") > 0)

active_days = active_days.withColumn(
    "prev_date",
    F.lag("date", 1).over(patient_date_window)
).withColumn(
    "days_between_visits",
    F.when(
        F.col("prev_date").isNotNull(),
        F.datediff("date", "prev_date")
    )
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 7.2 Collect Gaps per Person-Year

# COMMAND ----------

# Collect all gaps per person-year for median and norm calculations
gaps_collected = active_days.filter(
    F.col("days_between_visits").isNotNull()
).groupBy("person_id", "year").agg(
    F.collect_list("days_between_visits").alias("gaps_list"),
    F.count("days_between_visits").alias("num_gaps"),
    F.min("date").alias("first_visit"),
    F.max("date").alias("last_visit")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 7.3 Define Regularity Computation UDF

# COMMAND ----------

compute_regularity_udf = make_compute_regularity_udf()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 7.4 Apply Regularity Calculation

# COMMAND ----------

# Apply regularity calculation
regularity = gaps_collected.withColumn(
    "regularity_metrics",
    compute_regularity_udf(F.col("gaps_list"))
).select(
    "person_id",
    "year",
    F.col("regularity_metrics.irregularity_l1").alias("irregularity_l1"),
    F.col("regularity_metrics.irregularity_l2").alias("irregularity_l2"),
    F.col("regularity_metrics.central_diff").alias("central_diff"),
    F.col("num_gaps"),
    F.col("first_visit"),
    F.col("last_visit")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 7.5 Add Observation Span Context

# COMMAND ----------

# Add observation-window-relative boundary test
regularity = regularity.join(
    observation_window.select("person_id", "year", "year_first_date", "year_last_date", "days_observed"),
    ["person_id", "year"],
    "left"
).withColumn(
    "observation_span_days",
    F.datediff(F.col("last_visit"), F.col("first_visit"))
).select(
    "person_id",
    "year",
    F.col("irregularity_l1"),
    F.col("irregularity_l2"),
    F.col("central_diff"),
    F.col("observation_span_days")
)


# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Comorbidity and Breadth Features

# COMMAND ----------

# MAGIC %md
# MAGIC ### 8.1 Calculate Max CCI and Comorbidity Flags per Person-Year

# COMMAND ----------

cci_features = unified_daily.groupBy("person_id", "year").agg(
    # Comorbidity burden
    F.max("cci_total").alias("max_cci"),
    # Specific comorbidity flags
    F.max("cci_myocardial_infarction").alias("has_mi"),
    F.max("cci_congestive_heart_failure").alias("has_chf"),
    F.max(F.greatest(F.col("cci_diabetes_uncomplicated"), F.col("cci_diabetes_complicated"))).alias("has_diabetes"),
    F.max(F.greatest(F.col("cci_cancer"), F.col("cci_metastatic_cancer"))).alias("has_cancer"),
    F.max("cci_renal").alias("has_ckd"),
    # Clinical breadth - count of distinct conditions, drugs, procedures seen during the year
    F.size(F.array_distinct(F.flatten(F.collect_list("condition_concept_ids")))).alias("unique_condition_count"),
    F.size(F.array_distinct(F.flatten(F.collect_list("drug_concept_ids")))).alias("unique_drug_count"),
    F.size(F.array_distinct(F.flatten(F.collect_list("procedure_concept_ids")))).alias("unique_procedure_count")
)


# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Assemble Feature Table

# COMMAND ----------

# MAGIC %md
# MAGIC ### 9.1 Join All Feature Sets

# COMMAND ----------

# Join all feature sets
person_year_features = observation_window
person_year_features = person_year_features.join(visit_frequency, ["person_id", "year"], "left")
person_year_features = person_year_features.join(hospitalization, ["person_id", "year"], "left")
person_year_features = person_year_features.join(regularity, ["person_id", "year"], "left")
person_year_features = person_year_features.join(cci_features, ["person_id", "year"], "left")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 9.2 Fill Null Values

# COMMAND ----------

# Fill nulls for volume features
person_year_features = person_year_features.fillna(0, subset=[
    "visit_count", "inpatient_visit_count", "hospitalized_days"
])

# Add has_valid_regularity indicator (1 if enough visits to compute regularity, 0 otherwise)
person_year_features = person_year_features.withColumn(
    "has_valid_regularity",
    F.when(F.col("irregularity_l1").isNotNull(), 1).otherwise(0)
)


# COMMAND ----------

# MAGIC %md
# MAGIC ### 9.3 Validate Final Feature Table

# COMMAND ----------

# Data quality check - verify no unexpected nulls in critical features
null_check = person_year_features.select(
    F.sum(F.when(F.col("visit_count").isNull(), 1).otherwise(0)).alias("null_visits"),
    F.sum(F.when(F.col("inpatient_visit_count").isNull(), 1).otherwise(0)).alias("null_ip")
).collect()[0]

if null_check.null_visits > 0 or null_check.null_ip > 0:
    raise ValueError(
        f"Unexpected nulls in critical features after fillna: "
        f"null_visits={null_check.null_visits}, null_ip={null_check.null_ip}"
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Final Schema Definition

# COMMAND ----------

# MAGIC %md
# MAGIC ### 10.1 Define Column Order

# COMMAND ----------

# Define final column order
final_columns = [
    # Identifiers
    "person_id", "year",

    # Observation window (for filtering and context)
    "year_first_date", "year_last_date", "days_observed",

    # CLUSTERING FEATURES (5 core for GMM, 4 for rules-based)
    "visit_count",
    "inpatient_visit_count",
    "hospitalized_days",
    "irregularity_l1",
    "irregularity_l2",
    "has_valid_regularity",

    # Comorbidity and clinical complexity
    "max_cci", "has_mi", "has_chf", "has_diabetes", "has_cancer", "has_ckd",
    "unique_condition_count", "unique_drug_count", "unique_procedure_count"
]

# COMMAND ----------

# MAGIC %md
# MAGIC ### 10.2 Select Final Columns

# COMMAND ----------

person_year_features = person_year_features.select(final_columns)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 11. Add Prior-Year Features

# COMMAND ----------

# MAGIC %md
# MAGIC ### 11.1 Define Person Window

# COMMAND ----------

# Define person-window ordered by year (for LAG operations)
person_window = Window.partitionBy("person_id").orderBy("year")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 11.2 Add LAG Columns for Longitudinal Analysis

# COMMAND ----------

# Add prior year columns using LAG (NULL for first year per person)
person_year_features = person_year_features.withColumn(
    "prior_year_max_cci",
    F.lag("max_cci", 1).over(person_window)
).withColumn(
    "prior_year_visit_count",
    F.lag("visit_count", 1).over(person_window)
).withColumn(
    "prior_year_inpatient_visit_count",
    F.lag("inpatient_visit_count", 1).over(person_window)
)


# COMMAND ----------

# MAGIC %md
# MAGIC ## 12. Save Output Table

# COMMAND ----------

# Save to Delta
output_db = "cohort_2a"
output_table = "archetype_features_yearly"

person_year_features.write \
    .format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .partitionBy("year") \
    .saveAsTable(f"{output_db}.{output_table}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 13. Cleanup

# COMMAND ----------

unified_daily.unpersist()

