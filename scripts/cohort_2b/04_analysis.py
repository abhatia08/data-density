# Databricks notebook source
# MAGIC %md
# MAGIC **Purpose**: Cohort 2b cluster analysis and summary (rules, GMM-4, GMM-7), mirroring cohort_1/04 but scoped to a single cohort.

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

input_db = "cohort_2b"

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Load data

# COMMAND ----------

features = spark.table(f"{input_db}.archetype_features_yearly")
rules = spark.table(f"{input_db}.rules_yearly")
gmm4 = spark.table(f"{input_db}.gmm_4_yearly")
gmm7 = spark.table(f"{input_db}.gmm_7_yearly")

for df_name, df in [("rules", rules), ("gmm4", gmm4), ("gmm7", gmm7)]:
    if "training_data" not in df.columns:
        locals()[df_name] = df.withColumn("training_data", F.lit(0))

rules = rules.withColumnRenamed("archetype", "rules_vanilla_cluster")

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
characterization_data.write.format("delta").mode("overwrite").option("overwriteSchema", "true") \
    .saveAsTable(f"{input_db}.yearly_archetypes_allmethods")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Basic cluster counts

# COMMAND ----------

def cluster_summary(df, cluster_col):
    return df.groupBy(cluster_col).agg(
        F.count("*").alias("person_years"),
        F.countDistinct("person_id").alias("unique_persons")
    ).orderBy(F.desc("person_years"))

summary_rules = cluster_summary(characterization_data, "rules_vanilla_cluster")
summary_gmm4 = cluster_summary(characterization_data, "gmm_4_cluster")
summary_gmm7 = cluster_summary(characterization_data, "gmm_7_cluster")

print("Rules (vanilla) distribution:")
display(summary_rules)
print("GMM-4 distribution:")
display(summary_gmm4)
print("GMM-7 distribution:")
display(summary_gmm7)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Concept breadth / coding density by cluster

# COMMAND ----------

def concept_stats(df, cluster_col):
    return df.groupBy(cluster_col).agg(
        F.mean("unique_condition_count").alias("mean_unique_condition_count"),
        F.mean("unique_drug_count").alias("mean_unique_drug_count"),
        F.mean("unique_procedure_count").alias("mean_unique_procedure_count"),
        F.mean("total_unique_concepts").alias("mean_total_unique_concepts"),
        F.mean("total_data_points").alias("mean_total_data_points"),
        F.mean("total_nomatch_concepts").alias("mean_total_nomatch_concepts")
    ).orderBy(cluster_col)

concept_rules = concept_stats(characterization_data.join(rules.select("person_id", "year", "rules_vanilla_cluster"), ["person_id", "year"]), "rules_vanilla_cluster")
concept_gmm4 = concept_stats(characterization_data.join(gmm4.select("person_id", "year", "gmm_4_cluster"), ["person_id", "year"]), "gmm_4_cluster")
concept_gmm7 = concept_stats(characterization_data.join(gmm7.select("person_id", "year", "gmm_7_cluster"), ["person_id", "year"]), "gmm_7_cluster")

print("Concept breadth by cluster (rules):")
display(concept_rules)
print("Concept breadth by cluster (gmm_4):")
display(concept_gmm4)
print("Concept breadth by cluster (gmm_7):")
display(concept_gmm7)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Cleanup

# COMMAND ----------

characterization_data.unpersist()
features.unpersist()

