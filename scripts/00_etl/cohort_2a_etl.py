# Databricks notebook source
# MAGIC %md
# MAGIC **Create date**: 7/11/2025
# MAGIC
# MAGIC **Author**: Abhishek Bhatia
# MAGIC
# MAGIC **This program loads the Core OMDP Concept tables and the OMDP standardized patient data for Cohort 2A for the data density project.**

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1. Configure Data Storage Link

# COMMAND ----------

# Configure Spark to connect to Azure Data Lake Storage
# Note: Use Azure Key Vault or Databricks secrets for production
spark.conf.set(
    "fs.azure.account.key.<storage-account>.dfs.core.windows.net",
    dbutils.secrets.get(scope="<secret-scope>", key="<storage-key>")
)

# COMMAND ----------

# Confirm Data Lake accessibility
dbutils.fs.ls("abfss://<container>@<storage-account>.dfs.core.windows.net/")

# COMMAND ----------


# MAGIC %md
# MAGIC ### 2. Create Cohort 2A Database  and load tables

# COMMAND ----------

# Define the 'cohort_2a' database and point to the same ADLS Gen2 root path
cohort_db = "cohort_2a"
base_path = "abfss://<container>@<storage-account>.dfs.core.windows.net/COHORT_2A_OMOP_TABLES"

# Create the cohort_2a database if it doesn't already exist
spark.sql(f"CREATE DATABASE IF NOT EXISTS {cohort_db}")

# List of cohort tables (those prefixed with 'V_')
cohort_tables = [
    "V_CARE_SITE",
    "V_CONDITION_OCCURRENCE",
    "V_DEATH",
    "V_DRUG_EXPOSURE",
    "V_MEASUREMENT",
    "V_OBSERVATION",
    "V_PERSON",
    "V_PROCEDURE_OCCURRENCE",
    "V_PROVIDER",
    "V_VISIT_DETIAL", # NOTE THAT THIS SPELLING ERROR IS RETAINED BECAUSE THAT'S WHAT THE FILE IS CALLED
    "V_VISIT_OCCURRENCE"
]

# COMMAND ----------

# ETL loop - read each CSV into a DataFrame and write as a Delta table in cohort_2a
for table in cohort_tables:

    # Read the CSV file from ADLS Gen2
    df = spark.read.csv(
        f"{base_path}/{table}.csv",
        header=True,
        inferSchema=True
    )

    # Standardize table name: lowercase and strip 'V_' prefix
    clean_name = table.lower().replace("v_", "")

    # Write DataFrame to Delta table in cohort_2a (overwrite mode)
    df.write.format("delta") \
      .mode("overwrite") \
      .saveAsTable(f"{cohort_db}.{clean_name}")

    # Print confirmation
    print(f"Loaded cohort_2a table '{clean_name}' into database '{cohort_db}'.")



# COMMAND ----------

# MAGIC %md
# MAGIC **Note:** We need to fix the typo in the `visit_detial` table name, and rename it to `visit_detail`. The original spelling error is retained because the data were uploaded

# COMMAND ----------

# After ingestion, fix the 'visit_detial' typo for this cohort
if spark.catalog.tableExists(f"{cohort_db}.visit_detial"):
    # create or replace the correctly named table from the misspelled one
    spark.sql(f"""
        CREATE OR REPLACE TABLE {cohort_db}.visit_detail
        AS SELECT * FROM {cohort_db}.visit_detial
        """)
    # drop the old, misspelled table
    spark.sql(f"DROP TABLE {cohort_db}.visit_detial")
    print(f"Replaced '{cohort_db}.visit_detail' with data from '{cohort_db}.visit_detial'")
else:
    # no action if the typo table does not exist
    print(f"No '{cohort_db}.visit_detial' table found; nothing to rename.")


# COMMAND ----------

# MAGIC %md
# MAGIC ### 3. Verify Successful Table Creation

# COMMAND ----------

# Verify that all cohort tables were created
spark.sql(f"SHOW TABLES IN {cohort_db}").show()


# COMMAND ----------


# MAGIC %md
# MAGIC ### 4. Cohort 2A Analysis - Patient Counts and Language Distribution
# MAGIC

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, countDistinct

# Set the database
cohort_db = "cohort_2a"


print("=" * 60)
print("4. COHORT 2A - SUMMARY STATISTICS")
print("=" * 60)

# 4.1 Total Patient Count
total_patients = spark.sql(f"SELECT COUNT(DISTINCT person_id) FROM {cohort_db}.person").collect()[0][0]
print(f"\n4.1 Total patients: {total_patients}")

# 4.2 Language Distribution
print("\n4.2 Language Distribution:")
print("-" * 40)

language_query = f"""
SELECT
    COALESCE(o.value_as_string, 'Not Recorded') as language,
    COUNT(DISTINCT p.person_id) as patient_count,
    ROUND(COUNT(DISTINCT p.person_id) * 100.0 / {total_patients}, 1) as percentage
FROM {cohort_db}.person p
LEFT JOIN {cohort_db}.observation o
    ON p.person_id = o.person_id
    AND o.observation_concept_id IN (
        4182947, -- Preferred language
        4152285, -- Language spoken
        40758030 -- Primary language
    )
GROUP BY COALESCE(o.value_as_string, 'Not Recorded')
ORDER BY patient_count DESC
"""

spark.sql(language_query).show(truncate=False)

print("\n" + "=" * 60)