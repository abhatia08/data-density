# Databricks notebook source
# MAGIC %md
# MAGIC **Create date**: 7/11/2025
# MAGIC
# MAGIC **Author**: Abhishek Bhatia
# MAGIC
# MAGIC **This program loads the Core OMDP Concept tables from the Cohort 1 dir (they are standard across all cohorts) for the data density project.**

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
# MAGIC ### 2. Create vocab database and load core tables

# COMMAND ----------

# Define the 'vocab' database and the ADLS Gen2 path where the core CSVs live
core_db    = "vocab"
base_path = "abfss://<container>@<storage-account>.dfs.core.windows.net/COHORT_1_OMOP_TABLES"

# Create the vocab database if it doesn't already exist
spark.sql(f"CREATE DATABASE IF NOT EXISTS {core_db}")

# COMMAND ----------

# List of core OMDP tables (no 'V_' prefix)
core_tables = [
    "CONCEPT",
    "CONCEPT_ANCESTOR",
    "CONCEPT_RELATIONSHIP",
    "VOCABULARY"
]

# Load each core CSV into a Delta table in the vocab database
for table in core_tables:
    # Read the CSV file into a Spark DataFrame
    df = spark.read.csv(
        f"{base_path}/{table}.csv",
        header=True,
        inferSchema=True
    )

    # Standardize the table name (lowercase)
    clean_name = table.lower()

    # Write the DataFrame as a Delta table (overwrite mode)
    df.write.format("delta") \
      .mode("overwrite") \
      .saveAsTable(f"{core_db}.{clean_name}")

    # Confirmation message
    print(f"Loaded core table '{clean_name}' into database '{core_db}'.")

# Verify that all core tables were created
spark.sql(f"SHOW TABLES IN {core_db}").show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3. Verify Successful Table Creation

# COMMAND ----------

# Verify that all core tables were created
spark.sql(f"SHOW TABLES IN {core_db}").show()






