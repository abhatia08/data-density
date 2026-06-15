# Databricks notebook source
# MAGIC %md
# MAGIC **Purpose**: Person-date unified table for cohort_2a (cohort_1 spec)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Setup and Configuration

# COMMAND ----------

# PySpark
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import ArrayType, IntegerType, StringType

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.1 Load Shared Utilities

# COMMAND ----------

# MAGIC %run ./99_utils

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.2 Configure Logging and Optimization

# COMMAND ----------

VERBOSE = get_verbose(default=True)
gate_prints(VERBOSE)
configure_spark_optimizations()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.3 Clear Downstream Delta Tables

# COMMAND ----------

# Set to True to clear all downstream tables (keeps only core OMOP tables)
CLEAR_DOWNSTREAM_TABLES = True

# Define core tables that should be preserved (loaded by ETL/OMOP)
core_tables = [
    "care_site",
    "condition_occurrence",
    "death",
    "drug_exposure",
    "measurement",
    "observation",
    "person",
    "procedure_occurrence",
    "provider",
    "visit_detail",
    "visit_occurrence"
]

# OMOP Standard Concept IDs for visit type classification
# 9201 = Inpatient Visit, 262 = Emergency Room and Inpatient Visit
INPATIENT_VISIT_CONCEPT_IDS = [9201, 262]

# Clear downstream tables if requested
clear_downstream_tables("cohort_2a", core_tables, use_widget=False, verbose=True, clear_downstream=CLEAR_DOWNSTREAM_TABLES)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Load OMOP Tables

# COMMAND ----------

# Define databases
cohort_db = "cohort_2a"
vocab_db = "vocab"

# Load cohort tables
person_df = spark.table(f"{cohort_db}.person")
visit_occurrence_df = spark.table(f"{cohort_db}.visit_occurrence")
condition_df = spark.table(f"{cohort_db}.condition_occurrence")
drug_df = spark.table(f"{cohort_db}.drug_exposure")
measurement_df = spark.table(f"{cohort_db}.measurement")
procedure_df = spark.table(f"{cohort_db}.procedure_occurrence")
observation_df = spark.table(f"{cohort_db}.observation")

# Load vocabulary tables
concept_df = spark.table(f"{vocab_db}.concept")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Extract Dates from Each Domain

# COMMAND ----------

# Condition occurrences
condition_daily = condition_df.select(
    F.col("person_id"),
    F.to_date("condition_start_date").alias("date"),
    F.col("condition_concept_id").alias("concept_id"),
    F.col("condition_source_value").alias("source_code"),
    F.col("condition_source_concept_vocabulary_id").alias("vocab_id"),
    F.col("visit_occurrence_id")
).filter(F.col("date").isNotNull()).cache()

# Drug exposures
drug_daily = drug_df.select(
    F.col("person_id"),
    F.to_date("drug_exposure_start_date").alias("date"),
    F.col("drug_concept_id").alias("concept_id"),
    F.col("visit_occurrence_id")
).filter(F.col("date").isNotNull()).cache()

# Measurements
measurement_daily = measurement_df.select(
    F.col("person_id"),
    F.to_date("measurement_date").alias("date"),
    F.col("measurement_concept_id").alias("concept_id"),
    F.col("visit_occurrence_id")
).filter(F.col("date").isNotNull()).cache()

# Procedures
procedure_daily = procedure_df.select(
    F.col("person_id"),
    F.to_date("procedure_date").alias("date"),
    F.col("procedure_concept_id").alias("concept_id"),
    F.col("visit_occurrence_id")
).filter(F.col("date").isNotNull()).cache()

# Observations
observation_daily = observation_df.select(
    F.col("person_id"),
    F.to_date("observation_date").alias("date"),
    F.col("observation_concept_id").alias("concept_id"),
    F.col("visit_occurrence_id")
).filter(F.col("date").isNotNull()).cache()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Aggregate Concept IDs by Day

# COMMAND ----------

# Conditions
condition_arrays = condition_daily.groupBy("person_id", "date").agg(
    F.collect_list("concept_id").alias("condition_concept_ids"),
    F.collect_list("source_code").alias("condition_source_codes"),
    F.collect_list("vocab_id").alias("condition_vocab_ids"),
    F.collect_set("visit_occurrence_id").alias("condition_visit_ids")
).cache()

# Drugs
drug_arrays = drug_daily.groupBy("person_id", "date").agg(
    F.collect_list("concept_id").alias("drug_concept_ids"),
    F.collect_set("visit_occurrence_id").alias("drug_visit_ids")
).cache()

# Measurements
measurement_arrays = measurement_daily.groupBy("person_id", "date").agg(
    F.collect_list("concept_id").alias("measurement_concept_ids"),
    F.collect_set("visit_occurrence_id").alias("measurement_visit_ids")
).cache()

# Procedures
procedure_arrays = procedure_daily.groupBy("person_id", "date").agg(
    F.collect_list("concept_id").alias("procedure_concept_ids"),
    F.collect_set("visit_occurrence_id").alias("procedure_visit_ids")
).cache()

# Observations
observation_arrays = observation_daily.groupBy("person_id", "date").agg(
    F.collect_list("concept_id").alias("observation_concept_ids"),
    F.collect_set("visit_occurrence_id").alias("observation_visit_ids")
).cache()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Visit Counts per Day

# COMMAND ----------

# Explode visits into all dates within the visit span to capture full hospitalization length
visit_daily = (
    visit_occurrence_df
    .select(
        F.col("person_id"),
        F.col("visit_occurrence_id"),
        F.col("visit_concept_id"),
        F.to_date("visit_start_date").alias("start_date"),
        F.to_date("visit_end_date").alias("end_date")
    )
    .filter(F.col("start_date").isNotNull())
    .withColumn("end_date",
        F.when(F.col("end_date").isNull() | (F.col("end_date") < F.col("start_date")),
               F.col("start_date"))
        .otherwise(F.col("end_date"))
    )
    .withColumn("date", F.explode(F.expr("sequence(start_date, end_date, interval 1 day)")))
    .withColumn(
        "is_hospitalized",
        F.when(F.col("visit_concept_id").isin(*INPATIENT_VISIT_CONCEPT_IDS), 1).otherwise(0)
    )
).cache()

# Count DISTINCT visits per person-date
visit_counts = visit_daily.groupBy("person_id", "date").agg(
    F.countDistinct("visit_occurrence_id").alias("visit_count"),
    F.collect_set("visit_occurrence_id").alias("visit_occurrence_ids"),
    F.max("is_hospitalized").alias("hospitalized_flag")
).cache()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Build Person-Date Spine

# COMMAND ----------

concept_dates = (
    condition_arrays.select("person_id", "date")
    .union(drug_arrays.select("person_id", "date"))
    .union(measurement_arrays.select("person_id", "date"))
    .union(procedure_arrays.select("person_id", "date"))
    .union(observation_arrays.select("person_id", "date"))
    .distinct()
)

complete_person_dates = (
    visit_counts.select("person_id", "date")
    .union(concept_dates.select("person_id", "date"))
    .distinct()
)

complete_person_dates.cache()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Charlson Comorbidity Index (CCI)

# COMMAND ----------

# Step 1: Filter to problem list conditions from same calendar year
condition_start_filter = (
    condition_df.alias("vco")
    .join(
        visit_occurrence_df.alias("visits"),
        F.col("vco.visit_occurrence_id") == F.col("visits.visit_occurrence_id"),
        "inner"
    )
    .filter(F.col("vco.condition_type_concept_id") == 32840)  # EHR problem list only
    .withColumn("condition_year", F.year("vco.condition_start_date"))
    .withColumn("visit_year", F.year("visits.visit_start_date"))
    .filter(F.col("condition_year") == F.col("visit_year"))
    .select(
        F.col("visits.visit_occurrence_id").alias("visit_occurrence_id"),
        F.col("visits.person_id"),
        F.col("vco.condition_start_date"),
        F.col("vco.condition_end_date"),
        F.col("vco.condition_source_value"),
        F.col("vco.condition_source_concept_vocabulary_id"),
        F.col("visit_year")
    )
    .distinct()
)

# Step 2: Assign CCI categories based on ICD codes (Quan et al. 2005 mappings)
conditions = condition_start_filter.withColumn(
    "clean_code",
    F.upper(F.regexp_replace(F.col("condition_source_value"), "\\.", ""))
).withColumn(
    "cc_group",
    F.when((F.substring(F.col("clean_code"), 1, 3).isin('410','412')) |
           (F.substring(F.col("clean_code"), 1, 3).isin('I21','I22')) |
           (F.substring(F.col("clean_code"), 1, 4) == 'I252'), 1)
    .when((F.substring(F.col("clean_code"), 1, 5).isin('39891','40201','40211','40291','40401','40403','40411','40413','40491','40493')) |
          (F.substring(F.col("clean_code"), 1, 3) == '428') |
          (F.substring(F.col("clean_code"), 1, 4).isin('I099','I110','I130','I132','I255','I420','I425','I426','I427','I428','I429','P290')) |
          (F.substring(F.col("clean_code"), 1, 3).isin('I43','I50')), 2)
    .when((F.substring(F.col("clean_code"), 1, 4).isin('0930','4373','4431','4432','4433','4434','4435','4436','4437','4438','4439','4471','5571','5579','V434')) |
          (F.substring(F.col("clean_code"), 1, 3).isin('440','441')) |
          (F.col("clean_code") == '36234') |
          (F.substring(F.col("clean_code"), 1, 3).isin('I70','I71')), 3)
    .when((F.substring(F.col("clean_code"), 1, 3).isin('430','431','432','433','434','435','436','437','438')) |
          (F.col("clean_code") == '36234') |
          (F.substring(F.col("clean_code"), 1, 3).isin('G45','G46','I60','I61','I62','I63','I64','I65','I66','I67','I68','I69')) |
          (F.substring(F.col("clean_code"), 1, 4) == 'H340'), 4)
    .when((F.substring(F.col("clean_code"), 1, 5).isin('29010','29011')) |
          (F.substring(F.col("clean_code"), 1, 3) == '290') |
          (F.substring(F.col("clean_code"), 1, 4).isin('F051','G311','G312')) |
          (F.substring(F.col("clean_code"), 1, 3).isin('F00','F01','F02','F03','G30')), 5)
    .when((F.substring(F.col("clean_code"), 1, 3).isin('490','491','492','493','494','495','496','500','501','502','503','504','505')) |
          (F.substring(F.col("clean_code"), 1, 4).isin('4168','4169','5064','5081','5088')) |
          (F.substring(F.col("clean_code"), 1, 3).isin('348','341','342','343','344','345','346','347','360','361','362','363','364','365','366','367','I27')) |
          (F.substring(F.col("clean_code"), 1, 4).isin('I270','I279','J684','J701','J703')) |
          (F.substring(F.col("clean_code"), 1, 3).isin('I28','J40','J41','J42','J43','J44','J45','J46','J47','J60','J61','J62','J63','J64','J65','J66','J67')), 6)
    .when((F.substring(F.col("clean_code"), 1, 4).isin('4465','7100','7101','7102','7103','7104','7140','7141','7142','7148')) |
          (F.substring(F.col("clean_code"), 1, 3).isin('725')) |
          (F.substring(F.col("clean_code"), 1, 4).isin('M315','M351','M353','M360')) |
          (F.substring(F.col("clean_code"), 1, 3).isin('M05','M32','M33','M34','M06')), 7)
    .when((F.substring(F.col("clean_code"), 1, 3).isin('531','532','533','534')) |
          (F.substring(F.col("clean_code"), 1, 3).isin('K25','K26','K27','K28')), 8)
    .when((F.substring(F.col("clean_code"), 1, 5).isin('07022','07023','07032','07033','07044','07054')) |
          (F.substring(F.col("clean_code"), 1, 3).isin('570','571')) |
          (F.substring(F.col("clean_code"), 1, 4).isin('K700','K701','K702','K703','K709','K713','K714','K715','K717','K760','K762','K763','K764','K768','K769','Z944','0706','0709','5733','5734','5738','5739','V427')) |
          (F.substring(F.col("clean_code"), 1, 3).isin('B18','K73','K74')), 9)
    .when((F.substring(F.col("clean_code"), 1, 4).isin('2500','2501','2502','2503','2508','2509')) |
          (F.substring(F.col("clean_code"), 1, 4).isin('E100','E101','E106','E108','E109','E110','E111','E116','E118','E119','E120','E121','E126','E128','E129','E130','E131','E136','E138','E139','E140','E141','E146','E148','E149')), 10)
    .when((F.substring(F.col("clean_code"), 1, 4).isin('2504','2505','2506','2507')) |
          (F.substring(F.col("clean_code"), 1, 4).isin('E102','E103','E104','E105','E107','E112','E113','E114','E115','E117','E122','E123','E124','E125','E127','E132','E133','E134','E135','E137','E142','E143','E144','E145','E147')), 11)
    .when((F.substring(F.col("clean_code"), 1, 4).isin('3341','3440','3441','3442','3443','3444','3445','3446','3449')) |
          (F.substring(F.col("clean_code"), 1, 3).isin('342','343')) |
          (F.substring(F.col("clean_code"), 1, 4).isin('G041','G114','G801','G802','G830','G831','G832','G833','G834','G839')) |
          (F.substring(F.col("clean_code"), 1, 3).isin('G81','G82')), 12)
    .when((F.substring(F.col("clean_code"), 1, 5).isin('40301','40311','40391','40402','40403','40412','40413','40492','40493')) |
          (F.substring(F.col("clean_code"), 1, 3).isin('582','585','586','V56','N18','N19')) |
          (F.substring(F.col("clean_code"), 1, 4).isin('N052','N053','N054','N055','N056','N057','N250','I120','I131','N032','N033','N034','N035','N036','N037','Z490','Z491','Z492','Z940','Z992','5830','5831','5832','5834','5836','5837','5880','V420','V451')), 13)
    .when((F.substring(F.col("clean_code"), 1, 3).isin('140','141','142','143','144','145','146','147','148','149','150','151','152','153','154','155','156','157','158','159','160','161','162','163','164','165','170','171','172','174','175','176','179','180','181','182','183','184','185','186','187','188','189','190','191','192','193','194','195','200','201','202','203','204','205','206','207','208')) |
          (F.substring(F.col("clean_code"), 1, 4) == '2386') |
          (F.substring(F.col("clean_code"), 1, 3).isin('C00','C01','C02','C03','C04','C05','C06','C07','C08','C09','C10','C11','C12','C13','C14','C15','C16','C17','C18','C19','C20','C21','C22','C23','C24','C25','C26','C30','C31','C32','C33','C37','C38','C39','C40','C41','C43','C45','C46','C47','C48','C49','C50','C51','C52','C53','C54','C55','C56','C57','C58','C60','C61','C62','C63','C64','C65','C66','C67','C68','C69','C70','C71','C72','C73','C74','C75','C76','C81','C82','C83','C84','C85','C88','C90','C91','C92','C93','C94','C95','C96','C97')), 14)
    .when((F.substring(F.col("clean_code"), 1, 4).isin('4560','4561','4562','5722','5723','5724','5728')) |
          (F.substring(F.col("clean_code"), 1, 4).isin('K704','K711','K721','K729','K765','K766','K767','I850','I859','I864','I982')), 15)
    .when((F.substring(F.col("clean_code"), 1, 3).isin('196','197','198','199')) |
          (F.substring(F.col("clean_code"), 1, 3).isin('C77','C78','C79','C80')), 16)
    .when((F.substring(F.col("clean_code"), 1, 3).isin('042','043','044')) |
          (F.substring(F.col("clean_code"), 1, 3).isin('B20','B21','B22','B24')), 17)
    .otherwise(0)
)

# Step 3: Add double-counting rows for codes that belong to multiple categories
conditions_expanded = conditions.union(
    conditions.filter(
        (F.substring(F.col("clean_code"), 1, 5).isin('40403','40413','40493')) |
        (F.substring(F.col("clean_code"), 1, 4) == '4573')
    ).withColumn(
        "cc_group",
        F.when(F.substring(F.col("clean_code"), 1, 5).isin('40403','40413','40493'), 13)
        .when(F.substring(F.col("clean_code"), 1, 4) == '4573', 4)
        .otherwise(0)
    )
)

# Step 4: Filter out ICD-9 V-codes when vocab is ICD10CM (per Quan 2005)
conditions_filtered = conditions_expanded.filter(
    ~((F.col("clean_code").like('V434%') |
       F.col("clean_code").like('V4276%') |
       F.col("clean_code").like('V420%') |
       F.col("clean_code").like('V451%') |
       F.col("clean_code").like('V56%')) &
      (F.col("condition_source_concept_vocabulary_id") == 'ICD10CM'))
)

# Step 5: Get distinct cc_groups per visit (no duplicates within a visit)
no_duplicate_conditions = (
    conditions_filtered
    .filter(F.col("cc_group") > 0)
    .select("visit_occurrence_id", "cc_group")
    .distinct()
)

# Step 6: Apply weights and calculate CCI per visit
weights_df = no_duplicate_conditions.withColumn(
    "cc_weight",
    F.when(F.col("cc_group").isin(1, 2, 3, 4, 5, 6, 7, 8, 9, 10), 1)
    .when(F.col("cc_group").isin(11, 12, 13, 14), 2)
    .when(F.col("cc_group") == 15, 3)
    .when(F.col("cc_group").isin(16, 17), 6)
    .otherwise(0)
)

# Step 7: Sum weights per visit and join with visit info
cci_per_visit = (
    weights_df
    .groupBy("visit_occurrence_id")
    .agg(
        F.sum("cc_weight").alias("cci_total"),
        F.max(F.when(F.col("cc_group") == 1, 1).otherwise(0)).alias("cci_myocardial_infarction"),
        F.max(F.when(F.col("cc_group") == 2, 1).otherwise(0)).alias("cci_congestive_heart_failure"),
        F.max(F.when(F.col("cc_group") == 3, 1).otherwise(0)).alias("cci_peripheral_vascular"),
        F.max(F.when(F.col("cc_group") == 4, 1).otherwise(0)).alias("cci_cerebrovascular"),
        F.max(F.when(F.col("cc_group") == 5, 1).otherwise(0)).alias("cci_dementia"),
        F.max(F.when(F.col("cc_group") == 6, 1).otherwise(0)).alias("cci_chronic_pulmonary"),
        F.max(F.when(F.col("cc_group") == 7, 1).otherwise(0)).alias("cci_connective_tissue"),
        F.max(F.when(F.col("cc_group") == 8, 1).otherwise(0)).alias("cci_peptic_ulcer"),
        F.max(F.when(F.col("cc_group") == 9, 1).otherwise(0)).alias("cci_mild_liver"),
        F.max(F.when(F.col("cc_group") == 10, 1).otherwise(0)).alias("cci_diabetes_uncomplicated"),
        F.max(F.when(F.col("cc_group") == 11, 1).otherwise(0)).alias("cci_diabetes_complicated"),
        F.max(F.when(F.col("cc_group") == 12, 1).otherwise(0)).alias("cci_paralysis"),
        F.max(F.when(F.col("cc_group") == 13, 1).otherwise(0)).alias("cci_renal"),
        F.max(F.when(F.col("cc_group") == 14, 1).otherwise(0)).alias("cci_cancer"),
        F.max(F.when(F.col("cc_group") == 15, 1).otherwise(0)).alias("cci_severe_liver"),
        F.max(F.when(F.col("cc_group") == 16, 1).otherwise(0)).alias("cci_metastatic_cancer"),
        F.max(F.when(F.col("cc_group") == 17, 1).otherwise(0)).alias("cci_hiv_aids")
    )
    .join(
        visit_occurrence_df.select("visit_occurrence_id", "person_id", F.to_date("visit_start_date").alias("visit_date")),
        on="visit_occurrence_id",
        how="inner"
    )
)

# Step 8: Aggregate CCI to person-date level (max across all visits on that date)
cci_daily = (
    cci_per_visit
    .groupBy("person_id", "visit_date")
    .agg(
        F.max("cci_total").alias("cci_total"),
        F.max("cci_myocardial_infarction").alias("cci_myocardial_infarction"),
        F.max("cci_congestive_heart_failure").alias("cci_congestive_heart_failure"),
        F.max("cci_peripheral_vascular").alias("cci_peripheral_vascular"),
        F.max("cci_cerebrovascular").alias("cci_cerebrovascular"),
        F.max("cci_dementia").alias("cci_dementia"),
        F.max("cci_chronic_pulmonary").alias("cci_chronic_pulmonary"),
        F.max("cci_connective_tissue").alias("cci_connective_tissue"),
        F.max("cci_peptic_ulcer").alias("cci_peptic_ulcer"),
        F.max("cci_mild_liver").alias("cci_mild_liver"),
        F.max("cci_diabetes_uncomplicated").alias("cci_diabetes_uncomplicated"),
        F.max("cci_diabetes_complicated").alias("cci_diabetes_complicated"),
        F.max("cci_paralysis").alias("cci_paralysis"),
        F.max("cci_renal").alias("cci_renal"),
        F.max("cci_cancer").alias("cci_cancer"),
        F.max("cci_severe_liver").alias("cci_severe_liver"),
        F.max("cci_metastatic_cancer").alias("cci_metastatic_cancer"),
        F.max("cci_hiv_aids").alias("cci_hiv_aids")
    )
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Build Daily Table (wide)

# COMMAND ----------

unified_daily = complete_person_dates.repartition("person_id").join(
    visit_counts,
    on=["person_id", "date"],
    how="left"
)

unified_daily = unified_daily.fillna(0, subset=["visit_count", "hospitalized_flag"])
unified_daily = unified_daily.withColumn(
    "visit_occurrence_ids",
    F.coalesce("visit_occurrence_ids", F.array().cast("array<bigint>"))
)

# Left-join each domain's aggregated concept arrays
unified_daily = unified_daily.join(
    F.broadcast(condition_arrays),
    on=["person_id", "date"],
    how="left"
)
unified_daily = unified_daily.join(
    F.broadcast(drug_arrays),
    on=["person_id", "date"],
    how="left"
)
unified_daily = unified_daily.join(
    F.broadcast(measurement_arrays),
    on=["person_id", "date"],
    how="left"
)
unified_daily = unified_daily.join(
    F.broadcast(procedure_arrays),
    on=["person_id", "date"],
    how="left"
)
unified_daily = unified_daily.join(
    F.broadcast(observation_arrays),
    on=["person_id", "date"],
    how="left"
)

# Left-join CCI data
unified_daily = unified_daily.join(
    cci_daily.withColumnRenamed("visit_date", "date"),
    on=["person_id", "date"],
    how="left"
)

# Final schema selection with coalescing for null handling
unified_daily = unified_daily.select(
    "person_id",
    "date",
    "visit_count",
    "visit_occurrence_ids",
    "hospitalized_flag",
    F.coalesce("condition_concept_ids", F.array().cast(ArrayType(IntegerType()))).alias("condition_concept_ids"),
    F.coalesce("condition_source_codes", F.array().cast(ArrayType(StringType()))).alias("condition_source_codes"),
    F.coalesce("condition_vocab_ids", F.array().cast(ArrayType(StringType()))).alias("condition_vocab_ids"),
    F.coalesce("drug_concept_ids", F.array().cast(ArrayType(IntegerType()))).alias("drug_concept_ids"),
    F.coalesce("measurement_concept_ids", F.array().cast(ArrayType(IntegerType()))).alias("measurement_concept_ids"),
    F.coalesce("procedure_concept_ids", F.array().cast(ArrayType(IntegerType()))).alias("procedure_concept_ids"),
    F.coalesce("observation_concept_ids", F.array().cast(ArrayType(IntegerType()))).alias("observation_concept_ids"),
    F.coalesce("cci_total", F.lit(0)).alias("cci_total"),
    F.coalesce("cci_myocardial_infarction", F.lit(0)).alias("cci_myocardial_infarction"),
    F.coalesce("cci_congestive_heart_failure", F.lit(0)).alias("cci_congestive_heart_failure"),
    F.coalesce("cci_peripheral_vascular", F.lit(0)).alias("cci_peripheral_vascular"),
    F.coalesce("cci_cerebrovascular", F.lit(0)).alias("cci_cerebrovascular"),
    F.coalesce("cci_dementia", F.lit(0)).alias("cci_dementia"),
    F.coalesce("cci_chronic_pulmonary", F.lit(0)).alias("cci_chronic_pulmonary"),
    F.coalesce("cci_connective_tissue", F.lit(0)).alias("cci_connective_tissue"),
    F.coalesce("cci_peptic_ulcer", F.lit(0)).alias("cci_peptic_ulcer"),
    F.coalesce("cci_mild_liver", F.lit(0)).alias("cci_mild_liver"),
    F.coalesce("cci_diabetes_uncomplicated", F.lit(0)).alias("cci_diabetes_uncomplicated"),
    F.coalesce("cci_diabetes_complicated", F.lit(0)).alias("cci_diabetes_complicated"),
    F.coalesce("cci_paralysis", F.lit(0)).alias("cci_paralysis"),
    F.coalesce("cci_renal", F.lit(0)).alias("cci_renal"),
    F.coalesce("cci_cancer", F.lit(0)).alias("cci_cancer"),
    F.coalesce("cci_severe_liver", F.lit(0)).alias("cci_severe_liver"),
    F.coalesce("cci_metastatic_cancer", F.lit(0)).alias("cci_metastatic_cancer"),
    F.coalesce("cci_hiv_aids", F.lit(0)).alias("cci_hiv_aids")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Add Summary Stats

# COMMAND ----------

unified_daily = (
    unified_daily
    # per-domain total counts
    .withColumn("condition_count", F.size("condition_concept_ids"))
    .withColumn("drug_count", F.size("drug_concept_ids"))
    .withColumn("measurement_count", F.size("measurement_concept_ids"))
    .withColumn("procedure_count", F.size("procedure_concept_ids"))
    .withColumn("observation_count", F.size("observation_concept_ids"))
    # overall total data points
    .withColumn(
        "total_data_points",
        F.col("condition_count")
        + F.col("drug_count")
        + F.col("measurement_count")
        + F.col("procedure_count")
        + F.col("observation_count")
    )
    # per-domain unique counts
    .withColumn("condition_unique_ct", F.size(F.array_distinct("condition_concept_ids")))
    .withColumn("drug_unique_ct", F.size(F.array_distinct("drug_concept_ids")))
    .withColumn("measurement_unique_ct", F.size(F.array_distinct("measurement_concept_ids")))
    .withColumn("procedure_unique_ct", F.size(F.array_distinct("procedure_concept_ids")))
    .withColumn("observation_unique_ct", F.size(F.array_distinct("observation_concept_ids")))
    # overall total unique concepts
    .withColumn(
        "total_unique_concepts",
        F.col("condition_unique_ct")
        + F.col("drug_unique_ct")
        + F.col("measurement_unique_ct")
        + F.col("procedure_unique_ct")
        + F.col("observation_unique_ct")
    )
    # per-domain "no matching concept" counts (concept_id = 0)
    .withColumn("condition_nomatch_ct", count_no_match("condition_concept_ids"))
    .withColumn("drug_nomatch_ct", count_no_match("drug_concept_ids"))
    .withColumn("measurement_nomatch_ct", count_no_match("measurement_concept_ids"))
    .withColumn("procedure_nomatch_ct", count_no_match("procedure_concept_ids"))
    .withColumn("observation_nomatch_ct", count_no_match("observation_concept_ids"))
    # overall total "no-match" concepts
    .withColumn(
        "total_nomatch_concepts",
        F.col("condition_nomatch_ct")
        + F.col("drug_nomatch_ct")
        + F.col("measurement_nomatch_ct")
        + F.col("procedure_nomatch_ct")
        + F.col("observation_nomatch_ct")
    )
    # FLAG: has any clinical concepts
    .withColumn(
        "has_clinical_concepts",
        F.when(
            (F.col("condition_count") > 0) |
            (F.col("drug_count") > 0) |
            (F.col("measurement_count") > 0) |
            (F.col("procedure_count") > 0) |
            (F.col("observation_count") > 0),
            1
        ).otherwise(0)
    )
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Add Person-Level Data

# COMMAND ----------

# Step 1: Join person table and construct birth_date
ud = unified_daily.alias("ud")
pd = person_df.select(
    F.col("person_id").alias("pd_person_id"),
    F.col("year_of_birth").alias("pd_year"),
    F.col("month_of_birth").alias("pd_month"),
    F.col("day_of_birth").alias("pd_day"),
    F.col("race_concept_id").alias("pd_race_cid"),
    F.col("ethnicity_concept_id").alias("pd_eth_cid"),
    F.col("gender_concept_id").alias("pd_gender_cid")
).alias("pd")

unified_daily = (
    ud.join(pd, F.col("ud.person_id") == F.col("pd.pd_person_id"), "left")
    .withColumn("birth_date",
        F.when(
            F.col("pd_month").isNotNull() & F.col("pd_day").isNotNull(),
            F.to_date(F.concat_ws("-", "pd_year", "pd_month", "pd_day"))
        ).when(
            F.col("pd_month").isNotNull(),
            F.to_date(F.concat_ws("-", "pd_year", "pd_month", F.lit("01")))
        ).otherwise(
            F.to_date(F.concat_ws("-", "pd_year", F.lit("07"), F.lit("01")))
        )
    )
    .drop("pd_person_id", "pd_year", "pd_month", "pd_day")
)

# Step 2: calculate age_at_day
unified_daily = unified_daily.withColumn(
    "age_at_day",
    F.floor(F.months_between(F.col("date"), F.col("birth_date")) / 12)
)

# Step 3: map gender
clg = concept_df.select(
    F.col("concept_id").alias("cl_gender_cid"),
    F.col("concept_name").alias("cl_gender_name")
).alias("clg")

unified_daily = (
    unified_daily.alias("ud")
    .join(clg, F.col("ud.pd_gender_cid") == F.col("clg.cl_gender_cid"), "left")
    .withColumn(
        "gender",
        F.when(F.lower(F.col("cl_gender_name")) == "female", "Female")
        .when(F.lower(F.col("cl_gender_name")) == "male", "Male")
        .otherwise("Unknown")
    )
    .drop("pd_gender_cid", "cl_gender_cid", "cl_gender_name")
)

# Step 4: map ethnicity
cle = concept_df.select(
    F.col("concept_id").alias("cl_eth_cid"),
    F.col("concept_name").alias("cl_eth_name")
).alias("cle")

unified_daily = (
    unified_daily.alias("ud")
    .join(cle, F.col("ud.pd_eth_cid") == F.col("cle.cl_eth_cid"), "left")
    .withColumn(
        "ethnicity",
        F.when(F.lower(F.col("cl_eth_name")).like("%not hispanic%"), "Not Hispanic or Latino")
        .when(F.lower(F.col("cl_eth_name")).like("%hispanic%"), "Hispanic or Latino")
        .when(F.col("cl_eth_name").isNull(), "Unknown")
        .otherwise("Not Hispanic or Latino")
    )
    .drop("pd_eth_cid", "cl_eth_cid", "cl_eth_name")
)

# Step 5: map race
clr = concept_df.select(
    F.col("concept_id").alias("cl_race_cid"),
    F.col("concept_name").alias("cl_race_name")
).alias("clr")

unified_daily = (
    unified_daily.alias("ud")
    .join(clr, F.col("ud.pd_race_cid") == F.col("clr.cl_race_cid"), "left")
    .withColumn(
        "race",
        F.when(F.lower(F.col("cl_race_name")).like("%asians%"), "Asian")
        .when(F.lower(F.col("cl_race_name")).like("%white%"), "White")
        .when(F.lower(F.col("cl_race_name")).like("%black%"), "Black")
        .when(F.lower(F.col("cl_race_name")).like("%american indian%"), "American Indian")
        .when(F.lower(F.col("cl_race_name")).like("%hawaiian%"), "Hawaiian")
        .otherwise("Unknown")
    )
    .drop("pd_race_cid", "cl_race_cid", "cl_race_name")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 10.1 Finalize Column Order

# COMMAND ----------

# drop any column starting with those join prefixes
drop_prefixes = ["pd_", "cl_"]
keep_cols = [c for c in unified_daily.columns if not any(c.startswith(p) for p in drop_prefixes)]

unified_daily = unified_daily.select(*keep_cols)

# Define column groups
date_col = ["date"]
person_level_cols = ["person_id", "birth_date", "age_at_day", "gender", "ethnicity", "race"]
visit_cols = ["visit_count", "visit_occurrence_ids"]
hospital_cols = ["hospitalized_flag"]
concept_flag_cols = ["has_clinical_concepts"]
domain_array_cols = [
    "condition_concept_ids", "condition_source_codes", "condition_vocab_ids",
    "drug_concept_ids", "measurement_concept_ids",
    "procedure_concept_ids", "observation_concept_ids"
]
per_day_count_cols = [
    "condition_count", "drug_count", "measurement_count",
    "procedure_count", "observation_count"
]
unique_ct_cols = [
    "condition_unique_ct", "drug_unique_ct", "measurement_unique_ct",
    "procedure_unique_ct", "observation_unique_ct"
]
nomatch_ct_cols = [
    "condition_nomatch_ct", "drug_nomatch_ct", "measurement_nomatch_ct",
    "procedure_nomatch_ct", "observation_nomatch_ct"
]
totals_group = ["total_data_points", "total_unique_concepts", "total_nomatch_concepts"]
cci_cols = [
    "cci_total", "cci_myocardial_infarction", "cci_congestive_heart_failure",
    "cci_peripheral_vascular", "cci_cerebrovascular", "cci_dementia",
    "cci_chronic_pulmonary", "cci_connective_tissue", "cci_peptic_ulcer",
    "cci_mild_liver", "cci_diabetes_uncomplicated", "cci_diabetes_complicated",
    "cci_paralysis", "cci_renal", "cci_cancer", "cci_severe_liver",
    "cci_metastatic_cancer", "cci_hiv_aids"
]

all_defined = set(person_level_cols + date_col + visit_cols + hospital_cols +
                  concept_flag_cols + domain_array_cols + per_day_count_cols + unique_ct_cols +
                  nomatch_ct_cols + totals_group + cci_cols)
other_cols = [c for c in unified_daily.columns if c not in all_defined]

unified_daily = unified_daily.select(
    *person_level_cols,
    *date_col,
    *visit_cols,
    *hospital_cols,
    *concept_flag_cols,
    *domain_array_cols,
    *per_day_count_cols,
    *unique_ct_cols,
    *nomatch_ct_cols,
    *totals_group,
    *cci_cols,
    *other_cols
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 11. Aggregate by Week/Month/Year

# COMMAND ----------

unified_temporal = (
    unified_daily
    .withColumn("year", F.year("date"))
    .withColumn("month", F.date_format(F.date_trunc("month", "date"), "yyyy-MM"))
    .withColumn("week", F.date_format(F.date_trunc("week", "date"), "yyyy-MM-dd"))
)

unified_weekly = aggregate_temporal_unified(unified_temporal, "week")
unified_monthly = aggregate_temporal_unified(unified_temporal, "month")
unified_yearly = aggregate_temporal_unified(unified_temporal, "year")

unified_weekly = finalize_unified_temporal(unified_weekly)
unified_monthly = finalize_unified_temporal(unified_monthly)
unified_yearly = finalize_unified_temporal(unified_yearly)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 12. Save Tables

# COMMAND ----------

# Save all tables to Delta, overwriting both data and schema
output_db = "cohort_2a"
spark.sql(f"CREATE DATABASE IF NOT EXISTS {output_db}")

# Save daily table
unified_daily.withColumn("year", F.year("date")).write \
    .format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .partitionBy("year") \
    .saveAsTable(f"{output_db}.unified_daily")

# Save weekly table
unified_weekly.write \
    .format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .partitionBy("period") \
    .saveAsTable(f"{output_db}.unified_weekly")

# Save monthly table
unified_monthly.write \
    .format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .partitionBy("period") \
    .saveAsTable(f"{output_db}.unified_monthly")

# Save yearly table
unified_yearly.write \
    .format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .partitionBy("period") \
    .saveAsTable(f"{output_db}.unified_yearly")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 13. Cleanup

# COMMAND ----------

condition_daily.unpersist()
drug_daily.unpersist()
measurement_daily.unpersist()
procedure_daily.unpersist()
observation_daily.unpersist()
condition_arrays.unpersist()
drug_arrays.unpersist()
measurement_arrays.unpersist()
procedure_arrays.unpersist()
observation_arrays.unpersist()
visit_daily.unpersist()
visit_counts.unpersist()
complete_person_dates.unpersist()
