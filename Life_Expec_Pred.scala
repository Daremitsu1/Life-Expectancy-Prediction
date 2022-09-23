// Databricks notebook source
// MAGIC %md
// MAGIC #### 1. Load Data in Dataframe using Custom Schema

// COMMAND ----------

// MAGIC %scala
// MAGIC 
// MAGIC import org.apache.spark.sql.Encoders

// COMMAND ----------

case class life(Country: String,
Year: Int,
Status: String,
Life_expectancy: Double,
Adult_Mortality: Int,
infant_deaths: Int,
Alcohol: Double,
percentage_expenditure: Double,
Hepatitis_B: Int,
Measles: Int,
BMI: Double,
under_five_deaths: Int,
Polio: Int,
Total_expenditure: Double,
Diphtheria: Int,
HIV_AIDS: Double,
GDP: Double,
Population: Double,
thinness_1_19_years: Double,
thinness_5_9_years: Double,
Income_composition_of_resources: Double,
Schooling: Double)

// COMMAND ----------

val lifeSchema = Encoders.product[life].schema

// COMMAND ----------

val lifeDF = spark.read.schema(lifeSchema).option("header", "true").csv("/FileStore/tables/Life_Expectancy_Data.csv")

// COMMAND ----------

display(lifeDF)

// COMMAND ----------

// MAGIC %md
// MAGIC 
// MAGIC #### 2. Count Data Records

// COMMAND ----------

// MAGIC %scala
// MAGIC 
// MAGIC lifeDF.count()

// COMMAND ----------

// MAGIC %md
// MAGIC 
// MAGIC #### 3. Statistics of Data

// COMMAND ----------

// MAGIC %scala
// MAGIC 
// MAGIC display(lifeDF.describe())

// COMMAND ----------


