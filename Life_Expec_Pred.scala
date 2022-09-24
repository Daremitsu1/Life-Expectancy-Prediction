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

val lifeDF = spark.read.schema(lifeSchema).option("header", "true").csv("/FileStore/tables/Life_Expectancy_Data-1.csv")

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

// MAGIC %scala
// MAGIC 
// MAGIC lifeDF.printSchema()

// COMMAND ----------

// MAGIC %md
// MAGIC #### 4. Exploratory Data Analysis or EDA

// COMMAND ----------

// MAGIC %md
// MAGIC ##### 4.1. Creating Temporary View

// COMMAND ----------

// MAGIC %scala
// MAGIC 
// MAGIC lifeDF.createOrReplaceTempView("LifeData")

// COMMAND ----------

// MAGIC %md
// MAGIC ##### 4.2. Display Data from View

// COMMAND ----------

// MAGIC %sql
// MAGIC 
// MAGIC select * from LifeData;

// COMMAND ----------

// MAGIC %md
// MAGIC 
// MAGIC ##### 4.3. Histogram of Life Expectancy

// COMMAND ----------

// MAGIC %sql
// MAGIC 
// MAGIC select life_expectancy from LifeData;

// COMMAND ----------

// MAGIC %md
// MAGIC ##### 4.4. Histogram of Adult Mortality

// COMMAND ----------

// MAGIC %sql
// MAGIC 
// MAGIC select Adult_Mortality from LifeData;

// COMMAND ----------

// MAGIC %md
// MAGIC 
// MAGIC ##### 4.5. Histograms of Infant Deaths

// COMMAND ----------

// MAGIC %sql
// MAGIC 
// MAGIC select infant_deaths from LifeData

// COMMAND ----------



// COMMAND ----------



// COMMAND ----------



// COMMAND ----------

// MAGIC %md
// MAGIC ##### 4.20. Life Expectancy w.r.t Year

// COMMAND ----------

// MAGIC %sql
// MAGIC 
// MAGIC select avg(Life_expectancy), Year from LifeData group by Year order by Year asc

// COMMAND ----------

// MAGIC %md
// MAGIC #### 5. Collecting all String Columns into an Array

// COMMAND ----------

// MAGIC %scala
// MAGIC 
// MAGIC var StringfeatureCol = Array("Country", "Status")

// COMMAND ----------

import org.apache.spark.ml.feature.StringIndexer

val df = spark.createDataFrame(
Seq((0, "a"), (1, "b"), (2, "c"), (3, "a"), (4, "a"), (5, "c"))
).toDF("id", "category")

val indexer = new StringIndexer()
.setInputCol("category")
.setOutputCol("categoryIndex")

val indexed = indexer.fit(df).transform(df)

indexed.show()

// COMMAND ----------

// MAGIC %md
// MAGIC #### 6. Define the Pipeline

// COMMAND ----------

// MAGIC %scala
// MAGIC 
// MAGIC import org.apache.spark.ml.attribute.Attribute
// MAGIC import org.apache.spark.ml.feature.{IndexToString, StringIndexer}
// MAGIC import org.apache.spark.ml.{Pipeline, PipelineModel}
// MAGIC 
// MAGIC val indexers = StringfeatureCol.map { colName =>
// MAGIC new StringIndexer().setInputCol(colName).setHandleInvalid("skip").setOutputCol(colName + "_indexed")
// MAGIC }
// MAGIC 
// MAGIC val pipeline = new Pipeline()
// MAGIC .setStages(indexers)
// MAGIC 
// MAGIC val LifeDF = pipeline.fit(lifeDF).transform(lifeDF)

// COMMAND ----------

// 6.1. Print the Schema

%scala

LifeDF.printSchema()

// COMMAND ----------

// 6.2. Split the data

%scala

val splits = LifeDF.randomSplit(Array(0.7, 0.3))
val train = splits(0)
val test = splits(1)
val train_rows = train.count()
val test_rows = test.count()
println("Training Rows: " + train_rows + " Testing Rows: " + test_rows)

// COMMAND ----------

// MAGIC %md
// MAGIC 
// MAGIC #### 7. VectorAssembler()

// COMMAND ----------

// MAGIC %scala
// MAGIC 
// MAGIC import org.apache.spark.ml.feature.VectorAssembler
// MAGIC 
// MAGIC val assembler = new VectorAssembler().setInputCols(Array("Country_indexed", "Year", "Status_indexed", "Adult_Mortality", "infant_deaths", "Alcohol", "percentage_expenditure", "Hepatitis_B", "Measles", "BMI", "under_five_deaths", "Polio", "Total_expenditure", "Diphtheria", "HIV_AIDS", "GDP", "Population", "thinness_1_19_years", "thinness_5_9_years", "Income_composition_of_resources", "Schooling")).setOutputCol("features").setHandleInvalid("skip")
// MAGIC 
// MAGIC val training = assembler.transform(train).select($"features", $"Life_expectancy".alias("label"))
// MAGIC 
// MAGIC training.show()

// COMMAND ----------

// MAGIC %md
// MAGIC 
// MAGIC ##### 7.1. Train the linear regression model

// COMMAND ----------

// MAGIC %scala
// MAGIC 
// MAGIC import org.apache.spark.sql.types._
// MAGIC import org.apache.spark.sql.functions._
// MAGIC 
// MAGIC import org.apache.spark.ml.Pipeline
// MAGIC import org.apache.spark.ml.feature.VectorAssembler
// MAGIC import org.apache.spark.ml.regression.LinearRegression
// MAGIC 
// MAGIC val lr = new LinearRegression().setLabelCol("label").setFeaturesCol("features")
// MAGIC 
// MAGIC val model = lr.fit(training)
// MAGIC 
// MAGIC println("Model Trained!")

// COMMAND ----------

// MAGIC %md
// MAGIC 
// MAGIC ##### 7.2. Prepare the testing data

// COMMAND ----------

// MAGIC %scala
// MAGIC 
// MAGIC val testing = assembler.transform(test).select($"features", $"Life_expectancy".alias("trueLabel"))
// MAGIC testing.show()

// COMMAND ----------

// MAGIC %md
// MAGIC 
// MAGIC ##### 7.3. Test the Model

// COMMAND ----------

// MAGIC %scala
// MAGIC 
// MAGIC val prediction = model.transform(testing)
// MAGIC val predicted = prediction.select("features", "prediction", "trueLabel")
// MAGIC predicted.show(100)

// COMMAND ----------

// MAGIC %md
// MAGIC 
// MAGIC #### 8. Root-Mean-Square-Error(RMSE)

// COMMAND ----------

import org.apache.spark.ml.evaluation.RegressionEvaluator

val evaluator = new RegressionEvaluator().setLabelCol("trueLabel").setPredictionCol("prediction").setMetricName("rmse")
val rmse = evaluator.evaluate(prediction)
println("Root Mean Square Error (RMSE): " + (rmse))

// COMMAND ----------


