# Databricks notebook source
# MAGIC %md
# MAGIC ## Overview: Evaluating risk for loan approvals using a gradient boosted tree classifier
# MAGIC ### Data: Public data from Lending Club including all funded loans from 2012 to 2017

# COMMAND ----------

# DBTITLE 1,Imports
from pyspark.sql.functions import regexp_replace, substring, trim, round, col
from pyspark.ml.classification import GBTClassifier
from pyspark.mllib.evaluation import BinaryClassificationMetrics
from pyspark.ml.linalg import Vectors

# COMMAND ----------

# DBTITLE 1,Functions
def extract(row):
  return (row.net,) + tuple(row.probability.toArray().tolist()) +  (row.label,) + (row.prediction,)

def score(model,data):
  pred = model.transform(data).select("net", "probability", "label", "prediction")
  pred = pred.rdd.map(extract).toDF(["net", "p0", "p1", "label", "prediction"])
  return pred 

def auc(pred):
  metric = BinaryClassificationMetrics(pred.select("p1", "label").rdd)
  return metric.areaUnderROC

# COMMAND ----------

# DBTITLE 1,Load Data
# location of loanstats_2012_2017.parquet
SOURCE = "/databricks-datasets/samples/lending_club/parquet/"

# Read parquet
df = spark.read.parquet(SOURCE)

# split the data
(loan_sample, loan_other) = df.randomSplit([0.025, 0.975], seed=123)

# Select only the columns needed
loan_col = ["loan_status", "int_rate", "revol_util", "issue_d", "earliest_cr_line", "emp_length", "verification_status", "total_pymnt",
       "loan_amnt", "grade", "annual_inc", "dti", "addr_state", "term", "home_ownership", "purpose", "application_type", "delinq_2yrs", "total_acc"]
loan_sample = loan_sample.select(loan_col)

# COMMAND ----------

# Print out number of loans
print(str(loan_sample.count()) + " available loans")

# COMMAND ----------

display(loan_sample)

# COMMAND ----------

# DBTITLE 1,Munge Data
# Create bad loan label, this will include charged off, defaulted, and late repayments on loans
loan_sample_categorical = (loan_sample
                   .filter(col('loan_status').isin(["Default", "Charged Off", "Fully Paid"]))
                   .withColumn("bad_loan", (col('loan_status') != "Fully Paid").cast("string"))
                )
# Map multiple categories into one
loan_stats_categorical = loan_sample_categorical.withColumn('verification_status', trim(regexp_replace('verification_status', 'Source Verified', 'Verified')))

# Turning string interest rate and revoling util columns into numeric columns
loan_sample_numeric = (loan_sample_categorical
                   .withColumn('int_rate', regexp_replace('int_rate', '%', '').cast('float'))
                   .withColumn('revol_util', regexp_replace('revol_util', '%', '').cast('float'))
                   .withColumn('issue_year',  substring('issue_d', 5, 4).cast('double') )
                   .withColumn('earliest_year', substring('earliest_cr_line', 5, 4).cast('double'))
                )
loan_sample_numeric = loan_sample_numeric.withColumn('credit_length_in_years', (col('issue_year') - col('earliest_year')))

# Converting emp_length column into numeric
loan_sample_numeric = loan_sample_numeric.withColumn('emp_length', trim(regexp_replace('emp_length', "([ ]*+[a-zA-Z].*)|(n/a)", "") ))
loan_sample_numeric = loan_sample_numeric.withColumn('emp_length', trim(regexp_replace('emp_length', "< 1", "0") ))
loan_sample_numeric = loan_sample_numeric.withColumn('emp_length', trim(regexp_replace('emp_length', "10\\+", "10") ).cast('float'))

# Calculate the total amount of money earned or lost per loan
loan_stats = loan_sample_numeric.withColumn('net', round(col('total_pymnt') - col('loan_amnt'), 2))

# COMMAND ----------

display(loan_stats)

# COMMAND ----------

# DBTITLE 1,Train/Test Split
# Create train/test split
myY = "bad_loan"
categorical_col = ["term", "home_ownership", "purpose", "addr_state",
                   "verification_status","application_type"]
numeric_col = ["loan_amnt","emp_length", "annual_inc", "dti", "delinq_2yrs",
               "revol_util", "total_acc", "credit_length_in_years"]
myX = categorical_col + numeric_col

loan_stats2 = loan_stats.select(myX + [myY, "int_rate", "net", "issue_year"])
train = loan_stats2.filter(col('issue_year') <= 2015).cache()
valid = loan_stats2.filter(col('issue_year') > 2015).cache()

# COMMAND ----------

# DBTITLE 1,Build GBT Pipeline
# Establish stages for our GBT model
indexers = map(lambda c: StringIndexer(inputCol=c, outputCol=c+"_idx", handleInvalid = 'keep'), categorical_col)
imputers = Imputer(inputCols = numeric_col, outputCols = numeric_col)
featureCols = list(map(lambda c: c+"_idx", categorical_col)) + numeric_col

# Define vector assemblers
model_matrix_stages = (list(indexers) + [imputers] +
                         [VectorAssembler(inputCols=featureCols, outputCol="features"), StringIndexer(inputCol="bad_loan", outputCol="label")])

# Define a GBT model
gbt = GBTClassifier(featuresCol="features",
                    labelCol="label",
                    lossType = "logistic",
                    maxBins = 52,
                    maxIter=20,
                    maxDepth=5)

# Chain indexer and GBT in a Pipeline
pipeline = Pipeline(stages=model_matrix_stages+[gbt])

# Train model
gbt_model = pipeline.fit(train)

# COMMAND ----------

# DBTITLE 1,Score Data
gbt_train = score(gbt_model, train)
gbt_valid = score(gbt_model, valid)

print ("GBT Training AUC :" + str(auc(gbt_train)))
print ("GBT Validation AUC :" + str(auc(gbt_valid)))

# COMMAND ----------

display(gbt_valid)

# COMMAND ----------


