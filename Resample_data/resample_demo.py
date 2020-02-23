# Databricks notebook source
# MAGIC %md ## Overview:
# MAGIC Comparing **interpolation** between pandas UDF and pure spark sql 

# COMMAND ----------

# DBTITLE 1,Import Packages
import pandas as pd
import numpy as np

import pyspark.sql.functions as f
from pyspark.sql import Window
from pyspark.sql.functions import pandas_udf
from pyspark.sql.functions import PandasUDFType


# COMMAND ----------

# DBTITLE 1,Functions
@pandas_udf("group string, num0 long, num1 long, date date, exist string", functionType=PandasUDFType.GROUPED_MAP)
def resampler(df):
    '''
    Resamples date and interpolates the rest of the columns.
    '''
    
    # sets the data column as the index and resamples by day
    result = (df.set_index(
        pd.to_datetime(df['date'])).resample("24H").last().drop("date", axis=1).reset_index())  
    result["exist"] = np.where(result['exist'].isnull(), "false", "true")
    # backfills object types and interpolates the rest
    result = result.apply(lambda x: x.bfill() if x.dtype.kind in 'O' else x.interpolate())
    
    return result

# COMMAND ----------

df = pd.DataFrame({"group": ["apple", "apple", "apple", "banana", "banana", "cat", "cat", "cat"],
                   "num0": [3, 3, 5, 1, 2, 3, 3, 5],
                   "num1": [1000, 1000, 2000, 30, 10, 1000, 1000, 2000],
                   "date": ["2017-01-01", "2017-01-03", "2017-01-05", "2018-01-01", "2018-01-08","2017-01-01", "2017-01-02", "2017-01-05"],
                   "exist": ["true", "true", "true", "true", "true", "true", "true", "true"]
                  })

df = spark.createDataFrame(df).withColumn("date", f.col("date").cast("date"))

# COMMAND ----------

# MAGIC %md ### Pure Spark SQL Method

# COMMAND ----------

# Return the min and max Dates per group
df_group = (
    df
    .groupby('group')
    .agg(f.min('date').alias('start'), f.max('date').alias('end'))
)
# Take the difference in days between the max and min dates then fill in that may days per group
df_resample = (
    df_group
    .withColumn('diffDays', f.datediff('end', 'start'))
    .withColumn('repeat', f.expr("split(repeat(',', diffDays), ',')"))
    .select('*', f.posexplode('repeat').alias('date', 'val'))
    .drop('repeat', 'val', 'diffDays')
    .withColumn('date', f.expr('date_add(start, date)'))
    .drop('start', 'end')
)
df_merge = (
    df_resample
    .join(df, on=['group', 'date'], how='left')
    .orderBy(['group', 'date'])
    .withColumn('id', f.when(f.col('exist').isNotNull(), f.monotonically_increasing_id()))
    .withColumn('exist', f.when(f.col('exist').isNull(), 'false').otherwise('true'))
)

# COMMAND ----------

# Set the window functions for Interpolation
w1 = Window.partitionBy("group").orderBy("date")
w2 = Window.partitionBy("group_id").orderBy("date")
w3 = Window.partitionBy("group_id")
w4 = Window.partitionBy("group").orderBy("id")

# Set the columns needed to interpolate
result = df_merge \
    .withColumn("group_id", f.last(f.col("id"), ignorenulls=True).over(w1)) \
    .withColumn("i", f.row_number().over(w2) - 1) \
    .withColumn("dx", f.max(f.col("i") + 1).over(w3))

for name in ["num0", "num1"]:
  result = result.withColumn("next_value_{}".format(name), f.lead(name, 1).over(w4))

# Interpolate the columns and drop the created variables
for name in ["num0", "num1"]:
  result = result \
        .withColumn("value0_{}".format(name), f.first(name).over(w2))\
        .withColumn("dy_{}".format(name), f.first(f.col("next_value_{}".format(name)) - f.col(name)).over(w2))\
        .withColumn("{}".format(name), 
        f.when(f.isnull("{}".format(name)), 
               (f.col("value0_{}".format(name)) + f.col("i") * f.col("dy_{}".format(name)) / f.col("dx")).cast(result.schema["{}".format(name)].dataType)) \
               .otherwise(f.col(name))) \
        .withColumn("{}".format(name), f.last("{}".format(name), ignorenulls=True).over(w1)) \
        .drop("next_value_{}".format(name), "value0_{}".format(name), "dy_{}".format(name), "value0_{}".format(name))

result = result \
  .drop("next_position", "group_id", "i", "id", "dx")

# COMMAND ----------

display(result)

# COMMAND ----------

# MAGIC %md ### Using Pandas UDF Method

# COMMAND ----------

test_pandas_udf = df.groupby(["group"]).apply(resampler)

# COMMAND ----------

display(test_pandas_udf)

# COMMAND ----------


