# Databricks notebook source
# DBTITLE 1,Installs
dbutils.library.installPyPI('hdbscan', '0.8.24')

# COMMAND ----------

# DBTITLE 1,Imports
import hdbscan

from operator import attrgetter
from scipy.sparse import csr_matrix, csc_matrix

from pyspark.sql.functions import col
from pyspark.sql.types import StructType, StructField, LongType, StringType, FloatType

from pyspark.ml import Pipeline
from pyspark.ml.linalg import Vectors, _convert_to_vector, VectorUDT
from pyspark.ml.feature import OneHotEncoderEstimator, VectorAssembler, PCA

# COMMAND ----------

# DBTITLE 1,Functions
# } Spark Utility {

def add_column_index(df): 
    '''Adds a column index to an existing dataframe.'''
    
    new_schema = StructType(
        df.schema.fields + [StructField('ColumnIndex', LongType(), False),]
    )
    new_df = df.rdd.zipWithIndex().map(
        lambda row: row[0] + (row[1],)).toDF(schema=new_schema)
    return new_df

# } Feature Vector {

def dense_to_sparse(vector):
    return _convert_to_vector(csc_matrix(vector.toArray()).T)

def explode(row):
    '''Return the row, column and values on a sparse vector.'''
    
    vec, row_index = row
    for col_index, value in zip(vec.indices, vec.values):
        yield row_index, col_index, value
        
def feature_to_matrix(df, feature_col):
    '''
    Creates a sparse matrix out of rows of sparse vectors.
    
    :param df: dataframe with feature vector
    :param feature_col: name of column with feature vector
    :return:  List of lists of data feeds
    '''
    
    features = df.rdd.map(attrgetter(feature_col))
    indexed_features = features.zipWithIndex()

    entries = indexed_features.flatMap(explode)
    row_indices, col_indices, data = zip(*entries.collect())

    shape = (
        df.count(),
        df.rdd.map(attrgetter(feature_col)).first().size
    )

    mat = csr_matrix((data, (row_indices, col_indices)), shape=shape)
    return mat

# } UDF {

to_sparse = udf(dense_to_sparse, VectorUDT())

# COMMAND ----------

# DBTITLE 1,ML Pipeline
# Create DataFrame
df = spark.createDataFrame([
    (0.0, 1.0),
    (1.0, 0.0),
    (2.0, 1.0),
    (0.0, 2.0),
    (0.0, 1.0),
    (2.0, 0.0)
], ['categoryIndex1', 'categoryIndex2'])

# Data pipeline for feature creation
encoder = OneHotEncoderEstimator(
    inputCols=['categoryIndex1'],
    outputCols=['categoryVec1']
)
assembler = VectorAssembler(
    inputCols=['categoryVec1', 'categoryIndex2'],
    outputCol='features'
)
# pca = PCA(
#     k=2,
#     inputCol='features',
#     outputCol='pcaFeatures'
# )
pipeline = Pipeline(stages=[encoder, assembler])
pipelineFit = pipeline.fit(df)
output = pipelineFit.transform(df)

# Transform all dense matrix to sparse
output_sparse = output.withColumn('features', to_sparse(col('features')))
# Add a row index to join results later
output_index = add_column_index(output_sparse)

# COMMAND ----------

feature_mat = feature_to_matrix(output_index, 'features')

# COMMAND ----------

# DBTITLE 1,Clustering
# Run clustering method
clusterer = hdbscan.HDBSCAN()
clusterer.fit(feature_mat.todense())

# COMMAND ----------

# DBTITLE 1,Cluster Result
# Transform clustered results to Spark DataFrame and join on original
schema = StructType([
    StructField('group_name', StringType()),
    StructField('cluster_prob', FloatType())
])
df_cluster = spark.createDataFrame(
    zip(clusterer.labels_.tolist(), clusterer.probabilities_.tolist()),
    schema=schema
)
df_cluster_index = add_column_index(df_cluster)

df_cluster_results = (
    output_index
    .select('categoryIndex1', 'categoryIndex2', 'ColumnIndex')
    .join(df_cluster_index, output_index.ColumnIndex == df_cluster_index.ColumnIndex, 'inner')
    .drop('ColumnIndex')
)

# COMMAND ----------

display(df_cluster_results)

# COMMAND ----------


