# Databricks notebook source
# MAGIC %md ## Overview:
# MAGIC Demo of replacing Jinja templates within a column using Pyspark

# COMMAND ----------

# DBTITLE 1,Install Packages
dbutils.library.installPyPI('jinja2')

# COMMAND ----------

# DBTITLE 1,Import Packages
import re
import pandas as pd

from jinja2 import Template
from pyspark.sql.functions import array, struct, lit

# COMMAND ----------

# DBTITLE 1,Functions
@udf('string')
def keyword_gen(col_struct, col_list):
    ''' 
    Render Jinja templates row wise.
    
    :param col_struct:  Dataframe struct type
    :param col_list:  List of column names
    :return:  String
    '''
    
    col_dict = {}
    for col in col_list:
        col_dict[col] = col_struct[col] if col_struct[col] else 'OUTPUT MISSING'
       
    t = Template('{templates}'.format(templates=col_dict['templates']))
    sent = t.render(col_dict)
    if re.search(r'\b(OUTPUT MISSING)\b', sent, flags=re.IGNORECASE):
        return ''
    return sent

# COMMAND ----------

# Create sample dataframe
df = pd.DataFrame({'col0': ['apple0', 'apple1', '', 'apple3'],
                   'col1': ['', 'banana1', 'banana2', 'banana3'],
                   'templates': ['{{col1}} text', '{{col1|upper}} text', 'this {{col0|title}} to {{col1|title}}', 'col0 {{col0|upper}} to {{col1|title}} col1']
                  })

df = spark.createDataFrame(df)

# COMMAND ----------

df_template = df.withColumn('render', keyword_gen(struct(df.columns), array([lit(x) for x in df.columns])))
display(df_template)

# COMMAND ----------


