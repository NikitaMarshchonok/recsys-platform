from pyspark.sql import SparkSession
from pyspark.sql.functions import(
    count, mean, sttdev, min, max, col

)
spark = SparkSession.builder.appName('RecSys Feature Engineering').master('local[*]').get0rCreate()

spark.sparkContext.setLogLevel('ERROR')
print('Spark Session created succsessfully')
