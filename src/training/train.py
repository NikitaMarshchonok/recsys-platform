import mlflow
import mlflow.spark
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import col
import os


# Spark сессия
spark = SparkSession.builder \
    .appName("RecSys Training") \
    .master("local[*]") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

# MLflow — куда записывать эксперименты
mlflow.set_tracking_uri("http://localhost:5001")
mlflow.set_experiment("recsys_als")

print("Сессии созданы")

# Читаем оценки
ratings = spark.read \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .csv("data/raw/ratings.csv")

# Делим 80% train / 20% test
train, test = ratings.randomSplit([0.8, 0.2], seed=42)

print(f"Train: {train.count()} записей")
print(f"Test: {test.count()} записей")