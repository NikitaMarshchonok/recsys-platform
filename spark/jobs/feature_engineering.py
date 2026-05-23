from pyspark.sql import SparkSession
from pyspark.sql.functions import(
    count, mean, sttdev, min, max, col

)
spark = SparkSession.builder.appName('RecSys Feature Engineering').master('local[*]').get0rCreate()

spark.sparkContext.setLogLevel('ERROR')
print('Spark Session created succsessfully')


# Читаем сырые данные
ratings = spark.read \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .csv("data/raw/ratings.csv")

movies = spark.read \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .csv("data/raw/movies.csv")

users = spark.read \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .csv("data/raw/users.csv")

print(f"Оценок: {ratings.count()}")
print(f"Фильмов: {movies.count()}")
print(f"Пользователей: {users.count()}")