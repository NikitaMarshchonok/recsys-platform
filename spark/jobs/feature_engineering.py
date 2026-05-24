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


# Считаем характеристики каждого пользователя
# Это и есть feature engineering
user_features = ratings.groupBy("userId").agg(
    count("movieId").alias("total_rated"),      # сколько фильмов оценил
    mean("rating").alias("avg_rating"),          # средняя оценка
    stddev("rating").alias("rating_stddev"),     # насколько разброс оценок
    min("rating").alias("min_rating"),           # самая низкая оценка
    max("rating").alias("max_rating"),           # самая высокая оценка
)

# stddev может быть null если пользователь оценил только 1 фильм
# заменяем null на 0
user_features = user_features.fillna(0, subset=["rating_stddev"])

print("=== Фичи пользователей ===")
user_features.show(5)

# Считаем характеристики каждого фильма
movie_features = ratings.groupBy("movieId").agg(
    count("userId").alias("total_ratings"),     # сколько людей оценили
    mean("rating").alias("avg_rating"),          # средняя оценка фильма
    stddev("rating").alias("rating_stddev"),     # насколько спорный фильм
)

movie_features = movie_features.fillna(0, subset=["rating_stddev"])

# Добавляем жанр из таблицы movies
movie_features = movie_features.join(movies, on="movieId", how="left")

print("=== Фичи фильмов ===")
movie_features.show(5)

# Сохраняем обработанные фичи
os.makedirs("data/processed", exist_ok=True)

user_features.toPandas().to_csv(
    "data/processed/user_features.csv", index=False
)
movie_features.toPandas().to_csv(
    "data/processed/movie_features.csv", index=False
)

print("Фичи сохранены в data/processed/")
print(f"Пользователей с фичами: {user_features.count()}")
print(f"Фильмов с фичами: {movie_features.count()}")

spark.stop()
print("Готово!")