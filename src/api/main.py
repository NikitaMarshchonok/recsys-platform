from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALSModel
import pandas as pd

# Создаём приложение
app = FastAPI(
    title="RecSys API",
    description="Рекомендательная система фильмов",
    version="1.0.0",
)

# Запускаем Spark и загружаем модель при старте
spark = SparkSession.builder \
    .appName("RecSys API") \
    .master("local[*]") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

# Загружаем обученную модель
model = ALSModel.load("models/als_model")

# Загружаем список фильмов
movies = pd.read_csv("data/raw/movies.csv")

print("API готов к работе!")

# Схема запроса — что принимаем
class RecommendRequest(BaseModel):
    user_id: int
    n_recommendations: int = 10  # по умолчанию 10 фильмов

# Схема ответа — что возвращаем
class MovieRecommendation(BaseModel):
    movie_id: int
    title: str
    genres: str
    predicted_rating: float

# Главная страница — проверка что API работает
@app.get("/")
def root():
    return {"status": "ok", "message": "RecSys API работает"}

# Эндпоинт здоровья — для мониторинга
@app.get("/health")
def health():
    return {"status": "healthy"}