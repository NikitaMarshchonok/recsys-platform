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

@app.post("/recommend", response_model=list[MovieRecommendation])
def recommend(request: RecommendRequest):
    # Проверяем что пользователь существует
    if request.user_id < 1 or request.user_id > 500:
        raise HTTPException(
            status_code=404,
            detail=f"Пользователь {request.user_id} не найден"
        )

    # Создаём DataFrame с одним пользователем
    user_df = spark.createDataFrame(
        [(request.user_id,)],
        ["userId"]
    )

    # Получаем топ-N рекомендаций от модели
    recs = model.recommendForUserSubset(
        user_df,
        request.n_recommendations
    )

    # Разворачиваем список рекомендаций
    recs_list = recs.collect()[0]["recommendations"]

    # Собираем результат
    result = []
    for rec in recs_list:
        movie_id = rec["movieId"]
        movie_info = movies[movies["movieId"] == movie_id].iloc[0]
        result.append(MovieRecommendation(
            movie_id=movie_id,
            title=movie_info["title"],
            genres=movie_info["genres"],
            predicted_rating=round(float(rec["rating"]), 2),
        ))

    return result