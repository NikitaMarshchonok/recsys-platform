from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALSModel
import pandas as pd
import psycopg2
import time
from datetime import datetime

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

def get_db_connection():
    return psycopg2.connect(
        host="localhost",
        database="recsys",
        user="recsys",
        password="recsys",
        port=5434,
    )

def init_db():
    conn = get_db_connection()
    with conn:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS request_logs (
                    id SERIAL PRIMARY KEY,
                    endpoint VARCHAR(50),
                    user_id INTEGER,
                    n_recommendations INTEGER,
                    response_time_ms FLOAT,
                    n_results INTEGER,
                    created_at TIMESTAMP DEFAULT NOW()
                )
            """)
    conn.close()

# Создаём таблицу при старте
init_db()
print("База данных инициализирована")

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
    start_time = time.time()  # начинаем замер времени

    if request.user_id < 1 or request.user_id > 500:
        raise HTTPException(
            status_code=404,
            detail=f"Пользователь {request.user_id} не найден"
        )

    user_df = spark.createDataFrame(
        [(request.user_id,)],
        ["userId"]
    )

    recs = model.recommendForUserSubset(
        user_df,
        request.n_recommendations
    )

    recs_list = recs.collect()[0]["recommendations"]

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

    # Считаем время и пишем в базу
    response_time = (time.time() - start_time) * 1000  # в миллисекундах

    try:
        conn = get_db_connection()
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO request_logs
                        (endpoint, user_id, n_recommendations, response_time_ms, n_results)
                    VALUES (%s, %s, %s, %s, %s)
                    """,
                    ("/recommend", request.user_id, request.n_recommendations,
                     response_time, len(result))
                )
        conn.close()
    except Exception as e:
        print(f"Ошибка логирования: {e}")
        # не ломаем API если логирование упало

    return result


@app.get("/similar_movies/{movie_id}")
def similar_movies(movie_id: int, n: int = 5):
    movie_features = pd.read_csv("data/processed/movie_features.csv")
    
    # Проверяем что фильм существует
    target = movie_features[movie_features["movieId"] == movie_id]
    if target.empty:
        raise HTTPException(status_code=404, detail=f"Фильм {movie_id} не найден")
    
    # Находим жанр
    genre = target.iloc[0]["genres"]
    
    # Похожие фильмы того же жанра
    similar = movie_features[movie_features["genres"] == genre] \
        .sort_values("avg_rating", ascending=False) \
        .head(n)
    
    return [
        {
            "movie_id": int(row["movieId"]),
            "title": row["title"],
            "genres": row["genres"],
            "avg_rating": round(row["avg_rating"], 2),
        }
        for _, row in similar.iterrows()
    ]

@app.get("/stats")
def stats():
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            cur.execute("""
                SELECT 
                    COUNT(*) as total_requests,
                    AVG(response_time_ms) as avg_response_ms,
                    MIN(response_time_ms) as min_response_ms,
                    MAX(response_time_ms) as max_response_ms
                FROM request_logs
                WHERE endpoint = '/recommend'
            """)
            row = cur.fetchone()
        conn.close()

        return {
            "total_requests": row[0],
            "avg_response_ms": round(row[1] or 0, 2),
            "min_response_ms": round(row[2] or 0, 2),
            "max_response_ms": round(row[3] or 0, 2),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))