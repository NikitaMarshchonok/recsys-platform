from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
import logging
import pandas as pd
import time
import uuid

from src.api.config import get_settings

settings = get_settings()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("recsys.api")

spark = None
model = None
movies = None
valid_user_ids = None

# Создаём приложение
app = FastAPI(
    title="RecSys API",
    description="Рекомендательная система фильмов",
    version="1.0.0",
)


@app.middleware("http")
async def request_tracing_middleware(request, call_next):
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    start_time = time.perf_counter()
    status_code = 500

    try:
        response = await call_next(request)
        status_code = response.status_code
    except Exception:
        latency_ms = (time.perf_counter() - start_time) * 1000
        logger.exception(
            "request_failed request_id=%s method=%s path=%s status_code=%s latency_ms=%.2f",
            request_id,
            request.method,
            request.url.path,
            status_code,
            latency_ms,
        )
        raise

    latency_ms = (time.perf_counter() - start_time) * 1000
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Response-Time-ms"] = f"{latency_ms:.2f}"
    logger.info(
        "request_completed request_id=%s method=%s path=%s status_code=%s latency_ms=%.2f",
        request_id,
        request.method,
        request.url.path,
        status_code,
        latency_ms,
    )

    return response


def get_db_connection():
    import psycopg2

    return psycopg2.connect(
        host=settings.db_host,
        database=settings.db_name,
        user=settings.db_user,
        password=settings.db_password,
        port=settings.db_port,
    )

def init_db():
    try:
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
    except Exception as e:
        print(f"PostgreSQL недоступен, логирование отключено: {e}")


def load_recommendation_resources():
    global spark, model

    if spark is None or model is None:
        from pyspark.sql import SparkSession
        from pyspark.ml.recommendation import ALSModel

        spark = SparkSession.builder \
            .appName("RecSys API") \
            .master("local[*]") \
            .getOrCreate()
        spark.sparkContext.setLogLevel("ERROR")
        model = ALSModel.load(str(settings.model_path))

    return spark, model, load_movie_catalog(), load_valid_user_ids()


def load_movie_catalog():
    global movies

    if movies is None:
        movies = pd.read_csv(settings.movies_path)

    return movies


def load_valid_user_ids():
    global valid_user_ids

    if valid_user_ids is None:
        users = pd.read_csv(settings.users_path)
        valid_user_ids = set(users["userId"].astype(int))

    return valid_user_ids


def select_top_movies(n: int, genre: str | None = None):
    movie_features = pd.read_csv(settings.movie_features_path)

    if genre is not None:
        movie_features = movie_features[
            movie_features["genres"].str.lower() == genre.lower()
        ]

    return movie_features.sort_values(
        ["avg_rating", "total_ratings"],
        ascending=[False, False],
    ).head(n)

# Схема запроса — что принимаем
class RecommendRequest(BaseModel):
    user_id: int
    n_recommendations: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Количество рекомендаций от 1 до 50",
    )
    fallback_to_top: bool = Field(
        default=False,
        description="Вернуть top-rated fallback для неизвестного пользователя",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "user_id": 1,
                    "n_recommendations": 5,
                    "fallback_to_top": False,
                }
            ]
        }
    }

# Схема ответа — что возвращаем
class MovieRecommendation(BaseModel):
    movie_id: int
    title: str
    genres: str
    predicted_rating: float


class SimilarMovieResponse(BaseModel):
    movie_id: int
    title: str
    genres: str
    avg_rating: float


class TopMovieResponse(BaseModel):
    movie_id: int
    title: str
    genres: str
    avg_rating: float
    total_ratings: int


class GenreSummaryResponse(BaseModel):
    genre: str
    movie_count: int
    avg_rating: float


class UserProfileResponse(BaseModel):
    user_id: int
    total_rated: int
    avg_rating: float
    rating_stddev: float
    min_rating: float
    max_rating: float


class RootResponse(BaseModel):
    status: str
    message: str


class HealthResponse(BaseModel):
    status: str


class ReadinessResponse(BaseModel):
    status: str
    checks: dict[str, bool]


class VersionResponse(BaseModel):
    service: str
    version: str


class StatsResponse(BaseModel):
    total_requests: int
    avg_response_ms: float
    min_response_ms: float
    max_response_ms: float


class MetricsResponse(BaseModel):
    spark_loaded: bool
    model_loaded: bool
    movie_catalog_loaded: bool
    user_catalog_loaded: bool
    cached_movies: int
    cached_users: int


class CatalogSummaryResponse(BaseModel):
    total_movies: int
    total_genres: int
    total_ratings: int
    avg_rating: float
    min_avg_rating: float
    max_avg_rating: float
    most_rated_movie_id: int | None
    most_rated_title: str | None

# Главная страница — проверка что API работает
@app.get("/", response_model=RootResponse)
def root():
    return {"status": "ok", "message": "RecSys API работает"}

# Эндпоинт здоровья — для мониторинга
@app.get("/health", response_model=HealthResponse)
def health():
    return {"status": "healthy"}


@app.get("/version", response_model=VersionResponse)
def version():
    return {"service": app.title, "version": app.version}


@app.get("/ready", response_model=ReadinessResponse)
def readiness():
    checks = {
        "model": settings.model_path.exists(),
        "movies": settings.movies_path.exists(),
        "users": settings.users_path.exists(),
        "user_features": settings.user_features_path.exists(),
        "movie_features": settings.movie_features_path.exists(),
    }

    if not all(checks.values()):
        raise HTTPException(
            status_code=503,
            detail={"status": "not_ready", "checks": checks},
        )

    return {"status": "ready", "checks": checks}


@app.get("/metrics", response_model=MetricsResponse)
def metrics():
    return {
        "spark_loaded": spark is not None,
        "model_loaded": model is not None,
        "movie_catalog_loaded": movies is not None,
        "user_catalog_loaded": valid_user_ids is not None,
        "cached_movies": 0 if movies is None else len(movies),
        "cached_users": 0 if valid_user_ids is None else len(valid_user_ids),
    }


@app.get("/catalog/summary", response_model=CatalogSummaryResponse)
def catalog_summary():
    movie_features = pd.read_csv(settings.movie_features_path)

    if movie_features.empty:
        return {
            "total_movies": 0,
            "total_genres": 0,
            "total_ratings": 0,
            "avg_rating": 0.0,
            "min_avg_rating": 0.0,
            "max_avg_rating": 0.0,
            "most_rated_movie_id": None,
            "most_rated_title": None,
        }

    most_rated = movie_features.sort_values(
        ["total_ratings", "avg_rating"],
        ascending=[False, False],
    ).iloc[0]

    return {
        "total_movies": int(len(movie_features)),
        "total_genres": int(movie_features["genres"].nunique()),
        "total_ratings": int(movie_features["total_ratings"].sum()),
        "avg_rating": round(float(movie_features["avg_rating"].mean()), 2),
        "min_avg_rating": round(float(movie_features["avg_rating"].min()), 2),
        "max_avg_rating": round(float(movie_features["avg_rating"].max()), 2),
        "most_rated_movie_id": int(most_rated["movieId"]),
        "most_rated_title": most_rated["title"],
    }


@app.post("/recommend", response_model=list[MovieRecommendation])
def recommend(request: RecommendRequest):
    start_time = time.time()  # начинаем замер времени

    init_db()
    users = load_valid_user_ids()

    if request.user_id not in users:
        if not request.fallback_to_top:
            raise HTTPException(
                status_code=404,
                detail=f"Пользователь {request.user_id} не найден"
            )

        top = select_top_movies(request.n_recommendations)
        result = [
            MovieRecommendation(
                movie_id=int(row["movieId"]),
                title=row["title"],
                genres=row["genres"],
                predicted_rating=round(float(row["avg_rating"]), 2),
            )
            for _, row in top.iterrows()
        ]
    else:
        spark_session, als_model, movie_catalog, _ = load_recommendation_resources()

        user_df = spark_session.createDataFrame(
            [(request.user_id,)],
            ["userId"]
        )

        recs = als_model.recommendForUserSubset(
            user_df,
            request.n_recommendations
        )

        rows = recs.collect()
        if not rows:
            raise HTTPException(
                status_code=404,
                detail=f"Для пользователя {request.user_id} нет рекомендаций"
            )

        recs_list = rows[0]["recommendations"]

        result = []
        for rec in recs_list:
            movie_id = rec["movieId"]
            movie_info = movie_catalog[movie_catalog["movieId"] == movie_id].iloc[0]
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


@app.get("/similar_movies/{movie_id}", response_model=list[SimilarMovieResponse])
def similar_movies(movie_id: int, n: int = Query(default=5, ge=1, le=50)):
    movie_features = pd.read_csv(settings.movie_features_path)
    
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


@app.get("/movies/top", response_model=list[TopMovieResponse])
def top_movies(
    n: int = Query(default=10, ge=1, le=50),
    genre: str | None = Query(default=None, min_length=1),
):
    top = select_top_movies(n, genre)

    return [
        {
            "movie_id": int(row["movieId"]),
            "title": row["title"],
            "genres": row["genres"],
            "avg_rating": round(row["avg_rating"], 2),
            "total_ratings": int(row["total_ratings"]),
        }
        for _, row in top.iterrows()
    ]


@app.get("/movies/genres", response_model=list[GenreSummaryResponse])
def movie_genres():
    movie_features = pd.read_csv(settings.movie_features_path)

    genre_stats = movie_features.groupby("genres").agg(
        movie_count=("movieId", "count"),
        avg_rating=("avg_rating", "mean"),
    ).reset_index()

    genre_stats = genre_stats.sort_values(
        ["movie_count", "avg_rating", "genres"],
        ascending=[False, False, True],
    )

    return [
        {
            "genre": row["genres"],
            "movie_count": int(row["movie_count"]),
            "avg_rating": round(row["avg_rating"], 2),
        }
        for _, row in genre_stats.iterrows()
    ]


@app.get("/users/{user_id}/profile", response_model=UserProfileResponse)
def user_profile(user_id: int):
    user_features = pd.read_csv(settings.user_features_path)

    target = user_features[user_features["userId"] == user_id]
    if target.empty:
        raise HTTPException(status_code=404, detail=f"Пользователь {user_id} не найден")

    row = target.iloc[0]
    return {
        "user_id": int(row["userId"]),
        "total_rated": int(row["total_rated"]),
        "avg_rating": round(row["avg_rating"], 2),
        "rating_stddev": round(row["rating_stddev"], 2),
        "min_rating": float(row["min_rating"]),
        "max_rating": float(row["max_rating"]),
    }


@app.get("/stats", response_model=StatsResponse)
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
