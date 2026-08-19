from difflib import SequenceMatcher
import json
import logging
import re
import time
from typing import Literal
import uuid

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, RedirectResponse
import pandas as pd

from src.api.config import get_settings
from src.api.feedback import RecommendationFeedbackStore
from src.api.schemas import (
    CatalogSummaryResponse,
    GenreSummaryResponse,
    HealthResponse,
    MetricsResponse,
    ModelInfoResponse,
    MovieDetailResponse,
    MovieRecommendation,
    RankingStrategy,
    ReadinessResponse,
    RecommendRequest,
    RecommendationFeedbackDeleteRequest,
    RecommendationFeedbackDeleteResponse,
    RecommendationFeedbackRequest,
    RecommendationFeedbackResponse,
    RecommendationFeedbackStateResponse,
    RecommendationFeedbackSummaryResponse,
    SimilarMovieResponse,
    StatsResponse,
    TopMovieResponse,
    UserProfileResponse,
    UserRatingHistoryResponse,
    VersionResponse,
)

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

web_app_path = settings.base_dir / "src" / "api" / "static" / "index.html"


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
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "same-origin"
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


feedback_store = RecommendationFeedbackStore(lambda: get_db_connection())


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


def directory_size_bytes(path):
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size

    return sum(file.stat().st_size for file in path.rglob("*") if file.is_file())


def load_model_card():
    if not settings.model_card_path.exists():
        return {}

    with settings.model_card_path.open(encoding="utf-8") as file:
        return json.load(file)


def rank_movies_by_confidence(movie_features: pd.DataFrame) -> pd.DataFrame:
    ranked = movie_features.copy()
    if ranked.empty:
        ranked["_ranking_score"] = pd.Series(dtype=float)
        return ranked

    catalog_rating = float(ranked["avg_rating"].mean())
    confidence_votes = max(
        float(ranked["total_ratings"].quantile(0.90)),
        1.0,
    )
    votes = ranked["total_ratings"].astype(float)

    # Bayesian shrinkage keeps tiny rating samples close to the catalog average.
    ranked["_ranking_score"] = (
        (votes / (votes + confidence_votes)) * ranked["avg_rating"]
        + (confidence_votes / (votes + confidence_votes)) * catalog_rating
    )
    return ranked


def fuzzy_title_score(query: str, title: str) -> float:
    normalized_query = " ".join(re.findall(r"\w+", query.casefold()))
    normalized_title = " ".join(re.findall(r"\w+", str(title).casefold()))
    if not normalized_query or not normalized_title:
        return 0.0

    title_candidates = [normalized_title, *normalized_title.split()]
    return max(
        SequenceMatcher(None, normalized_query, candidate).ratio()
        for candidate in title_candidates
    )


def select_top_movies(n: int, genre: str | None = None):
    movie_features = rank_movies_by_confidence(
        pd.read_csv(settings.movie_features_path)
    )

    if genre is not None:
        movie_features = movie_features[
            movie_features["genres"].str.lower() == genre.lower()
        ]

    return movie_features.sort_values(
        ["_ranking_score", "total_ratings", "title"],
        ascending=[False, False, True],
    ).head(n)


def clamp_model_rating(rating: float) -> float:
    return round(max(0.0, min(5.0, float(rating))), 2)


def primary_genre(genres: str) -> str:
    genre = str(genres).split(",")[0].strip()
    return genre or "Unknown"


def recommendation_reason(genres: str, source: str) -> str:
    genre = primary_genre(genres)
    if source == "fallback_top_diverse":
        return (
            "Cold-start top-rated fallback with genre-diversity re-ranking; "
            f"primary genre: {genre}."
        )
    if source == "fallback_top":
        return f"Cold-start fallback from top-rated catalog; primary genre: {genre}."
    if source == "als_diverse":
        return (
            "ALS collaborative filtering match with genre-diversity re-ranking; "
            f"primary genre: {genre}."
        )

    return (
        "ALS collaborative filtering match from similar-user rating patterns; "
        f"primary genre: {genre}."
    )


def rerank_for_genre_diversity(candidates: list[dict], n: int) -> list[dict]:
    selected = []
    deferred = []
    selected_genres = set()

    for candidate in candidates:
        genre = primary_genre(candidate["genres"]).lower()
        if genre not in selected_genres and len(selected) < n:
            selected.append(candidate)
            selected_genres.add(genre)
        else:
            deferred.append(candidate)

    for candidate in deferred:
        if len(selected) >= n:
            break
        selected.append(candidate)

    return selected[:n]


def build_recommendation(
    movie_id: int,
    title: str,
    genres: str,
    rating: float,
    source: RankingStrategy = "als",
):
    raw_rating = round(float(rating), 2)

    return MovieRecommendation(
        movie_id=int(movie_id),
        title=title,
        genres=genres,
        predicted_rating=clamp_model_rating(raw_rating),
        raw_predicted_rating=raw_rating,
        ranking_strategy=source,
        reason=recommendation_reason(genres, source),
    )


# Главная страница открывает демо-интерфейс для портфолио.
@app.get("/", response_class=RedirectResponse, include_in_schema=False)
def root():
    return RedirectResponse(url="/app")


@app.get("/app", response_class=FileResponse, include_in_schema=False)
def web_app():
    return FileResponse(web_app_path)


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
        "model_card": settings.model_card_path.exists(),
        "movies": settings.movies_path.exists(),
        "ratings": settings.ratings_path.exists(),
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


@app.get("/model/info", response_model=ModelInfoResponse)
def model_info():
    card = load_model_card()
    training = card.get("training", {})
    metrics = card.get("metrics", {})
    ranking_evaluation = card.get("ranking_evaluation", {})
    rating_scale = card.get("rating_scale", {})
    score_policy = card.get("score_policy", {})
    raw_score_policy = score_policy.get("raw_predicted_rating", "unbounded model output")
    predicted_score_policy = score_policy.get("predicted_rating", "user-facing score")
    ranking_k = ranking_evaluation.get("k")
    precision_at_k = (
        metrics.get(f"precision_at_{ranking_k}")
        if ranking_k is not None
        else None
    )
    recall_at_k = (
        metrics.get(f"recall_at_{ranking_k}")
        if ranking_k is not None
        else None
    )

    return {
        "model_name": card.get("model_name", "ALS recommender"),
        "algorithm": card.get("algorithm", "pyspark.ml.recommendation.ALS"),
        "dataset": card.get("dataset", "unknown"),
        "model_exists": settings.model_path.exists(),
        "model_card_available": bool(card),
        "model_size_mb": round(directory_size_bytes(settings.model_path) / 1024 / 1024, 2),
        "rmse": metrics.get("rmse"),
        "ranking_k": ranking_k,
        "precision_at_k": precision_at_k,
        "recall_at_k": recall_at_k,
        "sample_fraction": training.get("sample_fraction"),
        "train_ratings": training.get("train_ratings"),
        "test_ratings": training.get("test_ratings"),
        "rating_scale_min": rating_scale.get("min", 0.0),
        "rating_scale_max": rating_scale.get("max", 5.0),
        "score_policy": (
            f"predicted_rating: {predicted_score_policy}; "
            f"raw_predicted_rating: {raw_score_policy}"
        ),
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
    candidate_count = (
        min(request.n_recommendations * 3, 150)
        if request.diversify
        else request.n_recommendations
    )

    if request.user_id not in users:
        if not request.fallback_to_top:
            raise HTTPException(
                status_code=404,
                detail=f"Пользователь {request.user_id} не найден"
            )

        top = select_top_movies(candidate_count)
        candidates = [
            {
                "movie_id": int(row["movieId"]),
                "title": row["title"],
                "genres": row["genres"],
                "rating": row["avg_rating"],
            }
            for _, row in top.iterrows()
        ]
        source = "fallback_top_diverse" if request.diversify else "fallback_top"
    else:
        spark_session, als_model, movie_catalog, _ = load_recommendation_resources()

        user_df = spark_session.createDataFrame(
            [(request.user_id,)],
            ["userId"]
        )

        recs = als_model.recommendForUserSubset(
            user_df,
            candidate_count,
        )

        rows = recs.collect()
        if not rows:
            raise HTTPException(
                status_code=404,
                detail=f"Для пользователя {request.user_id} нет рекомендаций"
            )

        recs_list = rows[0]["recommendations"]

        movie_lookup = movie_catalog.set_index("movieId")
        candidates = []
        for rec in recs_list:
            movie_id = int(rec["movieId"])
            if movie_id not in movie_lookup.index:
                continue
            movie_info = movie_lookup.loc[movie_id]
            candidates.append({
                "movie_id": movie_id,
                "title": movie_info["title"],
                "genres": movie_info["genres"],
                "rating": rec["rating"],
            })
        source = "als_diverse" if request.diversify else "als"

    if request.diversify:
        candidates = rerank_for_genre_diversity(
            candidates,
            request.n_recommendations,
        )
    else:
        candidates = candidates[:request.n_recommendations]

    result = [
        build_recommendation(
            movie_id=candidate["movie_id"],
            title=candidate["title"],
            genres=candidate["genres"],
            rating=candidate["rating"],
            source=source,
        )
        for candidate in candidates
    ]

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


@app.post("/recommend/feedback", response_model=RecommendationFeedbackResponse)
def recommendation_feedback(feedback: RecommendationFeedbackRequest):
    storage = feedback_store.record(feedback)

    return {
        "status": "accepted",
        "storage": storage,
        "user_id": feedback.user_id,
        "movie_id": feedback.movie_id,
        "feedback": feedback.feedback,
        "ranking_strategy": feedback.ranking_strategy,
    }


@app.delete(
    "/recommend/feedback",
    response_model=RecommendationFeedbackDeleteResponse,
)
def delete_recommendation_feedback(feedback: RecommendationFeedbackDeleteRequest):
    storage = feedback_store.remove(feedback)

    return {
        "status": "removed",
        "storage": storage,
        "user_id": feedback.user_id,
        "movie_id": feedback.movie_id,
        "ranking_strategy": feedback.ranking_strategy,
    }


@app.get(
    "/recommend/feedback/users/{user_id}",
    response_model=list[RecommendationFeedbackStateResponse],
)
def recommendation_feedback_for_user(
    user_id: int,
    source: str = Query(default="web_app", min_length=1, max_length=50),
):
    return feedback_store.load_for_user(user_id, source)


@app.get(
    "/recommend/feedback/summary",
    response_model=RecommendationFeedbackSummaryResponse,
)
def recommendation_feedback_summary():
    return feedback_store.summarize()


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
            "confidence_score": round(float(row["_ranking_score"]), 2),
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


@app.get("/movies/search", response_model=list[MovieDetailResponse])
def search_movies(
    q: str = Query(default="", max_length=100),
    n: int = Query(default=10, ge=1, le=50),
    genre: str | None = Query(default=None, min_length=1),
    min_rating: float | None = Query(default=None, ge=0, le=5),
    sort_by: Literal["rating", "popularity", "title"] = Query(default="rating"),
):
    movie_features = rank_movies_by_confidence(
        pd.read_csv(settings.movie_features_path)
    )

    query = q.strip()
    matches = movie_features
    fuzzy_match = False

    if query:
        exact_matches = matches[
            matches["title"].str.contains(query, case=False, na=False)
        ]
        if exact_matches.empty and len(query) >= 4:
            matches = matches.assign(
                _search_score=matches["title"].map(
                    lambda title: fuzzy_title_score(query, title)
                )
            )
            matches = matches[matches["_search_score"] >= 0.78]
            fuzzy_match = True
        else:
            matches = exact_matches

    if genre is not None:
        matches = matches[matches["genres"].str.lower() == genre.lower()]

    if min_rating is not None:
        matches = matches[matches["avg_rating"] >= min_rating]

    sort_options = {
        "rating": (
            ["_ranking_score", "total_ratings", "title"],
            [False, False, True],
        ),
        "popularity": (
            ["total_ratings", "avg_rating", "title"],
            [False, False, True],
        ),
        "title": (["title"], [True]),
    }
    sort_columns, sort_order = sort_options[sort_by]
    if fuzzy_match:
        sort_columns = ["_search_score", *sort_columns]
        sort_order = [False, *sort_order]
    matches = matches.sort_values(sort_columns, ascending=sort_order).head(n)

    return [
        {
            "movie_id": int(row["movieId"]),
            "title": row["title"],
            "genres": row["genres"],
            "avg_rating": round(float(row["avg_rating"]), 2),
            "total_ratings": int(row["total_ratings"]),
            "confidence_score": round(float(row["_ranking_score"]), 2),
        }
        for _, row in matches.iterrows()
    ]


@app.get("/movies/discover", response_model=MovieDetailResponse)
def discover_movie(
    genre: str | None = Query(default=None, min_length=1),
    min_rating: float = Query(default=4.0, ge=0, le=5),
    min_ratings: int = Query(default=100, ge=1, le=1_000_000),
    seed: int | None = Query(default=None, ge=0, le=2_147_483_647),
):
    movie_features = rank_movies_by_confidence(
        pd.read_csv(settings.movie_features_path)
    )
    candidates = movie_features[
        (movie_features["avg_rating"] >= min_rating)
        & (movie_features["total_ratings"] >= min_ratings)
    ]

    if genre is not None:
        candidates = candidates[
            candidates["genres"].str.lower() == genre.lower()
        ]

    if candidates.empty:
        raise HTTPException(
            status_code=404,
            detail="Под текущие фильтры не найдено фильмов",
        )

    row = candidates.sample(n=1, random_state=seed).iloc[0]
    return {
        "movie_id": int(row["movieId"]),
        "title": row["title"],
        "genres": row["genres"],
        "avg_rating": round(float(row["avg_rating"]), 2),
        "total_ratings": int(row["total_ratings"]),
        "confidence_score": round(float(row["_ranking_score"]), 2),
    }


@app.get("/movies/{movie_id}", response_model=MovieDetailResponse)
def movie_detail(movie_id: int):
    movie_features = rank_movies_by_confidence(
        pd.read_csv(settings.movie_features_path)
    )

    target = movie_features[movie_features["movieId"] == movie_id]
    if target.empty:
        raise HTTPException(status_code=404, detail=f"Фильм {movie_id} не найден")

    row = target.iloc[0]
    return {
        "movie_id": int(row["movieId"]),
        "title": row["title"],
        "genres": row["genres"],
        "avg_rating": round(float(row["avg_rating"]), 2),
        "total_ratings": int(row["total_ratings"]),
        "confidence_score": round(float(row["_ranking_score"]), 2),
    }


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


@app.get("/users/{user_id}/history", response_model=list[UserRatingHistoryResponse])
def user_rating_history(user_id: int, n: int = Query(default=10, ge=1, le=50)):
    ratings = pd.read_csv(settings.ratings_path)

    user_ratings = ratings[ratings["userId"] == user_id]
    if user_ratings.empty:
        raise HTTPException(
            status_code=404,
            detail=f"У пользователя {user_id} нет истории оценок",
        )

    user_ratings = user_ratings.sort_values("timestamp", ascending=False).head(n)
    movie_catalog = load_movie_catalog()
    history = user_ratings.merge(movie_catalog, on="movieId", how="left")

    return [
        {
            "movie_id": int(row["movieId"]),
            "title": row["title"] if pd.notna(row["title"]) else f"Movie {int(row['movieId'])}",
            "genres": row["genres"] if pd.notna(row["genres"]) else "Unknown",
            "rating": round(float(row["rating"]), 2),
            "timestamp": int(row["timestamp"]),
        }
        for _, row in history.iterrows()
    ]


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
