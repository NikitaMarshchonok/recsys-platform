from typing import Literal

from pydantic import BaseModel, Field


RankingStrategy = Literal[
    "als",
    "als_diverse",
    "fallback_top",
    "fallback_top_diverse",
]
FeedbackRankingStrategy = Literal[
    "als",
    "als_diverse",
    "fallback_top",
    "fallback_top_diverse",
    "unknown",
]


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
    diversify: bool = Field(
        default=True,
        description="Переранжировать кандидатов для разнообразия основных жанров",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "user_id": 1,
                    "n_recommendations": 5,
                    "fallback_to_top": False,
                    "diversify": True,
                }
            ]
        }
    }


class RecommendationFeedbackRequest(BaseModel):
    user_id: int = Field(ge=1)
    movie_id: int = Field(ge=1)
    feedback: Literal["like", "dislike"]
    source: str = Field(default="web_app", min_length=1, max_length=50)
    ranking_strategy: FeedbackRankingStrategy = "unknown"


class RecommendationFeedbackDeleteRequest(BaseModel):
    user_id: int = Field(ge=1)
    movie_id: int = Field(ge=1)
    source: str = Field(default="web_app", min_length=1, max_length=50)
    ranking_strategy: FeedbackRankingStrategy = "unknown"


class MovieRecommendation(BaseModel):
    movie_id: int
    title: str
    genres: str
    predicted_rating: float
    raw_predicted_rating: float
    ranking_strategy: RankingStrategy
    reason: str


class RecommendationFeedbackResponse(BaseModel):
    status: str
    storage: str
    user_id: int
    movie_id: int
    feedback: str
    ranking_strategy: FeedbackRankingStrategy


class RecommendationFeedbackDeleteResponse(BaseModel):
    status: str
    storage: str
    user_id: int
    movie_id: int
    ranking_strategy: FeedbackRankingStrategy


class RecommendationFeedbackStateResponse(BaseModel):
    movie_id: int
    feedback: Literal["like", "dislike"]
    ranking_strategy: FeedbackRankingStrategy


class RecommendationStrategyFeedbackSummary(BaseModel):
    ranking_strategy: FeedbackRankingStrategy
    total_feedback: int
    likes: int
    dislikes: int
    like_rate: float


class RecommendationFeedbackSummaryResponse(BaseModel):
    total_feedback: int
    likes: int
    dislikes: int
    like_rate: float
    storage: Literal["postgres", "unavailable"]
    strategies: list[RecommendationStrategyFeedbackSummary]


class SimilarMovieResponse(BaseModel):
    movie_id: int
    title: str
    genres: str
    avg_rating: float
    similarity_score: float = Field(ge=0, le=1)
    similarity_method: Literal["als_cosine", "genre_overlap"]


class TopMovieResponse(BaseModel):
    movie_id: int
    title: str
    genres: str
    avg_rating: float
    total_ratings: int
    confidence_score: float


class MovieDetailResponse(BaseModel):
    movie_id: int
    title: str
    genres: str
    avg_rating: float
    total_ratings: int
    confidence_score: float


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


class UserRatingHistoryResponse(BaseModel):
    movie_id: int
    title: str
    genres: str
    rating: float
    timestamp: int


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


class ModelInfoResponse(BaseModel):
    model_name: str
    algorithm: str
    dataset: str
    model_exists: bool
    model_card_available: bool
    model_size_mb: float
    rmse: float | None
    ranking_k: int | None
    precision_at_k: float | None
    recall_at_k: float | None
    sample_fraction: float | None
    train_ratings: int | None
    test_ratings: int | None
    rating_scale_min: float
    rating_scale_max: float
    score_policy: str


class CatalogSummaryResponse(BaseModel):
    total_movies: int
    total_genres: int
    total_ratings: int
    avg_rating: float
    min_avg_rating: float
    max_avg_rating: float
    most_rated_movie_id: int | None
    most_rated_title: str | None
