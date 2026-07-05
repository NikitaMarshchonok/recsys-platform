import pytest
import pandas as pd
from fastapi.testclient import TestClient
import src.api.main as api_module
from src.api.main import app

# TestClient — это виртуальный браузер для тестов
# он отправляет запросы к API без реального сервера
client = TestClient(app)


class FakeSpark:
    def createDataFrame(self, data, columns):
        return {"data": data, "columns": columns}


class FakeRecommendations:
    def collect(self):
        return [{
            "recommendations": [
                {"movieId": 1, "rating": 4.8},
                {"movieId": 2, "rating": 4.5},
                {"movieId": 3, "rating": 4.2},
                {"movieId": 4, "rating": 4.0},
                {"movieId": 5, "rating": 3.9},
            ]
        }]


class FakeModel:
    def recommendForUserSubset(self, user_df, n_recommendations):
        return FakeRecommendations()


class FakeCursor:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def execute(self, query):
        self.query = query

    def fetchone(self):
        return (12, 42.4, 10.1, 88.9)


class FakeConnection:
    def cursor(self):
        return FakeCursor()

    def close(self):
        pass


@pytest.fixture(autouse=True)
def mock_api_resources(monkeypatch):
    movies = pd.DataFrame({
        "movieId": [1, 2, 3, 4, 5],
        "title": ["Movie 1", "Movie 2", "Movie 3", "Movie 4", "Movie 5"],
        "genres": ["Drama", "Comedy", "Action", "Sci-Fi", "Thriller"],
    })

    monkeypatch.setattr(api_module, "init_db", lambda: None)
    monkeypatch.setattr(
        api_module,
        "load_recommendation_resources",
        lambda: (FakeSpark(), FakeModel(), movies, {1}),
    )
    monkeypatch.setattr(api_module, "load_valid_user_ids", lambda: {1})


def test_stats_response(monkeypatch):
    monkeypatch.setattr(api_module, "get_db_connection", lambda: FakeConnection())

    response = client.get("/stats")

    assert response.status_code == 200
    assert response.json() == {
        "total_requests": 12,
        "avg_response_ms": 42.4,
        "min_response_ms": 10.1,
        "max_response_ms": 88.9,
    }


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_root_response():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "message": "RecSys API работает"}


def test_web_app_response():
    response = client.get("/app")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "RecSys Demo" in response.text
    assert "Recommendations" in response.text
    assert "Movie Detail" in response.text
    assert "User Context" in response.text
    assert "Similar Movies" in response.text
    assert "data-inspect-movie-id" in response.text
    assert "Horizon, River, Storm" in response.text
    assert "Quick searches" in response.text
    assert "data-search-query" in response.text
    assert 'aria-pressed="true">Top rated' in response.text
    assert "reset-search-button" in response.text
    assert "Reset filters" in response.text
    assert "from user history" in response.text


def test_version():
    response = client.get("/version")

    assert response.status_code == 200
    assert response.json() == {"service": "RecSys API", "version": "1.0.0"}


def test_request_tracing_headers():
    response = client.get("/health", headers={"X-Request-ID": "test-request"})
    assert response.headers["X-Request-ID"] == "test-request"
    assert float(response.headers["X-Response-Time-ms"]) >= 0


def test_readiness():
    response = client.get("/ready")
    assert response.status_code == 200
    assert response.json()["status"] == "ready"


def test_metrics(monkeypatch):
    monkeypatch.setattr(api_module, "spark", object())
    monkeypatch.setattr(api_module, "model", object())
    monkeypatch.setattr(api_module, "movies", pd.DataFrame({"movieId": [1, 2, 3]}))
    monkeypatch.setattr(api_module, "valid_user_ids", {1, 2})

    response = client.get("/metrics")

    assert response.status_code == 200
    assert response.json() == {
        "spark_loaded": True,
        "model_loaded": True,
        "movie_catalog_loaded": True,
        "user_catalog_loaded": True,
        "cached_movies": 3,
        "cached_users": 2,
    }


def test_catalog_summary(monkeypatch):
    movie_features = pd.DataFrame({
        "movieId": [1, 2, 3],
        "title": ["Movie 1", "Movie 2", "Movie 3"],
        "genres": ["Drama", "Action", "Drama"],
        "avg_rating": [4.5, 4.8, 3.9],
        "total_ratings": [100, 80, 120],
    })

    monkeypatch.setattr(api_module.pd, "read_csv", lambda path: movie_features)

    response = client.get("/catalog/summary")

    assert response.status_code == 200
    assert response.json() == {
        "total_movies": 3,
        "total_genres": 2,
        "total_ratings": 300,
        "avg_rating": 4.4,
        "min_avg_rating": 3.9,
        "max_avg_rating": 4.8,
        "most_rated_movie_id": 3,
        "most_rated_title": "Movie 3",
    }


def test_catalog_summary_empty_catalog(monkeypatch):
    movie_features = pd.DataFrame({
        "movieId": [],
        "title": [],
        "genres": [],
        "avg_rating": [],
        "total_ratings": [],
    })

    monkeypatch.setattr(api_module.pd, "read_csv", lambda path: movie_features)

    response = client.get("/catalog/summary")

    assert response.status_code == 200
    assert response.json() == {
        "total_movies": 0,
        "total_genres": 0,
        "total_ratings": 0,
        "avg_rating": 0.0,
        "min_avg_rating": 0.0,
        "max_avg_rating": 0.0,
        "most_rated_movie_id": None,
        "most_rated_title": None,
    }


def test_recommend_valid_user():
    response = client.post(
        "/recommend",
        json={"user_id": 1, "n_recommendations": 5}
    )
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) == 5
    assert "movie_id" in data[0]
    assert "title" in data[0]
    assert "predicted_rating" in data[0]


def test_recommend_openapi_example_uses_valid_user():
    schema = app.openapi()
    example = schema["components"]["schemas"]["RecommendRequest"]["examples"][0]

    assert example == {
        "user_id": 1,
        "n_recommendations": 5,
        "fallback_to_top": False,
    }


def test_recommend_unknown_user_can_fallback_to_top(monkeypatch):
    movie_features = pd.DataFrame({
        "movieId": [1, 2, 3],
        "title": ["Movie 1", "Movie 2", "Movie 3"],
        "genres": ["Drama", "Action", "Comedy"],
        "avg_rating": [4.5, 4.8, 4.8],
        "total_ratings": [100, 80, 120],
    })

    monkeypatch.setattr(api_module.pd, "read_csv", lambda path: movie_features)

    response = client.post(
        "/recommend",
        json={"user_id": 9999, "n_recommendations": 2, "fallback_to_top": True},
    )

    assert response.status_code == 200
    assert response.json() == [
        {"movie_id": 3, "title": "Movie 3", "genres": "Comedy", "predicted_rating": 4.8},
        {"movie_id": 2, "title": "Movie 2", "genres": "Action", "predicted_rating": 4.8},
    ]


def test_similar_movies_response(monkeypatch):
    movie_features = pd.DataFrame({
        "movieId": [1, 2, 3],
        "title": ["Movie 1", "Movie 2", "Movie 3"],
        "genres": ["Drama", "Drama", "Comedy"],
        "avg_rating": [4.2, 4.8, 3.9],
    })

    monkeypatch.setattr(api_module.pd, "read_csv", lambda path: movie_features)

    response = client.get("/similar_movies/1?n=2")

    assert response.status_code == 200
    assert response.json() == [
        {"movie_id": 2, "title": "Movie 2", "genres": "Drama", "avg_rating": 4.8},
        {"movie_id": 1, "title": "Movie 1", "genres": "Drama", "avg_rating": 4.2},
    ]


def test_similar_movies_openapi_response_schema():
    schema = app.openapi()

    assert schema["paths"]["/similar_movies/{movie_id}"]["get"]["responses"]["200"]["content"]["application/json"]["schema"] == {
        "items": {"$ref": "#/components/schemas/SimilarMovieResponse"},
        "type": "array",
        "title": "Response Similar Movies Similar Movies  Movie Id  Get",
    }


def test_top_movies_response(monkeypatch):
    movie_features = pd.DataFrame({
        "movieId": [1, 2, 3],
        "title": ["Movie 1", "Movie 2", "Movie 3"],
        "genres": ["Drama", "Action", "Comedy"],
        "avg_rating": [4.5, 4.8, 4.8],
        "total_ratings": [100, 80, 120],
    })

    monkeypatch.setattr(api_module.pd, "read_csv", lambda path: movie_features)

    response = client.get("/movies/top?n=2")

    assert response.status_code == 200
    assert response.json() == [
        {"movie_id": 3, "title": "Movie 3", "genres": "Comedy", "avg_rating": 4.8, "total_ratings": 120},
        {"movie_id": 2, "title": "Movie 2", "genres": "Action", "avg_rating": 4.8, "total_ratings": 80},
    ]


def test_top_movies_filters_by_genre(monkeypatch):
    movie_features = pd.DataFrame({
        "movieId": [1, 2, 3],
        "title": ["Movie 1", "Movie 2", "Movie 3"],
        "genres": ["Drama", "Action", "Drama"],
        "avg_rating": [4.5, 4.9, 4.8],
        "total_ratings": [100, 80, 120],
    })

    monkeypatch.setattr(api_module.pd, "read_csv", lambda path: movie_features)

    response = client.get("/movies/top?n=2&genre=drama")

    assert response.status_code == 200
    assert response.json() == [
        {"movie_id": 3, "title": "Movie 3", "genres": "Drama", "avg_rating": 4.8, "total_ratings": 120},
        {"movie_id": 1, "title": "Movie 1", "genres": "Drama", "avg_rating": 4.5, "total_ratings": 100},
    ]


def test_top_movies_openapi_response_schema():
    schema = app.openapi()

    assert schema["paths"]["/movies/top"]["get"]["responses"]["200"]["content"]["application/json"]["schema"] == {
        "items": {"$ref": "#/components/schemas/TopMovieResponse"},
        "type": "array",
        "title": "Response Top Movies Movies Top Get",
    }


def test_movie_genres_response(monkeypatch):
    movie_features = pd.DataFrame({
        "movieId": [1, 2, 3, 4],
        "title": ["Movie 1", "Movie 2", "Movie 3", "Movie 4"],
        "genres": ["Drama", "Action", "Drama", "Comedy"],
        "avg_rating": [4.0, 4.8, 5.0, 3.5],
        "total_ratings": [100, 80, 120, 30],
    })

    monkeypatch.setattr(api_module.pd, "read_csv", lambda path: movie_features)

    response = client.get("/movies/genres")

    assert response.status_code == 200
    assert response.json() == [
        {"genre": "Drama", "movie_count": 2, "avg_rating": 4.5},
        {"genre": "Action", "movie_count": 1, "avg_rating": 4.8},
        {"genre": "Comedy", "movie_count": 1, "avg_rating": 3.5},
    ]


def test_movie_detail_response(monkeypatch):
    movie_features = pd.DataFrame({
        "movieId": [1, 2],
        "title": ["Movie 1", "Movie 2"],
        "genres": ["Drama", "Action"],
        "avg_rating": [4.234, 4.8],
        "total_ratings": [100, 80],
    })

    monkeypatch.setattr(api_module.pd, "read_csv", lambda path: movie_features)

    response = client.get("/movies/1")

    assert response.status_code == 200
    assert response.json() == {
        "movie_id": 1,
        "title": "Movie 1",
        "genres": "Drama",
        "avg_rating": 4.23,
        "total_ratings": 100,
    }


def test_movie_detail_not_found(monkeypatch):
    movie_features = pd.DataFrame({
        "movieId": [1],
        "title": ["Movie 1"],
        "genres": ["Drama"],
        "avg_rating": [4.2],
        "total_ratings": [100],
    })

    monkeypatch.setattr(api_module.pd, "read_csv", lambda path: movie_features)

    response = client.get("/movies/999")

    assert response.status_code == 404
    assert response.json() == {"detail": "Фильм 999 не найден"}


def test_movie_search_response(monkeypatch):
    movie_features = pd.DataFrame({
        "movieId": [1, 2, 3],
        "title": ["Star Drama", "Star Action", "Other Movie"],
        "genres": ["Drama", "Action", "Drama"],
        "avg_rating": [4.4, 4.8, 5.0],
        "total_ratings": [100, 80, 120],
    })

    monkeypatch.setattr(api_module.pd, "read_csv", lambda path: movie_features)

    response = client.get("/movies/search?q=star&n=2")

    assert response.status_code == 200
    assert response.json() == [
        {
            "movie_id": 2,
            "title": "Star Action",
            "genres": "Action",
            "avg_rating": 4.8,
            "total_ratings": 80,
        },
        {
            "movie_id": 1,
            "title": "Star Drama",
            "genres": "Drama",
            "avg_rating": 4.4,
            "total_ratings": 100,
        },
    ]


def test_movie_search_allows_empty_query(monkeypatch):
    movie_features = pd.DataFrame({
        "movieId": [1, 2, 3],
        "title": ["Silent Horizon", "Neon Journey", "Broken River"],
        "genres": ["Drama", "Action", "Drama"],
        "avg_rating": [4.4, 4.9, 4.8],
        "total_ratings": [100, 80, 120],
    })

    monkeypatch.setattr(api_module.pd, "read_csv", lambda path: movie_features)

    response = client.get("/movies/search?n=2&min_rating=4.5")

    assert response.status_code == 200
    assert response.json() == [
        {
            "movie_id": 2,
            "title": "Neon Journey",
            "genres": "Action",
            "avg_rating": 4.9,
            "total_ratings": 80,
        },
        {
            "movie_id": 3,
            "title": "Broken River",
            "genres": "Drama",
            "avg_rating": 4.8,
            "total_ratings": 120,
        },
    ]


def test_movie_search_filters_by_genre_and_rating(monkeypatch):
    movie_features = pd.DataFrame({
        "movieId": [1, 2, 3],
        "title": ["Movie Alpha", "Movie Beta", "Movie Gamma"],
        "genres": ["Drama", "Action", "Drama"],
        "avg_rating": [4.4, 4.9, 4.8],
        "total_ratings": [100, 80, 120],
    })

    monkeypatch.setattr(api_module.pd, "read_csv", lambda path: movie_features)

    response = client.get("/movies/search?q=movie&genre=drama&min_rating=4.5")

    assert response.status_code == 200
    assert response.json() == [
        {
            "movie_id": 3,
            "title": "Movie Gamma",
            "genres": "Drama",
            "avg_rating": 4.8,
            "total_ratings": 120,
        },
    ]


def test_user_profile_response(monkeypatch):
    user_features = pd.DataFrame({
        "userId": [1, 2],
        "total_rated": [42, 12],
        "avg_rating": [3.756, 4.1],
        "rating_stddev": [0.812, 1.2],
        "min_rating": [1.0, 2.0],
        "max_rating": [5.0, 5.0],
    })

    monkeypatch.setattr(api_module.pd, "read_csv", lambda path: user_features)

    response = client.get("/users/1/profile")

    assert response.status_code == 200
    assert response.json() == {
        "user_id": 1,
        "total_rated": 42,
        "avg_rating": 3.76,
        "rating_stddev": 0.81,
        "min_rating": 1.0,
        "max_rating": 5.0,
    }


def test_user_profile_not_found(monkeypatch):
    user_features = pd.DataFrame({
        "userId": [1],
        "total_rated": [42],
        "avg_rating": [3.75],
        "rating_stddev": [0.8],
        "min_rating": [1.0],
        "max_rating": [5.0],
    })

    monkeypatch.setattr(api_module.pd, "read_csv", lambda path: user_features)

    response = client.get("/users/999/profile")

    assert response.status_code == 404


def test_user_rating_history_response(monkeypatch):
    ratings = pd.DataFrame({
        "userId": [1, 2, 1],
        "movieId": [1, 3, 2],
        "rating": [4.25, 3.0, 5.0],
        "timestamp": [100, 200, 300],
    })
    movies = pd.DataFrame({
        "movieId": [1, 2, 3],
        "title": ["Movie 1", "Movie 2", "Movie 3"],
        "genres": ["Drama", "Action", "Comedy"],
    })

    def fake_read_csv(path):
        if path == api_module.settings.ratings_path:
            return ratings
        if path == api_module.settings.movies_path:
            return movies
        raise AssertionError(f"Unexpected path: {path}")

    monkeypatch.setattr(api_module, "movies", None)
    monkeypatch.setattr(api_module.pd, "read_csv", fake_read_csv)

    response = client.get("/users/1/history?n=2")

    assert response.status_code == 200
    assert response.json() == [
        {
            "movie_id": 2,
            "title": "Movie 2",
            "genres": "Action",
            "rating": 5.0,
            "timestamp": 300,
        },
        {
            "movie_id": 1,
            "title": "Movie 1",
            "genres": "Drama",
            "rating": 4.25,
            "timestamp": 100,
        },
    ]


def test_user_rating_history_not_found(monkeypatch):
    ratings = pd.DataFrame({
        "userId": [1],
        "movieId": [1],
        "rating": [4.25],
        "timestamp": [100],
    })

    monkeypatch.setattr(api_module.pd, "read_csv", lambda path: ratings)

    response = client.get("/users/999/history")

    assert response.status_code == 404
    assert response.json() == {"detail": "У пользователя 999 нет истории оценок"}


def test_openapi_has_root_and_health_response_schemas():
    schema = app.openapi()

    assert schema["paths"]["/"]["get"]["responses"]["200"]["content"]["application/json"]["schema"] == {
        "$ref": "#/components/schemas/RootResponse"
    }
    assert schema["paths"]["/health"]["get"]["responses"]["200"]["content"]["application/json"]["schema"] == {
        "$ref": "#/components/schemas/HealthResponse"
    }


def test_recommend_invalid_user():
    response = client.post(
        "/recommend",
        json={"user_id": 9999, "n_recommendations": 5}
    )
    assert response.status_code == 404


def test_recommend_invalid_limit():
    response = client.post(
        "/recommend",
        json={"user_id": 1, "n_recommendations": 0}
    )
    assert response.status_code == 422
