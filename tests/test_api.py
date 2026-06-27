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

    assert example == {"user_id": 1, "n_recommendations": 5}


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
