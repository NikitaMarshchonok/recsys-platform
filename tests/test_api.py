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


def test_request_tracing_headers():
    response = client.get("/health", headers={"X-Request-ID": "test-request"})
    assert response.headers["X-Request-ID"] == "test-request"
    assert float(response.headers["X-Response-Time-ms"]) >= 0


def test_readiness():
    response = client.get("/ready")
    assert response.status_code == 200
    assert response.json()["status"] == "ready"


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
