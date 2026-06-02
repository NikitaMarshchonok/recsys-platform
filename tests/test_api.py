import pytest
from fastapi.testclient import TestClient
from src.api.main import app

# TestClient — это виртуальный браузер для тестов
# он отправляет запросы к API без реального сервера
client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


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