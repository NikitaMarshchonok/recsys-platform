import pytest
from fastapi.testclient import TestClient
from src.api.main import app

# TestClient — это виртуальный браузер для тестов
# он отправляет запросы к API без реального сервера
client = TestClient(app)


def test_health():
    # Отправляем GET запрос на /health
    response = client.get("/health")
    
    # Проверяем что статус 200 (всё хорошо)
    assert response.status_code == 200
    
    # Проверяем что в ответе есть поле status
    assert response.json()["status"] == "healthy"


    def test_recommend_valid_user():
    response = client.post(
        "/recommend",
        json={"user_id": 1, "n_recommendations": 5}
    )
    assert response.status_code == 200
    
    data = response.json()
    # проверяем что вернули список
    assert isinstance(data, list)
    # проверяем что вернули 5 фильмов
    assert len(data) == 5
    # проверяем что у каждого фильма есть нужные поля
    assert "movie_id" in data[0]
    assert "title" in data[0]
    assert "predicted_rating" in data[0]


def test_recommend_invalid_user():
    response = client.post(
        "/recommend",
        json={"user_id": 9999, "n_recommendations": 5}
    )
    # несуществующий пользователь должен вернуть 404
    assert response.status_code == 404