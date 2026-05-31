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