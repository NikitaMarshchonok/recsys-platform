PYTHON ?= python
JAVA_HOME ?= /opt/homebrew/opt/openjdk@17
MLFLOW_TRACKING_URI ?= file:./mlruns
API_PORT ?= 8001
MOVIELENS_DIR ?= data/external/ml-latest-small
MOVIELENS_MOVIES_FILE ?=
MOVIELENS_RATINGS_FILE ?=
RAW_DATA_DIR ?= data/raw

.PHONY: help install lint test coverage generate-data import-movielens features train evaluate-model pipeline run-api docker-up docker-down

help:
	@echo "Available commands:"
	@echo "  make install        Install Python dependencies"
	@echo "  make lint           Run Python lint checks"
	@echo "  make test           Run fast API tests"
	@echo "  make coverage       Run tests with branch coverage"
	@echo "  make generate-data  Generate synthetic ratings data"
	@echo "  make import-movielens Import MovieLens CSV files into data/raw"
	@echo "  make features       Run Spark feature engineering"
	@echo "  make train          Train the ALS model"
	@echo "  make evaluate-model Evaluate the saved ALS model"
	@echo "  make pipeline       Run data generation, features, and training"
	@echo "  make run-api        Start the FastAPI service"
	@echo "  make docker-up      Start Docker Compose services"
	@echo "  make docker-down    Stop Docker Compose services"

install:
	$(PYTHON) -m pip install -r requirements.txt

lint:
	$(PYTHON) -m ruff check .

test:
	$(PYTHON) -m pytest tests/ -v -p no:cacheprovider

coverage:
	$(PYTHON) -m pytest tests/ -v -p no:cacheprovider --cov=src --cov-report=term-missing

generate-data:
	$(PYTHON) scripts/generate_data.py

import-movielens:
	$(PYTHON) scripts/import_movielens.py --source-dir $(MOVIELENS_DIR) $(if $(MOVIELENS_MOVIES_FILE),--movies-file $(MOVIELENS_MOVIES_FILE),) $(if $(MOVIELENS_RATINGS_FILE),--ratings-file $(MOVIELENS_RATINGS_FILE),) --output-dir $(RAW_DATA_DIR)

features:
	JAVA_HOME=$(JAVA_HOME) $(PYTHON) spark/jobs/feature_engineering.py

train:
	JAVA_HOME=$(JAVA_HOME) MLFLOW_TRACKING_URI=$(MLFLOW_TRACKING_URI) $(PYTHON) -m src.training.train

evaluate-model:
	JAVA_HOME=$(JAVA_HOME) $(PYTHON) -m scripts.evaluate_model

pipeline: generate-data features train

run-api:
	JAVA_HOME=$(JAVA_HOME) uvicorn src.api.main:app --host 0.0.0.0 --port $(API_PORT)

docker-up:
	docker compose up -d

docker-down:
	docker compose down
