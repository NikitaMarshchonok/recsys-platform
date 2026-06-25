PYTHON ?= python
JAVA_HOME ?= /opt/homebrew/opt/openjdk@17
MLFLOW_TRACKING_URI ?= file:./mlruns
API_PORT ?= 8001

.PHONY: install test generate-data features train pipeline run-api docker-up docker-down

install:
	$(PYTHON) -m pip install -r requirements.txt

test:
	$(PYTHON) -m pytest tests/ -v -p no:cacheprovider

generate-data:
	$(PYTHON) scripts/generate_data.py

features:
	JAVA_HOME=$(JAVA_HOME) $(PYTHON) spark/jobs/feature_engineering.py

train:
	JAVA_HOME=$(JAVA_HOME) MLFLOW_TRACKING_URI=$(MLFLOW_TRACKING_URI) $(PYTHON) src/training/train.py

pipeline: generate-data features train

run-api:
	JAVA_HOME=$(JAVA_HOME) uvicorn src.api.main:app --host 0.0.0.0 --port $(API_PORT)

docker-up:
	docker compose up -d

docker-down:
	docker compose down
