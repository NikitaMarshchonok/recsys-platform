# RecSys Platform

Educational MLOps project for a movie recommendation system.

The project demonstrates a full recommendation workflow: data generation,
Spark feature engineering, ALS model training, experiment tracking, API
serving, request logging, orchestration, and automated tests.

## Architecture

```text
Raw Data
  -> Spark Feature Engineering
  -> ALS Model Training
  -> FastAPI Recommendation Service
  -> PostgreSQL Request Logging

MLflow tracks training parameters and metrics.
Airflow can orchestrate the feature and training pipeline.
```

## Stack

- Apache Spark / PySpark - feature engineering and ALS model training
- FastAPI - REST API for recommendations
- MLflow - experiment tracking
- PostgreSQL - request logging and API statistics
- Airflow - scheduled training pipeline
- Docker Compose - local infrastructure
- pytest - automated API tests

## Project Structure

```text
recsys-platform/
├── data/
│   ├── raw/          # generated movie, user, and rating data
│   └── processed/    # feature engineering outputs
├── spark/jobs/       # Spark feature engineering pipeline
├── src/
│   ├── api/          # FastAPI service
│   └── training/     # ALS training script
├── dags/             # Airflow DAG
├── models/           # saved ALS model
├── tests/            # API tests
└── docker-compose.yml
```

## Quick Start

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Generate sample data:

```bash
python scripts/generate_data.py
```

Run feature engineering:

```bash
python spark/jobs/feature_engineering.py
```

Train the ALS model:

```bash
MLFLOW_TRACKING_URI=file:./mlruns python src/training/train.py
```

Start the API:

```bash
uvicorn src.api.main:app --port 8001
```

Run tests:

```bash
python -m pytest
```

## API Endpoints

| Method | Endpoint | Description |
| --- | --- | --- |
| GET | `/` | Root status message |
| GET | `/health` | Lightweight health check |
| POST | `/recommend` | Personalized movie recommendations |
| GET | `/similar_movies/{movie_id}` | Similar movies by genre |
| GET | `/stats` | Request statistics from PostgreSQL |

Example recommendation request:

```bash
curl -X POST "http://localhost:8001/recommend" \
  -H "Content-Type: application/json" \
  -d '{"user_id": 1, "n_recommendations": 5}'
```

## Docker Infrastructure

Start PostgreSQL and Airflow services:

```bash
docker compose up -d
```

Airflow UI is available at:

```text
http://localhost:8083
```

Default credentials:

```text
username: admin
password: admin
```

## Notes

- The API loads Spark and the ALS model lazily when `/recommend` is called.
- PostgreSQL logging is optional: the API still works if the database is not running.
- Fast API tests mock the ML resources, so they run quickly without starting Spark.
- The full ML workflow is covered by the data generation, feature engineering, and training scripts.

## Current Results

- Synthetic dataset: 500 users, 200 movies, about 18k ratings after duplicate removal
- ALS model saved to `models/als_model`
- API tests: 3 passing tests for health, valid recommendations, and invalid users

## Author

Nikita Marshchonok

Telegram: `@nikitamarshchonok`
