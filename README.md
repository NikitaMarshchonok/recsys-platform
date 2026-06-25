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

Create local environment settings:

```bash
cp .env.example .env
```

Generate sample data:

```bash
python scripts/generate_data.py
```

Run feature engineering:

```bash
export JAVA_HOME=/opt/homebrew/opt/openjdk@17  # macOS/Homebrew example
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
| GET | `/ready` | Readiness check for model and data artifacts |
| POST | `/recommend` | Personalized movie recommendations |
| GET | `/similar_movies/{movie_id}` | Similar movies by genre |
| GET | `/stats` | Request statistics from PostgreSQL |

Example recommendation request:

```bash
curl -X POST "http://localhost:8001/recommend" \
  -H "Content-Type: application/json" \
  -d '{"user_id": 1, "n_recommendations": 5}'
```

Readiness check:

```bash
curl "http://localhost:8001/ready"
```

## Runtime Configuration

The API can be configured through environment variables:

| Variable | Default |
| --- | --- |
| `RECSYS_MODEL_PATH` | `models/als_model` |
| `RECSYS_MOVIES_PATH` | `data/raw/movies.csv` |
| `RECSYS_USERS_PATH` | `data/raw/users.csv` |
| `RECSYS_MOVIE_FEATURES_PATH` | `data/processed/movie_features.csv` |
| `RECSYS_DB_HOST` | `localhost` |
| `RECSYS_DB_NAME` | `recsys` |
| `RECSYS_DB_USER` | `recsys` |
| `RECSYS_DB_PASSWORD` | `recsys` |
| `RECSYS_DB_PORT` | `5434` |

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
- `/health` checks that the API process is alive; `/ready` checks that model and data artifacts exist.
- Every API response includes `X-Request-ID` and `X-Response-Time-ms` headers for tracing.
- PostgreSQL logging is optional: the API still works if the database is not running.
- Fast API tests mock the ML resources, so they run quickly without starting Spark.
- The full ML workflow is covered by the data generation, feature engineering, and training scripts.

## Current Results

- Synthetic dataset: 500 users, 200 movies, about 18k ratings after duplicate removal
- Latest local ALS RMSE: 1.4285
- ALS model saved to `models/als_model`
- API tests: 6 passing tests for health, readiness, tracing, valid recommendations, invalid users, and request validation

## Author

Nikita Marshchonok

Telegram: `@nikitamarshchonok`
