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
- GitHub Actions - fast API tests and ML pipeline smoke checks

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

Or import a real MovieLens catalog after placing `movies.csv` and `ratings.csv`
under `data/external/ml-latest-small/`:

```bash
make import-movielens
```

For MovieLens exports that use `movie.csv` and `rating.csv`, pass explicit files:

```bash
make import-movielens \
  MOVIELENS_MOVIES_FILE=data/raw/movie.csv \
  MOVIELENS_RATINGS_FILE=data/raw/rating.csv
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

Training writes the Spark model and a compact `models/als_model/model_card.json`
with RMSE, data split sizes, hyperparameters, and score policy.

Start the API:

```bash
uvicorn src.api.main:app --port 8001
```

Run tests:

```bash
python -m pytest
```

Common commands are also available through `make`:

```bash
make test
make import-movielens
make pipeline
make run-api
```

CI runs fast API tests separately from the heavier ML pipeline smoke check.

## Demo API Flow

Start the API:

```bash
make run-api
```

Open Swagger UI:

```text
http://127.0.0.1:8001/docs
```

Check that the API process is alive:

```text
GET /health
```

Check that model and data artifacts are available:

```text
GET /ready
```

Open the demo web UI:

```text
http://localhost:8001/app
```

Request personalized recommendations:

```json
{
  "user_id": 1,
  "n_recommendations": 5,
  "fallback_to_top": false,
  "diversify": true
}
```

Each recommendation includes `ranking_strategy` so clients can distinguish pure ALS,
diversity re-ranking, and cold-start fallback results.

## API Endpoints

| Method | Endpoint | Description |
| --- | --- | --- |
| GET | `/` | Redirects to the browser demo UI |
| GET | `/app` | Browser demo UI for catalog exploration, movie details, recommendations, user context, and similar movies |
| GET | `/health` | Lightweight health check |
| GET | `/version` | Service name and API version |
| GET | `/ready` | Readiness check for model and data artifacts |
| GET | `/metrics` | Lightweight runtime cache metrics |
| GET | `/model/info` | Model card summary with RMSE, data volume, score policy, and artifact size |
| GET | `/catalog/summary` | Catalog-level business summary for dashboards and demos |
| GET | `/movies/genres` | Available movie genres with catalog statistics |
| GET | `/movies/top` | Top-rated movies fallback ranking, optionally filtered by genre |
| GET | `/movies/search` | Search movies by title with optional genre and rating filters |
| GET | `/movies/{movie_id}` | Movie catalog detail for recommendation debugging |
| GET | `/users/{user_id}/profile` | User rating profile for recommendation debugging |
| GET | `/users/{user_id}/history` | Recent user rating history with movie metadata |
| POST | `/recommend` | Personalized recommendations with optional genre-diversity re-ranking |
| POST | `/recommend/feedback` | Capture like/dislike feedback for recommendation quality loops |
| GET | `/recommend/feedback/summary` | Aggregate overall and per-strategy feedback metrics |
| GET | `/similar_movies/{movie_id}` | Similar movies by genre |
| GET | `/stats` | Request statistics from PostgreSQL |

Example recommendation request:

```bash
curl -X POST "http://localhost:8001/recommend" \
  -H "Content-Type: application/json" \
  -d '{"user_id": 1, "n_recommendations": 5, "fallback_to_top": false, "diversify": true}'
```

Cold-start fallback for an unknown user:

```bash
curl -X POST "http://localhost:8001/recommend" \
  -H "Content-Type: application/json" \
  -d '{"user_id": 9999, "n_recommendations": 5, "fallback_to_top": true}'
```

Recommendation feedback:

```bash
curl -X POST "http://localhost:8001/recommend/feedback" \
  -H "Content-Type: application/json" \
  -d '{"user_id": 1, "movie_id": 1, "feedback": "like", "source": "web_app", "ranking_strategy": "als_diverse"}'
```

Feedback summary:

```bash
curl "http://localhost:8001/recommend/feedback/summary"
```

Readiness check:

```bash
curl "http://localhost:8001/ready"
```

Demo web UI:

```text
http://localhost:8001/app
```

Catalog summary:

```bash
curl "http://localhost:8001/catalog/summary"
```

Model card summary:

```bash
curl "http://localhost:8001/model/info"
```

Movie detail:

```bash
curl "http://localhost:8001/movies/1"
```

Movie search:

```bash
curl "http://localhost:8001/movies/search?q=Movie&genre=Drama&min_rating=3.0"
```

User rating history:

```bash
curl "http://localhost:8001/users/1/history?n=5"
```

## API Error Responses

Unknown users return `404` when `fallback_to_top` is disabled:

```json
{
  "detail": "Пользователь 9999 не найден"
}
```

Invalid request limits return `422`:

```json
{
  "detail": [
    {
      "loc": ["body", "n_recommendations"],
      "msg": "Input should be greater than or equal to 1",
      "type": "greater_than_equal"
    }
  ]
}
```

## Runtime Configuration

The API can be configured through environment variables:

| Variable | Default |
| --- | --- |
| `RECSYS_MODEL_PATH` | `models/als_model` |
| `RECSYS_MOVIES_PATH` | `data/raw/movies.csv` |
| `RECSYS_RATINGS_PATH` | `data/raw/ratings.csv` |
| `RECSYS_USERS_PATH` | `data/raw/users.csv` |
| `RECSYS_USER_FEATURES_PATH` | `data/processed/user_features.csv` |
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
- The API Docker image includes a container healthcheck against `/health`.
- Every API response includes `X-Request-ID` and `X-Response-Time-ms` headers for tracing.
- PostgreSQL logging is optional: the API still works if the database is not running.
- Fast API tests mock the ML resources, so they run quickly without starting Spark.
- The full ML workflow is covered by the data generation, feature engineering, and training scripts.

## Current Results

- Synthetic dataset: 500 users, 200 movies, about 18k ratings after duplicate removal
- Latest local ALS RMSE: 1.4285
- ALS model saved to `models/als_model`
- API tests: 30 passing tests for root, web UI, health, version, readiness, metrics, tracing, catalog summary, stats, recommendations, cold-start fallback, movie detail, movie search, movie genres, top movies, genre-filtered rankings, similar movies, user profiles, user rating history, OpenAPI schemas, OpenAPI examples, invalid users, and request validation

## Project Status

Implemented:

- Synthetic data generation
- Spark feature engineering
- ALS model training
- MLflow experiment tracking
- FastAPI recommendation serving
- Browser demo UI
- Request tracing and latency headers
- Health and readiness checks
- PostgreSQL request logging
- Airflow DAG for scheduled pipeline runs
- Docker, Makefile commands, and GitHub Actions CI

Future improvements:

- Real MovieLens dataset versioning
- Offline ranking metrics such as precision@k and recall@k
- Model registry promotion flow
- Batch recommendation cache
- Authentication and rate limiting
- Deployment manifests for a cloud runtime

## Author

Nikita Marshchonok

Telegram: `@nikitamarshchonok`
