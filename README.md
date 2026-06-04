# RecSys Platform

Production-grade movie recommendation system built with modern MLOps stack.

## Architecture
Raw Data → Spark Feature Engineering → ALS Model Training → FastAPI Service
↓
MLflow Tracking
↓
PostgreSQL Logging

## Stack

- **Apache Spark** — distributed feature engineering
- **PySpark ALS** — collaborative filtering model
- **MLflow** — experiment tracking and model registry
- **FastAPI** — REST API for recommendations
- **PostgreSQL** — request logging and observability
- **pytest** — automated API testing
- **Docker Compose** — infrastructure as code

## Project Structure
recsys-platform/
├── data/
│   ├── raw/          # generated dataset (500 users, 200 movies, 18k ratings)
│   └── processed/    # spark feature engineering output
├── spark/jobs/       # feature engineering pipeline
├── src/
│   ├── api/          # FastAPI recommendation service
│   └── training/     # ALS model training with MLflow
├── tests/            # pytest API tests
├── models/           # saved ALS model
└── docker-compose.yml

## Quick Start

```bash
# 1. Start infrastructure
docker compose up -d

# 2. Generate data
python scripts/generate_data.py

# 3. Run Spark feature engineering
export JAVA_HOME=/opt/homebrew/opt/openjdk@17
python spark/jobs/feature_engineering.py

# 4. Train model
python src/training/train.py

# 5. Start API
uvicorn src.api.main:app --port 8001
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | /health | Service health check |
| POST | /recommend | Get personalized recommendations |
| GET | /similar_movies/{id} | Get similar movies by genre |
| GET | /stats | Request statistics |

## Example Request

```bash
curl -X POST "http://localhost:8001/recommend" \
  -H "Content-Type: application/json" \
  -d '{"user_id": 1, "n_recommendations": 5}'
```

## Results

- RMSE: 1.05 on test set
- Average API response time: ~1.4s (local Spark)
- 3 automated tests covering core functionality