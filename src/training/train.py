import mlflow
import mlflow.spark
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from datetime import datetime, timezone
import json
import os

from src.evaluation.spark_metrics import evaluate_ranking_predictions


# Spark сессия
spark = SparkSession.builder \
    .appName("RecSys Training") \
    .master("local[*]") \
    .config("spark.driver.memory", "4g") \
    .config("spark.executor.memory", "4g") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

# MLflow — куда записывать эксперименты
mlflow.set_tracking_uri(
    os.getenv("MLFLOW_TRACKING_URI", "http://host.docker.internal:5001")
)
mlflow.set_experiment("recsys_als")

print("Сессии созданы")

# Читаем оценки
ratings = spark.read \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .csv("data/raw/ratings.csv")

# Берём 20% данных для обучения на локальной машине
ratings = ratings.sample(fraction=0.2, seed=42)
sampled_count = ratings.count()
print(f"После sampling: {sampled_count} записей")
# Делим 80% train / 20% test
train, test = ratings.randomSplit([0.8, 0.2], seed=42)

train_count = train.count()
test_count = test.count()
print(f"Train: {train_count} записей")
print(f"Test: {test_count} записей")

# Параметры модели
rank = 10        # размерность скрытых факторов
max_iter = 10    # количество итераций
reg_param = 0.1  # регуляризация (защита от переобучения)
ranking_k = 10
relevance_threshold = 4.0

with mlflow.start_run():
    # Записываем параметры
    mlflow.log_param("rank", rank)
    mlflow.log_param("max_iter", max_iter)
    mlflow.log_param("reg_param", reg_param)
    mlflow.log_param("ranking_k", ranking_k)
    mlflow.log_param("relevance_threshold", relevance_threshold)

    # Создаём и обучаем ALS модель
    als = ALS(
        rank=rank,
        maxIter=max_iter,
        regParam=reg_param,
        userCol="userId",
        itemCol="movieId",
        ratingCol="rating",
        coldStartStrategy="drop",  # игнорируем новых пользователей
    )

    model = als.fit(train)

    # Проверяем на тестовых данных
    predictions = model.transform(test)

    # Считаем RMSE — среднеквадратичная ошибка
    evaluator = RegressionEvaluator(
        metricName="rmse",
        labelCol="rating",
        predictionCol="prediction",
    )
    rmse = evaluator.evaluate(predictions)

    ranking_metrics = evaluate_ranking_predictions(
        predictions,
        test,
        k=ranking_k,
        relevance_threshold=relevance_threshold,
    )

    # Записываем метрики
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric(
        f"precision_at_{ranking_k}",
        ranking_metrics["precision_at_k"],
    )
    mlflow.log_metric(
        f"recall_at_{ranking_k}",
        ranking_metrics["recall_at_k"],
    )
    mlflow.log_metric(
        "ranking_evaluated_users",
        ranking_metrics["evaluated_users"],
    )

    print(f"RMSE: {rmse:.4f}")
    print(
        f"Precision@{ranking_k}: "
        f"{ranking_metrics['precision_at_k']:.4f}"
    )
    print(
        f"Recall@{ranking_k}: "
        f"{ranking_metrics['recall_at_k']:.4f}"
    )
    print("Эксперимент записан в MLflow!")

    # Сохраняем модель на диск
    model_path = "models/als_model"
    os.makedirs("models", exist_ok=True)
    model.write().overwrite().save(model_path)

    model_card = {
        "model_name": "MovieLens ALS recommender",
        "algorithm": "pyspark.ml.recommendation.ALS",
        "dataset": "MovieLens ratings",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "generated_by": "src/training/train.py",
        "rating_scale": {"min": 0.0, "max": 5.0},
        "score_policy": {
            "predicted_rating": "clipped to the user-facing 0-5 rating scale",
            "raw_predicted_rating": "unbounded ALS model output",
        },
        "ranking_evaluation": {
            "k": ranking_k,
            "relevance_threshold": relevance_threshold,
            "candidate_policy": "observed test interactions ranked by prediction",
            "evaluated_users": ranking_metrics["evaluated_users"],
        },
        "training": {
            "sample_fraction": 0.2,
            "seed": 42,
            "train_split": 0.8,
            "test_split": 0.2,
            "sampled_ratings": sampled_count,
            "train_ratings": train_count,
            "test_ratings": test_count,
        },
        "hyperparameters": {
            "rank": rank,
            "max_iter": max_iter,
            "reg_param": reg_param,
            "cold_start_strategy": "drop",
        },
        "metrics": {
            "rmse": round(float(rmse), 4),
            f"precision_at_{ranking_k}": round(
                float(ranking_metrics["precision_at_k"]),
                4,
            ),
            f"recall_at_{ranking_k}": round(
                float(ranking_metrics["recall_at_k"]),
                4,
            ),
        },
    }
    with open(os.path.join(model_path, "model_card.json"), "w", encoding="utf-8") as file:
        json.dump(model_card, file, indent=2)

    print(f"Модель сохранена в {model_path}")
    print(f"Model card сохранён в {model_path}/model_card.json")

spark.stop()
print("Готово!")
