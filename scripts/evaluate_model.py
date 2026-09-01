from __future__ import annotations

import argparse
import json
from pathlib import Path

from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALSModel
from pyspark.sql import SparkSession

from src.data.versioning import fingerprint_file
from src.evaluation.spark_metrics import evaluate_ranking_predictions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a saved ALS model on its reproducible holdout split.",
    )
    parser.add_argument(
        "--ratings-path",
        type=Path,
        default=Path("data/raw/ratings.csv"),
        help="Ratings CSV used to train the saved model.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("models/als_model"),
        help="Directory containing the saved Spark ALS model and model card.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=10,
        help="Ranking cutoff for precision and recall.",
    )
    parser.add_argument(
        "--relevance-threshold",
        type=float,
        default=4.0,
        help="Minimum rating treated as relevant.",
    )
    parser.add_argument(
        "--update-model-card",
        action="store_true",
        help="Atomically write the evaluation results to model_card.json.",
    )
    return parser.parse_args()


def load_training_config(model_path: Path) -> dict[str, float | int]:
    model_card_path = model_path / "model_card.json"
    with model_card_path.open(encoding="utf-8") as file:
        model_card = json.load(file)

    training = model_card.get("training", {})
    return {
        "sample_fraction": float(training.get("sample_fraction", 1.0)),
        "seed": int(training.get("seed", 42)),
        "train_split": float(training.get("train_split", 0.8)),
        "test_split": float(training.get("test_split", 0.2)),
    }


def evaluate_saved_model(
    ratings_path: Path,
    model_path: Path,
    *,
    k: int,
    relevance_threshold: float,
) -> dict[str, float | int | str]:
    training = load_training_config(model_path)
    training_data = fingerprint_file(ratings_path)
    spark = (
        SparkSession.builder.appName("RecSys Saved Model Evaluation")
        .master("local[*]")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    test = None
    predictions = None

    try:
        ratings = (
            spark.read.option("header", "true")
            .option("inferSchema", "true")
            .csv(str(ratings_path))
        )
        sampled = ratings.sample(
            fraction=training["sample_fraction"],
            seed=training["seed"],
        )
        _, test = sampled.randomSplit(
            [training["train_split"], training["test_split"]],
            seed=training["seed"],
        )
        test = test.cache()
        test_ratings = test.count()

        model = ALSModel.load(str(model_path))
        predictions = model.transform(test).cache()
        predicted_ratings = predictions.count()

        rmse = RegressionEvaluator(
            metricName="rmse",
            labelCol="rating",
            predictionCol="prediction",
        ).evaluate(predictions)
        ranking_metrics = evaluate_ranking_predictions(
            predictions,
            test,
            k=k,
            relevance_threshold=relevance_threshold,
        )

        return {
            "model_path": str(model_path),
            "training_data_path": training_data["path"],
            "training_data_sha256": training_data["sha256"],
            "training_data_size_bytes": training_data["size_bytes"],
            "test_ratings": test_ratings,
            "predicted_ratings": predicted_ratings,
            "prediction_coverage": round(
                predicted_ratings / test_ratings if test_ratings else 0.0,
                4,
            ),
            "rmse": round(float(rmse), 4),
            "ranking_k": k,
            "relevance_threshold": relevance_threshold,
            f"precision_at_{k}": round(
                float(ranking_metrics["precision_at_k"]),
                4,
            ),
            f"recall_at_{k}": round(
                float(ranking_metrics["recall_at_k"]),
                4,
            ),
            "evaluated_users": int(ranking_metrics["evaluated_users"]),
        }
    finally:
        if predictions is not None:
            predictions.unpersist()
        if test is not None:
            test.unpersist()
        spark.stop()


def update_model_card(
    model_path: Path,
    evaluation: dict[str, float | int | str],
) -> None:
    model_card_path = model_path / "model_card.json"
    with model_card_path.open(encoding="utf-8") as file:
        model_card = json.load(file)

    ranking_k = int(evaluation["ranking_k"])
    test_ratings = int(evaluation["test_ratings"])
    predicted_ratings = int(evaluation["predicted_ratings"])
    model_card["training_data"] = {
        "path": str(evaluation["training_data_path"]),
        "algorithm": "sha256",
        "sha256": str(evaluation["training_data_sha256"]),
        "size_bytes": int(evaluation["training_data_size_bytes"]),
    }
    model_card["ranking_evaluation"] = {
        "k": ranking_k,
        "relevance_threshold": float(evaluation["relevance_threshold"]),
        "candidate_policy": "observed test interactions ranked by prediction",
        "evaluated_users": int(evaluation["evaluated_users"]),
        "test_ratings": test_ratings,
        "predicted_ratings": predicted_ratings,
        "prediction_coverage": round(
            predicted_ratings / test_ratings if test_ratings else 0.0,
            4,
        ),
    }
    metrics = model_card.setdefault("metrics", {})
    metrics["rmse"] = float(evaluation["rmse"])
    metrics[f"precision_at_{ranking_k}"] = float(
        evaluation[f"precision_at_{ranking_k}"]
    )
    metrics[f"recall_at_{ranking_k}"] = float(
        evaluation[f"recall_at_{ranking_k}"]
    )

    temporary_path = model_card_path.with_suffix(".json.tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(model_card, file, indent=2)
        file.write("\n")
    temporary_path.replace(model_card_path)


def main() -> None:
    args = parse_args()
    metrics = evaluate_saved_model(
        args.ratings_path,
        args.model_path,
        k=args.k,
        relevance_threshold=args.relevance_threshold,
    )
    if args.update_model_card:
        update_model_card(args.model_path, metrics)
        metrics["model_card_updated"] = True
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
