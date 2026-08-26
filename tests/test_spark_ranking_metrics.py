import pytest
from pyspark.sql import SparkSession

from src.evaluation.spark_metrics import evaluate_ranking_predictions


@pytest.fixture(scope="module")
def spark():
    session = (
        SparkSession.builder.master("local[1]")
        .appName("ranking-metrics-tests")
        .config("spark.ui.enabled", "false")
        .config("spark.driver.host", "127.0.0.1")
        .getOrCreate()
    )
    session.sparkContext.setLogLevel("ERROR")
    yield session
    session.stop()


def test_evaluate_ranking_predictions_calculates_mean_metrics(spark):
    predictions = spark.createDataFrame([
        (1, 10, 0.9),
        (1, 20, 0.8),
        (1, 30, 0.7),
        (2, 40, 0.9),
        (2, 50, 0.8),
    ], ["userId", "movieId", "prediction"])
    test_ratings = spark.createDataFrame([
        (1, 10, 5.0),
        (1, 20, 2.0),
        (1, 30, 4.0),
        (2, 40, 3.0),
        (2, 50, 4.5),
    ], ["userId", "movieId", "rating"])

    metrics = evaluate_ranking_predictions(
        predictions,
        test_ratings,
        k=2,
        relevance_threshold=4.0,
    )

    assert metrics["precision_at_k"] == pytest.approx(0.5)
    assert metrics["recall_at_k"] == pytest.approx(0.75)
    assert metrics["evaluated_users"] == 2


def test_evaluate_ranking_predictions_counts_missing_predictions(spark):
    predictions = spark.createDataFrame([
        (2, 20, 0.9),
    ], ["userId", "movieId", "prediction"])
    test_ratings = spark.createDataFrame([
        (1, 10, 5.0),
    ], ["userId", "movieId", "rating"])

    metrics = evaluate_ranking_predictions(predictions, test_ratings, k=1)

    assert metrics == {
        "precision_at_k": 0.0,
        "recall_at_k": 0.0,
        "evaluated_users": 1,
    }


def test_evaluate_ranking_predictions_handles_no_relevant_items(spark):
    predictions = spark.createDataFrame([
        (1, 10, 0.9),
    ], ["userId", "movieId", "prediction"])
    test_ratings = spark.createDataFrame([
        (1, 10, 3.5),
    ], ["userId", "movieId", "rating"])

    metrics = evaluate_ranking_predictions(
        predictions,
        test_ratings,
        relevance_threshold=4.0,
    )

    assert metrics == {
        "precision_at_k": 0.0,
        "recall_at_k": 0.0,
        "evaluated_users": 0,
    }


def test_evaluate_ranking_predictions_breaks_score_ties_by_movie_id(spark):
    predictions = spark.createDataFrame([
        (1, 20, 0.9),
        (1, 10, 0.9),
    ], ["userId", "movieId", "prediction"])
    test_ratings = spark.createDataFrame([
        (1, 10, 5.0),
        (1, 20, 2.0),
    ], ["userId", "movieId", "rating"])

    metrics = evaluate_ranking_predictions(predictions, test_ratings, k=1)

    assert metrics["precision_at_k"] == pytest.approx(1.0)
    assert metrics["recall_at_k"] == pytest.approx(1.0)


def test_evaluate_ranking_predictions_rejects_non_positive_k():
    with pytest.raises(ValueError, match="k must be at least 1"):
        evaluate_ranking_predictions(None, None, k=0)
