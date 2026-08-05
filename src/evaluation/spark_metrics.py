from pyspark.ml.evaluation import RankingEvaluator
from pyspark.sql import DataFrame, Window
from pyspark.sql.functions import (
    array,
    col,
    collect_list,
    collect_set,
    row_number,
    sort_array,
    struct,
    transform,
    when,
)


def evaluate_ranking_predictions(
    predictions: DataFrame,
    test_ratings: DataFrame,
    *,
    k: int = 10,
    relevance_threshold: float = 4.0,
) -> dict[str, float | int]:
    """Evaluate ALS ordering over observed test interactions."""
    if k < 1:
        raise ValueError("k must be at least 1")

    ranking_window = Window.partitionBy("userId").orderBy(
        col("prediction").desc(),
        col("movieId").asc(),
    )
    ranked_items = (
        predictions.select("userId", "movieId", "prediction")
        .filter(col("prediction").isNotNull())
        .withColumn("_rank", row_number().over(ranking_window))
        .filter(col("_rank") <= k)
        .withColumn("_item", col("movieId").cast("double"))
        .groupBy("userId")
        .agg(
            sort_array(
                collect_list(struct("_rank", "_item"))
            ).alias("_ranked_items")
        )
        .select(
            "userId",
            transform(
                "_ranked_items",
                lambda item: item["_item"],
            ).alias("prediction"),
        )
    )
    relevant_items = (
        test_ratings.filter(col("rating") >= relevance_threshold)
        .groupBy("userId")
        .agg(
            collect_set(col("movieId").cast("double")).alias("label")
        )
    )
    evaluation_data = (
        relevant_items.join(ranked_items, on="userId", how="left")
        .withColumn(
            "prediction",
            when(
                col("prediction").isNull(),
                array().cast("array<double>"),
            ).otherwise(col("prediction")),
        )
        .select("prediction", "label")
    )

    evaluated_users = evaluation_data.count()
    if evaluated_users == 0:
        return {
            "precision_at_k": 0.0,
            "recall_at_k": 0.0,
            "evaluated_users": 0,
        }

    precision_evaluator = RankingEvaluator(
        predictionCol="prediction",
        labelCol="label",
        metricName="precisionAtK",
        k=k,
    )
    recall_evaluator = RankingEvaluator(
        predictionCol="prediction",
        labelCol="label",
        metricName="recallAtK",
        k=k,
    )
    return {
        "precision_at_k": float(precision_evaluator.evaluate(evaluation_data)),
        "recall_at_k": float(recall_evaluator.evaluate(evaluation_data)),
        "evaluated_users": evaluated_users,
    }

