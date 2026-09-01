import json

from scripts.evaluate_model import update_model_card


def test_update_model_card_writes_ranking_metrics_atomically(tmp_path):
    model_path = tmp_path / "als_model"
    model_path.mkdir()
    model_card_path = model_path / "model_card.json"
    model_card_path.write_text(
        json.dumps({
            "model_name": "Test ALS",
            "metrics": {"rmse": 1.0},
        }),
        encoding="utf-8",
    )
    evaluation = {
        "model_path": str(model_path),
        "training_data_path": "data/raw/ratings.csv",
        "training_data_sha256": "a" * 64,
        "training_data_size_bytes": 1234,
        "test_ratings": 12,
        "predicted_ratings": 11,
        "prediction_coverage": 0.9167,
        "rmse": 0.8,
        "ranking_k": 10,
        "relevance_threshold": 4.0,
        "precision_at_10": 0.25,
        "recall_at_10": 0.75,
        "evaluated_users": 3,
    }

    update_model_card(model_path, evaluation)

    updated = json.loads(model_card_path.read_text(encoding="utf-8"))
    assert updated["model_name"] == "Test ALS"
    assert updated["training_data"] == {
        "path": "data/raw/ratings.csv",
        "algorithm": "sha256",
        "sha256": "a" * 64,
        "size_bytes": 1234,
    }
    assert updated["metrics"] == {
        "rmse": 0.8,
        "precision_at_10": 0.25,
        "recall_at_10": 0.75,
    }
    assert updated["ranking_evaluation"] == {
        "k": 10,
        "relevance_threshold": 4.0,
        "candidate_policy": "observed test interactions ranked by prediction",
        "evaluated_users": 3,
        "test_ratings": 12,
        "predicted_ratings": 11,
        "prediction_coverage": 0.9167,
    }
    assert not (model_path / "model_card.json.tmp").exists()
