import pytest

from src.evaluation.metrics import precision_at_k, recall_at_k


def test_precision_at_k_counts_relevant_recommendations():
    assert precision_at_k([1, 2, 3, 4, 5], {2, 4, 8}, k=5) == pytest.approx(0.4)


def test_recall_at_k_counts_retrieved_relevant_items():
    assert recall_at_k([1, 2, 3, 4, 5], {2, 4, 8}, k=5) == pytest.approx(2 / 3)


def test_metrics_ignore_recommendations_below_k():
    assert precision_at_k([1, 2, 3], {3}, k=2) == 0.0
    assert recall_at_k([1, 2, 3], {3}, k=2) == 0.0


def test_precision_at_k_penalizes_short_recommendation_lists():
    assert precision_at_k([1], {1}, k=5) == pytest.approx(0.2)


def test_metrics_do_not_count_duplicate_recommendations_twice():
    assert precision_at_k([1, 1, 2], {1}, k=3) == pytest.approx(1 / 3)
    assert recall_at_k([1, 1, 2], {1}, k=3) == 1.0


def test_recall_at_k_returns_zero_without_relevant_items():
    assert recall_at_k([1, 2, 3], set(), k=3) == 0.0


@pytest.mark.parametrize("metric", [precision_at_k, recall_at_k])
def test_ranking_metrics_reject_non_positive_k(metric):
    with pytest.raises(ValueError, match="k must be at least 1"):
        metric([1, 2, 3], {1}, k=0)
