from collections.abc import Collection, Sequence
from typing import Hashable


def _validate_k(k: int) -> None:
    if k < 1:
        raise ValueError("k must be at least 1")


def _hits_at_k(
    recommended_items: Sequence[Hashable],
    relevant_items: Collection[Hashable],
    k: int,
) -> int:
    _validate_k(k)
    return len(set(recommended_items[:k]) & set(relevant_items))


def precision_at_k(
    recommended_items: Sequence[Hashable],
    relevant_items: Collection[Hashable],
    k: int = 10,
) -> float:
    """Return the share of top-k positions containing a relevant item."""
    return _hits_at_k(recommended_items, relevant_items, k) / k


def recall_at_k(
    recommended_items: Sequence[Hashable],
    relevant_items: Collection[Hashable],
    k: int = 10,
) -> float:
    """Return the share of relevant items retrieved in the top-k positions."""
    unique_relevant_items = set(relevant_items)
    if not unique_relevant_items:
        _validate_k(k)
        return 0.0

    return _hits_at_k(
        recommended_items,
        unique_relevant_items,
        k,
    ) / len(unique_relevant_items)

