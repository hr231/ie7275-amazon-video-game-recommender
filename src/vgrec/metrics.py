"""Top-K ranking metrics with binary relevance. Accuracy metrics return one value per user."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def hit_matrix(topk: np.ndarray, targets: Sequence[np.ndarray]) -> np.ndarray:
    """Bool (n_users, k): True where the ranked item is one of that user's targets."""
    if len(topk) != len(targets):
        raise ValueError(f"{len(topk)} ranked lists but {len(targets)} target sets")
    hits = np.zeros(topk.shape, dtype=bool)
    for row, (ranked, relevant) in enumerate(zip(topk, targets, strict=True)):
        hits[row] = np.isin(ranked, relevant)
    return hits


def precision_at_k(hits: np.ndarray) -> np.ndarray:
    return hits.sum(axis=1) / hits.shape[1]


def recall_at_k(hits: np.ndarray, n_targets: np.ndarray) -> np.ndarray:
    return hits.sum(axis=1) / n_targets


def ndcg_at_k(hits: np.ndarray, n_targets: np.ndarray) -> np.ndarray:
    """DCG / ideal DCG, where the ideal ranking puts min(n_targets, k) hits on top."""
    k = hits.shape[1]
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    ideal = np.cumsum(discounts)[np.minimum(n_targets, k) - 1]
    return (hits @ discounts) / ideal


def hit_rate_at_k(hits: np.ndarray) -> np.ndarray:
    return hits.any(axis=1).astype(np.float64)


def catalog_coverage(topk: np.ndarray, n_candidates: int) -> float:
    """Share of the candidate catalog that appears in at least one user's top-K."""
    return np.unique(topk).size / n_candidates


def personalization(topk: np.ndarray) -> float:
    """1 - mean |A ∩ B| / k over all user pairs; 0 = everyone gets the same list.

    Exact and O(n·k): summing c·(c-1) over each item's recommendation count c counts every
    ordered pair of users that share that item.
    """
    n, k = topk.shape
    if n < 2:
        return float("nan")
    counts = np.bincount(topk.ravel()).astype(np.float64)
    mean_overlap = (counts * (counts - 1)).sum() / (n * (n - 1) * k)
    return float(1.0 - mean_overlap)
