"""Implicit-feedback matrix factorization (Hu, Koren & Volinsky 2008) via the `implicit` library."""

from __future__ import annotations

import numpy as np
from threadpoolctl import threadpool_limits

from vgrec.config import SEED
from vgrec.models.base import Recommender
from vgrec.protocol import TrainData, recency_weights

WEIGHTINGS = ("binary", "bm25")


class ALS(Recommender):
    """Fit implicit's AlternatingLeastSquares on the history matrix, then score = U[users] @ V.T.

    half_life_days > 0: each interaction counts 0.5 ** (age_days / half_life_days) instead of 1
                        (protocol.recency_weights), so recent tastes dominate.
    weighting: "binary" keeps those values; "bm25" then down-weights very active users and
               very popular items. implicit then uses confidence = alpha * value for observed
               pairs and 1 for unobserved ones (not Hu et al.'s 1 + alpha * value).
    """

    name = "als"

    def __init__(self, factors: int = 128, regularization: float = 0.05, alpha: float = 10.0,
                 iterations: int = 20, weighting: str = "binary", half_life_days: float = 0.0,
                 bm25_k1: float = 100.0, bm25_b: float = 0.8, seed: int = SEED):
        if weighting not in WEIGHTINGS:
            raise ValueError(f"weighting must be one of {WEIGHTINGS}, got {weighting!r}")
        self.factors = factors
        self.regularization = regularization
        self.alpha = alpha
        self.iterations = iterations
        self.weighting = weighting
        self.half_life_days = half_life_days
        self.bm25_k1 = bm25_k1
        self.bm25_b = bm25_b
        self.seed = seed

    def fit(self, train: TrainData) -> ALS:
        from implicit.als import AlternatingLeastSquares
        from implicit.nearest_neighbours import bm25_weight

        user_items = recency_weights(train, self.half_life_days)
        if self.weighting == "bm25":
            user_items = bm25_weight(user_items, K1=self.bm25_k1, B=self.bm25_b).tocsr()
        model = AlternatingLeastSquares(
            factors=self.factors, regularization=self.regularization, alpha=self.alpha,
            iterations=self.iterations, random_state=self.seed, use_gpu=False)
        # implicit runs its own thread pool; nested multi-threaded BLAS calls slow it down badly.
        with threadpool_limits(1, "blas"):
            model.fit(user_items.astype(np.float32), show_progress=False)
        self.user_factors_ = np.ascontiguousarray(model.user_factors, dtype=np.float32)
        self.item_factors_ = np.ascontiguousarray(model.item_factors, dtype=np.float32)
        return self

    def score(self, users: np.ndarray) -> np.ndarray:
        return self.user_factors_[users] @ self.item_factors_.T
