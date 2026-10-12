"""Implicit-feedback matrix factorization (Hu, Koren & Volinsky 2008) via the `implicit` library.

STUB: fit() and score() are not implemented yet.
"""

from __future__ import annotations

import numpy as np

from vgrec.config import SEED
from vgrec.models.base import Recommender
from vgrec.protocol import TrainData

WEIGHTINGS = ("binary", "bm25")


class ALS(Recommender):
    """Fit implicit's AlternatingLeastSquares on the history matrix, then score = U[users] @ V.T.

    half_life_days > 0: each interaction counts 0.5 ** (age_days / half_life_days) instead of 1
                        (protocol.recency_weights), so recent tastes dominate.
    weighting: "binary" keeps those values; "bm25" then down-weights very active users and
               very popular items. Confidence = 1 + alpha * value.
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
        raise NotImplementedError("ALS.fit is not implemented yet")

    def score(self, users: np.ndarray) -> np.ndarray:
        raise NotImplementedError("ALS.score is not implemented yet")
