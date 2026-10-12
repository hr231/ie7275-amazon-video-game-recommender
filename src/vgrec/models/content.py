"""Content-based recommender: TF-IDF item vectors + per-user profile, scored by cosine similarity.

STUB: fit() and score() are not implemented yet.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from vgrec.models.base import Recommender
from vgrec.protocol import TrainData


class ContentTfidf(Recommender):
    """item vectors = TF-IDF(item_text), L2-normalized rows.
    user profile   = weighted sum of the user's history item vectors, L2-normalized.
    score          = profile @ item_vectors.T  (cosine similarity).

    half_life_days = 0 weights history items equally; > 0 weights them by
    0.5 ** (age_days / half_life_days) (protocol.recency_weights).
    """

    name = "content"

    def __init__(self, item_text: Sequence[str], max_features: int = 50_000, min_df: int = 2,
                 max_df: float = 0.5, ngram_max: int = 1, half_life_days: float = 0.0):
        self.item_text = list(item_text)
        self.max_features = max_features
        self.min_df = min_df
        self.max_df = max_df
        self.ngram_max = ngram_max
        self.half_life_days = half_life_days

    def fit(self, train: TrainData) -> ContentTfidf:
        raise NotImplementedError("ContentTfidf.fit is not implemented yet")

    def score(self, users: np.ndarray) -> np.ndarray:
        raise NotImplementedError("ContentTfidf.score is not implemented yet")
