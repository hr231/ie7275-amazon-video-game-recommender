"""Content-based recommender: TF-IDF item vectors + per-user profile, scored by cosine similarity."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from vgrec.models.base import Recommender
from vgrec.protocol import TrainData, recency_weights


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
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.preprocessing import normalize

        if len(self.item_text) != train.n_items:
            raise ValueError(f"{len(self.item_text)} item texts for {train.n_items} items")
        vectorizer = TfidfVectorizer(
            stop_words="english", max_features=self.max_features, min_df=self.min_df,
            max_df=self.max_df, ngram_range=(1, self.ngram_max), sublinear_tf=True,
            dtype=np.float32)
        self.item_vectors_ = vectorizer.fit_transform(self.item_text).tocsr()  # rows are L2-normalized
        self.vocabulary_size_ = len(vectorizer.vocabulary_)
        self._item_vectors_t = self.item_vectors_.T.tocsr()
        profiles = recency_weights(train, self.half_life_days) @ self.item_vectors_
        self.user_profiles_ = normalize(profiles).astype(np.float32).tocsr()
        return self

    def score(self, users: np.ndarray) -> np.ndarray:
        return (self.user_profiles_[users] @ self._item_vectors_t).toarray().astype(np.float32)
