"""Hybrid: per-user z-normalized blend of a collaborative model, the content model, and
(optionally) a recent-popularity prior."""

from __future__ import annotations

import numpy as np

from vgrec.models.base import Recommender
from vgrec.protocol import TrainData


def zscore_rows(scores: np.ndarray) -> np.ndarray:
    """Standardize each row; constant rows become all-zero instead of NaN."""
    scores = np.asarray(scores, dtype=np.float32)
    mean = scores.mean(axis=1, keepdims=True)
    std = scores.std(axis=1, keepdims=True)
    return np.divide(scores - mean, std, out=np.zeros_like(scores), where=std > 0)


class Hybrid(Recommender):
    """score = (1 - beta) * [w_u * z(cf) + (1 - w_u) * z(content)] + beta * z(popularity)

    z = per-user standardization over all items (constant rows -> 0).
    adaptive_k == 0: w_u = alpha for every user.
    adaptive_k  > 0: w_u = alpha * n_u / (n_u + adaptive_k), n_u = history length, so users with
                     short histories lean on content (the proposal's hypothesis).
    beta == 0 is the proposal's CF + content hybrid; beta > 0 needs `popularity`.
    fit_components=False reuses already-fitted components (cheap weight tuning).
    """

    name = "hybrid"

    def __init__(self, cf: Recommender, content: Recommender, popularity: Recommender | None = None,
                 alpha: float = 0.5, beta: float = 0.0, adaptive_k: float = 0.0,
                 fit_components: bool = True):
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        if not 0.0 <= beta <= 1.0:
            raise ValueError(f"beta must be in [0, 1], got {beta}")
        if beta > 0 and popularity is None:
            raise ValueError("beta > 0 needs a popularity model")
        if adaptive_k < 0:
            raise ValueError(f"adaptive_k must be >= 0, got {adaptive_k}")
        self.cf = cf
        self.content = content
        self.popularity = popularity
        self.alpha = alpha
        self.beta = beta
        self.adaptive_k = adaptive_k
        self.fit_components = fit_components

    def fit(self, train: TrainData) -> Hybrid:
        if self.fit_components:
            for model in (self.cf, self.content, self.popularity):
                if model is not None:
                    model.fit(train)
        self.history_lengths_ = np.diff(train.user_items.indptr).astype(np.float32)
        return self

    def weights(self, users: np.ndarray) -> np.ndarray:
        """CF weight w_u per user, in [0, alpha]."""
        if self.adaptive_k > 0:
            n = self.history_lengths_[users]
            return (self.alpha * n / (n + self.adaptive_k)).astype(np.float32)
        return np.full(len(users), self.alpha, dtype=np.float32)

    def components(self, users: np.ndarray) -> dict[str, np.ndarray]:
        """z-normalized component scores; compute once, then `blend` with many weightings."""
        parts = {"cf": zscore_rows(self.cf.score(users)),
                 "content": zscore_rows(self.content.score(users))}
        if self.popularity is not None:
            parts["popularity"] = zscore_rows(self.popularity.score(users))
        return parts

    def blend(self, parts: dict[str, np.ndarray], users: np.ndarray) -> np.ndarray:
        w = self.weights(users)[:, None]
        scores = w * parts["cf"] + (1.0 - w) * parts["content"]
        if self.beta > 0:
            scores = (1.0 - self.beta) * scores + self.beta * parts["popularity"]
        return scores.astype(np.float32)

    def score(self, users: np.ndarray) -> np.ndarray:
        return self.blend(self.components(users), users)
