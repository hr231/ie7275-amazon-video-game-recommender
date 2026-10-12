"""Hybrid: per-user z-normalized blend of a collaborative model, the content model, and
(optionally) a recent-popularity prior.

STUB: fit() and score() are not implemented yet.
"""

from __future__ import annotations

import numpy as np

from vgrec.models.base import Recommender
from vgrec.protocol import TrainData


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
        self.cf = cf
        self.content = content
        self.popularity = popularity
        self.alpha = alpha
        self.beta = beta
        self.adaptive_k = adaptive_k
        self.fit_components = fit_components

    def fit(self, train: TrainData) -> Hybrid:
        raise NotImplementedError("Hybrid.fit is not implemented yet")

    def score(self, users: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Hybrid.score is not implemented yet")
