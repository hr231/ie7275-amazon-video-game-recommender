"""The contract every recommender implements. The evaluator handles masking and top-K."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from vgrec.protocol import TrainData


class Recommender(ABC):
    name: str = "recommender"

    @abstractmethod
    def fit(self, train: TrainData) -> Recommender:
        """Learn from history only. Must return self."""

    @abstractmethod
    def score(self, users: np.ndarray) -> np.ndarray:
        """float32 array (len(users), n_items); higher = more relevant.

        Score every item. Do not mask seen or non-candidate items; `evaluate.rank` does that,
        identically for all models. Never return NaN.
        """
