"""Popularity baseline: the most-interacted items, optionally within a recent time window."""

from __future__ import annotations

import numpy as np

from vgrec.config import MS_PER_DAY
from vgrec.models.base import Recommender
from vgrec.protocol import TrainData


class Popularity(Recommender):
    """`window_days=0` counts the whole history; otherwise only the last N days before cutoff.

    Window counts are integers, so adding all-time counts scaled into [0, 1) breaks ties by
    long-run popularity without reordering items with different window counts.
    """

    name = "popularity"

    def __init__(self, window_days: float = 0):
        self.window_days = window_days

    def fit(self, train: TrainData) -> Popularity:
        items = train.interactions["item_idx"].to_numpy()
        all_time = np.bincount(items, minlength=train.n_items).astype(np.float64)
        if self.window_days > 0:
            recent = train.interactions["timestamp"].to_numpy() >= (
                train.cutoff_ms - self.window_days * MS_PER_DAY
            )
            counts = np.bincount(items[recent], minlength=train.n_items).astype(np.float64)
        else:
            counts = all_time.copy()
        self.item_scores_ = (counts + all_time / (all_time.max() + 1.0)).astype(np.float32)
        return self

    def score(self, users: np.ndarray) -> np.ndarray:
        return np.tile(self.item_scores_, (len(users), 1))
