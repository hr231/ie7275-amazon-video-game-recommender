"""Neural Collaborative Filtering: NeuMF = GMF branch + MLP branch (He et al., WWW 2017).

STUB: fit() and score() are not implemented yet.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from vgrec.config import SEED
from vgrec.models.base import Recommender
from vgrec.protocol import TrainData


class NeuMF(Recommender):
    """Trained with binary cross-entropy on observed pairs + `n_neg` uniformly sampled negatives
    per positive (resampled every epoch). half_life_days > 0 weights each positive, and the
    negatives drawn for it, by recency. `device="auto"` picks cuda > mps > cpu.
    `on_epoch_end(epoch, model)` lets a notebook record learning curves.
    """

    name = "ncf"

    def __init__(self, mf_dim: int = 32, mlp_layers: tuple[int, ...] = (64, 32, 16), n_neg: int = 4,
                 epochs: int = 10, batch_size: int = 4096, lr: float = 1e-3,
                 weight_decay: float = 0.0, dropout: float = 0.0, half_life_days: float = 0.0,
                 device: str = "auto", seed: int = SEED,
                 on_epoch_end: Callable[[int, NeuMF], None] | None = None):
        if mlp_layers[0] % 2:
            raise ValueError(f"mlp_layers[0] must be even (split between user and item), got {mlp_layers}")
        self.mf_dim = mf_dim
        self.mlp_layers = tuple(mlp_layers)
        self.n_neg = n_neg
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.half_life_days = half_life_days
        self.device = device
        self.seed = seed
        self.on_epoch_end = on_epoch_end

    def fit(self, train: TrainData) -> NeuMF:
        raise NotImplementedError("NeuMF.fit is not implemented yet")

    def score(self, users: np.ndarray) -> np.ndarray:
        raise NotImplementedError("NeuMF.score is not implemented yet")
