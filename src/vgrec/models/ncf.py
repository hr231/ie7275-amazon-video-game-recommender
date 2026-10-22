"""Neural Collaborative Filtering: NeuMF = GMF branch + MLP branch (He et al., WWW 2017)."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch
from torch import nn

from vgrec.config import SEED
from vgrec.models.base import Recommender
from vgrec.protocol import TrainData, recency_weights

PAIRS_PER_FORWARD = 1 << 20  # ~1M (user, item) pairs per scoring pass bounds memory


def pick_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class NeuMFNet(nn.Module):
    def __init__(self, n_users: int, n_items: int, mf_dim: int, mlp_layers: tuple[int, ...],
                 dropout: float):
        super().__init__()
        self.mf_user = nn.Embedding(n_users, mf_dim)
        self.mf_item = nn.Embedding(n_items, mf_dim)
        self.mlp_user = nn.Embedding(n_users, mlp_layers[0] // 2)
        self.mlp_item = nn.Embedding(n_items, mlp_layers[0] // 2)
        layers: list[nn.Module] = []
        for d_in, d_out in zip(mlp_layers[:-1], mlp_layers[1:], strict=True):
            layers += [nn.Linear(d_in, d_out), nn.ReLU(), nn.Dropout(dropout)]
        self.mlp = nn.Sequential(*layers)
        self.head = nn.Linear(mf_dim + mlp_layers[-1], 1)
        for emb in (self.mf_user, self.mf_item, self.mlp_user, self.mlp_item):
            nn.init.normal_(emb.weight, std=0.01)
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        gmf = self.mf_user(users) * self.mf_item(items)
        mlp = self.mlp(torch.cat([self.mlp_user(users), self.mlp_item(items)], dim=-1))
        return self.head(torch.cat([gmf, mlp], dim=-1)).squeeze(-1)  # logits


def sample_negatives(pos_users: np.ndarray, pos_keys: np.ndarray, item_pool: np.ndarray,
                     n_items: int, n_neg: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """n_neg uniform negatives per positive, redrawn while they collide with a known positive.

    pos_keys = sorted user * n_items + item for every positive pair.
    """
    users = np.repeat(pos_users, n_neg)
    items = item_pool[rng.integers(0, len(item_pool), size=len(users))]
    for _ in range(20):
        keys = users * n_items + items
        found = np.searchsorted(pos_keys, keys)
        clash = pos_keys[np.minimum(found, len(pos_keys) - 1)] == keys
        if not clash.any():
            break
        items[clash] = item_pool[rng.integers(0, len(item_pool), size=int(clash.sum()))]
    return users, items


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
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        self.device_ = pick_device(self.device)
        self.n_items_ = train.n_items
        coo = recency_weights(train, self.half_life_days).tocoo()
        pos_users, pos_items = coo.row.astype(np.int64), coo.col.astype(np.int64)
        pos_weights = coo.data.astype(np.float32)
        if len(pos_users) == 0:
            raise ValueError("no training interactions left to learn from (half_life_days too small?)")
        pos_keys = np.sort(pos_users * train.n_items + pos_items)
        item_pool = np.flatnonzero(np.diff(train.user_items.tocsc().indptr) > 0)

        self.net_ = NeuMFNet(train.n_users, train.n_items, self.mf_dim, self.mlp_layers,
                             self.dropout).to(self.device_)
        optimizer = torch.optim.Adam(self.net_.parameters(), lr=self.lr,
                                     weight_decay=self.weight_decay)
        loss_fn = nn.BCEWithLogitsLoss(reduction="none")
        self.loss_history_: list[float] = []
        for epoch in range(1, self.epochs + 1):
            neg_users, neg_items = sample_negatives(pos_users, pos_keys, item_pool, train.n_items,
                                                    self.n_neg, rng)
            users = np.concatenate([pos_users, neg_users])
            items = np.concatenate([pos_items, neg_items])
            labels = np.concatenate([np.ones(len(pos_users), np.float32),
                                     np.zeros(len(neg_users), np.float32)])
            # Each negative inherits the weight of the positive it was drawn for.
            weights = np.concatenate([pos_weights, np.repeat(pos_weights, self.n_neg)])
            order = torch.from_numpy(rng.permutation(len(users))).to(self.device_)
            users_t = torch.from_numpy(users).to(self.device_)[order]
            items_t = torch.from_numpy(items).to(self.device_)[order]
            labels_t = torch.from_numpy(labels).to(self.device_)[order]
            weights_t = torch.from_numpy(weights).to(self.device_)[order]
            self.net_.train()
            total = torch.zeros((), device=self.device_)
            for start in range(0, len(users), self.batch_size):
                batch = slice(start, start + self.batch_size)
                optimizer.zero_grad(set_to_none=True)
                losses = loss_fn(self.net_(users_t[batch], items_t[batch]), labels_t[batch])
                loss = (losses * weights_t[batch]).sum()
                (loss / len(losses)).backward()
                optimizer.step()
                total += loss.detach()
            self.loss_history_.append(float(total) / len(users))
            if self.on_epoch_end is not None:
                self.on_epoch_end(epoch, self)
        return self

    @torch.no_grad()
    def score(self, users: np.ndarray) -> np.ndarray:
        self.net_.eval()
        users = np.asarray(users, dtype=np.int64)
        out = np.empty((len(users), self.n_items_), dtype=np.float32)
        all_items = torch.arange(self.n_items_, device=self.device_)
        chunk = max(1, PAIRS_PER_FORWARD // self.n_items_)
        for start in range(0, len(users), chunk):
            block = torch.from_numpy(users[start : start + chunk]).to(self.device_)
            logits = self.net_(block.repeat_interleave(self.n_items_), all_items.repeat(len(block)))
            out[start : start + len(block)] = logits.view(len(block), self.n_items_).float().cpu().numpy()
        return out
