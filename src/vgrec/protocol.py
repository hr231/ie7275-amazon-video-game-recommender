"""Leakage-safe evaluation splits: what a model may train on vs. what it is scored against.

Stage "valid": history = train,         targets = valid  (cutoff T1) -> model selection
Stage "test":  history = train + valid, targets = test   (cutoff T2) -> final numbers

Targets drop rows whose user has no history (cold user), whose item never appears in the
history (cold item: no model can rank it under this protocol), or that the user already saw.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import scipy.sparse as sp

from vgrec.config import MS_PER_DAY, T1_MS, T2_MS
from vgrec.data import Dataset

STAGES: dict[str, tuple[tuple[str, ...], str, int]] = {
    "valid": (("train",), "valid", T1_MS),
    "test": (("train", "valid"), "test", T2_MS),
}


@dataclass(frozen=True)
class TrainData:
    """Everything a model is allowed to see: interactions strictly before `cutoff_ms`."""

    interactions: pd.DataFrame  # user_idx, item_idx, rating, timestamp
    n_users: int
    n_items: int
    cutoff_ms: int
    user_items: sp.csr_matrix = field(repr=False)  # binary float32 (n_users, n_items)


@dataclass(frozen=True)
class EvalSplit:
    stage: str
    train: TrainData
    eval_users: np.ndarray  # int64, sorted; every one has >= 1 history row and >= 1 target
    targets: list[np.ndarray]  # aligned with eval_users: sorted unique target item_idx
    candidate_mask: np.ndarray  # bool (n_items,): item appears in the history
    stats: dict[str, int]

    @property
    def n_items(self) -> int:
        return self.train.n_items

    @property
    def history_lengths(self) -> np.ndarray:
        return np.diff(self.train.user_items.indptr)[self.eval_users]


def interactions_to_csr(df: pd.DataFrame, n_users: int, n_items: int) -> sp.csr_matrix:
    """Binary user-item matrix (repeats collapse to 1)."""
    m = sp.csr_matrix(
        (np.ones(len(df), dtype=np.float32), (df["user_idx"].to_numpy(), df["item_idx"].to_numpy())),
        shape=(n_users, n_items),
    )
    m.sum_duplicates()
    m.data[:] = 1.0
    return m


def recency_weights(train: TrainData, half_life_days: float) -> sp.csr_matrix:
    """train.user_items with each interaction weighted by 0.5 ** (age_days / half_life_days).

    Age is measured back from the cutoff. half_life_days <= 0 returns the binary matrix.
    Weights that underflow to 0 are dropped (the interaction is effectively forgotten).
    """
    if half_life_days <= 0:
        return train.user_items
    df = train.interactions
    age_days = (train.cutoff_ms - df["timestamp"].to_numpy()) / MS_PER_DAY
    weights = np.power(0.5, age_days / half_life_days).astype(np.float32)
    m = sp.csr_matrix((weights, (df["user_idx"].to_numpy(), df["item_idx"].to_numpy())),
                      shape=(train.n_users, train.n_items))
    m.eliminate_zeros()
    return m


def make_split(ds: Dataset, stage: str) -> EvalSplit:
    if stage not in STAGES:
        raise ValueError(f"stage must be one of {sorted(STAGES)}, got {stage!r}")
    history_splits, target_split, cutoff = STAGES[stage]
    inter = ds.interactions
    history = inter.loc[inter["split"].isin(history_splits), ["user_idx", "item_idx", "rating", "timestamp"]]
    history = history.reset_index(drop=True)
    target = inter.loc[inter["split"] == target_split, ["user_idx", "item_idx", "timestamp"]]

    user_items = interactions_to_csr(history, ds.n_users, ds.n_items)
    candidate_mask = np.diff(user_items.tocsc().indptr) > 0
    history_len = np.diff(user_items.indptr)

    t_users = target["user_idx"].to_numpy()
    t_items = target["item_idx"].to_numpy()
    cold_user = history_len[t_users] == 0
    cold_item = ~candidate_mask[t_items]
    seen = np.asarray(user_items[t_users, t_items]).ravel() > 0
    keep = ~(cold_user | cold_item | seen)

    pairs = np.unique(np.stack([t_users[keep], t_items[keep]], axis=1), axis=0)
    eval_users, starts = np.unique(pairs[:, 0], return_index=True)
    targets = np.split(pairs[:, 1].astype(np.int64), starts[1:]) if len(pairs) else []

    stats = {
        "history_rows": len(history),
        "candidate_items": int(candidate_mask.sum()),
        "target_rows": len(target),
        "target_users": int(np.unique(t_users).size),
        "cold_user_rows": int(cold_user.sum()),
        "cold_item_rows": int((cold_item & ~cold_user).sum()),
        "seen_item_rows": int((seen & ~cold_user & ~cold_item).sum()),
        "kept_rows": int(keep.sum()),
        "eval_users": len(eval_users),
    }
    train = TrainData(history, ds.n_users, ds.n_items, cutoff, user_items)
    return EvalSplit(stage, train, eval_users.astype(np.int64), targets, candidate_mask, stats)
