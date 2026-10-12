"""One evaluation path for every model: fit -> score in batches -> mask -> top-K -> metrics."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
import pandas as pd

from vgrec.config import K
from vgrec.metrics import (
    catalog_coverage,
    hit_matrix,
    hit_rate_at_k,
    ndcg_at_k,
    personalization,
    precision_at_k,
    recall_at_k,
)
from vgrec.models.base import Recommender
from vgrec.protocol import EvalSplit


@dataclass
class EvalResult:
    model: str
    stage: str
    k: int
    per_user: pd.DataFrame  # user_idx, n_history, n_targets, hits, precision, recall, ndcg, hit_rate
    topk: np.ndarray  # (n_eval_users, k) item_idx, best first
    n_candidates: int
    fit_seconds: float
    score_seconds: float

    def summary(self) -> dict[str, float | int | str]:
        mean = self.per_user[["precision", "recall", "ndcg", "hit_rate"]].mean()
        k = self.k
        return {
            "model": self.model,
            "stage": self.stage,
            "users": len(self.per_user),
            f"precision@{k}": float(mean["precision"]),
            f"recall@{k}": float(mean["recall"]),
            f"ndcg@{k}": float(mean["ndcg"]),
            f"hit_rate@{k}": float(mean["hit_rate"]),
            "coverage": catalog_coverage(self.topk, self.n_candidates),
            "personalization": personalization(self.topk),
            "fit_s": round(self.fit_seconds, 2),
            "score_s": round(self.score_seconds, 2),
        }


def top_k(scores: np.ndarray, k: int) -> np.ndarray:
    """Indices of the k largest scores per row, sorted best-first."""
    idx = np.argpartition(-scores, k - 1, axis=1)[:, :k]
    order = np.argsort(-np.take_along_axis(scores, idx, axis=1), axis=1, kind="stable")
    return np.take_along_axis(idx, order, axis=1)


def mask_unavailable(scores: np.ndarray, users: np.ndarray, split: EvalSplit) -> np.ndarray:
    """In place: -inf for non-candidate items and for items each user already interacted with."""
    scores[:, ~split.candidate_mask] = -np.inf
    seen = split.train.user_items[users]
    scores[np.repeat(np.arange(len(users)), np.diff(seen.indptr)), seen.indices] = -np.inf
    return scores


def rank(model: Recommender, split: EvalSplit, k: int = K, batch_size: int = 1024) -> np.ndarray:
    """Top-k unseen candidate items for every eval user, shape (n_eval_users, k)."""
    out = np.empty((len(split.eval_users), k), dtype=np.int64)
    for start in range(0, len(split.eval_users), batch_size):
        users = split.eval_users[start : start + batch_size]
        scores = np.array(model.score(users), dtype=np.float32, copy=True)
        if scores.shape != (len(users), split.n_items):
            raise ValueError(f"{model.name}.score returned {scores.shape}, "
                             f"expected {(len(users), split.n_items)}")
        if np.isnan(scores).any():
            raise ValueError(f"{model.name}.score returned NaN")
        out[start : start + len(users)] = top_k(mask_unavailable(scores, users, split), k)
    return out


def evaluate(model: Recommender, split: EvalSplit, *, k: int = K, batch_size: int = 1024,
             fit: bool = True) -> EvalResult:
    """Fit `model` on split.train (unless fit=False), then score it against split.targets."""
    start = perf_counter()
    if fit:
        model.fit(split.train)
    fit_seconds = perf_counter() - start
    start = perf_counter()
    topk = rank(model, split, k=k, batch_size=batch_size)
    return result_from_topk(model.name, split, topk, fit_seconds, perf_counter() - start)


def result_from_topk(name: str, split: EvalSplit, topk: np.ndarray, fit_seconds: float = 0.0,
                     score_seconds: float = 0.0) -> EvalResult:
    """Per-user metrics for ranked lists produced by `rank` (or any equivalent loop)."""
    k = topk.shape[1]
    hits = hit_matrix(topk, split.targets)
    n_targets = np.array([len(t) for t in split.targets], dtype=np.int64)
    per_user = pd.DataFrame(
        {
            "user_idx": split.eval_users,
            "n_history": split.history_lengths,
            "n_targets": n_targets,
            "hits": hits.sum(axis=1),
            "precision": precision_at_k(hits),
            "recall": recall_at_k(hits, n_targets),
            "ndcg": ndcg_at_k(hits, n_targets),
            "hit_rate": hit_rate_at_k(hits),
        }
    )
    return EvalResult(name, split.stage, k, per_user, topk, int(split.candidate_mask.sum()),
                      fit_seconds, score_seconds)
