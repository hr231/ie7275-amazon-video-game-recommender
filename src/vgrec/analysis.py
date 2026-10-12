"""Result analysis: metrics by history length and paired significance tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from vgrec.config import SEED

# Buckets chosen from the eval-user history-length distribution (median 5, p10 = 2).
HISTORY_BINS = [0, 2, 4, 9, 19, np.inf]
HISTORY_LABELS = ["1-2", "3-4", "5-9", "10-19", "20+"]


def history_bucket(n_history: pd.Series | np.ndarray) -> pd.Categorical:
    return pd.cut(np.asarray(n_history), HISTORY_BINS, labels=HISTORY_LABELS)


def by_history_bucket(per_user: pd.DataFrame, metric: str = "ndcg") -> pd.DataFrame:
    """Mean `metric` and user count per history-length bucket."""
    buckets = history_bucket(per_user["n_history"])
    return per_user.groupby(buckets, observed=False)[metric].agg(["mean", "count"])


def paired_bootstrap(a: np.ndarray, b: np.ndarray, *, n_boot: int = 2000,
                     seed: int = SEED) -> dict[str, float]:
    """Mean of (a - b) over the same users, its 95% bootstrap CI, and a paired t-test p-value."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"paired arrays differ in shape: {a.shape} vs {b.shape}")
    diff = a - b
    rng = np.random.default_rng(seed)
    means = np.concatenate([
        diff[rng.integers(0, len(diff), size=(chunk, len(diff)))].mean(axis=1)
        for chunk in np.diff(np.r_[0:n_boot:200, n_boot])
    ])
    low, high = np.percentile(means, [2.5, 97.5])
    p_value = 1.0 if np.allclose(diff, 0) else float(stats.ttest_rel(a, b).pvalue)
    return {"mean_diff": float(diff.mean()), "ci_low": float(low), "ci_high": float(high),
            "p_value": p_value}
