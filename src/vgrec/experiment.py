"""Validation grid search -> refit best config on train+valid -> one test evaluation.

Everything lands in results/<model>/: valid_grid.csv, {valid,test}_summary.json,
{valid,test}_per_user.parquet, {valid,test}_topk.npy. `report` aggregates across models.
"""

from __future__ import annotations

import itertools
import json
import tomllib
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from vgrec.analysis import by_history_bucket, paired_bootstrap
from vgrec.config import CONFIG_PATH, RESULTS_DIR
from vgrec.data import Dataset
from vgrec.evaluate import EvalResult, evaluate
from vgrec.models import build_model
from vgrec.protocol import EvalSplit, make_split


def load_config(path: Path = CONFIG_PATH) -> dict:
    with open(path, "rb") as f:
        return tomllib.load(f)


def expand_grid(grid: dict[str, list]) -> list[dict]:
    """Cartesian product of a {param: [values]} grid, in a stable order."""
    keys = list(grid)
    return [dict(zip(keys, combo, strict=True)) for combo in itertools.product(*grid.values())]


def _jsonable(obj):
    return obj.item() if hasattr(obj, "item") else str(obj)


def save_result(result: EvalResult, out_dir: Path, params: dict) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {**result.summary(), "params": params}
    result.per_user.to_parquet(out_dir / f"{result.stage}_per_user.parquet", index=False)
    np.save(out_dir / f"{result.stage}_topk.npy", result.topk)
    (out_dir / f"{result.stage}_summary.json").write_text(
        json.dumps(summary, indent=2, default=_jsonable))
    return summary


def _print_row(params: dict, summary: dict, metric: str) -> None:
    print(f"  {params} -> {metric}={summary[metric]:.4f} "
          f"(fit {summary['fit_s']}s, score {summary['score_s']}s)", flush=True)


def run_model(name: str, ds: Dataset, cfg: dict, *, item_text: Sequence[str] | None = None,
              splits: tuple[EvalSplit, EvalSplit] | None = None, out_dir: Path = RESULTS_DIR,
              max_configs: int | None = None) -> dict:
    """Grid-search `name` on the valid stage, refit the winner on the test stage, save both."""
    k, batch_size = cfg["eval"]["k"], cfg["eval"]["batch_size"]
    metric = f"ndcg@{k}"
    valid, test = splits or (make_split(ds, "valid"), make_split(ds, "test"))
    rows, best = [], None
    for params in expand_grid(cfg[name]["grid"])[:max_configs]:
        result = evaluate(build_model(name, params, item_text=item_text), valid, k=k,
                          batch_size=batch_size)
        summary = result.summary()
        rows.append({**params, **summary})
        _print_row(params, summary, metric)
        if best is None or summary[metric] > best[0]:
            best = (summary[metric], params, result)
    _, best_params, best_valid = best
    model_dir = out_dir / name
    save_result(best_valid, model_dir, best_params)
    pd.DataFrame(rows).to_csv(model_dir / "valid_grid.csv", index=False)
    test_result = evaluate(build_model(name, best_params, item_text=item_text), test, k=k,
                           batch_size=batch_size)
    return save_result(test_result, model_dir, best_params)


def run_hybrid(name: str, ds: Dataset, cfg: dict, *, item_text: Sequence[str],
               splits: tuple[EvalSplit, EvalSplit] | None = None,
               out_dir: Path = RESULTS_DIR) -> dict:
    """Blend the already-tuned component models (cfg[name]) and tune the blend weights on valid."""
    raise NotImplementedError("experiment.run_hybrid is not implemented yet")


def report(out_dir: Path = RESULTS_DIR, cfg: dict | None = None) -> dict[str, pd.DataFrame]:
    """Aggregate every results/<model>/ into summary / by-history / significance CSVs."""
    cfg = cfg or load_config()
    summaries = [json.loads(p.read_text()) for p in sorted(out_dir.glob("*/*_summary.json"))]
    if not summaries:
        raise FileNotFoundError(f"no results in {out_dir}; run `vgrec run --model popularity` first")
    summary = pd.DataFrame(summaries)
    summary["params"] = summary["params"].map(json.dumps)
    summary = summary.sort_values(["stage", "model"]).reset_index(drop=True)

    per_user = {p.parent.name: pd.read_parquet(p) for p in sorted(out_dir.glob("*/test_per_user.parquet"))}
    buckets = pd.DataFrame({m: by_history_bucket(df)["mean"] for m, df in per_user.items()})
    if per_user:
        buckets.insert(0, "users", by_history_bucket(next(iter(per_user.values())))["count"])

    tests = []
    for a, b in cfg.get("report", {}).get("compare", []):
        if a in per_user and b in per_user:
            if not np.array_equal(per_user[a]["user_idx"], per_user[b]["user_idx"]):
                raise ValueError(f"{a} and {b} were evaluated on different users")
            tests.append({"a": a, "b": b, "metric": "ndcg",
                          **paired_bootstrap(per_user[a]["ndcg"], per_user[b]["ndcg"])})
    significance = pd.DataFrame(tests, columns=["a", "b", "metric", "mean_diff", "ci_low",
                                                "ci_high", "p_value"])

    summary.to_csv(out_dir / "summary.csv", index=False)
    buckets.to_csv(out_dir / "ndcg_by_history.csv")
    significance.to_csv(out_dir / "significance.csv", index=False)
    return {"summary": summary, "ndcg_by_history": buckets, "significance": significance}
