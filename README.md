# IE7275 Group 5: Hybrid Recommender for Amazon Video Games

Top-K (K = 10) personalized recommendation on **Amazon Reviews 2023, Video_Games 5-core**
(94,762 users · 25,612 games · 814,586 interactions) using the official absolute-timestamp
train/valid/test split. We compare a popularity baseline, implicit ALS, Neural Collaborative
Filtering (NeuMF), a TF-IDF content model, and CF + content hybrids on Precision@10,
Recall@10 and NDCG@10.

Team: Liam Campbell · Harshitkumar Brahmbhatt · Hemanth Karthik Devakivada

## Quickstart

Needs [uv](https://docs.astral.sh/uv/) (it installs Python 3.12 and every dependency).

```bash
uv sync                                 # create .venv from pyproject.toml / uv.lock
uv run vgrec download                   # ~117 MB into data/raw/ (splits + metadata)
uv run vgrec prepare                    # -> data/processed/*.parquet, prints a quality report
uv run vgrec stats                      # split sizes, cold-start counts, history buckets
uv run vgrec run --model popularity     # tune on valid, refit, evaluate on test
uv run vgrec report                     # results/summary.csv + by-history + significance
```

`make all` runs the whole pipeline once every model is implemented (`make help` lists targets).
Without `make` (e.g. Windows), run the commands above directly.

Prefer pip? `uv export --no-hashes --no-dev > requirements.txt`, then
`pip install -r requirements.txt && pip install -e .`

## What works today

| Piece | Status |
|---|---|
| Download, cleaning, id mapping, item text (`data.py`) | done |
| Leakage-safe valid/test splits (`protocol.py`) | done |
| Metrics + batched full-ranking evaluator (`metrics.py`, `evaluate.py`) | done |
| Grid search → refit → test runner, report tables (`experiment.py`) | done |
| Popularity baseline (`models/popularity.py`) | done: test NDCG@10 = 0.0149 |
| ALS, content TF-IDF, NeuMF, hybrids (`models/*.py`, `run_hybrid`) | **stubs: to implement next** |

## Layout

```
configs/experiment.toml   hyperparameter grids per model (searched on the valid stage)
src/vgrec/
  config.py               paths, URLs, split cutoffs T1/T2, K, SEED
  download.py data.py     raw files -> data/processed/{interactions,users,items}.parquet
  protocol.py             make_split(ds, "valid"|"test") -> EvalSplit(train, targets, ...)
  metrics.py evaluate.py  P/R/NDCG/HR@K, coverage, personalization; one ranking path for all
  analysis.py             metrics by history length, paired bootstrap significance
  experiment.py cli.py    `vgrec run` / `vgrec report`
  models/                 base.Recommender contract + one file per model
notebooks/                01_eda.ipynb, 02_results.ipynb (import from vgrec)
results/<model>/          valid_grid.csv, {valid,test}_summary.json, per-user parquet
```

## Evaluation protocol (one paragraph)

Tune on the **valid stage** (history = train, targets = valid interactions after T1 =
2021-08-11), then refit the best config on train+valid and evaluate **once** on the
**test stage** (targets after T2 = 2022-07-16). Every model ranks *all* items seen in the
history (full ranking, no negative sampling), minus the items the user already interacted
with. Target interactions by users with no history, or on items never seen in the history,
cannot be ranked by any model and are dropped (and counted in `vgrec stats`: under the test
stage that is 32% and a further 24% of target rows).

## Data source

Amazon Reviews 2023, McAuley Lab (UCSD): <https://amazon-reviews-2023.github.io/>.
Cite it as the dataset page asks.
