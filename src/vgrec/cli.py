"""Command-line entry point: `uv run vgrec <command>`."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from vgrec import download
from vgrec.config import CONFIG_PATH
from vgrec.models import MODEL_NAMES


def _stats() -> None:
    from vgrec.analysis import history_bucket
    from vgrec.data import load_dataset
    from vgrec.protocol import make_split

    ds = load_dataset()
    print(f"dataset: {ds.n_users:,} users, {ds.n_items:,} items, {len(ds.interactions):,} rows")
    print(ds.interactions.groupby("split", observed=True).agg(
        rows=("user_idx", "size"), users=("user_idx", "nunique"), items=("item_idx", "nunique")))
    for stage in ("valid", "test"):
        split = make_split(ds, stage)
        print(f"\n[{stage}] " + ", ".join(f"{k}={v:,}" for k, v in split.stats.items()))
        counts = pd.Series(history_bucket(split.history_lengths)).value_counts(sort=False)
        print("  eval users by history length:", counts.to_dict())


def _run(model: str, config: Path, max_configs: int | None) -> None:
    from vgrec.data import load_dataset, load_items
    from vgrec.experiment import load_config, run_hybrid, run_model
    from vgrec.protocol import make_split

    cfg = load_config(config)
    ds = load_dataset()
    splits = (make_split(ds, "valid"), make_split(ds, "test"))
    is_hybrid = model.startswith("hybrid")
    item_text = load_items()["item_text"].tolist() if is_hybrid or model == "content" else None
    print(f"== {model}: tuning on valid ({splits[0].stats['eval_users']:,} users), "
          f"then test ({splits[1].stats['eval_users']:,} users)")
    if is_hybrid:
        summary = run_hybrid(model, ds, cfg, item_text=item_text, splits=splits)
    else:
        summary = run_model(model, ds, cfg, item_text=item_text, splits=splits,
                            max_configs=max_configs)
    print("TEST:", {k: v for k, v in summary.items() if k != "params"})
    print("best params:", summary["params"])


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="vgrec", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("download", help="fetch the official split files + metadata")
    p.add_argument("--no-meta", action="store_true", help="skip the 103 MB metadata file")
    p.add_argument("--force", action="store_true", help="re-download existing files")

    sub.add_parser("prepare", help="raw files -> data/processed/*.parquet (+ quality report)")
    sub.add_parser("stats", help="print split / cold-start statistics")

    p = sub.add_parser("run", help="tune on valid, refit, evaluate on test, save to results/")
    p.add_argument("--model", required=True, choices=MODEL_NAMES)
    p.add_argument("--config", type=Path, default=CONFIG_PATH)
    p.add_argument("--max-configs", type=int, default=None, help="only the first N grid points")

    sub.add_parser("report", help="aggregate results/ into summary tables")

    args = parser.parse_args(argv)
    if args.command == "download":
        paths = download.download_all(include_meta=not args.no_meta, force=args.force)
        for name, path in paths.items():
            print(f"{name:>5}: {path}")
    elif args.command == "prepare":
        from vgrec.data import prepare

        for key, value in prepare().items():
            print(f"{key:>26}: {value:,}")
    elif args.command == "stats":
        _stats()
    elif args.command == "run":
        try:
            _run(args.model, args.config, args.max_configs)
        except NotImplementedError as exc:
            raise SystemExit(f"{args.model}: {exc}") from None
    elif args.command == "report":
        from vgrec.experiment import report

        with pd.option_context("display.width", 160, "display.max_columns", 20):
            for name, table in report().items():
                print(f"\n== {name}\n{table.to_string()}")


if __name__ == "__main__":
    main()
