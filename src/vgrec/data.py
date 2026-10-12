"""Load the official split files and item metadata into an integer-indexed Dataset."""

from __future__ import annotations

import gzip
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from vgrec.config import PROCESSED_DIR, RAW_DIR, SPLITS, T1_MS, T2_MS
from vgrec.download import raw_paths

SPLIT_DTYPE = pd.CategoricalDtype(list(SPLITS), ordered=True)
RAW_COLUMNS = ["user_id", "parent_asin", "rating", "timestamp"]
# details{} keys that describe the product (not shipping/pricing noise).
DETAIL_KEYS = ("Manufacturer", "Brand", "Type of item", "Rated", "Hardware Platform", "Platform")


@dataclass(frozen=True)
class Dataset:
    """All interactions with contiguous integer ids; `split` is the official split label."""

    interactions: pd.DataFrame  # user_idx:int32, item_idx:int32, rating:float32, timestamp:int64, split
    user_ids: np.ndarray  # user_idx -> raw user_id
    item_ids: np.ndarray  # item_idx -> parent_asin

    @property
    def n_users(self) -> int:
        return len(self.user_ids)

    @property
    def n_items(self) -> int:
        return len(self.item_ids)


def load_raw_interactions(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Concatenate the three official split CSVs, tagging each row with its split."""
    paths = raw_paths(raw_dir)
    dtypes = {"user_id": str, "parent_asin": str, "rating": "float32", "timestamp": "int64"}
    frames = [pd.read_csv(paths[s], dtype=dtypes).assign(split=s) for s in SPLITS]
    df = pd.concat(frames, ignore_index=True)
    df["split"] = df["split"].astype(SPLIT_DTYPE)
    return df


def quality_report(df: pd.DataFrame) -> dict[str, int]:
    """Counts of the problems the proposal asks us to inspect (all expected to be 0 here)."""
    expected = np.select([df["timestamp"] < T1_MS, df["timestamp"] < T2_MS], ["train", "valid"], "test")
    return {
        "rows": len(df),
        "missing_values": int(df[RAW_COLUMNS].isna().any(axis=1).sum()),
        "invalid_rating": int((df["rating"].notna() & ~df["rating"].between(1, 5)).sum()),
        "duplicate_user_item": int(df.duplicated(["user_id", "parent_asin"]).sum()),
        "split_boundary_violations": int((df["split"].astype(str).to_numpy() != expected).sum()),
    }


def clean_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """Drop incomplete rows and out-of-range ratings; keep the earliest repeat of a (user, item)."""
    df = df.dropna(subset=RAW_COLUMNS)
    df = df[df["rating"].between(1, 5)]
    df = df.sort_values("timestamp", kind="stable")
    df = df.drop_duplicates(["user_id", "parent_asin"], keep="first")
    return df.reset_index(drop=True)


def build_dataset(df: pd.DataFrame) -> Dataset:
    """Map raw string ids to contiguous ints (sorted, so ids are stable across runs)."""
    user_idx, user_ids = pd.factorize(df["user_id"], sort=True)
    item_idx, item_ids = pd.factorize(df["parent_asin"], sort=True)
    interactions = pd.DataFrame(
        {
            "user_idx": user_idx.astype(np.int32),
            "item_idx": item_idx.astype(np.int32),
            "rating": df["rating"].to_numpy(dtype=np.float32),
            "timestamp": df["timestamp"].to_numpy(dtype=np.int64),
            "split": pd.Categorical(df["split"], dtype=SPLIT_DTYPE),
        }
    )
    return Dataset(interactions, np.asarray(user_ids, dtype=object), np.asarray(item_ids, dtype=object))


def _join(value, sep: str = " ") -> str:
    """Flatten the metadata's str / list / None fields into one string."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return sep.join(str(v) for v in value if v)


def _slug(prefix: str, value: str) -> str:
    return prefix + re.sub(r"[^0-9a-z]+", "_", value.lower()).strip("_")


def load_metadata(path: Path, item_ids: np.ndarray) -> pd.DataFrame:
    """Read the metadata JSONL; return one row per item, aligned to item_idx (missing -> blanks)."""
    wanted = set(item_ids)
    rows: dict[str, dict] = {}
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            asin = rec.get("parent_asin")
            if asin not in wanted:
                continue
            details = rec.get("details") or {}
            rows[asin] = {
                "title": _join(rec.get("title")),
                "store": _join(rec.get("store")),
                "main_category": _join(rec.get("main_category")),
                "categories": _join(rec.get("categories"), " | "),
                "features": _join(rec.get("features"), "\n"),
                "description": _join(rec.get("description"), "\n"),
                "details": " | ".join(f"{k}: {details[k]}" for k in DETAIL_KEYS if details.get(k)),
                "average_rating": rec.get("average_rating"),
                "rating_number": rec.get("rating_number"),
                "price": rec.get("price"),
            }
    text_cols = ["title", "store", "main_category", "categories", "features", "description", "details"]
    columns = text_cols + ["average_rating", "rating_number", "price"]
    meta = pd.DataFrame.from_dict(rows, orient="index", columns=columns).reindex(list(item_ids))
    meta["has_metadata"] = meta.index.isin(list(rows))
    meta[text_cols] = meta[text_cols].fillna("")
    for col in ("average_rating", "rating_number", "price"):
        meta[col] = pd.to_numeric(meta[col], errors="coerce")
    meta.index.name = "parent_asin"
    return meta.reset_index()


def build_item_text(meta: pd.DataFrame) -> pd.Series:
    """One TF-IDF document per item. Category/store slugs keep 'PlayStation 4' distinct from '5'."""

    def doc(row) -> str:
        cats = [c for c in row["categories"].split(" | ") if c]
        tokens = [_slug("cat_", c) for c in cats]
        if row["store"]:
            tokens.append(_slug("store_", row["store"]))
        parts = [row["title"], row["store"], " ".join(cats), " ".join(tokens), row["details"],
                 row["features"], row["description"]]
        return "\n".join(p for p in parts if p)

    return meta.apply(doc, axis=1).rename("item_text")


def prepare(raw_dir: Path = RAW_DIR, out_dir: Path = PROCESSED_DIR) -> dict[str, int]:
    """raw files -> cleaned, indexed parquet files in data/processed. Returns the quality report."""
    raw = load_raw_interactions(raw_dir)
    report = quality_report(raw)
    ds = build_dataset(clean_interactions(raw))
    meta = load_metadata(raw_paths(raw_dir)["meta"], ds.item_ids)
    meta["item_text"] = build_item_text(meta)
    out_dir.mkdir(parents=True, exist_ok=True)
    ds.interactions.to_parquet(out_dir / "interactions.parquet", index=False)
    pd.DataFrame({"user_id": ds.user_ids.astype(str)}).to_parquet(out_dir / "users.parquet", index=False)
    meta.to_parquet(out_dir / "items.parquet", index=False)
    report["rows_after_cleaning"] = len(ds.interactions)
    report["items_missing_metadata"] = int((~meta["has_metadata"]).sum())
    return report


def load_dataset(processed_dir: Path = PROCESSED_DIR) -> Dataset:
    """Load what `prepare` wrote. Run `uv run vgrec prepare` first."""
    interactions = pd.read_parquet(processed_dir / "interactions.parquet")
    interactions["split"] = interactions["split"].astype(SPLIT_DTYPE)
    users = pd.read_parquet(processed_dir / "users.parquet")["user_id"].to_numpy(dtype=object)
    items = pd.read_parquet(processed_dir / "items.parquet", columns=["parent_asin"])
    return Dataset(interactions, users, items["parent_asin"].to_numpy(dtype=object))


def load_items(processed_dir: Path = PROCESSED_DIR) -> pd.DataFrame:
    """Item metadata + `item_text`, row i = item_idx i."""
    return pd.read_parquet(processed_dir / "items.parquet")
