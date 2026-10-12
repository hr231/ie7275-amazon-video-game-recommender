"""Project-wide constants: paths, download URLs, and the official split boundaries."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("VGREC_DATA_DIR", PROJECT_ROOT / "data"))
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = Path(os.environ.get("VGREC_RESULTS_DIR", PROJECT_ROOT / "results"))
CONFIG_PATH = PROJECT_ROOT / "configs" / "experiment.toml"

CATEGORY = "Video_Games"
BASE_URL = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023"
SPLITS = ("train", "valid", "test")
SPLIT_URLS = {
    split: f"{BASE_URL}/benchmark/5core/timestamp/{CATEGORY}.{split}.csv.gz" for split in SPLITS
}
META_URL = f"{BASE_URL}/raw/meta_categories/meta_{CATEGORY}.jsonl.gz"

# Official absolute-timestamp split (ms since epoch), shared by every category:
# train = (-inf, T1), valid = [T1, T2), test = [T2, +inf).
T1_MS = 1_628_643_414_042  # 2021-08-11 00:56:54 UTC
T2_MS = 1_658_002_729_837  # 2022-07-16 20:18:49 UTC
MS_PER_DAY = 86_400_000

K = 10
SEED = 42
