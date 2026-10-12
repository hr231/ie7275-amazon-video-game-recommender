"""Download the official split files and item metadata from the McAuley Lab server."""

from __future__ import annotations

import urllib.request
from pathlib import Path

from tqdm import tqdm

from vgrec.config import CATEGORY, META_URL, RAW_DIR, SPLIT_URLS


def download(url: str, dest: Path, *, force: bool = False, timeout: float = 60.0) -> Path:
    """Stream `url` to `dest` via a .part file; skip if `dest` already exists."""
    if dest.exists() and not force:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    with urllib.request.urlopen(url, timeout=timeout) as resp, open(tmp, "wb") as out:
        total = int(resp.headers.get("Content-Length") or 0)
        with tqdm(total=total or None, unit="B", unit_scale=True, desc=dest.name) as bar:
            while chunk := resp.read(1 << 20):
                out.write(chunk)
                bar.update(len(chunk))
    if total and tmp.stat().st_size != total:
        tmp.unlink()
        raise OSError(f"Truncated download of {url}: expected {total} bytes")
    tmp.replace(dest)
    return dest


def raw_paths(raw_dir: Path = RAW_DIR) -> dict[str, Path]:
    """Local paths of every raw file, keyed by 'train' / 'valid' / 'test' / 'meta'."""
    paths = {split: raw_dir / f"{CATEGORY}.{split}.csv.gz" for split in SPLIT_URLS}
    paths["meta"] = raw_dir / f"meta_{CATEGORY}.jsonl.gz"
    return paths


def download_all(raw_dir: Path = RAW_DIR, *, include_meta: bool = True, force: bool = False):
    """Fetch the three split CSVs (~13 MB) and, optionally, the metadata (~103 MB)."""
    paths = raw_paths(raw_dir)
    urls = dict(SPLIT_URLS, meta=META_URL)
    for name, url in urls.items():
        if name == "meta" and not include_meta:
            continue
        download(url, paths[name], force=force)
    return paths
