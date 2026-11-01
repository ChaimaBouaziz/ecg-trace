"""
Download PTB-XL (v1.0.3) from PhysioNet.

Two modes:
  --mode subset  (default) : metadata CSVs + N records at 100 Hz -> fast start
  --mode full              : the official zip (~1.7 GB), all 21,799 records

Usage (from the repo root, inside WSL):
  python scripts/download_ptbxl.py --n 2000
  python scripts/download_ptbxl.py --mode full
"""
from __future__ import annotations

import argparse
import sys
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

BASE = "https://physionet.org/files/ptb-xl/1.0.3/"
ZIP_URL = ("https://physionet.org/static/published-projects/ptb-xl/"
           "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3.zip")
OUT = Path("data/raw/ptbxl")


def fetch(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.rename(dest)


def download_subset(n: int, workers: int) -> None:
    for name in ("ptbxl_database.csv", "scp_statements.csv"):
        print(f"-> {name}")
        fetch(BASE + name, OUT / name)

    df = pd.read_csv(OUT / "ptbxl_database.csv", index_col="ecg_id")
    # stratified-ish sample: take records from every fold so train/test stay balanced
    k = min(max(1, n // 10), int(df["strat_fold"].value_counts().min()))
    sample = df.groupby("strat_fold").sample(n=k, random_state=42)
    files = [f"{p}.{ext}" for p in sample["filename_lr"] for ext in ("hea", "dat")]
    print(f"-> {len(sample)} records ({len(files)} files) at 100 Hz")

    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fetch, BASE + f, OUT / f): f for f in files}
        for fut in as_completed(futures):
            fut.result()
            done += 1
            if done % 200 == 0 or done == len(files):
                print(f"   {done}/{len(files)}", flush=True)
    sample.index.to_series().to_csv(OUT / "subset_ids.csv", index=False)


def download_full() -> None:
    zpath = OUT.parent / "ptbxl.zip"
    print("-> downloading full zip (~1.7 GB), this takes a while...")
    fetch(ZIP_URL, zpath)
    with zipfile.ZipFile(zpath) as z:
        root = z.namelist()[0].split("/")[0]
        z.extractall(OUT.parent)
    (OUT.parent / root).rename(OUT)
    zpath.unlink()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["subset", "full"], default="subset")
    ap.add_argument("--n", type=int, default=2000, help="records in subset mode")
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    try:
        download_subset(a.n, a.workers) if a.mode == "subset" else download_full()
    except Exception as e:  # network issues are the usual culprit
        sys.exit(f"Download failed: {e}")
    print(f"Done -> {OUT.resolve()}")
