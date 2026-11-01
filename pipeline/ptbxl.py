"""PTB-XL loading helpers (metadata, labels, signals)."""
from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb

DATA_DIR = Path("data/raw/ptbxl")
SUPERCLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]
QUALITY_COLS = ["baseline_drift", "static_noise", "burst_noise",
                "electrodes_problems", "extra_beats", "pacemaker"]


def load_metadata(data_dir: Path = DATA_DIR, only_downloaded: bool = True) -> pd.DataFrame:
    df = pd.read_csv(data_dir / "ptbxl_database.csv", index_col="ecg_id")
    df["scp_codes"] = df["scp_codes"].apply(ast.literal_eval)

    scp = pd.read_csv(data_dir / "scp_statements.csv", index_col=0)
    diag = scp[scp["diagnostic"] == 1]["diagnostic_class"]

    def superclasses(codes: dict) -> list[str]:
        return sorted({diag[c] for c in codes if c in diag.index})

    df["superclasses"] = df["scp_codes"].apply(superclasses)
    df["n_quality_issues"] = df[QUALITY_COLS].notna().sum(axis=1)

    if only_downloaded:
        mask = df["filename_lr"].apply(lambda p: (data_dir / f"{p}.dat").exists())
        df = df[mask]
    return df


def load_signal(row: pd.Series, data_dir: Path = DATA_DIR) -> tuple[np.ndarray, dict]:
    """Return (signal [1000 samples x 12 leads], meta) for a 100 Hz record."""
    rec = wfdb.rdrecord(str(data_dir / row["filename_lr"]))
    return rec.p_signal.astype(np.float32), {"fs": rec.fs, "leads": rec.sig_name}
