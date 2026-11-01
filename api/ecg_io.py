"""Read uploaded ECG files into a (n_samples, 12) array at 100 Hz."""
from __future__ import annotations

import io
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import resample_poly

TARGET_FS = 100
LEADS = ["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]


class ECGFormatError(ValueError):
    pass


def _orient(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2:
        raise ECGFormatError(f"expected a 2-D signal, got shape {x.shape}")
    if x.shape[1] != 12 and x.shape[0] == 12:
        x = x.T
    if x.shape[1] != 12:
        raise ECGFormatError(f"expected 12 leads, got shape {x.shape}")
    return x


def read_csv(data: bytes) -> np.ndarray:
    df = pd.read_csv(io.BytesIO(data))
    if all(str(c).replace(".", "").replace("-", "").isdigit() for c in df.columns):   # no header row
        df = pd.read_csv(io.BytesIO(data), header=None)
    df = df.loc[:, [c for c in df.columns if str(c).lower() not in ("time", "t", "time_s", "index")]]
    return _orient(df.to_numpy())


def read_npy(data: bytes) -> np.ndarray:
    return _orient(np.load(io.BytesIO(data), allow_pickle=False))


def read_wfdb(hea: bytes, dat: bytes, hea_name: str) -> tuple[np.ndarray, float]:
    import wfdb
    stem = Path(hea_name).stem
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / f"{stem}.hea").write_bytes(hea)
        header_text = hea.decode("utf-8", errors="replace").splitlines()
        dat_name = header_text[1].split()[0] if len(header_text) > 1 else f"{stem}.dat"
        (Path(d) / Path(dat_name).name).write_bytes(dat)
        rec = wfdb.rdrecord(str(Path(d) / stem))
    return _orient(rec.p_signal), float(rec.fs)


def to_target_fs(x: np.ndarray, fs: float) -> tuple[np.ndarray, list[str]]:
    notes = []
    if int(round(fs)) != TARGET_FS:
        fs_i = int(round(fs))
        g = np.gcd(fs_i, TARGET_FS)
        x = resample_poly(x, TARGET_FS // g, fs_i // g, axis=0)
        notes.append(f"resampled from {fs_i} Hz to {TARGET_FS} Hz")
    return x, notes


def load_upload(files: dict[str, tuple[str, bytes]], fs: float = TARGET_FS) -> tuple[np.ndarray, float, list[str]]:
    """files: {kind: (filename, bytes)} where kind in ecg | hea | dat. Returns (signal, fs, notes)."""
    if "hea" in files and "dat" in files:
        x, fs = read_wfdb(files["hea"][1], files["dat"][1], files["hea"][0])
    elif "ecg" in files:
        name, data = files["ecg"]
        ext = Path(name).suffix.lower()
        if ext == ".csv" or ext == ".txt":
            x = read_csv(data)
        elif ext == ".npy":
            x = read_npy(data)
        else:
            raise ECGFormatError(f"unsupported ECG file type '{ext}' (use .csv, .npy or .hea + .dat)")
    else:
        raise ECGFormatError("no ECG file received")
    x, notes = to_target_fs(x, fs)
    return x, float(TARGET_FS), notes