"""
Step 1 — ECG preparation (spec §4.2).

Input : raw signal (n_samples x 12 leads, mV) + sampling rate (+ optional PTB-XL quality flags)
Output: ProcessedECG = cleaned/normalised signal, beat segments, R-peaks and a short
        processing report (checks, parameters, quality score) — JSON-serialisable.

Pipeline:
  1. format check          (12 leads, expected length, expected fs)
  2. missing/invalid values (NaN/inf -> interpolate if few, fail if many; flat leads; extreme amplitudes)
  3. denoising             (Butterworth band-pass 0.5-40 Hz, zero-phase -> removes baseline drift + HF noise)
  4. R-peak detection      (multi-lead derivative envelope) -> heart rate, beats
  5. normalisation         (per-lead z-score)
  6. segmentation          (one window per beat, 0.25 s before / 0.45 s after the R peak)
  7. model-ready format    (float32 arrays)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
from scipy.signal import butter, find_peaks, sosfiltfilt

PREPROCESS_VERSION = "preprocess-1.1.0"   # 1.1.0: report keeps per-lead amplitude (lead_std_mv)

PARAMS: dict[str, Any] = {
    "expected_fs": 100,
    "expected_n_samples": 1000,
    "n_leads": 12,
    "bandpass_hz": [0.5, 40.0],
    "filter_order": 4,
    "max_nan_ratio": 0.05,          # above this the record is rejected
    "flat_lead_std_mv": 0.01,
    "extreme_amplitude_mv": 5.0,
    "hr_range_bpm": [30, 200],
    "min_beats": 4,
    "beat_window_s": [0.25, 0.45],  # before / after R peak
    "rpeak_lead": 1,                # lead II
}

# penalty per problem, used to build a 0..1 quality score (used by the validation module)
PENALTIES = {
    "nan_interpolated": 0.10,
    "flat_lead": 0.15,              # per flat lead
    "extreme_amplitude": 0.20,
    "hr_implausible": 0.30,
    "too_few_beats": 0.30,
    # PTB-XL annotator flags
    "baseline_drift": 0.05,
    "static_noise": 0.10,
    "burst_noise": 0.15,
    "electrodes_problems": 0.20,
}


class PreprocessingError(Exception):
    """Raised when the record cannot be processed at all (-> validation: REJECTED)."""


@dataclass
class ProcessedECG:
    signal: np.ndarray            # (n_samples, 12) float32, filtered + z-scored
    beats: np.ndarray             # (n_beats, win, 12) float32
    r_peaks: np.ndarray           # (n_beats_detected,) int, sample indices
    report: dict = field(default_factory=dict)   # processing report (JSON-serialisable)


# ---------------------------------------------------------------------------
def _bandpass(x: np.ndarray, fs: float) -> np.ndarray:
    lo, hi = PARAMS["bandpass_hz"]
    hi = min(hi, 0.45 * fs)   # stay below Nyquist
    sos = butter(PARAMS["filter_order"], [lo, hi], btype="bandpass", fs=fs, output="sos")
    return sosfiltfilt(sos, x, axis=0)


def _interpolate_nans(x: np.ndarray) -> np.ndarray:
    x = x.copy()
    idx = np.arange(x.shape[0])
    for j in range(x.shape[1]):
        bad = ~np.isfinite(x[:, j])
        if bad.all():
            x[:, j] = 0.0
        elif bad.any():
            x[bad, j] = np.interp(idx[bad], idx[~bad], x[~bad, j])
    return x


def detect_r_peaks(x_filt: np.ndarray, fs: float) -> np.ndarray:
    """Multi-lead envelope: robust even when R waves are negative in some leads."""
    env = np.abs(np.diff(x_filt, axis=0)).sum(axis=1)
    win = max(1, int(0.05 * fs))
    env = np.convolve(env, np.ones(win) / win, mode="same")
    height = 0.35 * np.percentile(env, 99)
    peaks, _ = find_peaks(env, height=height, distance=int(0.3 * fs))  # <= 200 bpm

    # refine: the R peak is the max |amplitude| on the reference lead near the envelope peak
    ref = np.abs(x_filt[:, PARAMS["rpeak_lead"]])
    r = int(0.06 * fs)
    refined = [max(0, p - r) + int(np.argmax(ref[max(0, p - r): p + r + 1])) for p in peaks]
    return np.unique(np.array(refined, dtype=int))


# ---------------------------------------------------------------------------
def preprocess_ecg(
    raw: np.ndarray,
    fs: float,
    quality_flags: Optional[dict[str, Any]] = None,
) -> ProcessedECG:
    checks: dict[str, Any] = {}
    warnings: list[str] = []
    penalty = 0.0

    # 1. format -------------------------------------------------------------
    raw = np.asarray(raw, dtype=np.float64)
    if raw.ndim != 2 or raw.shape[1] != PARAMS["n_leads"]:
        raise PreprocessingError(f"expected (n_samples, 12), got {raw.shape}")
    if fs != PARAMS["expected_fs"]:
        raise PreprocessingError(f"expected fs={PARAMS['expected_fs']} Hz, got {fs}")
    checks["shape"] = list(raw.shape)
    if raw.shape[0] != PARAMS["expected_n_samples"]:
        warnings.append(f"unexpected length {raw.shape[0]} samples")

    # 2. missing / invalid values ---------------------------------------------
    nan_ratio = float((~np.isfinite(raw)).mean())
    checks["nan_ratio"] = round(nan_ratio, 4)
    if nan_ratio > PARAMS["max_nan_ratio"]:
        raise PreprocessingError(f"too many missing values ({nan_ratio:.1%})")
    if nan_ratio > 0:
        raw = _interpolate_nans(raw)
        penalty += PENALTIES["nan_interpolated"]
        warnings.append("missing values interpolated")

    flat = [int(j) for j in np.where(raw.std(axis=0) < PARAMS["flat_lead_std_mv"])[0]]
    checks["flat_leads"] = flat
    if len(flat) == PARAMS["n_leads"]:
        raise PreprocessingError("all leads are flat")
    penalty += PENALTIES["flat_lead"] * len(flat)

    extreme = float((np.abs(raw) > PARAMS["extreme_amplitude_mv"]).mean())
    checks["extreme_amplitude_ratio"] = round(extreme, 4)
    if extreme > 0.01:
        penalty += PENALTIES["extreme_amplitude"]
        warnings.append("extreme amplitudes (>5 mV)")

    # 3. denoising ------------------------------------------------------------
    filt = _bandpass(raw, fs)

    # 4. R peaks / heart rate -------------------------------------------------
    r_peaks = detect_r_peaks(filt, fs)
    hr = float(60.0 * fs / np.mean(np.diff(r_peaks))) if len(r_peaks) > 1 else 0.0
    checks["n_r_peaks"] = int(len(r_peaks))
    checks["heart_rate_bpm"] = round(hr, 1)
    lo, hi = PARAMS["hr_range_bpm"]
    if not lo <= hr <= hi:
        penalty += PENALTIES["hr_implausible"]
        warnings.append(f"implausible heart rate ({hr:.0f} bpm)")

    # 5. normalisation (per-lead z-score; flat leads stay at 0) --------------
    mu = filt.mean(axis=0)
    sd = filt.std(axis=0)
    # keep the original scale: amplitude matters for diagnosis (e.g. hypertrophy)
    checks["lead_std_mv"] = [round(float(s), 4) for s in sd]
    sd[sd < 1e-6] = 1.0
    norm = ((filt - mu) / sd).astype(np.float32)

    # 6. segmentation ---------------------------------------------------------
    pre, post = (int(s * fs) for s in PARAMS["beat_window_s"])
    valid = [p for p in r_peaks if p - pre >= 0 and p + post <= len(norm)]
    beats = (np.stack([norm[p - pre: p + post] for p in valid]).astype(np.float32)
             if valid else np.zeros((0, pre + post, PARAMS["n_leads"]), np.float32))
    checks["n_beats"] = int(len(beats))
    if len(beats) < PARAMS["min_beats"]:
        penalty += PENALTIES["too_few_beats"]
        warnings.append(f"only {len(beats)} complete beats")

    # PTB-XL annotator flags (optional) --------------------------------------
    flags_present = []
    for k, v in (quality_flags or {}).items():
        if k in PENALTIES and v is not None and not (isinstance(v, float) and np.isnan(v)):
            flags_present.append(k)
            penalty += PENALTIES[k]
    checks["dataset_quality_flags"] = flags_present

    quality = round(max(0.0, 1.0 - penalty), 3)
    report = {
        "software_version": PREPROCESS_VERSION,
        "params": PARAMS,
        "status": "OK",
        "checks": checks,
        "warnings": warnings,
        "quality_score": quality,
        "output": {"signal_shape": list(norm.shape), "beats_shape": list(beats.shape)},
    }
    return ProcessedECG(signal=norm, beats=beats, r_peaks=r_peaks, report=report)


# ---------------------------------------------------------------------------
def prepare_report(text: str) -> dict:
    """Light clean-up of the medical report (content analysis happens in validation)."""
    clean = " ".join(str(text).replace("\r", " ").split()).lower()
    de_hits = sum(w in clean for w in ("sinusrhythmus", "lagetyp", "normales", "abnorm", "infarkt", "linksschenkel"))
    en_hits = sum(w in clean for w in ("sinus rhythm", "normal ecg", "abnormal", "infarct", "block", "wave"))
    return {
        "text_clean": clean,
        "language": "de" if de_hits > en_hits else "en",
        "n_chars": len(clean),
        "empty": len(clean) == 0,
    }