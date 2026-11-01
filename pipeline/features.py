"""
Step 2a — Feature extraction (artifact FEATURES, parent = PROCESSED).

From a ProcessedECG we build a fixed-length vector:
  - median beat of each lead (robust "average heartbeat"), downsampled x2  -> 35 x 12 = 420
  - per-lead amplitude in mV before normalisation (lead_std_mv)          -> 12
  - rhythm: heart rate, RR std, RR coefficient of variation, n_beats     -> 4
Total: 436 features, always in the same order (FEATURE_NAMES).
"""
from __future__ import annotations

import numpy as np

from pipeline.preprocess import ProcessedECG

FEATURES_VERSION = "features-1.0.0"
LEADS = ["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]
BEAT_LEN, STEP = 70, 2

FEATURE_NAMES: list[str] = (
    [f"beat_{lead}_{i}" for lead in LEADS for i in range(0, BEAT_LEN, STEP)]
    + [f"amp_{lead}" for lead in LEADS]
    + ["hr_bpm", "rr_std_s", "rr_cv", "n_beats"]
)


def extract_features(p: ProcessedECG, fs: float = 100.0) -> np.ndarray:
    if len(p.beats) > 0:
        median_beat = np.median(p.beats, axis=0)          # (70, 12)
    else:
        median_beat = np.zeros((BEAT_LEN, len(LEADS)), np.float32)
    beat_part = median_beat[::STEP].T.reshape(-1)         # lead-major order, matches FEATURE_NAMES

    amp = np.asarray(p.report["checks"].get("lead_std_mv", [0.0] * len(LEADS)), dtype=np.float32)

    rr = np.diff(p.r_peaks) / fs if len(p.r_peaks) > 1 else np.array([0.0])
    rr_mean = float(rr.mean())
    rhythm = np.array([
        p.report["checks"]["heart_rate_bpm"],
        float(rr.std()),
        float(rr.std() / rr_mean) if rr_mean > 0 else 0.0,
        len(p.beats),
    ], dtype=np.float32)

    vec = np.concatenate([beat_part, amp, rhythm]).astype(np.float32)
    assert vec.shape[0] == len(FEATURE_NAMES)
    return vec