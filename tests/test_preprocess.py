import json

import numpy as np
import pytest

from pipeline.preprocess import PreprocessingError, prepare_report, preprocess_ecg

FS = 100


def synthetic_ecg(hr_bpm=60, seconds=10, drift=True, noise=0.03, seed=0):
    """Gaussian 'QRS' spikes + T waves + baseline drift + noise, 12 leads (some inverted)."""
    rng = np.random.default_rng(seed)
    t = np.arange(seconds * FS) / FS
    beat_times = np.arange(0.5, seconds - 0.3, 60 / hr_bpm)
    one = np.zeros_like(t)
    for bt in beat_times:
        one += 1.2 * np.exp(-((t - bt) ** 2) / (2 * 0.012 ** 2))      # R
        one += 0.3 * np.exp(-((t - bt - 0.25) ** 2) / (2 * 0.04 ** 2))  # T
    signs = np.array([1, 1, 1, -1, 1, 1, -1, 1, 1, 1, 1, 1])
    x = one[:, None] * signs[None, :]
    if drift:
        x += 0.5 * np.sin(2 * np.pi * 0.15 * t)[:, None]
    x += rng.normal(0, noise, x.shape)
    return x, len(beat_times)


def test_clean_signal_ok():
    x, n = synthetic_ecg(hr_bpm=72)
    out = preprocess_ecg(x, FS)
    assert out.signal.shape == (1000, 12) and out.signal.dtype == np.float32
    assert abs(out.report["checks"]["n_r_peaks"] - n) <= 1
    assert 65 <= out.report["checks"]["heart_rate_bpm"] <= 79
    assert out.beats.shape[1:] == (70, 12)
    assert out.report["quality_score"] == 1.0
    np.testing.assert_allclose(out.signal.mean(axis=0), 0, atol=1e-4)  # z-scored
    json.dumps(out.report)  # must be JSON-serialisable (it will be hashed)


def test_bandpass_removes_baseline_drift():
    x, _ = synthetic_ecg(drift=True, noise=0)
    out = preprocess_ecg(x, FS)
    # low-frequency content should be ~gone: mean of each 2-s block close to 0
    blocks = out.signal[:, 0].reshape(5, 200).mean(axis=1)
    assert np.all(np.abs(blocks) < 0.3)


def test_few_nans_are_interpolated_and_penalised():
    x, _ = synthetic_ecg()
    x[100:110, 2] = np.nan
    out = preprocess_ecg(x, FS)
    assert np.isfinite(out.signal).all()
    assert out.report["quality_score"] < 1.0
    assert "missing values interpolated" in out.report["warnings"]


def test_many_nans_rejected():
    x, _ = synthetic_ecg()
    x[:200, :] = np.nan
    with pytest.raises(PreprocessingError):
        preprocess_ecg(x, FS)


def test_flat_lead_detected():
    x, _ = synthetic_ecg()
    x[:, 5] = 0.0
    out = preprocess_ecg(x, FS)
    assert out.report["checks"]["flat_leads"] == [5]
    assert out.report["quality_score"] <= 0.85


def test_wrong_format_rejected():
    with pytest.raises(PreprocessingError):
        preprocess_ecg(np.zeros((1000, 3)), FS)
    with pytest.raises(PreprocessingError):
        preprocess_ecg(synthetic_ecg()[0], 500)


def test_dataset_flags_penalise():
    x, _ = synthetic_ecg()
    out = preprocess_ecg(x, FS, {"static_noise": ", I-AVF", "burst_noise": np.nan})
    assert out.report["checks"]["dataset_quality_flags"] == ["static_noise"]
    assert out.report["quality_score"] == pytest.approx(0.9)


def test_prepare_report():
    r = prepare_report("Sinusrhythmus   Lagetyp normal\r\n normales EKG")
    assert r["language"] == "de" and r["text_clean"].startswith("sinusrhythmus lagetyp")
    assert prepare_report("sinus rhythm. normal ecg.")["language"] == "en"