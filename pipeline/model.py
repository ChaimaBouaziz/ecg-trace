"""
Step 2b — AI model: load a trained bundle and produce a PREDICTION artifact (spec §4.3).

A model "bundle" (saved with joblib by scripts/train_model.py) contains:
  estimator, classes, feature_names, model_name, model_version, metadata (metrics, versions,
  training data fingerprint...). Everything needed for reproducibility.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from pipeline.features import FEATURE_NAMES
from pipeline.hashing import sha256_file

DEFAULT_MODEL_PATH = Path("models/ecg_binary_v1.joblib")


def load_model(path: str | Path = DEFAULT_MODEL_PATH) -> dict[str, Any]:
    bundle = joblib.load(path)
    if bundle["feature_names"] != FEATURE_NAMES:
        raise ValueError("Model was trained with a different feature set — retrain it.")
    bundle["file_sha256"] = sha256_file(path)   # identifies exactly which model file was used
    return bundle


def predict(bundle: dict[str, Any], features: np.ndarray, input_artifact_id: str) -> dict:
    proba = bundle["estimator"].predict_proba(features.reshape(1, -1))[0]
    classes = [str(c) for c in bundle["classes"]]
    best = int(np.argmax(proba))
    return {
        "predicted_class": classes[best],
        "confidence": round(float(proba[best]), 4),
        "probabilities": {c: round(float(p), 4) for c, p in zip(classes, proba)},
        "model_name": bundle["model_name"],
        "model_version": bundle["model_version"],
        "model_file_sha256": bundle.get("file_sha256"),
        "input_artifact_id": input_artifact_id,
        "analysed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }