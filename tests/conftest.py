import joblib
import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier

from pipeline.features import FEATURE_NAMES, extract_features
from pipeline.preprocess import preprocess_ecg
from tests.test_preprocess import FS, synthetic_ecg


@pytest.fixture(scope="session")
def tiny_model_path(tmp_path_factory):
    """Small fake model: slow heart rate = NORM, fast = ABNORMAL."""
    X, y = [], []
    for i, hr in enumerate([50, 55, 60, 110, 120, 130] * 3):
        X.append(extract_features(preprocess_ecg(synthetic_ecg(hr_bpm=hr, seed=i)[0], FS)))
        y.append("NORM" if hr < 100 else "ABNORMAL")
    est = RandomForestClassifier(n_estimators=20, random_state=0).fit(np.stack(X), y)
    path = tmp_path_factory.mktemp("model") / "m.joblib"
    joblib.dump({"estimator": est, "classes": est.classes_.tolist(), "feature_names": FEATURE_NAMES,
                 "model_name": "test-model", "model_version": "0.0.1", "metadata": {}}, path)
    return path