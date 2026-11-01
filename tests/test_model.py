import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from pipeline.features import FEATURE_NAMES, extract_features
from pipeline.model import load_model, predict
from pipeline.preprocess import preprocess_ecg
from tests.test_preprocess import FS, synthetic_ecg


def test_features_shape_and_determinism():
    x, _ = synthetic_ecg(hr_bpm=70)
    f1 = extract_features(preprocess_ecg(x, FS))
    f2 = extract_features(preprocess_ecg(x, FS))
    assert f1.shape == (len(FEATURE_NAMES),) == (436,)
    assert np.isfinite(f1).all()
    np.testing.assert_array_equal(f1, f2)          # same input -> same features -> same hash
    assert abs(f1[FEATURE_NAMES.index("hr_bpm")] - 70) < 5


def test_predict_artifact_fields(tmp_path):
    # tiny fake model: slow vs fast heart rate
    X, y = [], []
    for i, hr in enumerate([50, 55, 60, 110, 120, 130] * 3):
        X.append(extract_features(preprocess_ecg(synthetic_ecg(hr_bpm=hr, seed=i)[0], FS)))
        y.append("NORM" if hr < 100 else "ABNORMAL")
    est = RandomForestClassifier(n_estimators=20, random_state=0).fit(np.stack(X), y)
    path = tmp_path / "m.joblib"
    joblib.dump({"estimator": est, "classes": est.classes_.tolist(), "feature_names": FEATURE_NAMES,
                 "model_name": "test-model", "model_version": "0.0.1", "metadata": {}}, path)

    bundle = load_model(path)
    feats = extract_features(preprocess_ecg(synthetic_ecg(hr_bpm=125, seed=99)[0], FS))
    out = predict(bundle, feats, input_artifact_id="FEATURES-ABC")
    assert out["predicted_class"] == "ABNORMAL"
    assert set(out["probabilities"]) == {"ABNORMAL", "NORM"}
    assert abs(sum(out["probabilities"].values()) - 1) < 1e-3
    assert out["confidence"] == max(out["probabilities"].values())
    assert out["input_artifact_id"] == "FEATURES-ABC" and len(out["model_file_sha256"]) == 64