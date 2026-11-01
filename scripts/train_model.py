"""
Step 2 — Train the baseline classifier: NORM vs ABNORMAL on PTB-XL (local records).

  python scripts/train_model.py

Split = official PTB-XL folds: 1-8 train, 9 validation (model choice), 10 test (final score).
Three candidates are compared on validation; the best one is saved to
models/ecg_binary_v1.joblib + a human-readable model card models/ecg_binary_v1.json.
"""
from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, roc_auc_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from pipeline.features import FEATURE_NAMES, FEATURES_VERSION, extract_features
from pipeline.hashing import sha256_json
from pipeline.preprocess import PREPROCESS_VERSION, PreprocessingError, preprocess_ecg
from pipeline.ptbxl import QUALITY_COLS, load_metadata, load_signal

MODEL_NAME = "ecg-binary-norm-vs-abnormal"
MODEL_VERSION = "1.0.0"
OUT = Path("models")
CACHE = Path("data/features_cache.npz")
CLASSES = np.array(["ABNORMAL", "NORM"])   # sklearn sorts labels alphabetically


def label_of(superclasses: list[str]) -> str | None:
    if not superclasses:
        return None                         # no diagnostic label -> excluded
    return "NORM" if superclasses == ["NORM"] else "ABNORMAL"


def build_dataset():
    df = load_metadata()
    df["label"] = df["superclasses"].apply(label_of)
    df = df[df["label"].notna()]

    if CACHE.exists():
        c = np.load(CACHE, allow_pickle=False)
        if (str(c["versions"]) == f"{PREPROCESS_VERSION}|{FEATURES_VERSION}"
                and set(c["ids"]) == set(df.index)):
            print(f"Using cached features ({CACHE})")
            order = {i: k for k, i in enumerate(c["ids"])}
            idx = [order[i] for i in df.index]
            return df, c["X"][idx]

    print(f"Extracting features for {len(df)} records...")
    X, keep, t0 = [], [], time.time()
    for ecg_id, row in df.iterrows():
        raw, meta = load_signal(row)
        try:
            p = preprocess_ecg(raw, meta["fs"], {k: row[k] for k in QUALITY_COLS})
        except PreprocessingError:
            continue
        X.append(extract_features(p, meta["fs"]))
        keep.append(ecg_id)
    df, X = df.loc[keep], np.stack(X)
    print(f"  done in {time.time() - t0:.0f}s")
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(CACHE, X=X, ids=np.array(keep), versions=f"{PREPROCESS_VERSION}|{FEATURES_VERSION}")
    return df, X


def evaluate(est, X, y) -> dict:
    proba = est.predict_proba(X)
    pred = est.classes_[proba.argmax(1)]
    return {
        "accuracy": round(accuracy_score(y, pred), 4),
        "f1_macro": round(f1_score(y, pred, average="macro"), 4),
        "roc_auc": round(roc_auc_score((y == "ABNORMAL").astype(int),
                                       proba[:, list(est.classes_).index("ABNORMAL")]), 4),
    }


def main():
    df, X = build_dataset()
    y = df["label"].to_numpy()
    fold = df["strat_fold"].to_numpy()
    tr, va, te = fold <= 8, fold == 9, fold == 10
    print(f"\nRecords: {len(y)}  (train {tr.sum()}, val {va.sum()}, test {te.sum()})")
    print("Class balance:", dict(zip(*np.unique(y, return_counts=True))))

    candidates = {
        "logreg": make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=3000)),
        "random_forest": RandomForestClassifier(n_estimators=400, min_samples_leaf=2,
                                                n_jobs=-1, random_state=42),
        "hist_gradient_boosting": HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05,
                                                                 random_state=42),
    }
    results = {}
    print("\nValidation (fold 9):")
    for name, est in candidates.items():
        est.fit(X[tr], y[tr])
        results[name] = evaluate(est, X[va], y[va])
        print(f"  {name:24s} {results[name]}")

    best = max(results, key=lambda k: results[k]["roc_auc"])
    est = candidates[best]
    test_metrics = evaluate(est, X[te], y[te])
    pred_te = est.predict(X[te])
    conf_te = est.predict_proba(X[te]).max(1)

    print(f"\nBest on validation: {best}")
    print(f"Test (fold 10): {test_metrics}")
    print(classification_report(y[te], pred_te, digits=3))
    cm = confusion_matrix(y[te], pred_te, labels=CLASSES)
    print("Confusion matrix (rows = true, cols = predicted):", CLASSES.tolist())
    print(cm)

    # how useful is the confidence score? (used by the validation module, step 3)
    print("\nConfidence vs accuracy on test:")
    conf_table = []
    for th in (0.5, 0.6, 0.7, 0.8, 0.9):
        m = conf_te >= th
        acc = float((pred_te[m] == y[te][m]).mean()) if m.any() else float("nan")
        conf_table.append({"threshold": th, "coverage": round(float(m.mean()), 3), "accuracy": round(acc, 3)})
        print(f"  conf >= {th:.1f}: {m.mean():5.1%} of cases, accuracy {acc:.3f}")

    OUT.mkdir(exist_ok=True)
    meta = {
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "algorithm": best,
        "task": "binary: NORM vs ABNORMAL (any of MI, STTC, CD, HYP)",
        "dataset": "PTB-XL v1.0.3, 100 Hz, local subset",
        "split": "strat_fold 1-8 train / 9 val / 10 test",
        "n_records": {"train": int(tr.sum()), "val": int(va.sum()), "test": int(te.sum())},
        "training_ids_sha256": sha256_json(sorted(int(i) for i in df.index[tr])),
        "validation_metrics": results,
        "test_metrics": test_metrics,
        "confusion_matrix_test": {"labels": CLASSES.tolist(), "matrix": cm.tolist()},
        "confidence_table_test": conf_table,
        "preprocess_version": PREPROCESS_VERSION,
        "features_version": FEATURES_VERSION,
        "n_features": len(FEATURE_NAMES),
        "sklearn_version": sklearn.__version__,
        "python_version": platform.python_version(),
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    bundle = {"estimator": est, "classes": est.classes_.tolist(), "feature_names": FEATURE_NAMES,
              "model_name": MODEL_NAME, "model_version": MODEL_VERSION, "metadata": meta}
    path = OUT / "ecg_binary_v1.joblib"
    joblib.dump(bundle, path)
    (OUT / "ecg_binary_v1.json").write_text(json.dumps(meta, indent=2))
    print(f"\nSaved {path} and model card {OUT / 'ecg_binary_v1.json'}")


if __name__ == "__main__":
    main()