"""
Step 3 check on real data: run model + validator on the TEST fold (10).

  python scripts/evaluate_validation.py

Answers 3 questions for the report:
  1. How well does the keyword reader understand the medical reports?
  2. How many cases are ACCEPTED / REVIEW_REQUIRED / REJECTED?
  3. Is the AI more accurate on ACCEPTED cases than on REVIEW cases? (it should be)
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from pipeline.explain import build_explanation
from pipeline.features import extract_features
from pipeline.model import load_model, predict
from pipeline.preprocess import PreprocessingError, prepare_report, preprocess_ecg
from pipeline.ptbxl import QUALITY_COLS, load_metadata, load_signal
from pipeline.validate import RULES, report_interpretation, validate
from scripts.train_model import label_of

OUT = Path("docs/validation"); OUT.mkdir(parents=True, exist_ok=True)

df = load_metadata()
df["label"] = df["superclasses"].apply(label_of)
df = df[df["label"].notna()]

# 1. report reader vs ground-truth label (all local records)
interp = df["report"].apply(lambda t: report_interpretation(prepare_report(t)["text_clean"]))
known = interp != "UNKNOWN"
print("1) Report keyword reader (all records)")
print(f"   clear reading for {known.mean():.1%} of reports")
print(f"   agreement with true label when clear: {(interp[known] == df['label'][known]).mean():.1%}")
print("   ", dict(Counter(zip(df["label"][known], interp[known]))))

# 2-3. full model + validation on test fold
bundle = load_model()
test = df[df["strat_fold"] == 10]
rows = []
for ecg_id, row in test.iterrows():
    raw, meta = load_signal(row)
    try:
        p = preprocess_ecg(raw, meta["fs"], {k: row[k] for k in QUALITY_COLS})
    except PreprocessingError:
        continue
    pr = predict(bundle, extract_features(p, meta["fs"]), input_artifact_id=f"FEATURES-{ecg_id}")
    rep = prepare_report(row["report"])
    v = validate(pr, p.report, rep, RULES["required_artifacts"], integrity_ok=True)
    rows.append(dict(ecg_id=ecg_id, label=row["label"], pred=pr["predicted_class"],
                     conf=pr["confidence"], quality=p.report["quality_score"],
                     status=v["status"], failed=",".join(v["failed_checks"])))
    if len(rows) == 1:
        e = build_explanation(pr, p.report, v)
        (OUT / "example_validation.json").write_text(json.dumps({"prediction": pr, "validation": v,
                                                                 "explanation": e}, indent=2, default=str))

r = pd.DataFrame(rows)
r["correct"] = r["label"] == r["pred"]
r.to_csv(OUT / "test_validation.csv", index=False)

print(f"\n2) Validation status on test fold ({len(r)} cases)")
print(r["status"].value_counts().to_string())
print("\n   Why review? (failed checks)")
print(r.loc[r["failed"] != "", "failed"].str.split(",").explode().value_counts().to_string())

print("\n3) AI accuracy by validation status")
print(r.groupby("status")["correct"].agg(["mean", "count"]).rename(columns={"mean": "accuracy"}).round(3).to_string())
print(f"\n   overall AI accuracy: {r['correct'].mean():.3f}")
print(f"\nDetails: {OUT / 'test_validation.csv'} — example: {OUT / 'example_validation.json'}")