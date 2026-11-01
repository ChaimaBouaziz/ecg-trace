import json

import pytest

from pipeline.explain import build_explanation, build_final_result
from pipeline.preprocess import prepare_report
from pipeline.validate import RULES, report_interpretation, validate

ALL = RULES["required_artifacts"]


def pred(cls="NORM", conf=0.95):
    return {"predicted_class": cls, "confidence": conf,
            "probabilities": {"NORM": conf, "ABNORMAL": 1 - conf},
            "model_name": "m", "model_version": "1.0.0"}


def proc(q=1.0, warnings=()):
    return {"quality_score": q, "warnings": list(warnings),
            "checks": {"heart_rate_bpm": 72.0, "n_beats": 11}}


@pytest.mark.parametrize("text,expected", [
    ("sinus rhythm. normal ecg.", "NORM"),
    ("sinusrhythmus normales ekg", "NORM"),
    ("sinusrhythmus linkstyp linksschenkelblock", "ABNORMAL"),
    ("sinus rhythm. non-specific t wave flattening in i, avl", "ABNORMAL"),
    ("sinusrhythmus a-v block i lagetyp normal qrs(t) abnorm inferiorer infarkt", "ABNORMAL"),
    ("sinus rhythm. normal axis.", "UNKNOWN"),
    ("", "UNKNOWN"),
])
def test_report_interpretation(text, expected):
    assert report_interpretation(text) == expected


def test_accepted():
    v = validate(pred(), proc(), prepare_report("sinus rhythm. normal ecg."), ALL)
    assert v["status"] == "ACCEPTED" and v["failed_checks"] == []


def test_low_confidence_review():
    v = validate(pred(conf=0.6), proc(), prepare_report("normal ecg"), ALL)
    assert v["status"] == "REVIEW_REQUIRED" and v["failed_checks"] == ["ai_confidence"]


def test_low_quality_review():
    v = validate(pred(), proc(q=0.6, warnings=["only 2 complete beats"]), prepare_report("normal ecg"), ALL)
    assert v["status"] == "REVIEW_REQUIRED" and "data_quality" in v["failed_checks"]


def test_report_contradiction_review():
    v = validate(pred("NORM"), proc(), prepare_report("inferiorer infarkt"), ALL)
    assert v["status"] == "REVIEW_REQUIRED" and v["failed_checks"] == ["report_coherence"]


def test_unclear_report_not_penalised():
    v = validate(pred("ABNORMAL"), proc(), prepare_report("sinus rhythm."), ALL)
    assert v["status"] == "ACCEPTED"


def test_missing_artifact_rejected():
    v = validate(pred(), proc(), prepare_report("normal ecg"), ALL[:-1])
    assert v["status"] == "REJECTED" and "artifacts_complete" in v["failed_checks"]


def test_integrity_failure_rejected():
    v = validate(pred(), proc(), prepare_report("normal ecg"), ALL, integrity_ok=False)
    assert v["status"] == "REJECTED"


def test_explanation_and_final():
    v = validate(pred("ABNORMAL", 0.7), proc(), prepare_report("infarkt"), ALL)
    e = build_explanation(pred("ABNORMAL", 0.7), proc(), v)
    f = build_final_result("CASE-1", pred("ABNORMAL", 0.7), v, e)
    assert "70%" in e["text"] and "Review required" in e["text"]
    assert f["review_required"] is True and f["predicted_class"] == "ABNORMAL"
    json.dumps([v, e, f])   # all JSON-serialisable -> hashable


def test_rejected_final_has_no_prediction():
    v = validate(None, None, None, [], integrity_ok=None)
    e = build_explanation(None, None, v)
    f = build_final_result("CASE-2", None, v, e)
    assert f["decision"] == "REJECTED" and f["predicted_class"] is None