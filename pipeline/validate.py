"""
Step 3 — Validation module (spec §4.4).

Simple, explicit rules — each one produces a check {name, passed, value, threshold, message}.
Decision logic (most severe wins):
  REJECTED         : a required artifact is missing, or an integrity check failed
  REVIEW_REQUIRED  : low AI confidence, low data quality, or ECG result contradicts the report
  ACCEPTED         : all configured checks pass (accepted *for the prototype*, not clinically)
"""
from __future__ import annotations

import re
from typing import Any, Optional

from pipeline.schemas import ArtifactType, ValidationStatus

VALIDATION_VERSION = "validation-1.0.0"

RULES: dict[str, Any] = {
    "min_confidence": 0.80,       # from the confidence table of the model card
    "min_quality_score": 0.80,    # from the preprocessing quality score
    "required_artifacts": [
        ArtifactType.RAW_ECG.value, ArtifactType.RAW_REPORT.value, ArtifactType.PROCESSED.value,
        ArtifactType.FEATURES.value, ArtifactType.PREDICTION.value,
    ],
}

# keyword lists for PTB-XL reports (English + German)
NORMAL_PATTERNS = [r"normal ecg", r"normales ekg", r"normal ekg", r"(?<!lagetyp )\bnormal\b(?! (axis|lagetyp|qrs))"]
ABNORMAL_PATTERNS = [
    r"infarct", r"infarkt", r"ischaem", r"ischem", r"isch[äa]mie",
    r"block", r"schenkelblock", r"hypertroph", r"\blvh\b", r"\brvh\b",
    r"st[- ]?(depression|elevation|senkung|hebung|segment)", r"t[- ]?wave", r"t-welle",
    r"abnorm", r"pathologisch", r"qrs\(t\) abnorm", r"repolarisation",
    r"atrial fibrillation", r"vorhofflimmern",
]


def report_interpretation(text_clean: str) -> str:
    """'ABNORMAL' / 'NORM' / 'UNKNOWN' from simple keyword matching."""
    t = text_clean.lower()
    if any(re.search(p, t) for p in ABNORMAL_PATTERNS):
        return "ABNORMAL"
    if any(re.search(p, t) for p in NORMAL_PATTERNS):
        return "NORM"
    return "UNKNOWN"


def _check(name, passed, value=None, threshold=None, message="") -> dict:
    return {"name": name, "passed": bool(passed), "value": value,
            "threshold": threshold, "message": message}


def validate(
    prediction: Optional[dict],
    processing_report: Optional[dict],
    report_info: Optional[dict],
    available_artifacts: list[str],
    integrity_ok: Optional[bool] = True,
) -> dict:
    """
    prediction        : output of pipeline.model.predict
    processing_report : ProcessedECG.report
    report_info       : output of pipeline.preprocess.prepare_report
    available_artifacts : artifact types already produced for this case
    integrity_ok      : result of re-hashing previous artifacts against the ledger
                        (True / False / None = not verified yet)
    """
    checks, severity = [], ValidationStatus.ACCEPTED

    # 1. completeness -------------------------------------------------------
    missing = [a for a in RULES["required_artifacts"] if a not in available_artifacts]
    checks.append(_check("artifacts_complete", not missing, missing, None,
                         "all required artifacts present" if not missing else f"missing: {missing}"))
    if missing:
        severity = ValidationStatus.REJECTED

    # 2. integrity of previous steps ----------------------------------------
    checks.append(_check("integrity", integrity_ok is True, integrity_ok, True,
                         {True: "previous artifacts match their recorded hashes",
                          False: "an artifact does not match its recorded hash",
                          None: "integrity not verified"}[integrity_ok]))
    if integrity_ok is False:
        severity = ValidationStatus.REJECTED

    if severity == ValidationStatus.REJECTED or prediction is None or processing_report is None:
        return _result(ValidationStatus.REJECTED, checks, None)

    # 3. AI confidence --------------------------------------------------------
    conf = prediction["confidence"]
    ok = conf >= RULES["min_confidence"]
    checks.append(_check("ai_confidence", ok, conf, RULES["min_confidence"],
                         f"confidence {conf:.2f}" + ("" if ok else " is below the threshold")))
    if not ok:
        severity = ValidationStatus.REVIEW_REQUIRED

    # 4. data quality ----------------------------------------------------------
    q = processing_report["quality_score"]
    ok = q >= RULES["min_quality_score"]
    msg = f"quality score {q:.2f}"
    if processing_report.get("warnings"):
        msg += " (" + "; ".join(processing_report["warnings"]) + ")"
    checks.append(_check("data_quality", ok, q, RULES["min_quality_score"], msg))
    if not ok:
        severity = ValidationStatus.REVIEW_REQUIRED

    # 5. coherence ECG result <-> medical report -------------------------------
    rep = report_interpretation(report_info["text_clean"]) if report_info and not report_info.get("empty") else "UNKNOWN"
    pred = prediction["predicted_class"]
    if rep == "UNKNOWN":
        checks.append(_check("report_coherence", True, rep, None,
                             "report does not state normal/abnormal clearly — not used"))
    else:
        ok = rep == pred
        checks.append(_check("report_coherence", ok, {"report": rep, "ai": pred}, None,
                             "AI result agrees with the report" if ok
                             else f"AI says {pred} but the report suggests {rep}"))
        if not ok:
            severity = ValidationStatus.REVIEW_REQUIRED

    return _result(severity, checks, rep)


def _result(status: ValidationStatus, checks: list[dict], report_says: Optional[str]) -> dict:
    return {
        "status": status.value,
        "checks": checks,
        "failed_checks": [c["name"] for c in checks if not c["passed"]],
        "report_interpretation": report_says,
        "rules": RULES,
        "software_version": VALIDATION_VERSION,
    }