"""
Step 3b — Explanation + final result (artifacts EXPLANATION and FINAL).

The explanation is deliberately simple and template-based: it states what the model predicted,
how confident it was, the key measurements it relied on and why the validator decided
accept / review / reject. Readable by a non-expert during the demo.
"""
from __future__ import annotations

from typing import Optional

EXPLAIN_VERSION = "explain-1.0.0"

STATUS_TEXT = {
    "ACCEPTED": "Accepted for the prototype: all configured checks passed.",
    "REVIEW_REQUIRED": "Review required: a human should look at this case.",
    "REJECTED": "Rejected: the processing chain is incomplete or an integrity problem was detected.",
}
CLASS_TEXT = {
    "NORM": "a normal ECG",
    "ABNORMAL": "an abnormal ECG (signs compatible with infarction, ST/T change, conduction disorder or hypertrophy)",
}


def build_explanation(prediction: Optional[dict], processing_report: Optional[dict],
                      validation: dict) -> dict:
    lines = []
    if prediction:
        lines.append(f"The model {prediction['model_name']} v{prediction['model_version']} "
                     f"classified this recording as {CLASS_TEXT.get(prediction['predicted_class'], prediction['predicted_class'])} "
                     f"with a confidence of {prediction['confidence']:.0%}.")
    if processing_report:
        c = processing_report["checks"]
        lines.append(f"Measured heart rate: {c['heart_rate_bpm']:.0f} bpm over {c['n_beats']} complete beats; "
                     f"signal quality score {processing_report['quality_score']:.2f}.")
    for chk in validation["checks"]:
        if not chk["passed"]:
            lines.append(f"Check '{chk['name']}' failed: {chk['message']}.")
    if validation.get("report_interpretation") in ("NORM", "ABNORMAL"):
        lines.append(f"The medical report was read as {validation['report_interpretation']}.")
    lines.append(STATUS_TEXT[validation["status"]])
    lines.append("This is an educational prototype, not a medical diagnosis.")

    return {"text": " ".join(lines), "lines": lines, "software_version": EXPLAIN_VERSION}


def build_final_result(case_id: str, prediction: Optional[dict], validation: dict,
                       explanation: dict) -> dict:
    status = validation["status"]
    return {
        "case_id": case_id,
        "decision": status,
        "predicted_class": prediction["predicted_class"] if prediction and status != "REJECTED" else None,
        "confidence": prediction["confidence"] if prediction and status != "REJECTED" else None,
        "review_required": status != "ACCEPTED",
        "summary": explanation["text"],
        "software_version": EXPLAIN_VERSION,
    }