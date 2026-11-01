"""
Spec §4.1 — personal identifiers are removed or replaced by the pseudonymised case id
BEFORE anything enters the pipeline (and therefore before anything is hashed or stored).
"""
from __future__ import annotations

import re

PSEUDO_VERSION = "pseudonymise-1.0.1"

# generic patterns that should never reach the pipeline
PATTERNS = {
    "email": re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"),
    "phone": re.compile(r"(?<![\w+])(?:\+?\d[\d .-]{7,}\d)(?![\w])"),   # not inside a word/id
    "date_of_birth": re.compile(r"\b(?:dob|né\(e\) le|born|geb\.?)\s*[:.]?\s*\d{1,2}[./-]\d{1,2}[./-]\d{2,4}", re.I),
}


def pseudonymise_text(text: str, case_id: str, identifiers: list[str] | None = None) -> tuple[str, dict]:
    """Replace known identifiers (name, patient id...) and generic PII patterns.
    Returns (clean text, report of what was replaced — counts only, never the values)."""
    out, counts = str(text), {}
    # 1) generic PII patterns first, 2) then known identifiers -> the inserted [CASE-…] tag is never re-scanned
    for name, pat in PATTERNS.items():
        out, n = pat.subn(f"[{name.upper()} REMOVED]", out)
        if n:
            counts[name] = n
    for ident in identifiers or []:
        ident = str(ident).strip()
        if len(ident) < 2:
            continue
        out, n = re.subn(re.escape(ident), f"[{case_id}]", out, flags=re.I)
        if n:
            counts["identifier"] = counts.get("identifier", 0) + n
    return out, {"software_version": PSEUDO_VERSION, "replacements": counts}