"""
Shared contract of the project.

Every module (AI, validation, storage, ledger, agents, API) exchanges
ArtifactRecord objects. The fields mirror section 4.6 of the spec:
what goes ON-CHAIN is exactly this record — never the medical data itself.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class ArtifactType(str, Enum):
    RAW_ECG = "RAW_ECG"            # original ECG signal (.dat/.hea or .npy)
    RAW_REPORT = "RAW_REPORT"      # original medical report (text)
    PROCESSED = "PROCESSED"        # cleaned / normalised signal
    FEATURES = "FEATURES"          # features or representation fed to the model
    PREDICTION = "PREDICTION"      # class, probabilities, model name/version
    VALIDATION = "VALIDATION"      # checks + status (ACCEPTED / REVIEW / REJECTED)
    EXPLANATION = "EXPLANATION"    # simple human-readable explanation
    FINAL = "FINAL"                # final prototype decision


class ValidationStatus(str, Enum):
    ACCEPTED = "ACCEPTED"          # "Accepté pour le prototype"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"  # "Révision requise"
    REJECTED = "REJECTED"          # "Rejeté"


class IntegrityStatus(str, Enum):
    VALID = "VALID"                # recomputed hash == on-chain hash
    MODIFIED = "MODIFIED"          # hash differs
    MISSING = "MISSING"            # off-chain file not found
    UNVERIFIABLE = "UNVERIFIABLE"  # no on-chain record / cannot compare


def new_case_id() -> str:
    """Pseudonymised case identifier (no link to patient identity)."""
    return f"CASE-{uuid.uuid4().hex[:12].upper()}"


def new_artifact_id(artifact_type: ArtifactType) -> str:
    return f"{artifact_type.value}-{uuid.uuid4().hex[:12].upper()}"


class ArtifactRecord(BaseModel):
    """One on-chain record = one artifact fingerprint + its provenance."""

    artifact_id: str
    case_id: str
    artifact_type: ArtifactType
    sha256: str = Field(..., min_length=64, max_length=64)
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )
    software_version: str                 # e.g. "preprocess-1.0.0" or "rf-ptbxl-1.0.0"
    parent_ids: list[str] = Field(default_factory=list)  # empty for RAW_* artifacts
    offchain_uri: str                     # e.g. "offchain/CASE-XXX/PROCESSED-YYY.npy"
    creator_id: str                       # component/agent identity, e.g. "agent.preprocess"
    tx_id: Optional[str] = None           # filled by the ledger after submission
    ledger_timestamp: Optional[str] = None  # set by the chaincode (transaction time)
    submitter: Optional[str] = None       # set by the chaincode: "<MSP>::<certificate identity>"

    @field_validator("sha256")
    @classmethod
    def _hex(cls, v: str) -> str:
        int(v, 16)  # raises if not hexadecimal
        return v.lower()