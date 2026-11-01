"""
Integrity verification (spec §4.5 + §6): recompute hashes and compare with the ledger.

Status per artifact:
  VALID         file exists and SHA-256(file) == hash on the ledger
  MODIFIED      file exists but the hash differs  -> tampering / corruption
  MISSING       ledger record exists but the off-chain file is gone
  UNVERIFIABLE  no ledger record for this artifact (or unreadable file / broken parent link)
"""
from __future__ import annotations

from ledger.base import LedgerClient
from pipeline.hashing import sha256_bytes
from pipeline.schemas import ArtifactRecord, IntegrityStatus
from pipeline.storage import OffchainStore


def verify_record(rec: ArtifactRecord, store: OffchainStore) -> dict:
    out = {"artifact_id": rec.artifact_id, "artifact_type": rec.artifact_type.value,
           "recorded_sha256": rec.sha256, "recomputed_sha256": None, "tx_id": rec.tx_id}
    try:
        if not store.exists(rec.offchain_uri):
            return {**out, "status": IntegrityStatus.MISSING.value}
        h = sha256_bytes(store.read(rec.offchain_uri))
    except (OSError, ValueError) as e:
        return {**out, "status": IntegrityStatus.UNVERIFIABLE.value, "error": str(e)}
    status = IntegrityStatus.VALID if h == rec.sha256 else IntegrityStatus.MODIFIED
    return {**out, "recomputed_sha256": h, "status": status.value}


def verify_artifact(artifact_id: str, ledger: LedgerClient, store: OffchainStore) -> dict:
    rec = ledger.get_artifact(artifact_id)
    if rec is None:
        return {"artifact_id": artifact_id, "status": IntegrityStatus.UNVERIFIABLE.value,
                "error": "no ledger record"}
    return verify_record(rec, store)


def verify_case(case_id: str, ledger: LedgerClient, store: OffchainStore) -> dict:
    history = ledger.get_case_history(case_id)
    if not history:
        return {"case_id": case_id, "overall": IntegrityStatus.UNVERIFIABLE.value, "artifacts": []}
    known = {r.artifact_id for r in history}
    results = []
    for rec in history:
        r = verify_record(rec, store)
        broken = [p for p in rec.parent_ids if p not in known]
        if broken and r["status"] == IntegrityStatus.VALID.value:
            r = {**r, "status": IntegrityStatus.UNVERIFIABLE.value, "error": f"unknown parents {broken}"}
        results.append(r)
    statuses = {r["status"] for r in results}
    overall = IntegrityStatus.VALID.value if statuses == {IntegrityStatus.VALID.value} else next(
        s.value for s in (IntegrityStatus.MODIFIED, IntegrityStatus.MISSING, IntegrityStatus.UNVERIFIABLE)
        if s.value in statuses)
    return {"case_id": case_id, "overall": overall, "artifacts": results}


def all_valid(records: list[ArtifactRecord], store: OffchainStore) -> bool:
    """Used by the validation step: are all previous artifacts still intact?"""
    return all(verify_record(r, store)["status"] == IntegrityStatus.VALID.value for r in records)