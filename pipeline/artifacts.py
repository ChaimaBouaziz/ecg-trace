"""
Step 4 — Artifact chain: serialize -> store off-chain -> hash -> record on the ledger.

    rec = recorder.record(ArtifactType.PROCESSED, payload, kind="npz",
                          parents=[raw.artifact_id], software_version="preprocess-1.1.0",
                          creator_id="agent.preprocess")

Every call produces exactly one ArtifactRecord on the ledger whose sha256 is the hash
of the stored file, and whose parent_ids link it to the artifacts it was computed from.
"""
from __future__ import annotations

from typing import Any

from ledger.base import LedgerClient
from pipeline.schemas import ArtifactRecord, ArtifactType, new_artifact_id
from pipeline.storage import OffchainStore, serialize


class ArtifactRecorder:
    def __init__(self, case_id: str, store: OffchainStore, ledger: LedgerClient):
        self.case_id = case_id
        self.store = store
        self.ledger = ledger
        self.records: list[ArtifactRecord] = []

    def record(self, artifact_type: ArtifactType, payload: Any, kind: str, parents: list[str],
               software_version: str, creator_id: str) -> ArtifactRecord:
        artifact_id = new_artifact_id(artifact_type)
        data, ext = serialize(payload, kind)
        uri, digest = self.store.save(self.case_id, artifact_id, data, ext)
        rec = ArtifactRecord(
            artifact_id=artifact_id, case_id=self.case_id, artifact_type=artifact_type,
            sha256=digest, software_version=software_version, parent_ids=parents,
            offchain_uri=uri, creator_id=creator_id,
        )
        tx_id = self.ledger.put_artifact(rec)
        rec = rec.model_copy(update={"tx_id": tx_id})
        self.records.append(rec)
        return rec

    def types_present(self) -> list[str]:
        return [r.artifact_type.value for r in self.records]