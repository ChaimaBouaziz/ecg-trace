"""
Ledger interface.  The rest of the code ONLY talks to this interface.

- FakeLedger  (ledger/fake_ledger.py)  -> used now, a local JSON file
- FabricLedger (ledger/fabric_client.py) -> step 6, same methods, real chaincode

The chaincode (step 6) must expose the same 3 operations:
  CreateArtifact(record_json) / ReadArtifact(artifact_id) / GetCaseHistory(case_id)
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from pipeline.schemas import ArtifactRecord


class LedgerError(Exception):
    pass


class LedgerClient(ABC):
    @abstractmethod
    def put_artifact(self, record: ArtifactRecord) -> str:
        """Store the record. Returns a transaction id. Must refuse duplicates."""

    @abstractmethod
    def get_artifact(self, artifact_id: str) -> ArtifactRecord | None:
        """Return the record or None if unknown."""

    @abstractmethod
    def get_case_history(self, case_id: str) -> list[ArtifactRecord]:
        """All records of a case, ordered by creation time."""
