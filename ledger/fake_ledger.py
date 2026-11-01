"""Append-only JSON ledger used until Hyperledger Fabric is plugged in."""
from __future__ import annotations

import json
import threading
import uuid
from pathlib import Path

from ledger.base import LedgerClient, LedgerError
from pipeline.schemas import ArtifactRecord


class FakeLedger(LedgerClient):
    def __init__(self, path: str | Path = "data/fake_ledger.json"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        if not self.path.exists():
            self.path.write_text("{}", encoding="utf-8")

    def _load(self) -> dict:
        return json.loads(self.path.read_text(encoding="utf-8"))

    def _save(self, state: dict) -> None:
        self.path.write_text(json.dumps(state, indent=2), encoding="utf-8")

    def put_artifact(self, record: ArtifactRecord) -> str:
        with self._lock:
            state = self._load()
            if record.artifact_id in state:
                # like the chaincode: an artifact can never be overwritten
                raise LedgerError(f"Artifact {record.artifact_id} already exists")
            tx_id = uuid.uuid4().hex
            stored = record.model_copy(update={"tx_id": tx_id})
            state[record.artifact_id] = stored.model_dump(mode="json")
            self._save(state)
            return tx_id

    def get_artifact(self, artifact_id: str) -> ArtifactRecord | None:
        raw = self._load().get(artifact_id)
        return ArtifactRecord(**raw) if raw else None

    def get_case_history(self, case_id: str) -> list[ArtifactRecord]:
        recs = [ArtifactRecord(**r) for r in self._load().values() if r["case_id"] == case_id]
        return sorted(recs, key=lambda r: r.created_at)
