"""
FabricLedger — same interface as FakeLedger, backed by Hyperledger Fabric
through the REST blockchain gateway (blockchain-gateway/server.js).
"""
from __future__ import annotations

import os

import httpx

from ledger.base import LedgerClient, LedgerError
from pipeline.schemas import ArtifactRecord


class FabricLedger(LedgerClient):
    def __init__(self, base_url: str | None = None, transport: httpx.BaseTransport | None = None,
                 timeout: float = 90.0):
        self.base_url = (base_url or os.getenv("FABRIC_GATEWAY_URL", "http://localhost:3001")).rstrip("/")
        self.http = httpx.Client(base_url=self.base_url, timeout=timeout, transport=transport)

    def _err(self, r: httpx.Response) -> LedgerError:
        try:
            msg = r.json().get("error", r.text)
        except ValueError:
            msg = r.text
        return LedgerError(f"Fabric gateway {r.status_code}: {msg}")

    def health(self) -> dict:
        r = self.http.get("/health")
        if r.status_code != 200:
            raise self._err(r)
        return r.json()

    def put_artifact(self, record: ArtifactRecord) -> str:
        body = record.model_dump(mode="json", exclude={"tx_id", "ledger_timestamp", "submitter"})
        try:
            r = self.http.post("/artifacts", json=body)
        except httpx.HTTPError as e:
            raise LedgerError(f"cannot reach Fabric gateway at {self.base_url}: {e}") from e
        if r.status_code != 201:
            raise self._err(r)
        return r.json()["tx_id"]

    def get_artifact(self, artifact_id: str) -> ArtifactRecord | None:
        r = self.http.get(f"/artifacts/{artifact_id}")
        if r.status_code == 404:
            return None
        if r.status_code != 200:
            raise self._err(r)
        return ArtifactRecord(**r.json())

    def get_case_history(self, case_id: str) -> list[ArtifactRecord]:
        r = self.http.get(f"/cases/{case_id}/history")
        if r.status_code != 200:
            raise self._err(r)
        return [ArtifactRecord(**x) for x in r.json()]

    def get_key_history(self, artifact_id: str) -> list[dict]:
        r = self.http.get(f"/artifacts/{artifact_id}/key-history")
        if r.status_code != 200:
            raise self._err(r)
        return r.json()