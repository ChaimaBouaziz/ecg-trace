"""FabricLedger tested against an in-memory imitation of the REST gateway (no Fabric needed)."""
import json
import re

import httpx
import pytest

from ledger.base import LedgerError
from ledger.factory import get_ledger
from ledger.fabric_client import FabricLedger
from ledger.fake_ledger import FakeLedger
from pipeline.hashing import sha256_text
from pipeline.schemas import ArtifactRecord, ArtifactType, new_case_id


def fake_gateway():
    state = {}

    def handler(req: httpx.Request) -> httpx.Response:
        path = req.url.path
        if req.method == "POST" and path == "/artifacts":
            rec = json.loads(req.content)
            if rec["artifact_id"] in state:
                return httpx.Response(409, json={"error": f"Artifact {rec['artifact_id']} already exists"})
            missing = [p for p in rec["parent_ids"] if p not in state]
            if missing:
                return httpx.Response(400, json={"error": f"Parent {missing[0]} does not exist"})
            rec |= {"tx_id": f"tx{len(state) + 1}", "ledger_timestamp": "2026-11-01T10:00:00.000Z",
                    "submitter": "Org1MSP::x509::CN=User1"}
            state[rec["artifact_id"]] = rec
            return httpx.Response(201, json=rec)
        if m := re.fullmatch(r"/artifacts/([^/]+)", path):
            rec = state.get(m[1])
            return httpx.Response(200, json=rec) if rec else httpx.Response(404, json={"error": "does not exist"})
        if m := re.fullmatch(r"/cases/([^/]+)/history", path):
            return httpx.Response(200, json=sorted((r for r in state.values() if r["case_id"] == m[1]),
                                                   key=lambda r: r["created_at"]))
        if path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        return httpx.Response(404, json={"error": "not found"})

    return httpx.MockTransport(handler)


def rec(case, parents=()):
    t = ArtifactType.PROCESSED if parents else ArtifactType.RAW_ECG
    return ArtifactRecord(artifact_id=f"{t.value}-{len(parents)}{case[-4:]}", case_id=case, artifact_type=t,
                          sha256=sha256_text("x"), software_version="t", parent_ids=list(parents),
                          offchain_uri="u", creator_id="test")


def test_roundtrip():
    led = FabricLedger("http://gw", transport=fake_gateway())
    case = new_case_id()
    raw = rec(case)
    assert led.put_artifact(raw) == "tx1"
    led.put_artifact(rec(case, [raw.artifact_id]))
    got = led.get_artifact(raw.artifact_id)
    assert got.tx_id == "tx1" and got.submitter.startswith("Org1MSP")
    assert len(led.get_case_history(case)) == 2
    assert led.get_artifact("NOPE") is None
    assert led.health()["status"] == "ok"


def test_errors_become_ledger_errors():
    led = FabricLedger("http://gw", transport=fake_gateway())
    case = new_case_id()
    raw = rec(case)
    led.put_artifact(raw)
    with pytest.raises(LedgerError, match="already exists"):
        led.put_artifact(raw)
    with pytest.raises(LedgerError, match="Parent"):
        led.put_artifact(rec(case, ["UNKNOWN"]))


def test_factory(monkeypatch, tmp_path):
    monkeypatch.setenv("FAKE_LEDGER_PATH", str(tmp_path / "l.json"))
    assert isinstance(get_ledger("fake"), FakeLedger)
    assert isinstance(get_ledger("fabric"), FabricLedger)
    with pytest.raises(ValueError):
        get_ledger("nope")