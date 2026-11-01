import numpy as np
import pytest

from ledger.base import LedgerError
from ledger.fake_ledger import FakeLedger
from pipeline.hashing import sha256_array, sha256_json, sha256_text
from pipeline.schemas import ArtifactRecord, ArtifactType, new_artifact_id, new_case_id


def test_json_hash_is_order_independent():
    assert sha256_json({"a": 1, "b": 2}) == sha256_json({"b": 2, "a": 1})


def test_array_hash_detects_single_change():
    x = np.random.default_rng(0).normal(size=(1000, 12)).astype(np.float32)
    h = sha256_array(x)
    assert h == sha256_array(x.copy())
    x[500, 3] += 1e-4
    assert h != sha256_array(x)


def test_text_hash_ignores_line_endings():
    assert sha256_text("a\r\nb") == sha256_text("a\nb")


def _record(case_id, parent=None):
    t = ArtifactType.RAW_ECG if parent is None else ArtifactType.PROCESSED
    return ArtifactRecord(
        artifact_id=new_artifact_id(t), case_id=case_id, artifact_type=t,
        sha256=sha256_text("x"), software_version="test-0.0.1",
        parent_ids=[parent] if parent else [], offchain_uri="offchain/x",
        creator_id="test",
    )


def test_fake_ledger_roundtrip_and_no_overwrite(tmp_path):
    led = FakeLedger(tmp_path / "ledger.json")
    case = new_case_id()
    raw = _record(case)
    tx = led.put_artifact(raw)
    proc = _record(case, parent=raw.artifact_id)
    led.put_artifact(proc)

    assert led.get_artifact(raw.artifact_id).tx_id == tx
    hist = led.get_case_history(case)
    assert [r.artifact_id for r in hist] == [raw.artifact_id, proc.artifact_id]
    assert hist[1].parent_ids == [raw.artifact_id]
    with pytest.raises(LedgerError):
        led.put_artifact(raw)
