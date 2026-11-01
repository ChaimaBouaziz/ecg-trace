import numpy as np
import pytest

from agents import orchestrator as orch
from agents.orchestrator import build_graph, run_case_agentic
from ledger.base import LedgerError
from ledger.fake_ledger import FakeLedger
from pipeline.model import load_model
from pipeline.pseudonymise import pseudonymise_text
from pipeline.schemas import ArtifactType as T
from pipeline.storage import OffchainStore
from tests.test_preprocess import FS, synthetic_ecg

NO_SLEEP = build_graph(sleep=lambda s: None)


@pytest.fixture
def env(tmp_path, tiny_model_path):
    return OffchainStore(tmp_path / "off"), FakeLedger(tmp_path / "ledger.json"), load_model(tiny_model_path)


def run(env, signal=None, report="sinus rhythm. normal ecg.", graph=NO_SLEEP, **kw):
    store, ledger, model = env
    sig = synthetic_ecg(hr_bpm=55)[0] if signal is None else signal
    return run_case_agentic(sig, FS, report, model, store, ledger, graph=graph, **kw)


def agents_status(res):
    return {t["agent"]: t["status"] for t in res["agent_trace"]}


def test_happy_path(env):
    res = run(env)
    assert res["status"] == "ACCEPTED" and res["audit"]["overall"] == "VALID"
    assert list(agents_status(res).values()) == ["ok"] * 7
    assert len(res["artifacts"]) == 8
    trace = res["_ctx"].final["agent_trace"]           # agents' log is inside the hashed FINAL
    assert [t["agent"] for t in trace][:6] == ["agent.ingest", "agent.preprocess", "agent.features",
                                               "agent.predict", "agent.integrity", "agent.validate"]
    assert all(t["artifacts"] for t in res["agent_trace"] if t["agent"] != "agent.integrity")


def test_bad_signal_routes_around_ai(env):
    sig = synthetic_ecg()[0]
    sig[:300] = np.nan
    res = run(env, signal=sig)
    st = agents_status(res)
    assert st["agent.preprocess"] == "rejected_input"
    assert "agent.features" not in st and "agent.predict" not in st     # orchestrator skipped them
    assert res["status"] == "REJECTED" and res["audit"]["overall"] == "VALID"


def test_agent_crash_is_traced_and_rejected(env, monkeypatch):
    def boom(ctx):
        raise RuntimeError("model file corrupted")
    monkeypatch.setitem(orch.AGENTS, "predict", (boom, "predict", False))
    res = run(env, graph=build_graph(sleep=lambda s: None))
    st = agents_status(res)
    assert st["agent.predict"] == "failed"
    assert res["status"] == "REJECTED"
    assert "artifacts_complete" in res["validation"]["failed_checks"]


def test_ledger_retry_then_success(env):
    store, ledger, model = env
    calls = {"n": 0}
    real = ledger.put_artifact

    def flaky(rec):
        calls["n"] += 1
        if calls["n"] == 1:
            raise LedgerError("gateway timeout")
        return real(rec)
    ledger.put_artifact = flaky
    res = run(env)
    ingest = res["agent_trace"][0]
    assert ingest["attempts"] == 2 and ingest["status"] == "ok"
    assert res["status"] == "ACCEPTED"


def test_ledger_down_stops_the_case(env):
    store, ledger, model = env

    def down(rec):
        raise LedgerError("connection refused")
    ledger.put_artifact = down
    res = run(env)
    assert res["status"] == "FAILED" and res["agent_trace"][0]["attempts"] == orch.MAX_ATTEMPTS
    assert len(res["agent_trace"]) == 1               # nothing else ran


def test_pseudonymisation_at_ingest(env):
    res = run(env, report="Patient Ben Salah Ahmed, tel +216 98 123 456. sinus rhythm. normal ecg.",
              patient_identifiers=["Ben Salah Ahmed"])
    store = env[0]
    raw = store.load(res["_ctx"].rec[T.RAW_REPORT].offchain_uri)
    assert "Ben Salah" not in raw and "98 123" not in raw and res["case_id"] in raw
    assert res["final"]["pseudonymisation"]["replacements"] == {"identifier": 1, "phone": 1}


def test_pseudonymise_text_unit():
    t, info = pseudonymise_text("mail: a.b@x.tn DOB: 01/02/1990", "CASE-1")
    assert "a.b@x.tn" not in t and "1990" not in t and set(info["replacements"]) == {"email", "date_of_birth"}


def test_mermaid_diagram():
    m = orch.mermaid()
    assert "ingest" in m and "audit" in m


def test_retry_does_not_duplicate_artifacts(env):
    store, ledger, model = env
    calls = {"n": 0}
    real = ledger.put_artifact

    def fail_second_write(rec):
        calls["n"] += 1
        if calls["n"] == 2:                      # RAW_ECG ok, RAW_REPORT fails once
            raise LedgerError("timeout")
        return real(rec)
    ledger.put_artifact = fail_second_write
    res = run(env)
    types = [a["artifact_type"] for a in res["artifacts"]]
    assert types.count("RAW_ECG") == 1 and len(types) == 8


def test_case_id_with_digits_is_not_taken_for_a_phone():
    # regression: CASE-D809888691A6 contains 9 digits in a row
    t, info = pseudonymise_text("Patient Foulen Ben Foulen, tel 98 123 456. normal ecg.", "CASE-D809888691A6",
                                ["Foulen Ben Foulen"])
    assert t == "Patient [CASE-D809888691A6], tel [PHONE REMOVED]. normal ecg."
    assert info["replacements"] == {"phone": 1, "identifier": 1}