import os

import numpy as np
import pytest

from ledger.fake_ledger import FakeLedger
from pipeline.model import load_model
from pipeline.run_case import run_case
from pipeline.schemas import ArtifactType as T
from pipeline.storage import OffchainStore
from pipeline.verify import verify_artifact, verify_case
from tests.test_preprocess import FS, synthetic_ecg


@pytest.fixture
def env(tmp_path, tiny_model_path):
    return OffchainStore(tmp_path / "offchain"), FakeLedger(tmp_path / "ledger.json"), load_model(tiny_model_path)


def _run(env, hr=55, report="sinus rhythm. normal ecg.", signal=None):
    store, ledger, model = env
    sig = signal if signal is not None else synthetic_ecg(hr_bpm=hr)[0]
    return run_case(sig, FS, report, model, store, ledger)


def test_full_chain_recorded(env):
    ctx = _run(env)
    _, ledger, _ = env
    hist = ledger.get_case_history(ctx.case_id)
    assert [r.artifact_type for r in hist] == [T.RAW_ECG, T.RAW_REPORT, T.PROCESSED, T.FEATURES,
                                               T.PREDICTION, T.VALIDATION, T.EXPLANATION, T.FINAL]
    assert all(r.tx_id for r in hist)
    # provenance links
    assert ctx.rec[T.PROCESSED].parent_ids == [ctx.rec[T.RAW_ECG].artifact_id, ctx.rec[T.RAW_REPORT].artifact_id]
    assert ctx.rec[T.FEATURES].parent_ids == [ctx.rec[T.PROCESSED].artifact_id]
    assert ctx.prediction["input_artifact_id"] == ctx.rec[T.FEATURES].artifact_id
    assert ctx.final["decision"] == "ACCEPTED" and ctx.final["predicted_class"] == "NORM"


def test_verify_all_valid(env):
    ctx = _run(env)
    store, ledger, _ = env
    v = verify_case(ctx.case_id, ledger, store)
    assert v["overall"] == "VALID" and len(v["artifacts"]) == 8


def test_tampering_detected(env):
    ctx = _run(env)
    store, ledger, _ = env
    p = store.path(ctx.rec[T.PREDICTION].offchain_uri)
    os.chmod(p, 0o600)
    p.write_bytes(p.read_bytes().replace(b'"NORM"', b'"ABNORMAL"', 1))   # attacker flips the result
    v = verify_case(ctx.case_id, ledger, store)
    assert v["overall"] == "MODIFIED"
    bad = [a for a in v["artifacts"] if a["status"] != "VALID"]
    assert [a["artifact_type"] for a in bad] == ["PREDICTION"]


def test_missing_and_unverifiable(env):
    ctx = _run(env)
    store, ledger, _ = env
    store.path(ctx.rec[T.FEATURES].offchain_uri).unlink()
    assert verify_case(ctx.case_id, ledger, store)["overall"] == "MISSING"
    assert verify_artifact("FEATURES-DOESNOTEXIST", ledger, store)["status"] == "UNVERIFIABLE"


def test_contradicting_report_needs_review(env):
    ctx = _run(env, hr=55, report="inferiorer infarkt")
    assert ctx.final["decision"] == "REVIEW_REQUIRED" and ctx.final["review_required"]


def test_bad_signal_rejected_but_traced(env):
    sig = synthetic_ecg()[0]
    sig[:300] = np.nan                     # too many missing values
    ctx = _run(env, signal=sig)
    assert ctx.final["decision"] == "REJECTED"
    types = [r.artifact_type for r in env[1].get_case_history(ctx.case_id)]
    assert types == [T.RAW_ECG, T.RAW_REPORT, T.VALIDATION, T.EXPLANATION, T.FINAL]
    assert verify_case(ctx.case_id, env[1], env[0])["overall"] == "VALID"