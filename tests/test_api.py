import io

import numpy as np
import pytest
from fastapi.testclient import TestClient

from api.main import create_app
from ledger.fake_ledger import FakeLedger
from pipeline.storage import OffchainStore
from tests.test_preprocess import synthetic_ecg


@pytest.fixture
def client(tmp_path, tiny_model_path, monkeypatch):
    monkeypatch.chdir(tmp_path)                       # tamper backups go to tmp_path/data/...
    monkeypatch.delenv("API_KEY", raising=False)
    app = create_app(OffchainStore(tmp_path / "off"), FakeLedger(tmp_path / "l.json"),
                     model_path=str(tiny_model_path), demo_mode=True)
    return TestClient(app)


def csv_bytes(hr=55):
    x, _ = synthetic_ecg(hr_bpm=hr)
    buf = io.StringIO()
    buf.write("I,II,III,AVR,AVL,AVF,V1,V2,V3,V4,V5,V6\n")
    np.savetxt(buf, x, delimiter=",", fmt="%.4f")
    return buf.getvalue().encode()


def upload(client, **extra):
    files = {"ecg": ("ecg.csv", csv_bytes(), "text/csv")}
    data = {"report": "Patient Foulen Ben Foulen. sinus rhythm. normal ecg.",
            "patient_identifiers": "Foulen Ben Foulen", **extra}
    r = client.post("/api/cases", files=files, data=data)
    assert r.status_code == 200, r.text
    return r.json()


def test_health(client):
    h = client.get("/api/health").json()
    assert h["ledger"] == "FakeLedger" and h["model"]["name"] == "test-model"


def test_upload_run_and_view(client):
    job = upload(client)
    s = client.get(f"/api/jobs/{job['job_id']}").json()
    assert s["status"] == "done" and s["result"]["status"] == "ACCEPTED"
    assert len(s["agent_trace"]) == 7
    c = client.get(f"/api/cases/{job['case_id']}").json()
    assert len(c["artifacts"]) == 8 and c["prediction"]["predicted_class"] == "NORM"
    assert "Foulen" not in c["report"] and job["case_id"] in c["report"]       # pseudonymised
    e = client.get(f"/api/cases/{job['case_id']}/ecg").json()
    assert len(e["raw"]) == 1000 and len(e["r_peaks"]) > 5
    assert client.get("/api/cases").json()[0]["decision"] == "ACCEPTED"


def test_tamper_verify_restore(client):
    cid = upload(client)["case_id"]
    assert client.post(f"/api/cases/{cid}/verify").json()["overall"] == "VALID"
    t = client.post(f"/api/cases/{cid}/tamper", json={"artifact_type": "PREDICTION"}).json()
    assert t["change"] == 'changed predicted_class "NORM" into "ABNORMAL"'
    v = client.post(f"/api/cases/{cid}/verify").json()
    assert v["overall"] == "MODIFIED"
    assert [a["artifact_type"] for a in v["artifacts"] if a["status"] != "VALID"] == ["PREDICTION"]
    client.post(f"/api/cases/{cid}/restore", json={"artifact_type": "PREDICTION"})
    assert client.post(f"/api/cases/{cid}/verify").json()["overall"] == "VALID"


def test_tamper_binary_artifact(client):
    cid = upload(client)["case_id"]
    client.post(f"/api/cases/{cid}/tamper", json={"artifact_type": "RAW_ECG"})
    assert client.post(f"/api/cases/{cid}/verify").json()["overall"] == "MODIFIED"


def test_bad_upload(client):
    r = client.post("/api/cases", files={"ecg": ("x.pdf", b"%PDF", "application/pdf")})
    assert r.status_code == 400
    r = client.post("/api/cases", files={"ecg": ("x.csv", b"a,b\n1,2\n", "text/csv")})
    assert r.status_code == 400


def test_resample_500hz(client):
    x, _ = synthetic_ecg(hr_bpm=60)
    x500 = np.repeat(x, 5, axis=0)                         # fake 500 Hz signal
    buf = io.BytesIO()
    np.save(buf, x500)
    r = client.post("/api/cases", files={"ecg": ("e.npy", buf.getvalue())}, data={"fs": "500", "report": "normal ecg"})
    assert r.status_code == 200 and "resampled" in r.json()["notes"][0]


def test_api_key_required(client, monkeypatch):
    monkeypatch.setenv("API_KEY", "secret")
    files = {"ecg": ("ecg.csv", csv_bytes(), "text/csv")}
    assert client.post("/api/cases", files=files).status_code == 401
    r = client.post("/api/cases", files=files, headers={"X-API-Key": "secret"}, data={"report": "x"})
    assert r.status_code == 200
    assert client.get("/api/health").status_code == 200    # reads stay open


def test_unknown_case(client):
    assert client.get("/api/cases/CASE-NOPE").status_code == 404
    assert client.get("/api/jobs/nope").status_code == 404


def test_ui_served(client):
    r = client.get("/")
    assert r.status_code == 200 and "ECG-Trace" in r.text


def test_double_tamper_stays_modified(client):
    cid = upload(client)["case_id"]
    for _ in range(2):
        client.post(f"/api/cases/{cid}/tamper", json={"artifact_type": "PREDICTION"})
    assert client.post(f"/api/cases/{cid}/verify").json()["overall"] == "MODIFIED"