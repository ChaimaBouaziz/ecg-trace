"""
Step 8 — FastAPI backend + web dashboard (spec §6 and §9).

Run (from the repo root, in WSL):
    uvicorn api.main:app --port 8000                  # FakeLedger
    LEDGER=fabric uvicorn api.main:app --port 8000    # Hyperledger Fabric (gateway must be running)
Then open http://localhost:8000 in your Windows browser.

Security (spec §7): if the env var API_KEY is set, every write operation requires the header
X-API-Key. Raw medical data never leaves the off-chain store except through these endpoints.
"""
from __future__ import annotations

import os
import re
import shutil
import threading
import time
import uuid
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

import numpy as np
from fastapi import BackgroundTasks, Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from agents.orchestrator import build_graph, case_summary, run_case_agentic
from api.ecg_io import ECGFormatError, load_upload
from ledger.base import LedgerClient, LedgerError
from ledger.factory import get_ledger
from pipeline.model import DEFAULT_MODEL_PATH, load_model
from pipeline.run_case import CaseContext
from pipeline.schemas import ArtifactType as T
from pipeline.storage import OffchainStore
from pipeline.verify import verify_case

UI_DIR = Path(__file__).resolve().parents[1] / "ui"
BACKUP_DIR = Path("data/tamper_backup")


class Job:
    def __init__(self, ctx: CaseContext):
        self.id = uuid.uuid4().hex[:12]
        self.ctx = ctx
        self.status = "running"          # running | done | error
        self.result: Optional[dict] = None
        self.error: Optional[str] = None
        self.created = time.time()


class TamperRequest(BaseModel):
    artifact_type: str = "PREDICTION"


def _public(summary: dict) -> dict:
    return {k: v for k, v in summary.items() if not k.startswith("_")}


def create_app(store: Optional[OffchainStore] = None, ledger: Optional[LedgerClient] = None,
               model_path: Optional[str] = None, demo_mode: Optional[bool] = None) -> FastAPI:
    app = FastAPI(title="ECG-Trace API", version="1.0.0",
                  description="Agentic ECG analysis with SHA-256 fingerprints anchored on Hyperledger Fabric")
    store = store or OffchainStore(os.getenv("OFFCHAIN_DIR", "data/offchain"))
    ledger = ledger or get_ledger()
    demo_mode = demo_mode if demo_mode is not None else os.getenv("DEMO_MODE", "1") == "1"
    jobs: dict[str, Job] = {}
    graph = build_graph()
    lock = threading.Lock()

    @lru_cache(maxsize=1)
    def model() -> dict:
        return load_model(model_path or os.getenv("MODEL_PATH", str(DEFAULT_MODEL_PATH)))

    def require_key(x_api_key: Optional[str] = Header(default=None)):
        expected = os.getenv("API_KEY")
        if expected and x_api_key != expected:
            raise HTTPException(401, "missing or invalid X-API-Key")

    def run_job(job: Job):
        try:
            res = run_case_agentic(None, None, None, None, store, ledger, graph=graph, ctx=job.ctx)
            job.result, job.status = res, "done"
        except Exception as e:                                     # never leave a job hanging
            job.error, job.status = f"{type(e).__name__}: {e}", "error"

    def start_job(signal, fs, report, background: BackgroundTasks, flags=None, identifiers=None) -> dict:
        ctx = CaseContext(raw_signal=signal, fs=fs, report_text=report, model_bundle=model(),
                          store=store, ledger=ledger, quality_flags=flags, patient_identifiers=identifiers)
        job = Job(ctx)
        with lock:
            jobs[job.id] = job
        background.add_task(run_job, job)
        return {"job_id": job.id, "case_id": ctx.case_id}

    # ---------------------------------------------------------------- info
    @app.get("/api/health")
    def health():
        info: dict[str, Any] = {"ledger": type(ledger).__name__, "demo_mode": demo_mode,
                                "auth_required": bool(os.getenv("API_KEY"))}
        try:
            info["ledger_status"] = ledger.health() if hasattr(ledger, "health") else {"status": "ok"}
        except Exception as e:
            info["ledger_status"] = {"status": "unreachable", "error": str(e)}
        try:
            m = model()
            info["model"] = {"name": m["model_name"], "version": m["model_version"],
                             "file_sha256": m["file_sha256"],
                             "test_metrics": m.get("metadata", {}).get("test_metrics")}
        except Exception as e:
            info["model"] = {"error": str(e)}
        return info

    # ---------------------------------------------------------------- samples (PTB-XL test fold)
    @lru_cache(maxsize=1)
    def samples_df():
        from pipeline.ptbxl import load_metadata
        df = load_metadata()
        return df[df["strat_fold"] == 10]

    @app.get("/api/samples")
    def samples(limit: int = 40):
        try:
            df = samples_df().head(limit)
        except Exception as e:
            raise HTTPException(404, f"PTB-XL not available: {e}")
        out = []
        for i, r in df.iterrows():
            age = r.get("age")
            out.append({"ecg_id": int(i), "labels": r["superclasses"], "report": str(r["report"])[:120],
                        "age": int(age) if isinstance(age, (int, float)) and age == age else None})
        return out

    def _sample(ecg_id: int):
        from pipeline.ptbxl import QUALITY_COLS, load_signal
        df = samples_df()
        if ecg_id not in df.index:
            raise HTTPException(404, f"ecg_id {ecg_id} not in the local test fold")
        row = df.loc[ecg_id]
        sig, meta = load_signal(row)
        return row, sig, meta["fs"], {k: row[k] for k in QUALITY_COLS}

    @app.post("/api/samples/{ecg_id}/run", dependencies=[Depends(require_key)])
    def run_sample(ecg_id: int, background: BackgroundTasks):
        row, sig, fs, flags = _sample(ecg_id)
        return start_job(sig, fs, row["report"], background, flags)

    @app.get("/api/samples/{ecg_id}/ecg.csv", response_class=PlainTextResponse)
    def sample_csv(ecg_id: int):
        _, sig, _, _ = _sample(ecg_id)
        lines = [",".join(["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"])]
        lines += [",".join(f"{v:.4f}" for v in r) for r in sig]
        return PlainTextResponse("\n".join(lines), headers={
            "Content-Disposition": f'attachment; filename="ptbxl_{ecg_id}_100hz.csv"'})

    @app.get("/api/samples/{ecg_id}/report.txt", response_class=PlainTextResponse)
    def sample_report(ecg_id: int):
        row, *_ = _sample(ecg_id)
        return PlainTextResponse(row["report"], headers={
            "Content-Disposition": f'attachment; filename="ptbxl_{ecg_id}_report.txt"'})

    # ---------------------------------------------------------------- upload
    @app.post("/api/cases", dependencies=[Depends(require_key)])
    async def upload_case(background: BackgroundTasks,
                          ecg: Optional[UploadFile] = File(None), hea: Optional[UploadFile] = File(None),
                          dat: Optional[UploadFile] = File(None), report_file: Optional[UploadFile] = File(None),
                          report: str = Form(""), fs: float = Form(100.0), patient_identifiers: str = Form("")):
        files = {}
        for kind, f in (("ecg", ecg), ("hea", hea), ("dat", dat)):
            if f is not None and f.filename:
                files[kind] = (f.filename, await f.read())
        try:
            signal, fs, notes = load_upload(files, fs)
        except ECGFormatError as e:
            raise HTTPException(400, str(e))
        except Exception as e:
            raise HTTPException(400, f"could not read the ECG file: {e}")
        text = report
        if report_file is not None and report_file.filename:
            text = (await report_file.read()).decode("utf-8", errors="replace")
        idents = [s.strip() for s in patient_identifiers.split(",") if s.strip()]
        out = start_job(signal, fs, text, background, None, idents)
        out["notes"] = notes
        return out

    # ---------------------------------------------------------------- jobs (live progress)
    @app.get("/api/jobs/{job_id}")
    def job_status(job_id: str):
        job = jobs.get(job_id)
        if not job:
            raise HTTPException(404, "unknown job")
        body = {"job_id": job.id, "case_id": job.ctx.case_id, "status": job.status, "error": job.error,
                "agent_trace": job.ctx.agent_trace, "n_artifacts": len(job.ctx.recorder.records)}
        if job.result:
            body["result"] = _public(job.result)
        return body

    # ---------------------------------------------------------------- cases
    def _load(rec) -> Any:
        try:
            return store.load(rec.offchain_uri) if store.exists(rec.offchain_uri) else None
        except Exception:
            return None

    @app.get("/api/cases")
    def list_cases(limit: int = 30):
        dirs = [d for d in store.root.iterdir() if d.is_dir() and d.name.startswith("CASE-")]
        dirs.sort(key=lambda d: d.stat().st_mtime, reverse=True)
        out = []
        for d in dirs[:limit]:
            final = next(d.glob("FINAL-*.json"), None)
            item = {"case_id": d.name, "created": d.stat().st_mtime, "decision": None, "predicted_class": None}
            if final:
                try:
                    import json
                    f = json.loads(final.read_bytes())
                    item.update(decision=f.get("decision"), predicted_class=f.get("predicted_class"))
                except Exception:
                    item["decision"] = "UNREADABLE"
            out.append(item)
        return out

    @app.get("/api/cases/{case_id}")
    def case_detail(case_id: str):
        try:
            history = ledger.get_case_history(case_id)
        except LedgerError as e:
            raise HTTPException(503, str(e))
        if not history:
            raise HTTPException(404, "no ledger record for this case (unknown case, or ledger was reset)")
        by_type = {r.artifact_type.value: r for r in history}
        get = lambda t: _load(by_type[t]) if t in by_type else None
        final = get("FINAL")
        return {
            "case_id": case_id,
            "artifacts": [r.model_dump(mode="json") for r in history],
            "prediction": get("PREDICTION"),
            "validation": get("VALIDATION"),
            "explanation": (get("EXPLANATION") or {}).get("text"),
            "final": final,
            "agent_trace": (final or {}).get("agent_trace", []),
            "report": get("RAW_REPORT"),
        }

    @app.get("/api/cases/{case_id}/ecg")
    def case_ecg(case_id: str, lead: int = 1):
        history = {r.artifact_type.value: r for r in ledger.get_case_history(case_id)}
        out: dict[str, Any] = {"fs": 100, "lead": lead}
        if "RAW_ECG" in history and (raw := _load(history["RAW_ECG"])) is not None:
            out["raw"] = np.round(np.asarray(raw)[:, lead], 4).tolist()
        if "PROCESSED" in history and (p := _load(history["PROCESSED"])) is not None:
            out["processed"] = np.round(p["signal"][:, lead], 4).tolist()
            out["r_peaks"] = p["r_peaks"].astype(int).tolist()
        return out

    @app.post("/api/cases/{case_id}/verify")
    def verify(case_id: str):
        try:
            return verify_case(case_id, ledger, store)
        except LedgerError as e:
            raise HTTPException(503, str(e))

    @app.get("/api/artifacts/{artifact_id}/key-history")
    def key_history(artifact_id: str):
        if not hasattr(ledger, "get_key_history"):
            return [{"tx_id": (r := ledger.get_artifact(artifact_id)) and r.tx_id, "is_delete": False}]
        return ledger.get_key_history(artifact_id)

    # ---------------------------------------------------------------- demo: tamper / restore
    def _record(case_id: str, artifact_type: str):
        recs = [r for r in ledger.get_case_history(case_id) if r.artifact_type.value == artifact_type]
        if not recs:
            raise HTTPException(404, f"no {artifact_type} artifact in {case_id}")
        return recs[0]

    @app.post("/api/cases/{case_id}/tamper", dependencies=[Depends(require_key)])
    def tamper(case_id: str, body: TamperRequest):
        if not demo_mode:
            raise HTTPException(403, "tampering is only available in DEMO_MODE")
        rec = _record(case_id, body.artifact_type)
        path = store.path(rec.offchain_uri)
        if not path.exists():
            raise HTTPException(404, "off-chain file is missing")
        backup = BACKUP_DIR / rec.artifact_id
        if not backup.exists():
            BACKUP_DIR.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, backup)
        os.chmod(path, 0o600)
        data = backup.read_bytes()          # always start from the original: clicking twice = same change
        m = re.search(rb'"(predicted_class|decision|status)":"([A-Z_]+)"', data) if path.suffix == ".json" else None
        if m:
            field, old = m.group(1).decode(), m.group(2).decode()
            swap = {"NORM": "ABNORMAL", "ABNORMAL": "NORM", "ACCEPTED": "REJECTED", "REJECTED": "ACCEPTED",
                    "REVIEW_REQUIRED": "ACCEPTED"}.get(old, "ACCEPTED")
            new = data[:m.start(2)] + swap.encode() + data[m.end(2):]
            what = f'changed {field} "{old}" into "{swap}"'
        elif path.suffix == ".json":
            new, what = data[:-1] + b',"edited":true}', "added a field to the JSON"
        elif path.suffix == ".txt":
            new, what = data + b" (edited)", "appended text to the report"
        else:
            b = bytearray(data)
            b[-1] ^= 0x01
            new, what = bytes(b), "flipped one bit of the last byte"
        path.write_bytes(new)
        return {"artifact_id": rec.artifact_id, "artifact_type": body.artifact_type, "change": what}

    @app.post("/api/cases/{case_id}/restore", dependencies=[Depends(require_key)])
    def restore(case_id: str, body: TamperRequest):
        rec = _record(case_id, body.artifact_type)
        backup = BACKUP_DIR / rec.artifact_id
        if not backup.exists():
            raise HTTPException(404, "nothing to restore for this artifact")
        path = store.path(rec.offchain_uri)
        if path.exists():
            os.chmod(path, 0o600)
        shutil.copy2(backup, path)
        os.chmod(path, 0o400)
        backup.unlink()
        return {"artifact_id": rec.artifact_id, "restored": True}

    # ---------------------------------------------------------------- UI
    if UI_DIR.is_dir():
        app.mount("/static", StaticFiles(directory=UI_DIR), name="static")

        @app.get("/", include_in_schema=False)
        def index():
            return FileResponse(UI_DIR / "index.html")

    return app


app = create_app()