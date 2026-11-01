"""
Step 5 — End-to-end processing of one case (sequential version).

Each stage is a separate function working on a shared CaseContext, so that in step 7
each one becomes an agent node (LangGraph) without rewriting the logic:

  ingest -> preprocess -> features -> predict -> validate -> explain/final

Artifact chain recorded on the ledger:
  RAW_ECG ─┐
           ├─> PROCESSED -> FEATURES -> PREDICTION ─┐
  RAW_REPORT ──────────────────────────────────────┴─> VALIDATION -> EXPLANATION -> FINAL
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from ledger.base import LedgerClient
from pipeline.artifacts import ArtifactRecorder
from pipeline.explain import EXPLAIN_VERSION, build_explanation, build_final_result
from pipeline.features import FEATURES_VERSION, extract_features
from pipeline.model import predict
from pipeline.preprocess import (PREPROCESS_VERSION, PreprocessingError, ProcessedECG,
                                 prepare_report, preprocess_ecg)
from pipeline.schemas import ArtifactRecord, ArtifactType as T, new_case_id
from pipeline.storage import OffchainStore
from pipeline.validate import VALIDATION_VERSION, validate
from pipeline.verify import all_valid

INGEST_VERSION = "ingest-1.0.0"


@dataclass
class CaseContext:
    raw_signal: np.ndarray
    fs: float
    report_text: str
    model_bundle: dict
    store: OffchainStore
    ledger: LedgerClient
    quality_flags: Optional[dict] = None
    case_id: str = field(default_factory=new_case_id)
    recorder: Optional[ArtifactRecorder] = None
    rec: dict[str, ArtifactRecord] = field(default_factory=dict)   # artifact type -> record
    processed: Optional[ProcessedECG] = None
    report_info: Optional[dict] = None
    features: Optional[np.ndarray] = None
    prediction: Optional[dict] = None
    validation: Optional[dict] = None
    explanation: Optional[dict] = None
    final: Optional[dict] = None
    error: Optional[str] = None
    log: list[str] = field(default_factory=list)

    def __post_init__(self):
        self.recorder = ArtifactRecorder(self.case_id, self.store, self.ledger)

    def add(self, t: T, payload: Any, kind: str, parents: list[T], version: str, creator: str):
        r = self.recorder.record(t, payload, kind, [self.rec[p].artifact_id for p in parents if p in self.rec],
                                 version, creator)
        self.rec[t] = r
        self.log.append(f"{creator}: {t.value} {r.artifact_id} sha256={r.sha256[:12]}… tx={r.tx_id[:10]}…")
        return r


# --- stages -----------------------------------------------------------------------
def stage_ingest(ctx: CaseContext) -> None:
    ctx.add(T.RAW_ECG, np.asarray(ctx.raw_signal, dtype=np.float32), "npy", [], INGEST_VERSION, "agent.ingest")
    ctx.add(T.RAW_REPORT, ctx.report_text, "text", [], INGEST_VERSION, "agent.ingest")


def stage_preprocess(ctx: CaseContext) -> None:
    ctx.report_info = prepare_report(ctx.report_text)
    try:
        p = preprocess_ecg(ctx.raw_signal, ctx.fs, ctx.quality_flags)
    except PreprocessingError as e:
        ctx.error = f"preprocessing failed: {e}"
        ctx.log.append(f"agent.preprocess: {ctx.error}")
        return
    ctx.processed = p
    ctx.add(T.PROCESSED, {"signal": p.signal, "beats": p.beats, "r_peaks": p.r_peaks,
                          "processing_report": p.report, "report_info": ctx.report_info},
            "npz", [T.RAW_ECG, T.RAW_REPORT], PREPROCESS_VERSION, "agent.preprocess")


def stage_features(ctx: CaseContext) -> None:
    if ctx.processed is None:
        return
    ctx.features = extract_features(ctx.processed, ctx.fs)
    ctx.add(T.FEATURES, ctx.features, "npy", [T.PROCESSED], FEATURES_VERSION, "agent.features")


def stage_predict(ctx: CaseContext) -> None:
    if ctx.features is None:
        return
    ctx.prediction = predict(ctx.model_bundle, ctx.features, ctx.rec[T.FEATURES].artifact_id)
    ctx.add(T.PREDICTION, ctx.prediction, "json", [T.FEATURES],
            f"{ctx.prediction['model_name']}-{ctx.prediction['model_version']}", "agent.predict")


def stage_validate(ctx: CaseContext) -> None:
    integrity_ok = all_valid(ctx.recorder.records, ctx.store)   # re-hash everything recorded so far
    ctx.validation = validate(ctx.prediction, ctx.processed.report if ctx.processed else None,
                              ctx.report_info, ctx.recorder.types_present(), integrity_ok)
    if ctx.error:
        ctx.validation["error"] = ctx.error
    ctx.add(T.VALIDATION, ctx.validation, "json", [T.PREDICTION, T.PROCESSED, T.RAW_REPORT, T.RAW_ECG],
            VALIDATION_VERSION, "agent.validate")


def stage_explain(ctx: CaseContext) -> None:
    ctx.explanation = build_explanation(ctx.prediction, ctx.processed.report if ctx.processed else None,
                                        ctx.validation)
    ctx.add(T.EXPLANATION, ctx.explanation, "json", [T.VALIDATION], EXPLAIN_VERSION, "agent.explain")
    ctx.final = build_final_result(ctx.case_id, ctx.prediction, ctx.validation, ctx.explanation)
    ctx.add(T.FINAL, ctx.final, "json", [T.EXPLANATION, T.VALIDATION, T.PREDICTION],
            EXPLAIN_VERSION, "agent.explain")


STAGES = [stage_ingest, stage_preprocess, stage_features, stage_predict, stage_validate, stage_explain]


def run_case(raw_signal, fs, report_text, model_bundle, store, ledger, quality_flags=None) -> CaseContext:
    ctx = CaseContext(raw_signal=raw_signal, fs=fs, report_text=report_text, model_bundle=model_bundle,
                      store=store, ledger=ledger, quality_flags=quality_flags)
    for stage in STAGES:
        stage(ctx)
    return ctx