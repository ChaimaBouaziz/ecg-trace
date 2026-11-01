"""
Step 7 — Agentic orchestration (spec §4.7) with LangGraph.

Specialised agents, coordinated by an orchestrator graph:

    ingest ─> preprocess ─┬─> features ─> predict ─┬─> integrity ─> validate ─> explain ─> audit ─> END
                          └──── (bad signal) ──────┘
    any agent ── ledger unreachable after retries ──> END  (status FAILED, nothing unverifiable is produced)

Each agent:
  - runs one stage of the pipeline (pipeline/run_case.py) without manual intervention,
  - retries automatically when the ledger is temporarily unavailable,
  - turns its own errors into a traced outcome (the validator then REJECTS the case),
  - logs what it did (agent, action, duration, artifacts created, outcome) in ctx.agent_trace.
The trace up to the explanation is stored inside the FINAL artifact, so it is hashed and anchored
on the blockchain like every other artifact. The audit agent finally re-verifies the whole case.
"""
from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Callable, Optional, TypedDict

from langgraph.graph import END, START, StateGraph

from ledger.base import LedgerClient, LedgerError
from pipeline import run_case as rc
from pipeline.run_case import CaseContext
from pipeline.storage import OffchainStore
from pipeline.verify import verify_case

ORCHESTRATOR_VERSION = "orchestrator-1.0.0"
MAX_ATTEMPTS = 3
BACKOFF_S = 1.0


class AgentState(TypedDict, total=False):
    ctx: CaseContext
    fatal: bool
    audit: dict


# name -> (stage function, role description, is the agent essential?)
# a failure of an essential agent stops the case; a failure of a non-essential one is traced
# and the case continues to validation, which will REJECT it (missing artifacts).
AGENTS: dict[str, tuple[Callable[[CaseContext], None], str, bool]] = {
    "ingest":     (rc.stage_ingest,     "receive ECG + report, pseudonymise, store raw artifacts", True),
    "preprocess": (rc.stage_preprocess, "check, clean, normalise and segment the ECG", False),
    "features":   (rc.stage_features,   "extract the feature vector for the model", False),
    "predict":    (rc.stage_predict,    "run the AI model and produce the prediction", False),
    "integrity":  (rc.stage_integrity,  "re-hash previous artifacts and compare with the ledger", True),
    "validate":   (rc.stage_validate,   "apply validation rules (accept / review / reject)", True),
    "explain":    (rc.stage_explain,    "write the explanation and the final result", True),
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def make_agent(name: str, fn: Callable[[CaseContext], None], role: str, essential: bool,
               sleep: Callable[[float], None] = time.sleep):
    def agent(state: AgentState) -> AgentState:
        ctx = state["ctx"]
        entry = {"agent": f"agent.{name}", "role": role, "started_at": _now(), "status": "running",
                 "attempts": 0, "artifacts": [], "message": ""}
        ctx.agent_trace.append(entry)
        before = {r.artifact_id for r in ctx.recorder.records}
        t0 = time.perf_counter()
        fatal = False
        for attempt in range(1, MAX_ATTEMPTS + 1):
            entry["attempts"] = attempt
            try:
                fn(ctx)
                entry["status"] = "ok"
                break
            except LedgerError as e:                       # transient: ledger / gateway unavailable
                entry["message"] = f"ledger error: {e}"
                if attempt == MAX_ATTEMPTS:
                    entry["status"] = "failed"
                    ctx.error = f"agent.{name}: ledger unavailable after {attempt} attempts"
                    fatal = True                           # cannot guarantee traceability -> stop
                else:
                    sleep(BACKOFF_S * attempt)
            except Exception as e:                         # logic / data error in this agent
                entry["status"] = "failed"
                entry["message"] = f"{type(e).__name__}: {e}"
                ctx.error = f"agent.{name}: {e}"
                fatal = essential
                break
        if entry["status"] == "ok" and name == "preprocess" and ctx.processed is None:
            entry["status"] = "rejected_input"
            entry["message"] = ctx.error or "signal could not be processed"
        if entry["status"] == "ok" and name in ("features", "predict") and ctx.error:
            entry["status"] = "skipped"
            entry["message"] = "skipped because a previous agent failed"
        entry["artifacts"] = [r.artifact_id for r in ctx.recorder.records if r.artifact_id not in before]
        entry["duration_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        return {"ctx": ctx, "fatal": fatal}

    agent.__name__ = f"agent_{name}"
    return agent


def audit_agent(state: AgentState) -> AgentState:
    """Final, independent check of the whole case on the ledger (not stored: it comes after FINAL)."""
    ctx = state["ctx"]
    audit = verify_case(ctx.case_id, ctx.ledger, ctx.store)
    ctx.log.append(f"agent.audit: case integrity = {audit['overall']}")
    return {"ctx": ctx, "audit": audit}


# ---- routing (the orchestrator's decisions) -------------------------------------------
def _stop_or(next_node: str):
    return lambda s: END if s.get("fatal") else next_node


def after_preprocess(state: AgentState) -> str:
    if state.get("fatal"):
        return END
    return "features" if state["ctx"].processed is not None else "integrity"   # bad signal -> skip AI


def build_graph(sleep: Callable[[float], None] = time.sleep):
    g = StateGraph(AgentState)
    for name, (fn, role, essential) in AGENTS.items():
        g.add_node(name, make_agent(name, fn, role, essential, sleep))
    g.add_node("audit", audit_agent)

    g.add_edge(START, "ingest")
    g.add_conditional_edges("ingest", _stop_or("preprocess"), ["preprocess", END])
    g.add_conditional_edges("preprocess", after_preprocess, ["features", "integrity", END])
    g.add_conditional_edges("features", _stop_or("predict"), ["predict", END])
    g.add_conditional_edges("predict", _stop_or("integrity"), ["integrity", END])
    g.add_conditional_edges("integrity", _stop_or("validate"), ["validate", END])
    g.add_conditional_edges("validate", _stop_or("explain"), ["explain", END])
    g.add_conditional_edges("explain", _stop_or("audit"), ["audit", END])
    g.add_edge("audit", END)
    return g.compile()


def run_case_agentic(raw_signal, fs, report_text, model_bundle, store: OffchainStore, ledger: LedgerClient,
                     quality_flags: Optional[dict] = None, patient_identifiers: Optional[list[str]] = None,
                     graph=None, ctx: Optional[CaseContext] = None) -> dict:
    """Pass a pre-built `ctx` to follow progress live (ctx.agent_trace grows as agents run)."""
    if ctx is None:
        ctx = CaseContext(raw_signal=raw_signal, fs=fs, report_text=report_text, model_bundle=model_bundle,
                          store=store, ledger=ledger, quality_flags=quality_flags,
                          patient_identifiers=patient_identifiers)
    out = (graph or build_graph()).invoke({"ctx": ctx, "fatal": False})
    return case_summary(out["ctx"], out.get("audit"), out.get("fatal", False))


def case_summary(ctx: CaseContext, audit: Optional[dict], fatal: bool) -> dict:
    """JSON-friendly result used by scripts and the API (step 8)."""
    return {
        "case_id": ctx.case_id,
        "status": "FAILED" if fatal else (ctx.final or {}).get("decision", "FAILED"),
        "error": ctx.error,
        "prediction": ctx.prediction,
        "validation": ctx.validation,
        "explanation": (ctx.explanation or {}).get("text"),
        "final": ctx.final,
        "artifacts": [r.model_dump(mode="json") for r in ctx.recorder.records],
        "agent_trace": ctx.agent_trace,
        "audit": audit,
        "log": ctx.log,
        "orchestrator_version": ORCHESTRATOR_VERSION,
        "_ctx": ctx,          # in-process use only (not serialised by the API)
    }


def mermaid() -> str:
    return build_graph().get_graph().draw_mermaid()