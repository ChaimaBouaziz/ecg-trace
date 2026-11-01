"""
Step 7 demo — the same case, processed by the multi-agent system (LangGraph).

  python scripts/run_agents_demo.py                          # FakeLedger
  LEDGER=fabric python scripts/run_agents_demo.py            # Hyperledger Fabric
  python scripts/run_agents_demo.py --ecg-id 42
  python scripts/run_agents_demo.py --fail predict           # simulate a crashing agent
  python scripts/run_agents_demo.py --bad-signal             # unusable ECG -> orchestrator skips the AI
  python scripts/run_agents_demo.py --mermaid                # write docs/agents_graph.md (graph diagram)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from agents import orchestrator as orch
from ledger.factory import get_ledger
from pipeline.model import load_model
from pipeline.ptbxl import QUALITY_COLS, load_metadata, load_signal
from pipeline.storage import OffchainStore

ICON = {"ok": "✔", "failed": "✘", "skipped": "↷", "rejected_input": "⚠", "running": "…"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ecg-id", type=int)
    ap.add_argument("--fail", choices=list(orch.AGENTS), help="make this agent crash")
    ap.add_argument("--bad-signal", action="store_true", help="corrupt the ECG (too many missing values)")
    ap.add_argument("--mermaid", action="store_true")
    a = ap.parse_args()

    if a.mermaid:
        out = Path("docs/agents_graph.md")
        out.write_text("# Agent orchestration graph\n\n```mermaid\n" + orch.mermaid() + "\n```\n")
        print(f"Diagram written to {out} (open it in VS Code preview or GitHub)")
        return

    if a.fail:
        def crash(ctx, _name=a.fail):
            raise RuntimeError(f"simulated crash of agent.{_name}")
        fn, role, essential = orch.AGENTS[a.fail]
        orch.AGENTS[a.fail] = (crash, role, essential)

    store, ledger = OffchainStore("data/offchain"), get_ledger()
    df = load_metadata()
    test = df[df["strat_fold"] == 10]
    ecg_id = a.ecg_id if a.ecg_id in df.index else test.index[1]
    row = df.loc[ecg_id]
    raw, meta = load_signal(row)
    if a.bad_signal:
        raw = raw.copy()
        raw[:300] = np.nan

    print(f"Ledger: {type(ledger).__name__}   Input: PTB-XL ecg_id={ecg_id} labels={row['superclasses']}")
    res = orch.run_case_agentic(raw, meta["fs"], row["report"], load_model(), store, ledger,
                                {k: row[k] for k in QUALITY_COLS})

    print(f"\nCase {res['case_id']} — agent trace:")
    print(f"  {'agent':18s} {'status':15s} {'try':>3s} {'ms':>8s}  artifacts / message")
    for t in res["agent_trace"]:
        detail = ", ".join(t["artifacts"]) or t["message"]
        print(f"  {ICON.get(t['status'], '?')} {t['agent']:16s} {t['status']:15s} {t['attempts']:>3d} "
              f"{t.get('duration_ms', 0):>8.1f}  {detail[:80]}")

    print(f"\nFinal status : {res['status']}")
    if res["prediction"]:
        print(f"Prediction   : {res['prediction']['predicted_class']} ({res['prediction']['confidence']:.2f})")
    if res["validation"]:
        print(f"Failed checks: {res['validation']['failed_checks']}")
    if res["explanation"]:
        print(f"Explanation  : {res['explanation']}")
    if res["error"]:
        print(f"Error        : {res['error']}")
    if res["audit"]:
        print(f"\nAudit agent  : integrity of the {len(res['audit']['artifacts'])} artifacts on the ledger = "
              f"{res['audit']['overall']}")
    print(f"\nRe-verify later: python scripts/run_case_demo.py --verify {res['case_id']}")


if __name__ == "__main__":
    main()