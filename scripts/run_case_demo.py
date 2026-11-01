"""
Steps 4-5 demo — the spec §9 scenario, without Fabric and UI yet (FakeLedger).

  python scripts/run_case_demo.py                      # process one test-fold ECG + verify
  python scripts/run_case_demo.py --ecg-id 42          # choose the record
  python scripts/run_case_demo.py --tamper             # + modify a stored file and detect it
  python scripts/run_case_demo.py --verify CASE-XXXX   # re-verify an existing case later
  LEDGER=fabric python scripts/run_case_demo.py --tamper   # same, on Hyperledger Fabric
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ledger.factory import get_ledger
from pipeline.model import load_model
from pipeline.ptbxl import QUALITY_COLS, load_metadata, load_signal
from pipeline.run_case import run_case
from pipeline.schemas import ArtifactType as T
from pipeline.storage import OffchainStore
from pipeline.verify import verify_case

store, ledger = OffchainStore("data/offchain"), get_ledger()   # LEDGER=fake (default) or LEDGER=fabric


def show_verification(case_id: str) -> None:
    v = verify_case(case_id, ledger, store)
    print(f"\nIntegrity check of {case_id}: overall = {v['overall']}")
    for a in v["artifacts"]:
        mark = "OK " if a["status"] == "VALID" else "!! "
        rec = (a.get("recorded_sha256") or "")[:16]
        new = (a.get("recomputed_sha256") or "-")[:16]
        print(f"  {mark}{a['artifact_type']:12s} {a['status']:12s} ledger={rec}  recomputed={new}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ecg-id", type=int)
    ap.add_argument("--tamper", action="store_true")
    ap.add_argument("--verify")
    a = ap.parse_args()

    print(f"Ledger: {type(ledger).__name__}")
    if a.verify:
        show_verification(a.verify)
        return

    df = load_metadata()
    test = df[df["strat_fold"] == 10]
    ecg_id = a.ecg_id if a.ecg_id in df.index else test.index[0]
    row = df.loc[ecg_id]
    raw, meta = load_signal(row)

    print(f"Input: PTB-XL ecg_id={ecg_id} (true labels {row['superclasses']})")
    print(f"Report: {row['report'][:100]}")
    ctx = run_case(raw, meta["fs"], row["report"], load_model(), store, ledger,
                   {k: row[k] for k in QUALITY_COLS})

    print(f"\nCase {ctx.case_id} — processing log:")
    for line in ctx.log:
        print("  " + line)

    print("\nHistory on the ledger (artifact -> parents):")
    for r in ledger.get_case_history(ctx.case_id):
        print(f"  {r.artifact_type.value:12s} {r.artifact_id:26s} tx={(r.tx_id or '')[:12]}  <- {', '.join(r.parent_ids) or '(input)'}")

    if ctx.prediction:
        print(f"\nPrediction : {ctx.prediction['predicted_class']} (confidence {ctx.prediction['confidence']:.2f})")
    print(f"Validation : {ctx.validation['status']}  failed={ctx.validation['failed_checks']}")
    print(f"Explanation: {ctx.explanation['text']}")

    show_verification(ctx.case_id)

    if a.tamper and T.PREDICTION in ctx.rec:
        p = store.path(ctx.rec[T.PREDICTION].offchain_uri)
        print(f"\n>>> TAMPERING: editing {p.name} (changing the predicted class)")
        os.chmod(p, 0o600)
        data = p.read_bytes()
        new = (data.replace(b'"NORM"', b'"XNORM"', 1) if b'"predicted_class":"NORM"' in data
               else data.replace(b'"ABNORMAL"', b'"NORM"', 1))
        p.write_bytes(new)
        show_verification(ctx.case_id)

    print(f"\nRe-verify later with: python scripts/run_case_demo.py --verify {ctx.case_id}")


if __name__ == "__main__":
    main()