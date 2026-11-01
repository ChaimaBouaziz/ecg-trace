"""
Step 1 check on real data.

  python scripts/run_preprocess.py              # one example + batch over all local records
  python scripts/run_preprocess.py --ecg-id 42  # pick the example record

Outputs in docs/preprocess/:
  example_<id>.png          raw vs filtered lead II with R peaks + overlaid beats
  example_<id>_report.json  processing report of that record
  batch_summary.csv         one line per record (quality, HR, beats, warnings)
  batch_quality.png         distribution of quality score and heart rate
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pipeline.preprocess import PreprocessingError, _bandpass, prepare_report, preprocess_ecg
from pipeline.ptbxl import QUALITY_COLS, load_metadata, load_signal

OUT = Path("docs/preprocess")
OUT.mkdir(parents=True, exist_ok=True)


def flags_of(row) -> dict:
    return {c: row[c] for c in QUALITY_COLS}


def plot_example(ecg_id, row):
    raw, meta = load_signal(row)
    fs = meta["fs"]
    out = preprocess_ecg(raw, fs, flags_of(row))
    filt = _bandpass(np.nan_to_num(raw.astype(float)), fs)
    t = np.arange(len(raw)) / fs

    fig, ax = plt.subplots(3, 1, figsize=(12, 9))
    ax[0].plot(t, raw[:, 1], lw=0.8); ax[0].set_title("Raw lead II (mV)")
    ax[1].plot(t, filt[:, 1], lw=0.8, label="filtered 0.5-40 Hz")
    ax[1].plot(out.r_peaks / fs, filt[out.r_peaks, 1], "rx", label="R peaks")
    ax[1].legend(loc="upper right"); ax[1].set_title(
        f"Filtered lead II — HR {out.report['checks']['heart_rate_bpm']} bpm")
    tb = (np.arange(out.beats.shape[1]) - 25) / fs
    for b in out.beats:
        ax[2].plot(tb, b[:, 1], lw=0.7, alpha=0.6)
    ax[2].axvline(0, color="k", ls="--", lw=0.6)
    ax[2].set_title(f"{len(out.beats)} segmented beats (z-scored), lead II"); ax[2].set_xlabel("s")
    fig.suptitle(f"ecg_id={ecg_id}  labels={row['superclasses']}  quality={out.report['quality_score']}")
    plt.tight_layout(); plt.savefig(OUT / f"example_{ecg_id}.png", dpi=110); plt.close()

    rep = {"ecg": out.report, "report_text": prepare_report(row["report"])}
    (OUT / f"example_{ecg_id}_report.json").write_text(json.dumps(rep, indent=2, ensure_ascii=False))
    print(json.dumps({k: rep["ecg"][k] for k in ("checks", "warnings", "quality_score")}, indent=2))


def batch(df):
    rows, t0 = [], time.time()
    for ecg_id, row in df.iterrows():
        raw, meta = load_signal(row)
        try:
            out = preprocess_ecg(raw, meta["fs"], flags_of(row))
            c = out.report["checks"]
            rows.append(dict(ecg_id=ecg_id, status="OK", quality=out.report["quality_score"],
                             hr=c["heart_rate_bpm"], n_beats=c["n_beats"],
                             flat_leads=len(c["flat_leads"]),
                             warnings="; ".join(out.report["warnings"]),
                             labels=",".join(row["superclasses"])))
        except PreprocessingError as e:
            rows.append(dict(ecg_id=ecg_id, status="FAILED", warnings=str(e)))
    res = pd.DataFrame(rows).set_index("ecg_id")
    res.to_csv(OUT / "batch_summary.csv")
    ok = res[res["status"] == "OK"]

    print(f"\nProcessed {len(res)} records in {time.time() - t0:.1f}s")
    print(f"FAILED          : {(res['status'] == 'FAILED').sum()}")
    print(f"quality = 1.0   : {(ok['quality'] == 1.0).sum()}")
    print(f"quality < 0.7   : {(ok['quality'] < 0.7).sum()}   <- would go to 'review required'")
    print(f"HR median       : {ok['hr'].median():.0f} bpm  (min {ok['hr'].min():.0f}, max {ok['hr'].max():.0f})")
    print(f"beats per record: median {ok['n_beats'].median():.0f}")
    print("\nMost common warnings:")
    w = ok["warnings"].str.split("; ").explode()
    w = w[w.astype(bool)].str.replace(r"\(.*\)|\d+", "", regex=True).str.strip()
    print(w.value_counts().head(6).to_string() if len(w) else "  none")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ok["quality"].plot.hist(bins=20, ax=ax[0], title="Quality score")
    ok["hr"].plot.hist(bins=40, ax=ax[1], title="Detected heart rate (bpm)")
    plt.tight_layout(); plt.savefig(OUT / "batch_quality.png", dpi=110); plt.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ecg-id", type=int, default=None)
    a = ap.parse_args()
    df = load_metadata()
    ecg_id = a.ecg_id if a.ecg_id in df.index else df.index[0]
    print(f"Example record: ecg_id={ecg_id}")
    plot_example(ecg_id, df.loc[ecg_id])
    batch(df)
    print(f"\nOutputs in {OUT.resolve()}")