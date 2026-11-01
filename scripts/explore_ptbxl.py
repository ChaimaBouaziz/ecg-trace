"""
Quick look at the downloaded PTB-XL data. Produces docs/exploration/*.png
and prints the numbers you will need for steps 1-3.

  python scripts/explore_ptbxl.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from pipeline.ptbxl import QUALITY_COLS, SUPERCLASSES, load_metadata, load_signal

OUT = Path("docs/exploration")
OUT.mkdir(parents=True, exist_ok=True)

df = load_metadata()
print(f"Records available locally : {len(df)}")
print(f"Patients                  : {df['patient_id'].nunique()}")

# --- labels -------------------------------------------------------------
counts = pd.Series({c: df["superclasses"].apply(lambda s: c in s).sum() for c in SUPERCLASSES})
n_labels = df["superclasses"].apply(len)
print("\nSuperclass counts (multi-label):\n", counts.to_string())
print(f"\nNo diagnostic label : {(n_labels == 0).sum()}")
print(f"Single label        : {(n_labels == 1).sum()}  <- easiest to start with")
print(f"Multiple labels     : {(n_labels > 1).sum()}")

counts.plot.bar(title="PTB-XL superclasses")
plt.tight_layout(); plt.savefig(OUT / "superclasses.png", dpi=120); plt.close()

# --- quality flags (useful for the validation module) --------------------
print("\nQuality flags (non-empty = problem noted):")
print(df[QUALITY_COLS].notna().sum().to_string())

# --- reports --------------------------------------------------------------
print("\nReport examples:")
for _, r in df.sample(5, random_state=0).iterrows():
    print(f"  [{','.join(r['superclasses']) or '-'}] {r['report'][:110]}")

# --- one signal -----------------------------------------------------------
row = df.iloc[0]
sig, meta = load_signal(row)
print(f"\nSignal shape {sig.shape}, fs={meta['fs']} Hz, leads={meta['leads']}")
fig, axes = plt.subplots(12, 1, figsize=(12, 14), sharex=True)
t = [i / meta["fs"] for i in range(sig.shape[0])]
for i, ax in enumerate(axes):
    ax.plot(t, sig[:, i], lw=0.8)
    ax.set_ylabel(meta["leads"][i], rotation=0, labelpad=18)
axes[-1].set_xlabel("time (s)")
fig.suptitle(f"ecg_id={row.name}  labels={row['superclasses']}  report: {row['report'][:60]}")
plt.tight_layout(); plt.savefig(OUT / "example_ecg.png", dpi=110); plt.close()
print(f"\nFigures saved in {OUT.resolve()}")
