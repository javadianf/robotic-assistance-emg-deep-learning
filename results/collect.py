#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Javadian. All rights reserved.
"""
Collect LOSO result JSONs into a single Markdown comparison table.

Reads every {model}_{cond}_{target}..._loso_{emg|kin}[_fN].json in RUNS and
prints one row per file that has an "aggregate" block. This is a reporting
tool, not a compute job: run it any time after some LOSO jobs have finished.

    export EMG_RUNS=/path/to/runs
    python3 collect.py
"""
import glob
import json
import os

RUNS = os.environ.get("EMG_RUNS", "./runs")

rows = []
for f in sorted(glob.glob(os.path.join(RUNS, "*_loso_*.json"))):
    with open(f) as fh:
        d = json.load(fh)
    a = d.get("aggregate")
    if not a:
        continue
    base = os.path.basename(f)[:-5]           # strip .json
    parts = base.split("_")                   # MODEL_COND_TARGET..._loso_KIN[_fN]
    i = parts.index("loso")
    target = "_".join(parts[2:i])
    inp = "EMG+kin" if parts[i + 1] == "kin" else "EMG"
    rows.append((d["model"], d["cond"], target, inp, len(d.get("folds", [])),
                 a["r2"]["mean"], a["r2"]["std"], a["rmse"]["mean"], a["pcc"]["mean"]))

print("| Model | Cond | Target | Input | Folds | R2 (mean +/- std) | RMSE | PCC |")
print("|---|---|---|---|---|---|---|---|")
for m, c, t, i, n, r, rs, rm, p in rows:
    print(f"| {m} | {c} | {t} | {i} | {n} | {r:.3f} +/- {rs:.3f} | {rm:.2f} | {p:.3f} |")

if not rows:
    print("\n(no LOSO result files found in", RUNS, ")")
