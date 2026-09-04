#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Javadian. All rights reserved.

"""
Diagnostic plots from an exported predictions file.

This script has no knowledge of any model, dataset, or feature pipeline. It
reads a CSV with columns "y_true" and "y_pred", and optionally "t" (time or
sample index) and "subject", and produces three standard regression
diagnostics: a scatter of predicted against true values, a residual plot, and
a time-series overlay if a time column is present.

Any two-column CSV of true and predicted values works, from any regression
project. Producing the input CSV, wherever the numbers come from, is outside
the scope of this file.

    python3 plot_predictions.py results.csv --out figs/
"""
import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_predictions(path):
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = set(reader.fieldnames or [])
        if not {"y_true", "y_pred"}.issubset(fieldnames):
            raise ValueError(
                f"{path}: expected columns 'y_true' and 'y_pred', "
                f"found {sorted(fieldnames)}"
            )
        has_t = "t" in fieldnames
        has_subject = "subject" in fieldnames

        y_true, y_pred, t, subject = [], [], [], []
        for row in reader:
            y_true.append(float(row["y_true"]))
            y_pred.append(float(row["y_pred"]))
            if has_t:
                t.append(float(row["t"]))
            if has_subject:
                subject.append(row["subject"])

    return {
        "y_true": np.array(y_true),
        "y_pred": np.array(y_pred),
        "t": np.array(t) if has_t else None,
        "subject": subject if has_subject else None,
    }


def compute_metrics(y_true, y_pred):
    err = y_true - y_pred
    ss_res = np.sum(err ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2) + 1e-12
    r2 = 1.0 - ss_res / ss_tot
    rmse = float(np.sqrt(np.mean(err ** 2)))
    pcc = float(np.corrcoef(y_true, y_pred)[0, 1])
    return {"r2": float(r2), "rmse": rmse, "pcc": pcc}


def plot_scatter(y_true, y_pred, metrics, out_path):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(y_true, y_pred, s=8, alpha=0.35, edgecolors="none")

    lo = min(y_true.min(), y_pred.min())
    hi = max(y_true.max(), y_pred.max())
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1, alpha=0.6, label="y = x")

    ax.set_xlabel("True value")
    ax.set_ylabel("Predicted value")
    ax.set_title(f"R2 = {metrics['r2']:.3f}   RMSE = {metrics['rmse']:.3f}   "
                 f"PCC = {metrics['pcc']:.3f}")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_residuals(y_true, y_pred, out_path):
    residual = y_true - y_pred
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.scatter(y_pred, residual, s=8, alpha=0.35, edgecolors="none")
    ax1.axhline(0, color="k", linewidth=1)
    ax1.set_xlabel("Predicted value")
    ax1.set_ylabel("Residual (true - predicted)")
    ax1.set_title("Residual vs prediction")
    ax1.grid(alpha=0.3)

    ax2.hist(residual, bins=40, alpha=0.75)
    ax2.set_xlabel("Residual")
    ax2.set_ylabel("Count")
    ax2.set_title(f"Residual distribution (std {residual.std():.3f})")
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_timeseries(t, y_true, y_pred, out_path):
    order = np.argsort(t)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t[order], y_true[order], label="true", linewidth=1)
    ax.plot(t[order], y_pred[order], label="predicted", linewidth=1, alpha=0.8)
    ax.set_xlabel("t")
    ax.set_ylabel("value")
    ax.set_title("Prediction overlay")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"  wrote {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("csv_path", help="CSV with y_true, y_pred, and optionally t")
    ap.add_argument("--out", default=".", help="output directory for figures")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    data = load_predictions(args.csv_path)
    metrics = compute_metrics(data["y_true"], data["y_pred"])

    print(f"n = {len(data['y_true'])}")
    print(f"R2   = {metrics['r2']:.4f}")
    print(f"RMSE = {metrics['rmse']:.4f}")
    print(f"PCC  = {metrics['pcc']:.4f}")

    plot_scatter(data["y_true"], data["y_pred"], metrics,
                 os.path.join(args.out, "scatter.png"))
    plot_residuals(data["y_true"], data["y_pred"],
                   os.path.join(args.out, "residuals.png"))

    if data["t"] is not None:
        plot_timeseries(data["t"], data["y_true"], data["y_pred"],
                        os.path.join(args.out, "timeseries.png"))
    else:
        print("  (no 't' column, skipping time-series overlay)")


if __name__ == "__main__":
    main()
