#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Javadian. All rights reserved.
"""
collect_results.py

Scan all run output folders, collect every result JSON, and produce:
  - a single combined CSV of all runs
  - a printed comparison table
  - figures: SNR-degradation curve, training curves, split comparison

Reads whatever exists; missing files are skipped silently:
  $RM_RUNS/lstm_out/run_*/test_metrics.json     (+ history.json)
  $RM_RUNS/gru_out/run_*/test_metrics.json      (+ history.json)
  $RM_RUNS/run_*/snr_sweep.json
  $RM_RUNS/finetune_out/run_*/finetune_results.json

Writes everything to OUT_DIR.

This is a reporting tool, not a compute job. Run it on the login node any
time after some runs have finished, no GPU, no SLURM needed:

    conda activate <your-env>
    python collect_results.py

It uses matplotlib with the non-interactive 'Agg' backend, so it works over
SSH with no display. If matplotlib is not installed:
    pip install matplotlib
"""

import os
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')                       # headless: no display needed
import matplotlib.pyplot as plt

#  PATHS 

_RUNS        = Path(os.environ.get('RM_RUNS', './runs'))
LSTM_DIR     = _RUNS / 'lstm_out'
GRU_DIR      = _RUNS / 'gru_out'
FINETUNE_DIR = _RUNS / 'finetune_out'
ARCHIVE_DIR  = _RUNS

OUT_DIR = _RUNS / 'report'

#  HELPERS 

def load_json(path: Path):
    """Return parsed JSON or None if the file is missing or unreadable."""
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def find_run_dirs(parent: Path):
    """Yield run_* subdirectories of `parent`, sorted, if parent exists."""
    if not parent.is_dir():
        return []
    return sorted(d for d in parent.iterdir()
                  if d.is_dir() and d.name.startswith('run_'))

#  COLLECT: single-model training runs (lstm / gru) 

def collect_training_runs():
    """
    Gather test_metrics.json from every lstm_out/run_* and gru_out/run_*.
    Returns a list of flat dicts, one per run.
    """
    rows = []
    for model_dir, model_name in [(LSTM_DIR, 'lstm'), (GRU_DIR, 'gru')]:
        for run_dir in find_run_dirs(model_dir):
            tm = load_json(run_dir / 'test_metrics.json')
            if tm is None:
                continue
            m = tm.get('test_metrics', {})
            rows.append({
                'run_dir'    : run_dir.name,
                'model'      : tm.get('model', model_name),
                'split_mode' : tm.get('split_mode', 'unknown'),
                'best_epoch' : tm.get('best_epoch', ''),
                'rmse'       : m.get('rmse', ''),
                'mae'        : m.get('mae', ''),
                'r2'         : m.get('r2', ''),
                'pearson_r'  : m.get('pearson_r', ''),
            })
    return rows

#  COLLECT: SNR sweeps 

def collect_snr_runs():
    """
    Gather snr_sweep.json files from the archive run folders.
    Returns list of (run_name, results_dict) where results_dict maps
    'clean'/'0dB'/... -> metric dict.
    """
    sweeps = []
    for run_dir in find_run_dirs(ARCHIVE_DIR):
        sj = load_json(run_dir / 'snr_sweep.json')
        if sj is None:
            continue
        sweeps.append((run_dir.name, sj.get('results', {})))
    return sweeps

#  COLLECT: finetune LOSO 

def collect_finetune_runs():
    """Gather finetune_results.json from finetune_out/run_*."""
    runs = []
    for run_dir in find_run_dirs(FINETUNE_DIR):
        fj = load_json(run_dir / 'finetune_results.json')
        if fj is None:
            continue
        runs.append((run_dir.name, fj))
    return runs

#  WRITE: combined CSV 

def write_training_csv(rows, path: Path):
    if not rows:
        return
    fields = ['run_dir', 'model', 'split_mode', 'best_epoch',
              'rmse', 'mae', 'r2', 'pearson_r']
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f'  wrote {path}')

#  PRINT: comparison tables 

def print_training_table(rows):
    if not rows:
        print('  (no lstm/gru training runs found)')
        return
    print(f'\n{"model":>8} {"split":>12} {"epoch":>7} '
          f'{"RMSE":>9} {"MAE":>9} {"R2":>9} {"PCC":>9}  run')
    print('  ' + '-' * 78)
    for r in rows:
        def fmt(v, p=4):
            return f'{v:.{p}f}' if isinstance(v, (int, float)) else f'{v:>}'
        print(f'{r["model"]:>8} {r["split_mode"]:>12} {str(r["best_epoch"]):>7} '
              f'{fmt(r["rmse"],3):>9} {fmt(r["mae"],3):>9} '
              f'{fmt(r["r2"]):>9} {fmt(r["pearson_r"]):>9}  {r["run_dir"]}')


def print_snr_table(sweeps):
    if not sweeps:
        print('  (no SNR sweeps found)')
        return
    for run_name, results in sweeps:
        print(f'\n  SNR sweep: {run_name}')
        print(f'  {"condition":>10} {"RMSE":>9} {"MAE":>9} {"R2":>9} {"PCC":>9}')
        order = ['clean'] + [k for k in results if k != 'clean']
        for cond in order:
            if cond not in results:
                continue
            m = results[cond]
            print(f'  {cond:>10} {m.get("rmse",0):>9.3f} '
                  f'{m.get("mae",0):>9.3f} {m.get("r2",0):>9.4f} '
                  f'{m.get("pearson_r",0):>9.4f}')


def print_finetune_table(runs):
    if not runs:
        print('  (no finetune runs found)')
        return
    for run_name, fj in runs:
        agg = fj.get('aggregate', {})
        print(f'\n  Finetune LOSO: {run_name}')
        for key in ('pretrained', 'finetuned'):
            if key not in agg:
                continue
            a = agg[key]
            print(f'    {key:>11}: '
                  f'R2 {a["r2"]["mean"]:.4f}+/-{a["r2"]["std"]:.4f}   '
                  f'RMSE {a["rmse"]["mean"]:.3f}+/-{a["rmse"]["std"]:.3f}   '
                  f'PCC {a["pearson_r"]["mean"]:.4f}+/-{a["pearson_r"]["std"]:.4f}')

#  FIGURES 

def plot_snr_curves(sweeps, out_path: Path):
    """One line per SNR sweep: R2 vs SNR level, with clean as a flat ref."""
    if not sweeps:
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    for run_name, results in sweeps:
        snr_keys = sorted((k for k in results if k.endswith('dB')),
                          key=lambda s: int(s[:-2]))
        if not snr_keys:
            continue
        xs   = [int(k[:-2]) for k in snr_keys]
        r2s  = [results[k]['r2'] for k in snr_keys]
        rmse = [results[k]['rmse'] for k in snr_keys]

        ax1.plot(xs, r2s, 'o-', label=run_name)
        ax2.plot(xs, rmse, 'o-', label=run_name)

        if 'clean' in results:
            ax1.axhline(results['clean']['r2'], ls='--', alpha=0.4)
            ax2.axhline(results['clean']['rmse'], ls='--', alpha=0.4)

    ax1.set_xlabel('SNR (dB)'); ax1.set_ylabel('Test R2')
    ax1.set_title('R2 vs noise level'); ax1.grid(alpha=0.3); ax1.legend(fontsize=8)
    ax2.set_xlabel('SNR (dB)'); ax2.set_ylabel('Test RMSE (N m)')
    ax2.set_title('RMSE vs noise level'); ax2.grid(alpha=0.3); ax2.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f'  wrote {out_path}')


def plot_training_curves(out_path: Path):
    """Overlay val-R2 training curves from every lstm/gru run with history."""
    curves = []
    for model_dir, model_name in [(LSTM_DIR, 'lstm'), (GRU_DIR, 'gru')]:
        for run_dir in find_run_dirs(model_dir):
            hist = load_json(run_dir / 'history.json')
            tm   = load_json(run_dir / 'test_metrics.json')
            if hist is None:
                continue
            split = tm.get('split_mode', '?') if tm else '?'
            label = f'{model_name}/{split}/{run_dir.name}'
            epochs = [h['epoch'] for h in hist]
            val_r2 = [h.get('val_r2', None) for h in hist]
            if any(v is None for v in val_r2):
                continue
            curves.append((label, epochs, val_r2))

    if not curves:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, epochs, val_r2 in curves:
        ax.plot(epochs, val_r2, label=label, alpha=0.85)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Validation R2')
    ax.set_title('Training curves (validation R2)')
    ax.grid(alpha=0.3); ax.legend(fontsize=7)
    ax.set_ylim(bottom=min(-0.5, ax.get_ylim()[0]))
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f'  wrote {out_path}')


def plot_split_comparison(rows, out_path: Path):
    """Bar chart of test R2 grouped by split mode, per model."""
    if not rows:
        return
    # Keep the most recent run per (model, split_mode).
    latest = {}
    for r in rows:
        if not isinstance(r['r2'], (int, float)):
            continue
        latest[(r['model'], r['split_mode'])] = r['r2']
    if not latest:
        return

    splits = ['sequence', 'recording', 'stratified']
    models = sorted({k[0] for k in latest})
    x = range(len(splits))
    width = 0.8 / max(len(models), 1)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for mi, model in enumerate(models):
        vals = [latest.get((model, s), 0) for s in splits]
        ax.bar([xi + mi * width for xi in x], vals, width, label=model)
    ax.axhline(0, color='k', lw=0.8)
    ax.set_xticks([xi + width * (len(models) - 1) / 2 for xi in x])
    ax.set_xticklabels(splits)
    ax.set_ylabel('Test R2')
    ax.set_title('Test R2 by split protocol')
    ax.grid(alpha=0.3, axis='y'); ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f'  wrote {out_path}')

#  MAIN 

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f'Output dir : {OUT_DIR}\n')

    training = collect_training_runs()
    sweeps   = collect_snr_runs()
    finetune = collect_finetune_runs()

    print(' Training runs (lstm / gru) ')
    print_training_table(training)
    write_training_csv(training, OUT_DIR / 'training_runs.csv')

    print('\n SNR sweeps ')
    print_snr_table(sweeps)

    print('\n Finetune LOSO ')
    print_finetune_table(finetune)

    print('\n Figures ')
    plot_snr_curves(sweeps,        OUT_DIR / 'fig_snr_curves.png')
    plot_training_curves(          OUT_DIR / 'fig_training_curves.png')
    plot_split_comparison(training, OUT_DIR / 'fig_split_comparison.png')

    print(f'\nDone. Tables and figures in {OUT_DIR}')
    if not (training or sweeps or finetune):
        print('NOTE: nothing found yet, run some jobs first, then re-run this.')


if __name__ == '__main__':
    main()
