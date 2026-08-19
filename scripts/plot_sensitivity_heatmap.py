"""
Two-panel threshold-sensitivity figure (Sweep 2, Exp 3: attack + straggler).

Fixes the confusion in the earlier single-panel version:
  - The earlier plot coloured by DETECTION RATE, which is a flat 1.0 everywhere
    (SNAS catches the 10x attacker at every threshold), so the whole map was one
    colour and looked broken. The informative quantity is the FALSE-POSITIVE RATE,
    which actually varies across the grid -- so we give it its own panel.
  - Invalid combinations (threshold_flag >= threshold_quarantine) were drawn as
    blank white, which readers mistook for "zero" (zero would be dark, not white).
    Here they are drawn in explicit grey and called out in the legend, so "no data"
    is never confused with "value 0".

Panel (a): detection rate  -- uniformly 100%, stated explicitly.
Panel (b): false-positive rate -- the real gradient; 61.7% of the space is at 0.

Reads results/sensitivity_sweep.csv (no experiments re-run).
Writes results/figures/fig_sensitivity.{pdf,png} and copies into writing contents/figures/.
"""
from __future__ import annotations

import csv
import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
IN_CSV = ROOT / "results" / "sensitivity_sweep.csv"
OUT_DIR = ROOT / "results" / "figures"
WRITING_FIGS = ROOT / "writing contents" / "figures"

DEFAULT_FLAG, DEFAULT_QUARANTINE = 0.22, 0.35
INVALID_COLOR = "#d9d9d9"  # neutral grey for flag >= quarantine cells

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "figure.dpi": 150,
})


def _edges(centers: list[float]) -> np.ndarray:
    c = np.array(centers, dtype=float)
    step = np.min(np.diff(c))
    return np.concatenate([[c[0] - step / 2], c[:-1] + np.diff(c) / 2, [c[-1] + step / 2]])


def build_grids(rows: list[dict]):
    tf = sorted(set(round(float(r["threshold_flag"]), 3) for r in rows))
    tq = sorted(set(round(float(r["threshold_quarantine"]), 4) for r in rows))
    dr = np.full((len(tq), len(tf)), np.nan)
    fpr = np.full((len(tq), len(tf)), np.nan)
    for r in rows:
        j = tf.index(round(float(r["threshold_flag"]), 3))
        i = tq.index(round(float(r["threshold_quarantine"]), 4))
        if r.get("detection_rate") not in (None, "", "nan"):
            dr[i, j] = float(r["detection_rate"])
        if r.get("false_positive_rate") not in (None, "", "nan"):
            fpr[i, j] = float(r["false_positive_rate"])
    return tf, tq, dr, fpr


def _panel(ax, tf, tq, grid, title, cbar_label):
    cmap = plt.cm.viridis.copy()
    cmap.set_bad(INVALID_COLOR)  # NaN (invalid) cells -> grey, not white
    masked = np.ma.masked_invalid(grid)
    mesh = ax.pcolormesh(_edges(tf), _edges(tq), masked, cmap=cmap,
                         vmin=0.0, vmax=1.0, shading="flat")
    # default operating point
    ax.scatter([DEFAULT_FLAG], [DEFAULT_QUARANTINE], marker="*", s=280,
               color="white", edgecolors="black", linewidths=0.9, zorder=5)
    ax.set_xlabel("threshold_flag")
    ax.set_ylabel("threshold_quarantine")
    ax.set_title(title)
    cb = plt.colorbar(mesh, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(cbar_label)
    return mesh


def main() -> None:
    if not IN_CSV.exists():
        print(f"ERROR: {IN_CSV} not found.")
        sys.exit(1)
    rows = [r for r in csv.DictReader(IN_CSV.open(encoding="utf-8"))
            if r.get("sweep_id") == "sweep2_thresholds"]
    if not rows:
        print("No sweep2 rows found.")
        sys.exit(1)

    tf, tq, dr, fpr = build_grids(rows)
    n_valid = int(np.sum(~np.isnan(fpr)))
    n_zero_fpr = int(np.sum(fpr == 0.0))

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12.5, 4.6))
    fig.subplots_adjust(wspace=0.55, bottom=0.18, top=0.84)
    _panel(axA, tf, tq, dr, "(a) Detection rate", "Detection rate (higher is better)")
    _panel(axB, tf, tq, fpr, "(b) False-positive rate", "False-positive rate (lower is better)")

    # annotate that detection is uniformly 100%
    axA.text(0.5, 0.02, "100% at all valid points",
             transform=axA.transAxes, ha="center", va="bottom", fontsize=9,
             color="white", weight="bold",
             bbox=dict(boxstyle="round,pad=0.25", fc="black", alpha=0.35, ec="none"))

    # shared legend: default star + invalid-region key
    star = plt.Line2D([], [], marker="*", color="white", markeredgecolor="black",
                      markersize=13, linestyle="none",
                      label=f"Default ({DEFAULT_FLAG}, {DEFAULT_QUARANTINE})")
    invalid = mpatches.Patch(facecolor=INVALID_COLOR, edgecolor="#888",
                             label="Invalid (flag $\\geq$ quarantine)")
    fig.legend(handles=[star, invalid], loc="lower center", ncol=2,
               frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.04))

    fig.suptitle("SNAS threshold sensitivity (Exp 3: attack + straggler). "
                 f"Detection is 100% everywhere; {n_zero_fpr}/{n_valid} "
                 f"({100*n_zero_fpr/n_valid:.1f}%) of valid settings also reach 0% false positives.",
                 fontsize=10.5, y=1.02)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    WRITING_FIGS.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        out = OUT_DIR / f"fig_sensitivity.{ext}"
        fig.savefig(out, bbox_inches="tight")
        shutil.copy(out, WRITING_FIGS / f"fig_sensitivity.{ext}")
        print(f"wrote {out} and copied to writing contents/figures/")
    plt.close(fig)
    print(f"Zero-FPR region: {n_zero_fpr}/{n_valid} = {100*n_zero_fpr/n_valid:.1f}%")


if __name__ == "__main__":
    main()
