"""
Generate the sensitivity heatmap from Sweep 2 results (Task 4).

Reads results/sensitivity_sweep.csv and produces:
  results/sensitivity_heatmap.png  — threshold_flag x threshold_quarantine
                                     colour = detection rate
                                     red contour = FPR=0 boundary
                                     white star = reported default (0.22, 0.35)

Usage:
    python scripts/plot_sensitivity_heatmap.py

Requires: matplotlib, numpy
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for server environments
import matplotlib.pyplot as plt
import numpy as np

ROOT        = Path(__file__).resolve().parents[1]
IN_CSV      = ROOT / "results" / "sensitivity_sweep.csv"
OUT_PNG     = ROOT / "results" / "sensitivity_heatmap.png"

# Reported defaults (white star on heatmap)
DEFAULT_FLAG        = 0.22
DEFAULT_QUARANTINE  = 0.35


def plot_heatmap(sweep2_results: list[dict], save_path: Path = OUT_PNG) -> None:
    tf_vals  = sorted(set(round(float(r["threshold_flag"]), 3)       for r in sweep2_results))
    tq_vals  = sorted(set(round(float(r["threshold_quarantine"]), 4) for r in sweep2_results))

    dr_grid  = np.full((len(tq_vals), len(tf_vals)), np.nan)
    fpr_grid = np.full_like(dr_grid, np.nan)

    for r in sweep2_results:
        tf = round(float(r["threshold_flag"]), 3)
        tq = round(float(r["threshold_quarantine"]), 4)
        try:
            i = tq_vals.index(tq)
            j = tf_vals.index(tf)
            dr  = r.get("detection_rate")
            fpr = r.get("false_positive_rate")
            if dr  not in (None, "nan", ""):
                dr_grid[i, j]  = float(dr)
            if fpr not in (None, "nan", ""):
                fpr_grid[i, j] = float(fpr)
        except (ValueError, TypeError):
            pass

    fig, ax = plt.subplots(figsize=(8, 6))

    im = ax.imshow(
        dr_grid,
        origin="lower",
        aspect="auto",
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
        extent=[min(tf_vals), max(tf_vals), min(tq_vals), max(tq_vals)],
    )

    # Red contour at FPR = 0 boundary
    if not np.all(np.isnan(fpr_grid)):
        try:
            ax.contour(
                tf_vals, tq_vals, fpr_grid,
                levels=[0.001],
                colors="red",
                linewidths=2,
                linestyles="--",
            )
        except Exception:
            pass

    # White star at reported default
    ax.scatter(
        [DEFAULT_FLAG], [DEFAULT_QUARANTINE],
        color="white",
        marker="*",
        s=300,
        edgecolors="black",
        linewidths=0.8,
        zorder=5,
        label=f"Default ({DEFAULT_FLAG}, {DEFAULT_QUARANTINE})",
    )

    cbar = fig.colorbar(im, ax=ax, label="Detection Rate", fraction=0.046, pad=0.04)
    ax.set_xlabel("threshold_flag", fontsize=12)
    ax.set_ylabel("threshold_quarantine", fontsize=12)
    ax.set_title(
        "SNAS Threshold Sensitivity (E4: Attack + Straggler)\n"
        "Red dashed line = FPR = 0 boundary",
        fontsize=11,
    )
    ax.legend(fontsize=9, loc="upper left")

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved heatmap to {save_path}")

    # Summary statistic
    zero_fpr_count = int(np.sum(fpr_grid == 0.0))
    total_valid    = int(np.sum(~np.isnan(fpr_grid)))
    if total_valid > 0:
        pct = 100.0 * zero_fpr_count / total_valid
        print(
            f"Zero-FPR region: {zero_fpr_count}/{total_valid} grid points "
            f"({pct:.1f}%) achieve 100% detection with 0% false positives."
        )


def main() -> None:
    if not IN_CSV.exists():
        print(f"ERROR: {IN_CSV} not found. Run run_sensitivity_sweep.py first.")
        sys.exit(1)

    rows = list(csv.DictReader(IN_CSV.open(encoding="utf-8")))
    sweep2 = [r for r in rows if r.get("sweep_id") == "sweep2_thresholds"]

    if not sweep2:
        print("No Sweep 2 results found in CSV. Check sweep_id column.")
        sys.exit(1)

    plot_heatmap(sweep2)


if __name__ == "__main__":
    main()
