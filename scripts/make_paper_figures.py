"""
Generate publication figures for the SENTINEL-SplitFed paper.

Design choices:
  - Okabe-Ito colorblind-safe categorical palette (fixed order, never cycled).
  - No chart titles; every figure's legend sits in a boxed panel BELOW the plot.
  - Experiment labels use the paper's gapless Exp 1-4 naming.
  - Serif fonts to match the LaTeX body; recessive grid; vector PDF + PNG.

Outputs -> results/figures/{fig_perround_auroc, fig_baseline_comparison,
                            fig_attack_magnitude, fig_ablation}.{pdf,png}
Also copies each into "writing contents/figures/" for Overleaf.
"""
from __future__ import annotations

import csv
import glob
import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIGDIR = ROOT / "results" / "figures"
WRITING_FIGS = ROOT / "writing contents" / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)

OKABE = {
    "blue":   "#0072B2",
    "orange": "#E69F00",
    "green":  "#009E73",
    "vermil": "#D55E00",
    "sky":    "#56B4E9",
    "purple": "#CC79A7",
    "yellow": "#F0E442",
    "black":  "#000000",
}

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#dddddd",
    "grid.linewidth": 0.6,
    "figure.dpi": 150,
})

# Paper's Exp 1-4 labels mapped to the internal experiment directories / data keys
EXP = [
    ("Exp 1 (gradient scaling)", "experiment_e1_gradient_scaling", "E1_gradient_scaling", OKABE["blue"],   "o"),
    ("Exp 2 (free rider)",        "experiment_e2_free_rider",       "E2_free_rider",       OKABE["vermil"], "s"),
    ("Exp 3 (attack + straggler)","experiment_e4_attack_plus_dropout","E4_attack_straggler",OKABE["green"], "^"),
    ("Exp 4 (honest straggler)",  "experiment_e6_dropout",          "E6_honest_straggler", OKABE["purple"], "D"),
]

BOX = dict(frameon=True, edgecolor="#bbbbbb", facecolor="white", framealpha=1.0, fancybox=False)

# Ten-seed clean asynchronous baseline from the P0-8 regeneration grid
# (results/p08_rerun_summary.json, cell "snas:clean"). Preferred over the older
# single-seed sync-FedAvg figure, which has no reproducible config on record.
CLEAN_BASELINE = 0.9651


def _save(fig, name: str) -> None:
    for ext in ("pdf", "png"):
        out = FIGDIR / f"{name}.{ext}"
        fig.savefig(out, bbox_inches="tight")
        WRITING_FIGS.mkdir(parents=True, exist_ok=True)
        shutil.copy(out, WRITING_FIGS / f"{name}.{ext}")
    plt.close(fig)
    print(f"  wrote {name}.pdf/.png (and copied to writing contents/figures/)")


# ---------------------------------------------------------------------------
# helpers to read per-round AUROC
# ---------------------------------------------------------------------------
def _best_seed42_csv(exp_dir: str) -> Path | None:
    candidates = glob.glob(str(ROOT / "results" / exp_dir / "*seed42*.csv"))
    best, best_rounds = None, -1
    for c in candidates:
        rows = list(csv.DictReader(open(c)))
        if not rows or "round" not in rows[0]:
            continue
        nr = max(int(r["round"]) for r in rows)
        if nr > best_rounds:
            best, best_rounds = Path(c), nr
    return best


def _mean_auroc_per_round(csv_path: Path) -> tuple[list[int], list[float]]:
    rows = list(csv.DictReader(csv_path.open()))
    per_round: dict[int, list[float]] = {}
    for r in rows:
        if r.get("val_auroc"):
            per_round.setdefault(int(r["round"]), []).append(float(r["val_auroc"]))
    rounds = sorted(per_round)
    return rounds, [float(np.mean(per_round[rd])) for rd in rounds]


# ---------------------------------------------------------------------------
# Figure 1: per-round AUROC -- all four experiments on ONE axes
# ---------------------------------------------------------------------------
def fig_perround_auroc() -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    for label, exp_dir, _key, color, marker in EXP:
        csv_path = _best_seed42_csv(exp_dir)
        if csv_path is None:
            continue
        rounds, means = _mean_auroc_per_round(csv_path)
        ax.plot(rounds, means, marker=marker, color=color, lw=1.8, ms=5, label=label)
    # Ten-seed clean baseline, not the retired single-seed 0.9649 sync figure.
    # In the clean configuration every client submits inside the window, so this
    # is also the synchronous reference -- and unlike 0.9649 it has a
    # reproducible config and ten seeds behind it.
    ax.axhline(CLEAN_BASELINE, color="#888", ls="--", lw=1.2,
               label=f"Clean baseline ({CLEAN_BASELINE})")
    ax.set_xlabel("Aggregation round")
    ax.set_ylabel("Mean AUROC across three clients")
    ax.set_xticks(range(1, 11))
    ax.set_ylim(0.79, 0.98)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3,
              fontsize=9, **BOX)
    fig.tight_layout()
    _save(fig, "fig_perround_auroc")


# ---------------------------------------------------------------------------
# Figure 2: baseline / aggregator comparison (no annotation arrow, legend below)
# ---------------------------------------------------------------------------
def fig_baseline_comparison() -> None:
    """Aggregator comparison. Values read from the regenerated 265-feature grid.

    Previously these were hardcoded at the pre-P0-8 266-feature numbers
    (SNAS 0.9598/0.9050/0.9581), which no longer match any table in the paper.
    Reading p08_rerun_summary.json keeps the figure and Table 3 in lockstep and
    means a rerun cannot silently leave the figure behind.
    """
    summary = json.load((ROOT / "results" / "p08_rerun_summary.json").open())
    cells = ["E1_gradient_scaling", "E2_free_rider", "E4_attack_straggler"]
    experiments = ["Exp 1\n(grad. scaling)", "Exp 2\n(free rider)", "Exp 3\n(attack+strag.)"]

    # Order fixes the legend and the bar order; colours are stable across the
    # paper (SNAS blue, Krum orange) so the new median arm takes an unused hue.
    series = [
        ("SNAS (ours)",       "snas",         OKABE["blue"]),
        ("Krum",              "krum",         OKABE["orange"]),
        ("Coordinate median", "median",       OKABE["sky"]),
        ("Trimmed mean",      "trimmed_mean", OKABE["green"]),
        ("FLTrust",           "fltrust",      OKABE["purple"]),
    ]
    data = {
        name: ([summary[f"{arm}:{c}"]["mean_auroc"] for c in cells], colour)
        for name, arm, colour in series
    }

    x = np.arange(len(experiments))
    n = len(data)
    w = 0.80 / n

    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    for i, (name, (vals, color)) in enumerate(data.items()):
        offset = (i - (n - 1) / 2) * w
        bars = ax.bar(x + offset, vals, w * 0.9, label=name, color=color,
                      edgecolor="white", linewidth=0.5)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.004, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=6, rotation=90, color="#333")
    ax.set_xticks(x)
    ax.set_xticklabels(experiments)
    ax.set_ylabel("Mean AUROC (rounds 8--10)")
    ax.set_ylim(0.80, 1.03)
    # Five entries stay on one row; ncol matches the series count as before.
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=5,
              fontsize=9, **BOX)
    fig.tight_layout()
    _save(fig, "fig_baseline_comparison")


# ---------------------------------------------------------------------------
# Figure 3: attack-magnitude sweep (two stacked panels, no titles, legend below)
# ---------------------------------------------------------------------------
def fig_attack_magnitude() -> None:
    rows = list(csv.DictReader((ROOT / "results" / "attack_magnitude_sweep.csv").open()))
    scale = [float(r["scale"]) for r in rows]
    dr = [float(r["detection_rate"]) * 100 for r in rows]
    auroc = [float(r["mean_auroc_r8_10"]) for r in rows]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.2, 5.4), sharex=True)

    # Panel (a): detection rate
    ax1.plot(scale, dr, "-o", color=OKABE["vermil"], lw=2, ms=6)
    ax1.set_ylabel("Detection rate (%)")
    ax1.set_ylim(-8, 112)
    ax1.set_yticks([0, 25, 50, 75, 100])
    # The sweep spans an order of magnitude either side of 1.0, and the gate
    # responds to |log(scale)|, so the curve is only symmetric on a log axis.
    # Shading marks detected / transition / evasion bands on both sides.
    ax1.axvspan(0.09, 0.20, color=OKABE["green"], alpha=0.10)
    ax1.axvspan(0.20, 0.40, color=OKABE["yellow"], alpha=0.18)
    ax1.axvspan(0.40, 2.50, color=OKABE["sky"], alpha=0.10)
    ax1.axvspan(2.50, 4.00, color=OKABE["yellow"], alpha=0.18)
    ax1.axvspan(4.00, 11.0, color=OKABE["green"], alpha=0.10)
    # regime labels pinned near the TOP so they never sit on the curve
    for xpos, txt in [(0.13, "detected"), (1.0, "evades\n(harmless)"), (6.6, "detected")]:
        ax1.text(xpos, 108, txt, ha="center", va="top", fontsize=8, color="#555")
    # Panel label sits low-left: the detection curve is pinned at 100% there, so
    # the bottom-left corner is the only region free of both curve and banding
    # text on this panel.
    ax1.text(0.015, 0.08, "(a)", transform=ax1.transAxes, fontsize=11, fontweight="bold")

    # Panel (b): AUROC
    ax2.plot(scale, auroc, "-s", color=OKABE["blue"], lw=2, ms=6)
    ax2.axhline(CLEAN_BASELINE, color="#888", ls="--", lw=1.2)
    ax2.set_xscale("log")
    ax2.set_xlim(0.09, 11.0)
    ax2.set_xticks(scale)
    ax2.set_xticklabels([f"{s:g}" for s in scale], fontsize=7.5)
    ax2.minorticks_off()
    ax2.set_xlabel("Gradient-scaling factor ($\\times$, log scale)")
    ax2.set_ylabel("Mean AUROC (rounds 8--10)")
    ax2.text(0.015, 0.90, "(b)", transform=ax2.transAxes, fontsize=11, fontweight="bold")

    handles = [
        Line2D([], [], color=OKABE["vermil"], marker="o", lw=2, label="Detection rate — panel (a)"),
        Line2D([], [], color=OKABE["blue"], marker="s", lw=2, label="Mean AUROC — panel (b)"),
        Line2D([], [], color="#888", ls="--", lw=1.2,
               label=f"Clean async baseline ({CLEAN_BASELINE})"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.045),
               ncol=3, fontsize=8.5, **BOX)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    _save(fig, "fig_attack_magnitude")


# ---------------------------------------------------------------------------
# Figure 4: gate ablation (clean value labels above error bars, legend below)
# ---------------------------------------------------------------------------
def fig_ablation() -> None:
    """2x2 factorial ablation of the anomaly gate and the norm clip.

    Replaces the earlier two-bar gate ablation, which drew only the both-on and
    both-off corners and read multiseed_summary.json / gap_closing/nodetect.json
    -- both pre-P0-8 266-feature files, so the old figure disagreed numerically
    with every table in the current paper. Source is now the same artifact the
    table is built from.

    Bar order is the factorial order (neither, clip, gate, both) so the reader
    reads left-to-right as defences are added. SNAS keeps blue and the
    no-detection corner keeps vermilion, matching their colours elsewhere.
    """
    fac = json.load((ROOT / "results" / "p1_1_factorial_ablation.json").open())["experiments"]

    labels = ["Exp 1\n(grad. scaling)", "Exp 2\n(free rider)",
              "Exp 3\n(attack+strag.)", "Exp 4\n(honest strag.)"]
    keys = [k for _, _, k, _, _ in EXP]
    arms = [
        ("Neither",   "nodetect", OKABE["vermil"]),
        ("Clip only", "cliponly", OKABE["orange"]),
        ("Gate only", "gateonly", OKABE["sky"]),
        ("Both (SNAS)", "snas",   OKABE["blue"]),
    ]

    x = np.arange(len(labels))
    n = len(arms)
    w = 0.80 / n

    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    for i, (name, arm, colour) in enumerate(arms):
        means = [fac[k]["cells"][arm]["mean"] for k in keys]
        stds = [fac[k]["cells"][arm]["std"] for k in keys]
        offset = (i - (n - 1) / 2) * w
        bars = ax.bar(x + offset, means, w * 0.9, yerr=stds, capsize=2, label=name,
                      color=colour, edgecolor="white", linewidth=0.5)
        # Rotated labels, as in fig_baseline_comparison: four bars per group are
        # too narrow for horizontal text, and rotating keeps them clear of the
        # error-bar caps.
        for bar, m, s in zip(bars, means, stds):
            ax.text(bar.get_x() + bar.get_width() / 2, m + s + 0.004, f"{m:.3f}",
                    ha="center", va="bottom", fontsize=6, rotation=90, color="#333")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Mean AUROC (rounds 8--10)")
    ax.set_ylim(0.77, 1.03)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=4,
              fontsize=9, **BOX)
    fig.tight_layout()
    _save(fig, "fig_ablation")


if __name__ == "__main__":
    print("Generating paper figures ->", FIGDIR)
    fig_perround_auroc()
    fig_baseline_comparison()
    fig_attack_magnitude()
    fig_ablation()
    print("Done.")
