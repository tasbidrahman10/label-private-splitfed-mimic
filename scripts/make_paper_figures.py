"""
Generate publication figures for the SENTINEL-SplitFed paper.

Design choices (per dataviz guidance + academic-print conventions):
  - Okabe-Ito colorblind-safe categorical palette.
  - No dual-axis charts: the attack-magnitude figure uses two stacked panels that
    share the x-axis, instead of overlaying two y-scales.
  - Serif fonts to match the LaTeX body; recessive grid; thin marks; direct legends.
  - Vector PDF for the manuscript + PNG for quick on-screen inspection.

Outputs -> results/figures/{fig_attack_magnitude,fig_baseline_comparison,fig_perround_auroc}.{pdf,png}
"""
from __future__ import annotations

import csv
import glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIGDIR = ROOT / "results" / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)

# Okabe-Ito colorblind-safe palette (fixed order, never cycled)
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
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#dddddd",
    "grid.linewidth": 0.6,
    "legend.frameon": False,
    "figure.dpi": 150,
})


def _save(fig, name: str) -> None:
    for ext in ("pdf", "png"):
        fig.savefig(FIGDIR / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {name}.pdf / .png")


# ---------------------------------------------------------------------------
# Figure: attack-magnitude sweep (two stacked panels, shared x-axis)
# ---------------------------------------------------------------------------
def fig_attack_magnitude() -> None:
    rows = list(csv.DictReader((ROOT / "results" / "attack_magnitude_sweep.csv").open()))
    scale = [float(r["scale"]) for r in rows]
    dr = [float(r["detection_rate"]) * 100 for r in rows]
    auroc = [float(r["mean_auroc_r8_10"]) for r in rows]
    quar = [r["ever_quarantined"].strip().lower() == "true" for r in rows]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5.2, 4.6), sharex=True)

    # Panel (a): detection rate
    ax1.plot(scale, dr, "-o", color=OKABE["vermil"], lw=2, ms=6)
    ax1.axhspan(0, 0, color="none")
    ax1.set_ylabel("Detection rate (%)")
    ax1.set_ylim(-5, 105)
    ax1.set_yticks([0, 25, 50, 75, 100])
    # annotate the three regimes
    ax1.axvspan(1.0, 2.5, color=OKABE["sky"], alpha=0.10)
    ax1.axvspan(2.5, 4.0, color=OKABE["yellow"], alpha=0.18)
    ax1.axvspan(4.0, 10.5, color=OKABE["green"], alpha=0.10)
    ax1.text(1.75, 52, "evades\n(harmless)", ha="center", va="center", fontsize=8, color="#555")
    ax1.text(3.25, 52, "transition", ha="center", va="center", fontsize=8, color="#555")
    ax1.text(7.0, 52, "detected", ha="center", va="center", fontsize=8, color="#555")
    ax1.set_title("(a) Attacker detection rate vs. scaling magnitude")

    # Panel (b): AUROC
    ax2.plot(scale, auroc, "-s", color=OKABE["blue"], lw=2, ms=6, label="Mean AUROC (R8-10)")
    ax2.axhline(0.9649, color="#888", ls="--", lw=1, label="Sync baseline (0.9649)")
    ax2.set_xlabel("Gradient-scaling factor ($\\times$)")
    ax2.set_ylabel("AUROC")
    ax2.set_title("(b) Model utility vs. scaling magnitude")
    ax2.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    _save(fig, "fig_attack_magnitude")


# ---------------------------------------------------------------------------
# Figure: baseline comparison grouped bars (matches Table 4, n=10 for SNAS/Krum)
# ---------------------------------------------------------------------------
def fig_baseline_comparison() -> None:
    # n=10 means for SNAS/Krum (from significance_tests.csv); single-seed for the
    # degenerate trimmed-mean/FLTrust (from baseline_comparison.csv).
    experiments = ["E1\n(grad. scaling)", "E2\n(free rider)", "E4\n(attack+strag.)"]
    data = {
        "SNAS (ours)":  ([0.9598, 0.9050, 0.9581], OKABE["blue"]),
        "Krum":         ([0.9653, 0.9654, 0.9653], OKABE["orange"]),
        "Trimmed mean": ([0.9539, 0.8227, 0.9541], OKABE["green"]),
        "FLTrust":      ([0.9539, 0.8227, 0.9541], OKABE["purple"]),
    }
    x = np.arange(len(experiments))
    n = len(data)
    w = 0.80 / n

    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    for i, (name, (vals, color)) in enumerate(data.items()):
        offset = (i - (n - 1) / 2) * w
        bars = ax.bar(x + offset, vals, w * 0.92, label=name, color=color, edgecolor="white", linewidth=0.5)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.004, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=6.5, rotation=90, color="#333")

    ax.set_xticks(x)
    ax.set_xticklabels(experiments)
    ax.set_ylabel("Mean AUROC (R8-10)")
    ax.set_ylim(0.80, 1.0)
    ax.set_title("Aggregator comparison: raw AUROC hides Krum's federation collapse")
    ax.legend(ncol=2, fontsize=8, loc="lower center")
    # annotation that Krum collapses to one client
    ax.annotate("Krum: single-client\nselection (880/880 rounds)",
                xy=(1 + (1 - (n - 1) / 2) * w, 0.9654), xytext=(1.15, 0.86),
                fontsize=7, color=OKABE["vermil"],
                arrowprops=dict(arrowstyle="->", color=OKABE["vermil"], lw=0.8))
    fig.tight_layout()
    _save(fig, "fig_baseline_comparison")


# ---------------------------------------------------------------------------
# Figure: per-round AUROC for E1/E2/E4/E6 (2x2 small multiples)
# ---------------------------------------------------------------------------
def _best_seed42_csv(exp_dir: str) -> Path | None:
    """Pick the seed42 per-round CSV with the most rounds (>=10 preferred)."""
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
    means = [float(np.mean(per_round[rd])) for rd in rounds]
    return rounds, means


def fig_perround_auroc() -> None:
    panels = [
        ("E1: gradient scaling ($10\\times$)", "experiment_e1_gradient_scaling", OKABE["blue"]),
        ("E2: free rider", "experiment_e2_free_rider", OKABE["vermil"]),
        ("E4: attack + straggler", "experiment_e4_attack_plus_dropout", OKABE["green"]),
        ("E6: honest straggler", "experiment_e6_dropout", OKABE["purple"]),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.0), sharex=True)
    for ax, (title, exp_dir, color) in zip(axes.ravel(), panels):
        csv_path = _best_seed42_csv(exp_dir)
        if csv_path is None:
            ax.set_title(title + " (no data)")
            continue
        rounds, means = _mean_auroc_per_round(csv_path)
        ax.plot(rounds, means, "-o", color=color, lw=1.8, ms=4)
        ax.axhline(0.9649, color="#888", ls="--", lw=0.9)
        ax.set_title(title, fontsize=9.5)
        ax.set_ylim(min(0.90, min(means) - 0.01), 0.98)
        ax.set_xticks(rounds[::2] if len(rounds) > 6 else rounds)
    for ax in axes[-1]:
        ax.set_xlabel("Aggregation round")
    for ax in axes[:, 0]:
        ax.set_ylabel("Mean AUROC")
    fig.suptitle("Per-round mean AUROC across three clients (seed 42; dashed = sync baseline)",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    _save(fig, "fig_perround_auroc")


if __name__ == "__main__":
    print("Generating paper figures ->", FIGDIR)
    fig_attack_magnitude()
    fig_baseline_comparison()
    fig_perround_auroc()
    print("Done.")
