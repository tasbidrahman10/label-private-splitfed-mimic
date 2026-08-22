"""
P1-1 — 2x2 factorial analysis of the anomaly gate and the norm clip.

The paper previously reported only the two extreme corners: both defences on
(`snas`) and both off (`nodetect`). That supports "the pipeline helps" but not
"the gate helps" or "clipping helps", because a single ON/OFF contrast cannot
attribute the benefit to either factor or detect that they overlap.

This reads the four arms of the completed grid

    nodetect   gate OFF, clip OFF        cliponly   gate OFF, clip ON
    gateonly   gate ON,  clip OFF        snas       gate ON,  clip ON

and reports, per experiment, the two main effects and their interaction.

Everything is **paired by seed**. All four arms run the same ten seeds through
configs that differ only in `results_dir`, so seed s in one arm shares its
initialisation and minibatch order with seed s in the others. Pairing removes
seed variance from every contrast, which matters here because the effects on
E1/E4 are ~0.003 AUROC against a between-seed SD of ~0.0002.

Definitions (all in AUROC, positive = the defence helps):

    gate effect   = mean over clip levels of (gate ON - gate OFF)
                  = ((gateonly - nodetect) + (snas - cliponly)) / 2
    clip effect   = mean over gate levels of (clip ON - clip OFF)
                  = ((cliponly - nodetect) + (snas - gateonly)) / 2
    interaction   = (snas - gateonly) - (cliponly - nodetect)

A negative interaction means the two defences are **sub-additive** — they
overlap, each recovering damage the other would also have caught, so their
combined benefit is less than the sum of the parts. That is the expected shape
for two mechanisms defending the same attack, and it is worth stating rather
than implying the benefits simply add.

Effect sizes are paired Cohen's d (mean difference / SD of the per-seed
differences), matching the convention `significance_tests_p08.py` adopted.
Do not mix them with the pooled independent-samples d.

Usage:
    python scripts/analyze_factorial_ablation.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
PROGRESS_PATH = ROOT / "results" / "p08_rerun_progress.json"
OUT_CSV = ROOT / "results" / "p1_1_factorial_ablation.csv"
OUT_JSON = ROOT / "results" / "p1_1_factorial_ablation.json"

# arm -> (gate enabled, clip enabled). Mirrors FACTORIAL_ARMS in rerun_p08.py.
ARMS = {
    "nodetect": (False, False),
    "cliponly": (False, True),
    "gateonly": (True, False),
    "snas":     (True, True),
}
EXPERIMENTS = [
    ("E1_gradient_scaling", "E1 gradient scaling"),
    ("E2_free_rider",       "E2 free rider"),
    ("E4_attack_straggler", "E3 attack + straggler"),
    ("E6_honest_straggler", "E4 honest straggler"),
]


def paired(progress: dict, arm_a: str, arm_b: str, experiment: str):
    """Per-seed (a, b) values present in both arms, ordered by seed."""
    a = progress.get(f"{arm_a}:{experiment}", {})
    b = progress.get(f"{arm_b}:{experiment}", {})
    seeds = sorted(
        s for s in set(a) & set(b)
        if a.get(s) is not None and b.get(s) is not None
    )
    return seeds, np.array([a[s] for s in seeds]), np.array([b[s] for s in seeds])


def contrast(progress: dict, better: str, worse: str, experiment: str) -> dict:
    """Paired comparison of two arms: better - worse."""
    seeds, hi, lo = paired(progress, better, worse, experiment)
    if len(seeds) < 2:
        return {"n_seeds": len(seeds), "delta": None}
    diff = hi - lo
    wins = int(np.sum(diff > 0))
    result = {
        "n_seeds": len(seeds),
        "delta": round(float(np.mean(diff)), 6),
        "delta_std": round(float(np.std(diff, ddof=1)), 6),
        "wins": wins,
        "win_rate": f"{wins}/{len(seeds)}",
        "cohens_d_paired": (
            round(float(np.mean(diff) / np.std(diff, ddof=1)), 4)
            if np.std(diff, ddof=1) > 1e-12 else None
        ),
    }
    # Wilcoxon is undefined when every paired difference is exactly zero.
    if np.allclose(diff, 0.0):
        result["wilcoxon_p"] = None
        result["ttest_p"] = None
        result["note"] = "all paired differences zero"
        return result
    try:
        result["wilcoxon_p"] = float(stats.wilcoxon(hi, lo).pvalue)
    except ValueError as exc:
        result["wilcoxon_p"] = None
        result["note"] = str(exc)
    result["ttest_p"] = float(stats.ttest_rel(hi, lo).pvalue)
    return result


def main() -> None:
    progress = json.loads(PROGRESS_PATH.read_text())

    report: dict[str, dict] = {}
    rows: list[dict] = []

    for experiment, label in EXPERIMENTS:
        cells = {}
        for arm in ARMS:
            values = [
                v for v in progress.get(f"{arm}:{experiment}", {}).values()
                if v is not None
            ]
            cells[arm] = {
                "n_seeds": len(values),
                "mean": round(float(np.mean(values)), 6) if values else None,
                "std": round(float(np.std(values, ddof=1)), 6) if len(values) > 1 else None,
            }
        if any(c["mean"] is None for c in cells.values()):
            continue

        # Simple effects
        gate_at_noclip = contrast(progress, "gateonly", "nodetect", experiment)
        gate_at_clip   = contrast(progress, "snas", "cliponly", experiment)
        clip_at_nogate = contrast(progress, "cliponly", "nodetect", experiment)
        clip_at_gate   = contrast(progress, "snas", "gateonly", experiment)
        both           = contrast(progress, "snas", "nodetect", experiment)

        # Main effects and interaction, computed per seed then averaged.
        seeds, snas_v, nod_v = paired(progress, "snas", "nodetect", experiment)
        _, gate_v, _ = paired(progress, "gateonly", "nodetect", experiment)
        _, clip_v, _ = paired(progress, "cliponly", "nodetect", experiment)

        gate_main = ((gate_v - nod_v) + (snas_v - clip_v)) / 2.0
        clip_main = ((clip_v - nod_v) + (snas_v - gate_v)) / 2.0
        interaction = (snas_v - gate_v) - (clip_v - nod_v)

        def summarise(values: np.ndarray) -> dict:
            return {
                "mean": round(float(np.mean(values)), 6),
                "std": round(float(np.std(values, ddof=1)), 6),
                "n_seeds": int(values.size),
                "ttest_p": float(stats.ttest_1samp(values, 0.0).pvalue),
            }

        report[experiment] = {
            "label": label,
            "cells": cells,
            "gate_main_effect": summarise(gate_main),
            "clip_main_effect": summarise(clip_main),
            "interaction": summarise(interaction),
            "simple_effects": {
                "gate_given_no_clip": gate_at_noclip,
                "gate_given_clip": gate_at_clip,
                "clip_given_no_gate": clip_at_nogate,
                "clip_given_gate": clip_at_gate,
                "both_vs_neither": both,
            },
        }

        for arm, (gate_on, clip_on) in ARMS.items():
            rows.append({
                "experiment": label,
                "arm": arm,
                "gate": "on" if gate_on else "off",
                "clip": "on" if clip_on else "off",
                "auroc_mean": cells[arm]["mean"],
                "auroc_std": cells[arm]["std"],
                "n_seeds": cells[arm]["n_seeds"],
            })

    OUT_JSON.write_text(json.dumps(
        {
            "description": (
                "P1-1 2x2 factorial ablation of the SNAS anomaly gate and the "
                "norm clip, paired by seed across ten seeds per cell."
            ),
            "arms": {a: {"gate": g, "clip": c} for a, (g, c) in ARMS.items()},
            "experiments": report,
        }, indent=2))

    with OUT_CSV.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # --- console output ---
    print("=== 2x2 cells (AUROC, 10 seeds) ===")
    print(f"{'experiment':24s} {'neither':>9s} {'clip only':>10s} {'gate only':>10s} {'both':>9s}")
    print("-" * 68)
    for experiment, data in report.items():
        c = data["cells"]
        print(f"{data['label']:24s} {c['nodetect']['mean']:9.4f} {c['cliponly']['mean']:10.4f} "
              f"{c['gateonly']['mean']:10.4f} {c['snas']['mean']:9.4f}")

    print("\n=== main effects and interaction (paired, AUROC) ===")
    print(f"{'experiment':24s} {'gate':>10s} {'clip':>10s} {'interaction':>13s}  {'both-neither':>13s}")
    print("-" * 78)
    for experiment, data in report.items():
        g = data["gate_main_effect"]["mean"]
        c = data["clip_main_effect"]["mean"]
        i = data["interaction"]["mean"]
        b = data["simple_effects"]["both_vs_neither"]["delta"]
        print(f"{data['label']:24s} {g:+10.4f} {c:+10.4f} {i:+13.4f}  {b:+13.4f}")

    print("\n=== per-contrast detail (delta, wins, Wilcoxon p) ===")
    for experiment, data in report.items():
        print(f"\n{data['label']}")
        for name, s in data["simple_effects"].items():
            if s.get("delta") is None:
                continue
            p = s.get("wilcoxon_p")
            p_txt = "n/a" if p is None else f"{p:.5f}"
            note = f"   [{s['note']}]" if s.get("note") else ""
            print(f"   {name:20s} {s['delta']:+.4f}  wins {s['win_rate']:>5s}  "
                  f"p={p_txt}  d={s.get('cohens_d_paired')}{note}")

    print(f"\nWrote {OUT_CSV}\n      {OUT_JSON}")


if __name__ == "__main__":
    main()
