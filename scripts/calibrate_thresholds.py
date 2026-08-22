"""
P1-7 — separate threshold calibration from evaluation.

The problem
-----------
delta_flag = 0.22 and delta_quarantine = 0.35 were chosen by scanning a grid of
threshold pairs for good detection and false-positive rates, and those same
rates are then reported as the result. A reviewer reads "100% DR, 0% FPR" as a
statement partly about how the thresholds were picked rather than about the
method. This script fixes the thresholds on a declared calibration set and
reports performance on data that had no influence on them.

Two splits, because they answer different questions
---------------------------------------------------
**By attack type (primary).** Calibrate on the clean runs plus gradient scaling
-- the attack the thresholds were originally tuned against -- and evaluate on
free rider, attack+straggler, label flip and backdoor. This asks the question
that matters: do thresholds fitted to one attack transfer to attacks that had
no say in choosing them? P1-2 already suggests they do not, since label flip and
backdoor are flagged every round but almost never quarantined.

**By seed (secondary).** Calibrate on five seeds, evaluate on the other five,
across all cells. This is a weaker test -- every attack still appears in the
calibration set -- but it separates "the thresholds are overfitted to these
particular runs" from "the thresholds are overfitted to this attack family".
Reported together so the two failure modes are not confused.

Calibration rules (mechanical, declared in advance)
---------------------------------------------------
    delta_flag  = max honest SNAS on the calibration set, plus a 5% margin.
                  The smallest threshold that raises no false positive on
                  calibration data. Uses only honest observations, so it is
                  well-defined even from clean runs alone.

    delta_quar  = 25th percentile of calibration attacker SNAS.
                  Set so that most, not all, attacker rounds in calibration
                  cross it -- picking the minimum would fit the single
                  lowest-scoring round, which is exactly the overfitting this
                  item exists to avoid.

Both are computed from the calibration set only, then applied unchanged to the
evaluation set. The paper's current pair is reported alongside for comparison,
and an oracle per-attack threshold shows how much is left on the table.

Warm-up rounds are excluded throughout, matching report_detection_rates.py.

Usage
-----
    python scripts/calibrate_thresholds.py \\
        --log results/label_private_splitfed/server/server_metrics.jsonl \\
              results/label_private_splitfed/server/server_metrics_nihal_p08.jsonl
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT_JSON = ROOT / "results" / "p1_7_threshold_calibration.json"
OUT_CSV = ROOT / "results" / "p1_7_threshold_calibration.csv"

# Only the full-SNAS arm: these are the thresholds the paper reports.
CALIBRATION_CELLS = ["snas:clean", "snas:E1_gradient_scaling"]
EVALUATION_CELLS = [
    "snas:E2_free_rider",
    "snas:E4_attack_straggler",
    "snas:E3_label_flip",
    "snas:E5_backdoor",
]
HONEST_ONLY_CELLS = ["snas:clean", "snas:E6_honest_straggler"]

CALIBRATION_SEEDS = {42, 7, 123, 2024, 31415}
EVALUATION_SEEDS = {1001, 2002, 3003, 4004, 5005}

PAPER_FLAG, PAPER_QUARANTINE = 0.22, 0.35
FLAG_MARGIN = 1.05          # 5% above the highest honest calibration score
QUARANTINE_PERCENTILE = 25  # of calibration attacker scores


def load_detection_module():
    spec = importlib.util.spec_from_file_location(
        "report_detection_rates", ROOT / "scripts" / "report_detection_rates.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def collect(module, args):
    """(cell, seed) -> list of (snas, is_attacker) for post-warm-up rounds."""
    grid = module.load_grid()
    runs = module.discover_runs(grid, args.since)
    blocks = module.read_gate_blocks(args.since, args.log)

    observations: dict[tuple[str, int], list] = defaultdict(list)
    for block in blocks:
        block_start = datetime.fromtimestamp(block[0]["timestamp"])
        candidates = [r for r in runs if r[0] <= block_start]
        if not candidates:
            continue
        started, cell, seed, attackers = candidates[-1]
        if (block_start - started).total_seconds() / 60.0 > args.tolerance:
            continue
        for record in block:
            if record["round"] <= module.WARMUP_ROUNDS:
                continue
            snas = record.get("snas")
            if snas is None or not np.isfinite(snas):
                continue
            observations[(cell, seed)].append(
                (float(snas), record["client_id"] in attackers))
    return observations


def pool(observations, cells=None, seeds=None):
    """(honest scores, attacker scores) over the selected cells and seeds."""
    honest, attacker = [], []
    for (cell, seed), rows in observations.items():
        if cells is not None and cell not in cells:
            continue
        if seeds is not None and seed not in seeds:
            continue
        for score, is_attacker in rows:
            (attacker if is_attacker else honest).append(score)
    return np.array(honest), np.array(attacker)


def calibrate(honest: np.ndarray, attacker: np.ndarray) -> tuple[float, float]:
    """Thresholds from calibration data only, by the declared rules."""
    flag = float(honest.max() * FLAG_MARGIN) if honest.size else PAPER_FLAG
    quarantine = (float(np.percentile(attacker, QUARANTINE_PERCENTILE))
                  if attacker.size else PAPER_QUARANTINE)
    # A quarantine threshold below the flag threshold is meaningless; the gate
    # checks quarantine first, so it must be the stricter of the two.
    quarantine = max(quarantine, flag)
    return flag, quarantine


def evaluate(honest, attacker, flag, quarantine) -> dict:
    return {
        "n_attacker_obs": int(attacker.size),
        "n_honest_obs": int(honest.size),
        "detection_rate": round(float((attacker > flag).mean()), 4) if attacker.size else None,
        "quarantine_rate": round(float((attacker > quarantine).mean()), 4) if attacker.size else None,
        "false_positive_rate": round(float((honest > flag).mean()), 4) if honest.size else None,
        "false_quarantine_rate": round(float((honest > quarantine).mean()), 4) if honest.size else None,
    }


def oracle_threshold(honest, attacker) -> dict:
    """Best achievable on this cell: the highest honest score is the ceiling."""
    if not honest.size or not attacker.size:
        return {}
    ceiling = float(honest.max())
    return {
        "honest_max": round(ceiling, 4),
        "attacker_min": round(float(attacker.min()), 4),
        "separable": bool(attacker.min() > ceiling),
        "margin": round(float(attacker.min() - ceiling), 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--since", default="2026-08-17 00:00")
    parser.add_argument("--tolerance", type=float, default=4.0)
    parser.add_argument("--log", nargs="+", type=Path)
    args = parser.parse_args()
    args.since = datetime.strptime(args.since, "%Y-%m-%d %H:%M")

    module = load_detection_module()
    observations = collect(module, args)

    report: dict = {"rules": {
        "delta_flag": f"max honest SNAS on calibration x {FLAG_MARGIN}",
        "delta_quarantine": f"{QUARANTINE_PERCENTILE}th percentile of calibration attacker SNAS",
        "paper_values": {"flag": PAPER_FLAG, "quarantine": PAPER_QUARANTINE},
    }}
    rows: list[dict] = []

    # ---------- primary split: by attack type ----------
    cal_h, cal_a = pool(observations, cells=CALIBRATION_CELLS)
    flag, quarantine = calibrate(cal_h, cal_a)
    report["split_by_attack"] = {
        "calibration_cells": CALIBRATION_CELLS,
        "evaluation_cells": EVALUATION_CELLS,
        "calibration_honest_obs": int(cal_h.size),
        "calibration_attacker_obs": int(cal_a.size),
        "calibration_honest_max": round(float(cal_h.max()), 4) if cal_h.size else None,
        "derived_flag": round(flag, 4),
        "derived_quarantine": round(quarantine, 4),
        "held_out": {},
    }
    for cell in EVALUATION_CELLS:
        honest, attacker = pool(observations, cells=[cell])
        derived = evaluate(honest, attacker, flag, quarantine)
        paper = evaluate(honest, attacker, PAPER_FLAG, PAPER_QUARANTINE)
        report["split_by_attack"]["held_out"][cell] = {
            "derived_thresholds": derived,
            "paper_thresholds": paper,
            "oracle": oracle_threshold(honest, attacker),
        }
        rows.append({
            "split": "by_attack", "cell": cell,
            "flag": round(flag, 4), "quarantine": round(quarantine, 4),
            **{f"derived_{k}": v for k, v in derived.items()},
            **{f"paper_{k}": v for k, v in paper.items()},
        })

    # false positives measured on the honest-only cells, held out from calibration
    for cell in HONEST_ONLY_CELLS:
        if cell in CALIBRATION_CELLS:
            continue
        honest, attacker = pool(observations, cells=[cell])
        report["split_by_attack"]["held_out"][cell] = {
            "derived_thresholds": evaluate(honest, attacker, flag, quarantine),
            "paper_thresholds": evaluate(honest, attacker, PAPER_FLAG, PAPER_QUARANTINE),
        }

    # ---------- secondary split: by seed ----------
    all_cells = CALIBRATION_CELLS + EVALUATION_CELLS
    cal_h_s, cal_a_s = pool(observations, cells=all_cells, seeds=CALIBRATION_SEEDS)
    flag_s, quarantine_s = calibrate(cal_h_s, cal_a_s)
    eval_h_s, eval_a_s = pool(observations, cells=all_cells, seeds=EVALUATION_SEEDS)
    report["split_by_seed"] = {
        "calibration_seeds": sorted(CALIBRATION_SEEDS),
        "evaluation_seeds": sorted(EVALUATION_SEEDS),
        "derived_flag": round(flag_s, 4),
        "derived_quarantine": round(quarantine_s, 4),
        "held_out_seeds": {
            "derived_thresholds": evaluate(eval_h_s, eval_a_s, flag_s, quarantine_s),
            "paper_thresholds": evaluate(eval_h_s, eval_a_s, PAPER_FLAG, PAPER_QUARANTINE),
        },
    }
    rows.append({
        "split": "by_seed", "cell": "all (held-out seeds)",
        "flag": round(flag_s, 4), "quarantine": round(quarantine_s, 4),
        **{f"derived_{k}": v for k, v in
           evaluate(eval_h_s, eval_a_s, flag_s, quarantine_s).items()},
        **{f"paper_{k}": v for k, v in
           evaluate(eval_h_s, eval_a_s, PAPER_FLAG, PAPER_QUARANTINE).items()},
    })

    # ---------- the global separability window ----------
    # Across every attack cell, the highest honest score and the lowest attacker
    # score bound the set of flag thresholds that give 100% DR at 0% FPR
    # simultaneously on all of them. If that window is non-empty, one global
    # flag threshold suffices and the paper's value can be checked against it.
    attack_cells = ["snas:E1_gradient_scaling"] + EVALUATION_CELLS
    worst_honest, worst_attacker, per_cell = 0.0, float("inf"), {}
    for cell in attack_cells + HONEST_ONLY_CELLS:
        honest, attacker = pool(observations, cells=[cell])
        if honest.size:
            worst_honest = max(worst_honest, float(honest.max()))
        if attacker.size:
            worst_attacker = min(worst_attacker, float(attacker.min()))
        per_cell[cell] = {
            "honest_max": round(float(honest.max()), 4) if honest.size else None,
            "attacker_min": round(float(attacker.min()), 4) if attacker.size else None,
        }
    report["global_flag_window"] = {
        "lower_bound_highest_honest": round(worst_honest, 4),
        "upper_bound_lowest_attacker": round(worst_attacker, 4),
        "window_is_non_empty": bool(worst_attacker > worst_honest),
        "width": round(worst_attacker - worst_honest, 4),
        "paper_flag_inside": bool(worst_honest < PAPER_FLAG < worst_attacker),
        "per_cell": per_cell,
    }

    OUT_JSON.write_text(json.dumps(report, indent=2))
    with OUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # ---------- console ----------
    split = report["split_by_attack"]
    print("=== PRIMARY SPLIT — calibrate on attack type, evaluate on unseen attacks ===")
    print(f"  calibration: {', '.join(CALIBRATION_CELLS)}")
    print(f"    honest obs {split['calibration_honest_obs']}, max honest SNAS "
          f"{split['calibration_honest_max']}")
    print(f"    -> delta_flag       = {split['derived_flag']}   (paper: {PAPER_FLAG})")
    print(f"    -> delta_quarantine = {split['derived_quarantine']}   (paper: {PAPER_QUARANTINE})")
    print()
    print(f"  {'held-out cell':26s} {'DR':>7s} {'quar':>7s} {'FPR':>7s} | "
          f"{'paper DR':>8s} {'paper quar':>10s} | {'sep?':>5s} {'margin':>7s}")
    print("  " + "-" * 88)
    for cell in EVALUATION_CELLS:
        entry = split["held_out"][cell]
        d, p, o = entry["derived_thresholds"], entry["paper_thresholds"], entry["oracle"]
        print(f"  {cell:26s} {d['detection_rate']:>7.1%} {d['quarantine_rate']:>7.1%} "
              f"{d['false_positive_rate']:>7.1%} | {p['detection_rate']:>8.1%} "
              f"{p['quarantine_rate']:>10.1%} | {str(o.get('separable')):>5s} "
              f"{o.get('margin'):>7}")
    for cell in HONEST_ONLY_CELLS:
        if cell in CALIBRATION_CELLS:
            continue
        d = split["held_out"][cell]["derived_thresholds"]
        print(f"  {cell:26s} {'--':>7s} {'--':>7s} {d['false_positive_rate']:>7.1%} | "
              f"{'--':>8s} {'--':>10s} |   (honest-only)")

    window = report["global_flag_window"]
    print()
    print("=== GLOBAL FLAG-THRESHOLD WINDOW (all attacks + honest-only cells) ===")
    print(f"  highest honest score anywhere : {window['lower_bound_highest_honest']}")
    print(f"  lowest attacker score anywhere: {window['upper_bound_lowest_attacker']}")
    print(f"  -> any flag threshold in ({window['lower_bound_highest_honest']}, "
          f"{window['upper_bound_lowest_attacker']}) gives 100% DR at 0% FPR on ALL cells")
    print(f"  window width: {window['width']}   paper's 0.22 inside: "
          f"{window['paper_flag_inside']}")

    seed_split = report["split_by_seed"]
    print()
    print("=== SECONDARY SPLIT — calibrate on 5 seeds, evaluate on the other 5 ===")
    print(f"  -> delta_flag = {seed_split['derived_flag']}, "
          f"delta_quarantine = {seed_split['derived_quarantine']}")
    d = seed_split["held_out_seeds"]["derived_thresholds"]
    p = seed_split["held_out_seeds"]["paper_thresholds"]
    print(f"  held-out seeds, derived: DR {d['detection_rate']:.1%}  "
          f"quar {d['quarantine_rate']:.1%}  FPR {d['false_positive_rate']:.1%}")
    print(f"  held-out seeds, paper:   DR {p['detection_rate']:.1%}  "
          f"quar {p['quarantine_rate']:.1%}  FPR {p['false_positive_rate']:.1%}")
    print(f"\nWrote {OUT_CSV}\n      {OUT_JSON}")


if __name__ == "__main__":
    main()
