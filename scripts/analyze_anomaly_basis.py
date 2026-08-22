"""
P1-5 — does the detector still work when scored on updates instead of weights?

The review's deepest technical objection is that `cos(theta_i, theta^g)` is
dominated by the shared base: every client starts the round from theta^g and
takes a small step, so the cosine is near 1 for honest and adversarial clients
alike and the discriminative part is a perturbation on top. The objection is
correct about the mechanism. The open question is whether re-scoring on the
update delta_i = theta_i - theta^g detects *better*.

Experimental design
-------------------
The `deltadet` arm is identical to `nodetect` in every respect except
`snas_anomaly_basis`, and both run with the gate off. With the gate off no
client is ever excluded, so at a given seed the two arms follow the *same*
training trajectory -- verified: their final AUROCs agree to 4 dp, exactly
(0.9578/0.9578, 0.8185/0.8185, 0.9567/0.9567 at seed 42). The two bases are
therefore compared on the same rounds, the same weights and the same attacker,
with only the scoring function changed.

Scoring the arms with the gate off is deliberate. The absolute basis's
thresholds (flag 0.22, quarantine 0.35) were calibrated for it and there is no
reason they should transfer, so comparing gate *decisions* would measure
calibration rather than signal. Instead this measures the separability of the
score itself, which is threshold-free:

  roc_auc         rank-based separation of attacker from honest observations.
                  1.0 = every attacker round scores above every honest round;
                  0.5 = no information.
  dr_at_zero_fpr  the largest detection rate reachable at zero false
                  positives, i.e. what a perfectly recalibrated threshold
                  could achieve. This is the decision-relevant number: if it
                  is 1.0 the basis is usable and only needs re-tuning; if it
                  is low, no threshold rescues it.
  separation      mean attacker score / mean honest score.

Warm-up rounds are excluded (see report_detection_rates.py).

Usage
-----
    python scripts/analyze_anomaly_basis.py \\
        --log results/label_private_splitfed/server/server_metrics.jsonl \\
              results/label_private_splitfed/server/server_metrics_nihal_p08.jsonl
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT_JSON = ROOT / "results" / "p1_5_anomaly_basis.json"
OUT_CSV = ROOT / "results" / "p1_5_anomaly_basis.csv"

# (absolute-basis arm, delta-basis arm) -- identical configs but for the basis.
ARM_PAIRS = [("nodetect", "deltadet")]
EXPERIMENTS = ["E1_gradient_scaling", "E2_free_rider", "E4_attack_straggler"]


def load_detection_module():
    spec = importlib.util.spec_from_file_location(
        "report_detection_rates", ROOT / "scripts" / "report_detection_rates.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Rank-based AUC (Mann-Whitney U), ties averaged. No sklearn dependency."""
    positives, negatives = scores[labels == 1], scores[labels == 0]
    if positives.size == 0 or negatives.size == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(scores.size, dtype=float)
    ranks[order] = np.arange(1, scores.size + 1)
    # average ranks within tied groups
    sorted_scores = scores[order]
    start = 0
    for index in range(1, scores.size + 1):
        if index == scores.size or sorted_scores[index] != sorted_scores[start]:
            ranks[order[start:index]] = ranks[order[start:index]].mean()
            start = index
    rank_sum = ranks[labels == 1].sum()
    return float((rank_sum - positives.size * (positives.size + 1) / 2)
                 / (positives.size * negatives.size))


def dr_at_zero_fpr(scores: np.ndarray, labels: np.ndarray) -> float:
    """Detection rate achievable while flagging no honest observation.

    A threshold strictly above the highest honest score is the only one with
    zero false positives, so this is the fraction of attacker observations
    above that ceiling.
    """
    positives, negatives = scores[labels == 1], scores[labels == 0]
    if positives.size == 0 or negatives.size == 0:
        return float("nan")
    return float((positives > negatives.max()).mean())


def collect(module, args):
    """(cell, seed) -> list of (client_id, round, snas, is_attacker)."""
    grid = module.load_grid()
    runs = module.discover_runs(grid, args.since)
    blocks = module.read_gate_blocks(args.since or runs[0][0], args.log)

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
            if snas is None:
                continue
            observations[(cell, seed)].append(
                (record["client_id"], record["round"], float(snas),
                 record["client_id"] in attackers))
    return observations


def split_finite(pooled):
    """(usable observations, count of non-finite ones).

    Non-finite scores are excluded from the metrics but counted and reported.
    They are not noise: with the gate off, a 10x attacker compounds unchecked
    for ten rounds and the encoder overflows float32, so the update vector
    stops being finite. The absolute basis never shows this because its norm
    term caps at 1.0 (min(log_ratio / LOG_CAP, 1.0)), which absorbs an infinite
    ratio; the delta term had no equivalent cap when these runs were produced.
    """
    usable = [o for o in pooled if np.isfinite(o[2])]
    return usable, len(pooled) - len(usable)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--since", default="2026-08-17 00:00")
    parser.add_argument("--tolerance", type=float, default=12.0)
    parser.add_argument("--log", nargs="+", type=Path)
    args = parser.parse_args()
    args.since = datetime.strptime(args.since, "%Y-%m-%d %H:%M")

    module = load_detection_module()
    observations = collect(module, args)

    report, rows = {}, []
    for absolute_arm, delta_arm in ARM_PAIRS:
        for experiment in EXPERIMENTS:
            entry = {}
            for arm, basis in ((absolute_arm, "absolute"), (delta_arm, "delta")):
                pooled = [o for (cell, _seed), obs in observations.items()
                          if cell == f"{arm}:{experiment}" for o in obs]
                if not pooled:
                    continue
                pooled, n_nonfinite = split_finite(pooled)
                if not pooled:
                    continue
                scores = np.array([o[2] for o in pooled])
                labels = np.array([1 if o[3] else 0 for o in pooled])
                seeds = {s for (cell, s) in observations if cell == f"{arm}:{experiment}"}
                attacker_mean = float(scores[labels == 1].mean())
                honest_mean = float(scores[labels == 0].mean())
                entry[basis] = {
                    "arm": arm,
                    "n_seeds": len(seeds),
                    "n_attacker_obs": int((labels == 1).sum()),
                    "n_honest_obs": int((labels == 0).sum()),
                    "mean_attacker": round(attacker_mean, 4),
                    "mean_honest": round(honest_mean, 4),
                    "separation": round(attacker_mean / honest_mean, 2) if honest_mean else None,
                    "roc_auc": round(roc_auc(scores, labels), 4),
                    "dr_at_zero_fpr": round(dr_at_zero_fpr(scores, labels), 4),
                    "n_nonfinite_excluded": n_nonfinite,
                }
            if len(entry) == 2:
                report[experiment] = entry
                for basis, values in entry.items():
                    rows.append({"experiment": experiment, "basis": basis, **values})

    OUT_JSON.write_text(json.dumps({
        "design": ("deltadet and nodetect share a training trajectory at each seed "
                   "(gate off in both); only the anomaly basis differs."),
        "warmup_rounds_excluded": module.WARMUP_ROUNDS,
        "experiments": report,
    }, indent=2))
    if rows:
        import csv as _csv
        with OUT_CSV.open("w", newline="", encoding="utf-8") as handle:
            writer = _csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    print(f"{'experiment':22s} {'basis':9s} {'seeds':>5s} {'atk':>8s} {'honest':>8s} "
          f"{'sep':>6s} {'ROC AUC':>8s} {'DR@0FPR':>8s}")
    print("-" * 82)
    for experiment, entry in report.items():
        for basis in ("absolute", "delta"):
            v = entry[basis]
            flag = f"   [{v['n_nonfinite_excluded']} non-finite excluded]" if v["n_nonfinite_excluded"] else ""
            print(f"{experiment:22s} {basis:9s} {v['n_seeds']:>5d} "
                  f"{v['mean_attacker']:>8.4f} {v['mean_honest']:>8.4f} "
                  f"{v['separation']:>6.2f} {v['roc_auc']:>8.4f} "
                  f"{v['dr_at_zero_fpr']:>8.1%}{flag}")
        print()
    print(f"Wrote {OUT_CSV}\n      {OUT_JSON}")


if __name__ == "__main__":
    main()
