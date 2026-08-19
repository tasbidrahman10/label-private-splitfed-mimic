"""
Paired significance tests on the P0-8 regeneration grid (n=3, 265 features).

Compares SNAS against each robust-aggregation baseline on every experiment they
share, at ten seeds, reading per-seed AUROCs from results/p08_rerun_summary.json.

Supersedes scripts/run_significance_tests.py, which hardcodes five seeds of
pre-P0-8 266-feature values and only covers Krum.

Tests reported per pair:
  - Wilcoxon signed-rank (paired, two-sided)
  - Paired t-test
  - Cohen's d for paired samples, sign convention SNAS - baseline
    (positive = SNAS higher)

`cohens_d_paired` is the reported figure: mean difference over the SD of the
differences, which is the correct estimator for a paired design.
`cohens_d_pooled` is retained only so the pre-P0-8 manuscript numbers (computed
with the independent-samples formula) can be traced; do not mix the two.

Pairing is by seed, not by sorted position: the same seed's SNAS and baseline
runs are matched, which is what makes the paired tests valid.

At n=10 the two-sided Wilcoxon floor is p = 2/2^10 ~= 0.00195, reached when every
pair agrees in direction. A result at exactly that value means "as significant as
this test can report at this sample size", not a coincidence.

Usage:
    python scripts/significance_tests_p08.py

Output: results/significance_tests_p08.{csv,json}
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import ttest_rel, wilcoxon

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "results" / "p08_rerun_summary.json"
OUT_CSV = ROOT / "results" / "significance_tests_p08.csv"
OUT_JSON = ROOT / "results" / "significance_tests_p08.json"

BASELINES = ["krum", "trimmed_mean", "fltrust"]
EXPERIMENTS = ["E1_gradient_scaling", "E2_free_rider", "E4_attack_straggler"]

# Gate ablation: SNAS (gate on) vs the same pipeline with the anomaly gate and
# norm clipping disabled. E6 is included here but not above, because it has no
# attacker and so tests whether the gate costs anything when it is not needed.
ABLATION_EXPERIMENTS = EXPERIMENTS + ["E6_honest_straggler"]


def paired_values(summary: dict, a: str, b: str) -> tuple[list[float], list[float], list[str]]:
    """Return (a_values, b_values, seeds) aligned on seeds present in both."""
    pa = summary.get(a, {}).get("per_seed", {})
    pb = summary.get(b, {}).get("per_seed", {})
    seeds = [s for s in pa if s in pb and pa[s] is not None and pb[s] is not None]
    seeds.sort(key=int)
    return [pa[s] for s in seeds], [pb[s] for s in seeds], seeds


def compare(snas: list[float], base: list[float]) -> dict:
    s, b = np.array(snas), np.array(base)
    diff = s - b

    if np.allclose(diff, 0):
        w_stat = w_p = float("nan")
    else:
        w_stat, w_p = wilcoxon(s, b)
    t_stat, t_p = ttest_rel(s, b)

    # Two conventions, because they give materially different numbers and the
    # paper must pick one and stay consistent:
    #   paired  - mean difference over the SD of the differences. Correct for a
    #             paired design and what we report.
    #   pooled  - mean difference over the average of the two groups' SDs. This
    #             is the independent-samples formula; scripts/run_significance_tests.py
    #             used it, so the manuscript's existing "d between -12 and -16"
    #             figures are on this scale.
    d = float(diff.mean() / diff.std(ddof=1)) if diff.std(ddof=1) > 0 else float("nan")
    pooled_sd = np.sqrt((s.std(ddof=1) ** 2 + b.std(ddof=1) ** 2) / 2)
    d_pooled = float(diff.mean() / pooled_sd) if pooled_sd > 0 else float("nan")

    return {
        "n_pairs": len(s),
        "snas_mean": float(s.mean()),
        "snas_std": float(s.std(ddof=1)),
        "baseline_mean": float(b.mean()),
        "baseline_std": float(b.std(ddof=1)),
        "mean_diff": float(diff.mean()),
        "snas_wins": int((diff > 0).sum()),
        "wilcoxon_stat": float(w_stat),
        "wilcoxon_p": float(w_p),
        "t_stat": float(t_stat),
        "t_p": float(t_p),
        "cohens_d_paired": d,
        "cohens_d_pooled": d_pooled,
        "significant_at_05": bool(w_p < 0.05) if not np.isnan(w_p) else False,
    }


def main() -> None:
    if not SUMMARY.exists():
        raise SystemExit(f"{SUMMARY} not found — run scripts/rerun_p08.py first.")
    summary = json.loads(SUMMARY.read_text())

    rows: list[dict] = []
    print("Paired significance tests — SNAS vs baselines (n=3, 265 features)\n")

    for exp in EXPERIMENTS:
        print(f"=== {exp} ===")
        for base in BASELINES:
            s_vals, b_vals, seeds = paired_values(summary, f"snas:{exp}", f"{base}:{exp}")
            if len(seeds) < 2:
                print(f"  {base:13s} skipped — only {len(seeds)} paired seed(s)")
                continue

            st = compare(s_vals, b_vals)
            direction = "SNAS higher" if st["mean_diff"] > 0 else "baseline higher"
            print(
                f"  {base:13s} SNAS {st['snas_mean']:.4f}+/-{st['snas_std']:.4f}  "
                f"vs {st['baseline_mean']:.4f}+/-{st['baseline_std']:.4f}  "
                f"| diff {st['mean_diff']:+.4f} ({direction}, SNAS wins {st['snas_wins']}/{st['n_pairs']})"
            )
            print(
                f"  {'':13s} Wilcoxon p={st['wilcoxon_p']:.5f}  t p={st['t_p']:.2e}  "
                f"d_paired={st['cohens_d_paired']:+.2f} (d_pooled={st['cohens_d_pooled']:+.2f})  "
                f"sig={st['significant_at_05']}"
            )
            rows.append({"experiment": exp, "baseline": base, "seeds": ",".join(seeds), **st})
        print()

    print("=== Gate ablation: SNAS (gate on) vs no-detection ===")
    for exp in ABLATION_EXPERIMENTS:
        s_vals, b_vals, seeds = paired_values(summary, f"snas:{exp}", f"nodetect:{exp}")
        if len(seeds) < 2:
            print(f"  {exp:22s} skipped — only {len(seeds)} paired seed(s)")
            continue
        st = compare(s_vals, b_vals)
        print(
            f"  {exp:22s} gate on {st['snas_mean']:.4f}+/-{st['snas_std']:.4f}  "
            f"off {st['baseline_mean']:.4f}+/-{st['baseline_std']:.4f}  "
            f"| benefit {st['mean_diff']:+.4f} ({st['snas_wins']}/{st['n_pairs']} seeds)"
        )
        print(
            f"  {'':22s} Wilcoxon p={st['wilcoxon_p']:.5f}  t p={st['t_p']:.2e}  "
            f"d_paired={st['cohens_d_paired']:+.2f}"
        )
        rows.append({"experiment": exp, "baseline": "nodetect",
                     "seeds": ",".join(seeds), **st})
    print()

    OUT_JSON.write_text(json.dumps(rows, indent=2))
    if rows:
        with OUT_CSV.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    print(f"Saved {OUT_CSV.relative_to(ROOT)} and {OUT_JSON.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
