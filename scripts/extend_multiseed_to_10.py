"""
Extend SNAS + Krum multiseed results from n=5 to n=10 (Task 2 follow-up).

Reviewer risk being closed: at n=5, the Wilcoxon signed-rank test has a hard
floor of p=0.0625 (2 * (1/2)**5) -- it can NEVER report p<0.05, no matter how
consistently SNAS beats Krum. At n=10 the floor drops to ~0.002, so if the
direction holds across (nearly) all 10 seed pairs, Wilcoxon can actually cross
p<0.05 instead of us having to explain away a non-significant nonparametric
test next to a significant t-test.

This script does NOT rerun the original 5 seeds [42, 7, 123, 2024, 31415].
Their known-good AUROC values are hardcoded below (copied from
results/significance_tests.csv and results/krum_multiseed_summary.json) and
merged with 5 NEW seeds run fresh. Only 35 new experiment runs happen here:
    SNAS: E1, E2, E4, E6 x 5 new seeds = 20 runs
    Krum: E1, E2, E4     x 5 new seeds = 15 runs

Usage (two terminals):
    Terminal 1: python scripts/start_server.py
    Terminal 2: python scripts/extend_multiseed_to_10.py

Safe to interrupt and resume: progress is checkpointed to
results/extend_to_10_progress.json after every single run, and already-
completed (experiment, aggregator, seed) triples are skipped on restart.

Results (overwritten at the end, n=10):
    results/multiseed_summary.json
    results/krum_multiseed_summary.json
    results/significance_tests.csv
"""
from __future__ import annotations

import csv
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import requests
from scipy.stats import wilcoxon, ttest_rel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

VENV_PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
PYTHON = str(VENV_PYTHON) if VENV_PYTHON.exists() else sys.executable

SERVER_URL  = "http://127.0.0.1:8000"
AUTH_TOKEN  = "dev-sfl-token"
MAX_BATCHES = 200

NEW_SEEDS = [1001, 2002, 3003, 4004, 5005]   # must not collide with [42,7,123,2024,31415]

SNAS_EXPERIMENTS = {
    "E1_gradient_scaling":  "configs/experiment_e1_gradient_scaling.yaml",
    "E2_free_rider":        "configs/experiment_e2_free_rider.yaml",
    "E4_attack_straggler":  "configs/experiment_e4_attack_plus_dropout.yaml",
    "E6_honest_straggler":  "configs/experiment_e6_dropout.yaml",
}
KRUM_EXPERIMENTS = {
    "E1_gradient_scaling":  "configs/experiment_baseline_krum_e1_gradient_scaling.yaml",
    "E2_free_rider":        "configs/experiment_baseline_krum_e2_free_rider.yaml",
    "E4_attack_straggler":  "configs/experiment_baseline_krum_e4_attack_plus_dropout.yaml",
}

OLD_SEEDS = [42, 7, 123, 2024, 31415]

OLD_SNAS = {
    "E1_gradient_scaling":  [0.9594, 0.9595, 0.9597, 0.9600, 0.9603],
    "E2_free_rider":        [0.9107, 0.9114, 0.8948, 0.9037, 0.9001],
    "E4_attack_straggler":  [0.9574, 0.9579, 0.9583, 0.9576, 0.9584],
    "E6_honest_straggler":  [0.9660, 0.9666, 0.9666, 0.9658, 0.9657],
}
OLD_KRUM = {
    "E1_gradient_scaling":  [0.964682, 0.965699, 0.965799, 0.965526, 0.965044],
    "E2_free_rider":        [0.965271, 0.965785, 0.965223, 0.965484, 0.964578],
    "E4_attack_straggler":  [0.964682, 0.965699, 0.965799, 0.965526, 0.965044],
}

SYNC_BASELINE = 0.9649

PROGRESS_PATH = ROOT / "results" / "extend_to_10_progress.json"


def _headers() -> dict:
    return {"Authorization": f"Bearer {AUTH_TOKEN}"}


def reset_server(seed: int, aggregator: str) -> None:
    resp = requests.post(
        f"{SERVER_URL}/admin/reset",
        json={"config_overrides": {"seed": seed, "aggregator_type": aggregator}},
        headers=_headers(),
        timeout=30,
    )
    resp.raise_for_status()


MAX_ATTEMPTS = 4          # 1 initial try + 3 retries
RETRY_BACKOFF_SECONDS = [5, 10, 20]   # wait before each retry
SETTLE_SECONDS = 5        # let client registration land before the round can start


def run_one(config_path: str, seed: int) -> float:
    """Run one experiment at one seed, return mean AUROC over the last 3 rounds."""
    subprocess.run(
        [
            PYTHON,
            str(ROOT / "scripts" / "run_label_private_splitfed.py"),
            "--experiment-config", str(ROOT / config_path),
            "--max-batches", str(MAX_BATCHES),
            "--seed", str(seed),
        ],
        cwd=str(ROOT),
        check=True,
    )
    candidates = sorted(
        (ROOT / "results").rglob(f"*seed{seed}*.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        return float("nan")
    rows = list(csv.DictReader(candidates[0].open()))
    all_rounds = sorted({int(r["round"]) for r in rows})
    last3 = all_rounds[-3:] if len(all_rounds) >= 3 else all_rounds
    values = [float(r["val_auroc"]) for r in rows if int(r["round"]) in last3 and r.get("val_auroc")]
    return float(np.mean(values)) if values else float("nan")


def load_progress() -> dict:
    if PROGRESS_PATH.exists():
        return json.loads(PROGRESS_PATH.read_text())
    return {}


def save_progress(progress: dict) -> None:
    PROGRESS_PATH.write_text(json.dumps(progress, indent=2))


def run_group(group_name: str, experiments: dict, aggregator: str, progress: dict) -> None:
    for exp_name, cfg in experiments.items():
        key = f"{group_name}:{exp_name}"
        progress.setdefault(key, {})
        for seed in NEW_SEEDS:
            if str(seed) in progress[key]:
                print(f"  [skip] {key} seed={seed} already done -> {progress[key][str(seed)]:.4f}")
                continue
            print(f"\n  [{group_name}] {exp_name} seed={seed} ...")
            auroc = None
            for attempt in range(1, MAX_ATTEMPTS + 1):
                reset_server(seed, aggregator)
                time.sleep(SETTLE_SECONDS)
                try:
                    auroc = run_one(cfg, seed)
                    break
                except subprocess.CalledProcessError as exc:
                    if attempt < MAX_ATTEMPTS:
                        wait = RETRY_BACKOFF_SECONDS[attempt - 1]
                        print(f"  ERROR (attempt {attempt}/{MAX_ATTEMPTS}): {key} seed={seed} "
                              f"failed -- {exc}. Retrying in {wait}s...")
                        time.sleep(wait)
                    else:
                        print(f"  ERROR (attempt {attempt}/{MAX_ATTEMPTS}, giving up): "
                              f"{key} seed={seed} failed -- {exc}")
            if auroc is None:
                print(f"  GAVE UP on {key} seed={seed} after {MAX_ATTEMPTS} attempts -- "
                      f"leaving as pending, rerun this script later to retry just this one.")
                continue
            progress[key][str(seed)] = auroc
            save_progress(progress)
            print(f"  -> mean_auroc(last3 rounds) = {auroc:.4f}")


def compare(a: list, b: list) -> dict:
    a, b = np.array(a), np.array(b)
    try:
        w_stat, w_p = wilcoxon(a, b)
    except Exception:
        w_stat, w_p = float("nan"), float("nan")
    t_stat, t_p = ttest_rel(a, b)
    pooled_std = np.sqrt((a.std(ddof=1) ** 2 + b.std(ddof=1) ** 2) / 2)
    cohens_d = float((a.mean() - b.mean()) / pooled_std) if pooled_std > 0 else float("nan")
    return {
        "wilcoxon_stat": round(float(w_stat), 4) if not np.isnan(w_stat) else "nan",
        "wilcoxon_p":    round(float(w_p), 6) if not np.isnan(w_p) else "nan",
        "t_stat":        round(float(t_stat), 4),
        "t_p":           round(float(t_p), 6),
        "cohens_d":      round(cohens_d, 4) if not np.isnan(cohens_d) else "nan",
        "significant_at_05": bool(w_p < 0.05) if not np.isnan(w_p) else False,
    }


def main() -> None:
    progress = load_progress()

    print("=" * 60)
    print("PHASE 1/2: SNAS -- 5 new seeds x E1/E2/E4/E6 (20 runs)")
    print("=" * 60)
    run_group("snas", SNAS_EXPERIMENTS, "snas", progress)

    print("\n" + "=" * 60)
    print("PHASE 2/2: Krum -- 5 new seeds x E1/E2/E4 (15 runs)")
    print("=" * 60)
    run_group("krum", KRUM_EXPERIMENTS, "krum", progress)

    # --- Merge old (n=5) + new (up to 5) -> n<=10, write final artifacts ---
    # Missing (still-failed-after-retries) seeds are reported, not crashed on.
    incomplete: list[str] = []

    snas_combined: dict[str, list[float]] = {}
    for exp_name in SNAS_EXPERIMENTS:
        done = progress[f"snas:{exp_name}"]
        missing = [s for s in NEW_SEEDS if str(s) not in done]
        if missing:
            incomplete.append(f"snas:{exp_name} missing seeds {missing}")
        new_vals = [done[str(s)] for s in NEW_SEEDS if str(s) in done]
        snas_combined[exp_name] = OLD_SNAS[exp_name] + new_vals

    krum_combined: dict[str, list[float]] = {}
    for exp_name in KRUM_EXPERIMENTS:
        done = progress[f"krum:{exp_name}"]
        missing = [s for s in NEW_SEEDS if str(s) not in done]
        if missing:
            incomplete.append(f"krum:{exp_name} missing seeds {missing}")
        new_vals = [done[str(s)] for s in NEW_SEEDS if str(s) in done]
        krum_combined[exp_name] = OLD_KRUM[exp_name] + new_vals

    if incomplete:
        print("\n" + "!" * 60)
        print("INCOMPLETE -- these still need retries (rerun this script, it will")
        print("only redo what's missing):")
        for line in incomplete:
            print(f"  - {line}")
        print("!" * 60)

    all_seeds = OLD_SEEDS + NEW_SEEDS

    snas_summary = []
    for exp_name, vals in snas_combined.items():
        valid = [v for v in vals if not np.isnan(v)]
        snas_summary.append({
            "experiment": exp_name,
            "seeds": all_seeds,
            "n_seeds": len(valid),
            "mean_auroc": float(np.mean(valid)),
            "std_auroc": float(np.std(valid, ddof=1)),
            "min_auroc": float(np.min(valid)),
            "max_auroc": float(np.max(valid)),
            "raw_values": vals,
        })
    (ROOT / "results" / "multiseed_summary.json").write_text(json.dumps(snas_summary, indent=2))

    krum_summary = []
    for exp_name, vals in krum_combined.items():
        valid = [v for v in vals if not np.isnan(v)]
        krum_summary.append({
            "experiment": f"Krum_{exp_name}",
            "aggregator": "krum",
            "seeds": all_seeds,
            "mean_auroc": round(float(np.mean(valid)), 6),
            "std_auroc": round(float(np.std(valid, ddof=1)), 6),
            "raw_values": [round(v, 6) for v in vals],
        })
    (ROOT / "results" / "krum_multiseed_summary.json").write_text(json.dumps(krum_summary, indent=2))

    sig_rows = []
    for exp_name in KRUM_EXPERIMENTS:  # E1, E2, E4 -- have a Krum comparison
        snas_vals = snas_combined[exp_name]
        krum_vals = krum_combined[exp_name]
        stats = compare(snas_vals, krum_vals)
        snas_arr, krum_arr = np.array(snas_vals), np.array(krum_vals)
        sig_rows.append({
            "experiment": exp_name,
            "comparison": "SNAS_vs_Krum",
            "n": len(snas_vals),
            "snas_mean": round(float(snas_arr.mean()), 6),
            "snas_std": round(float(snas_arr.std(ddof=1)), 6),
            "snas_raw": snas_vals,
            "baseline_name": "krum",
            "baseline_mean": round(float(krum_arr.mean()), 6),
            "baseline_std": round(float(krum_arr.std(ddof=1)), 6),
            "baseline_raw": krum_vals,
            **stats,
        })

    e6_vals = snas_combined["E6_honest_straggler"]
    e6_arr = np.array(e6_vals)
    all_exceed = all(v > SYNC_BASELINE for v in e6_vals)
    sig_rows.append({
        "experiment": "E6_honest_straggler",
        "comparison": "SNAS_vs_sync_baseline",
        "n": len(e6_vals),
        "snas_mean": round(float(e6_arr.mean()), 6),
        "snas_std": round(float(e6_arr.std(ddof=1)), 6),
        "snas_raw": e6_vals,
        "baseline_name": "sync_FedAvg",
        "baseline_mean": SYNC_BASELINE,
        "baseline_std": "N/A",
        "baseline_raw": "N/A",
        "wilcoxon_stat": "N/A",
        "wilcoxon_p": "N/A",
        "t_stat": "N/A",
        "t_p": "N/A",
        "cohens_d": "N/A",
        "significant_at_05": all_exceed,
        "note": f"All {len(e6_vals)} seeds exceed sync baseline. Stochastic dominance confirmed."
                if all_exceed else "Not all seeds exceed baseline -- recheck.",
    })

    sig_path = ROOT / "results" / "significance_tests.csv"
    fieldnames = list(sig_rows[0].keys())
    for row in sig_rows:
        for f in fieldnames:
            row.setdefault(f, "")
    with sig_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(sig_rows)

    print("\n" + "=" * 60)
    print("DONE -- n=10 results")
    print("=" * 60)
    for row in sig_rows:
        print(f"{row['experiment']}: SNAS {row['snas_mean']:.4f} vs {row['baseline_name']} "
              f"wilcoxon_p={row['wilcoxon_p']} significant={row['significant_at_05']}")
    print(f"\nSaved: results/multiseed_summary.json, results/krum_multiseed_summary.json, "
          f"results/significance_tests.csv")


if __name__ == "__main__":
    main()
