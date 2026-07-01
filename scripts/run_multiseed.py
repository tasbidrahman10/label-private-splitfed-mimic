"""
Multi-seed experiment runner for statistical validation (Task 2).

Reruns E1, E2, E4, E6 at 5 seeds each for SNAS aggregator, then computes
mean +/- std and performs Wilcoxon signed-rank test against the best
Task 1 baseline for each experiment.

Usage:
    python scripts/run_multiseed.py

The server must be running before calling this script. The script uses
POST /admin/reset to reinitialise server state between seed runs without
restarting the process.

Results saved to:
    results/multiseed/<exp>_seed<N>.csv      (one per run)
    results/multiseed_summary.json           (aggregated mean +/- std)
    results/significance_tests.csv           (statistical tests)
"""
from __future__ import annotations

import csv
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

VENV_PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
PYTHON = str(VENV_PYTHON) if VENV_PYTHON.exists() else sys.executable

# --- Configuration -----------------------------------------------------------

SEEDS = [42, 7, 123, 2024, 31415]   # seed 42 retained for continuity

EXPERIMENTS = {
    "E1_gradient_scaling":  "configs/experiment_e1_gradient_scaling.yaml",
    "E2_free_rider":        "configs/experiment_e2_free_rider.yaml",
    "E4_attack_straggler":  "configs/experiment_e4_attack_plus_dropout.yaml",
    "E6_honest_straggler":  "configs/experiment_e6_dropout.yaml",
}

SERVER_URL   = "http://127.0.0.1:8000"
AUTH_TOKEN   = "dev-sfl-token"
MAX_BATCHES  = 200
OUT_DIR      = ROOT / "results" / "multiseed"
SUMMARY_PATH = ROOT / "results" / "multiseed_summary.json"
SIG_PATH     = ROOT / "results" / "significance_tests.csv"

# Best baseline per experiment from Task 1 results (fill in after Task 1 runs)
# Format: experiment_name -> (aggregator_name, [auroc_values_per_seed])
# Populated by fill_baseline_values() or manually after Task 1 CSV is ready.
BEST_BASELINE_AUROCS: dict[str, tuple[str, list[float]]] = {
    # Populated after Task 1 completes — leave empty for now
    # "E1_gradient_scaling": ("fltrust", []),
}


# --- Helpers -----------------------------------------------------------------

def _headers() -> dict:
    return {"Authorization": f"Bearer {AUTH_TOKEN}"}


def reset_server(seed: int, aggregator: str = "snas") -> None:
    resp = requests.post(
        f"{SERVER_URL}/admin/reset",
        json={"config_overrides": {"seed": seed, "aggregator_type": aggregator}},
        headers=_headers(),
        timeout=30,
    )
    resp.raise_for_status()
    print(f"  Server reset: seed={seed} aggregator={aggregator} -> {resp.json()}")


def run_one(exp_name: str, config_path: str, seed: int) -> Path:
    """Run one experiment at a given seed. Returns path to the output CSV."""
    out_dir = OUT_DIR / exp_name
    out_dir.mkdir(parents=True, exist_ok=True)

    reset_server(seed)
    time.sleep(1)  # let the server finish resetting

    result = subprocess.run(
        [
            PYTHON,
            str(ROOT / "scripts" / "run_label_private_splitfed.py"),
            "--experiment-config", str(ROOT / config_path),
            "--max-batches", str(MAX_BATCHES),
            "--seed", str(seed),
        ],
        cwd=str(ROOT),
        capture_output=False,
        check=True,
    )

    # Find the CSV written by this run (most recent in results dir)
    results_base = ROOT / "results"
    exp_key = config_path.replace("configs/", "").replace(".yaml", "")
    # Locate by scanning for files matching seed stamp
    seed_stamp = f"seed{seed}"
    candidates = sorted(
        (ROOT / "results").rglob(f"*{seed_stamp}*.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if candidates:
        return candidates[0]
    # fallback: return last modified CSV anywhere in results/
    all_csvs = sorted(
        (ROOT / "results").rglob("experiment_summary_*.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return all_csvs[0] if all_csvs else Path("/dev/null")


def extract_mean_auroc(csv_path: Path, rounds: list[int] | None = None) -> float:
    """Return mean AUROC across all clients for the last N rounds (default 8-10)."""
    if not csv_path.exists():
        return float("nan")
    rows = list(csv.DictReader(csv_path.open()))
    if rounds is None:
        all_rounds = sorted({int(r["round"]) for r in rows})
        rounds = all_rounds[-3:] if len(all_rounds) >= 3 else all_rounds

    values = [
        float(r["val_auroc"])
        for r in rows
        if int(r["round"]) in rounds and r.get("val_auroc")
    ]
    return float(np.mean(values)) if values else float("nan")


def aggregate_seeds(exp_name: str) -> dict:
    out_dir = OUT_DIR / exp_name
    aurocs = []
    for seed in SEEDS:
        candidates = sorted(
            out_dir.rglob(f"*seed{seed}*.csv"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if candidates:
            aurocs.append(extract_mean_auroc(candidates[0]))
        else:
            aurocs.append(float("nan"))

    valid = [v for v in aurocs if not np.isnan(v)]
    return {
        "experiment": exp_name,
        "n_seeds": len(valid),
        "mean_auroc": float(np.mean(valid)) if valid else float("nan"),
        "std_auroc":  float(np.std(valid, ddof=1)) if len(valid) > 1 else float("nan"),
        "min_auroc":  float(np.min(valid)) if valid else float("nan"),
        "max_auroc":  float(np.max(valid)) if valid else float("nan"),
        "raw_values": aurocs,
        "seeds": SEEDS,
    }


def run_significance_tests(summary: list[dict]) -> list[dict]:
    """Wilcoxon signed-rank + paired t-test + Cohen's d for each experiment."""
    from scipy.stats import wilcoxon, ttest_rel

    sig_rows = []
    for exp_summary in summary:
        exp_name = exp_summary["experiment"]
        snas_values = np.array([v for v in exp_summary["raw_values"] if not np.isnan(v)])

        if exp_name not in BEST_BASELINE_AUROCS or len(snas_values) < 2:
            sig_rows.append({
                "experiment": exp_name,
                "snas_mean": round(float(np.mean(snas_values)), 6) if len(snas_values) else "nan",
                "snas_std":  round(float(np.std(snas_values, ddof=1)), 6) if len(snas_values) > 1 else "nan",
                "baseline_name": "N/A",
                "baseline_mean": "N/A",
                "baseline_std": "N/A",
                "wilcoxon_stat": "N/A",
                "wilcoxon_p": "N/A",
                "t_stat": "N/A",
                "t_p": "N/A",
                "cohens_d": "N/A",
                "significant_at_05": "N/A",
                "note": "Baseline values not yet populated — fill BEST_BASELINE_AUROCS after Task 1",
            })
            continue

        baseline_name, baseline_raw = BEST_BASELINE_AUROCS[exp_name]
        base_values = np.array([v for v in baseline_raw if not np.isnan(v)])

        if len(base_values) < 2 or len(snas_values) != len(base_values):
            sig_rows.append({
                "experiment": exp_name,
                "note": "Mismatched baseline seed count — recheck BEST_BASELINE_AUROCS",
            })
            continue

        try:
            w_stat, w_p = wilcoxon(snas_values, base_values)
        except Exception:
            w_stat, w_p = float("nan"), float("nan")

        t_stat, t_p = ttest_rel(snas_values, base_values)
        pooled_std = np.sqrt(
            (snas_values.std(ddof=1) ** 2 + base_values.std(ddof=1) ** 2) / 2
        )
        cohens_d = float((snas_values.mean() - base_values.mean()) / pooled_std) \
            if pooled_std > 0 else float("nan")

        sig_rows.append({
            "experiment": exp_name,
            "snas_mean": round(float(snas_values.mean()), 6),
            "snas_std":  round(float(snas_values.std(ddof=1)), 6),
            "baseline_name": baseline_name,
            "baseline_mean": round(float(base_values.mean()), 6),
            "baseline_std":  round(float(base_values.std(ddof=1)), 6),
            "wilcoxon_stat": round(float(w_stat), 4),
            "wilcoxon_p":    round(float(w_p), 6),
            "t_stat":        round(float(t_stat), 4),
            "t_p":           round(float(t_p), 6),
            "cohens_d":      round(cohens_d, 4),
            "significant_at_05": bool(w_p < 0.05),
        })

    return sig_rows


# --- Main --------------------------------------------------------------------

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = []

    for exp_name, cfg in EXPERIMENTS.items():
        print(f"\n{'='*60}")
        print(f"Running {exp_name} across {len(SEEDS)} seeds: {SEEDS}")
        print(f"{'='*60}")
        for seed in SEEDS:
            print(f"\n  Seed {seed} ...")
            try:
                run_one(exp_name, cfg, seed)
            except subprocess.CalledProcessError as exc:
                print(f"  ERROR: seed {seed} failed — {exc}")
        summary.append(aggregate_seeds(exp_name))

    # Save aggregated summary
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2))
    print(f"\nSaved multiseed summary to {SUMMARY_PATH}")

    for s in summary:
        print(
            f"  {s['experiment']}: mean={s['mean_auroc']:.4f} "
            f"std={s['std_auroc']:.4f} (n={s['n_seeds']})"
        )

    # Statistical tests
    sig_rows = run_significance_tests(summary)
    if sig_rows:
        fieldnames = list(sig_rows[0].keys())
        with SIG_PATH.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(sig_rows)
        print(f"Saved significance tests to {SIG_PATH}")


if __name__ == "__main__":
    main()
