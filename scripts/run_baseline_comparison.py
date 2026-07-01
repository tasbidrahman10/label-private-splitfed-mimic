"""
Run all 9 baseline comparison experiments automatically (Task 1).

Runs E1, E2, E4 against Krum, Trimmed Mean, and FLTrust — 9 total runs.
Uses POST /admin/reset between experiments so only ONE server start is needed.

Usage:
    Terminal 1: python scripts/start_server.py
    Terminal 2: python scripts/run_baseline_comparison.py

Results saved to:
    results/baseline_krum/e1_gradient_scaling/
    results/baseline_krum/e2_free_rider/
    results/baseline_krum/e4_attack_plus_dropout/
    results/baseline_trimmed_mean/...
    results/baseline_fltrust/...
    results/baseline_comparison.csv   <-- consolidated table
"""
from __future__ import annotations

import csv
import json
import subprocess
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

VENV_PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
PYTHON = str(VENV_PYTHON) if VENV_PYTHON.exists() else sys.executable

SERVER_URL = "http://127.0.0.1:8000"
AUTH_TOKEN = "dev-sfl-token"
MAX_BATCHES = 200

AGGREGATORS = ["krum", "trimmed_mean", "fltrust"]

EXPERIMENTS = {
    "e1_gradient_scaling":    "configs/experiment_baseline_{agg}_e1_gradient_scaling.yaml",
    "e2_free_rider":          "configs/experiment_baseline_{agg}_e2_free_rider.yaml",
    "e4_attack_plus_dropout": "configs/experiment_baseline_{agg}_e4_attack_plus_dropout.yaml",
}

# SNAS reference results (from existing single-seed runs)
SNAS_REFERENCE = {
    "e1_gradient_scaling":    0.9597,
    "e2_free_rider":          0.9107,
    "e4_attack_plus_dropout": 0.9574,
}

OUT_CSV = ROOT / "results" / "baseline_comparison.csv"


def _headers() -> dict:
    return {"Authorization": f"Bearer {AUTH_TOKEN}"}


def reset_server(aggregator: str, results_dir: str) -> None:
    """Reset server state with new aggregator and results directory."""
    resp = requests.post(
        f"{SERVER_URL}/admin/reset",
        json={"config_overrides": {
            "aggregator_type": aggregator,
            "seed": 42,
            "results_dir": results_dir,
        }},
        headers=_headers(),
        timeout=30,
    )
    resp.raise_for_status()
    print(f"  Server reset -> aggregator={aggregator}")


def run_one(agg: str, exp_name: str) -> tuple[float, str]:
    """Run one experiment. Returns (mean_auroc_r8_10, csv_path)."""
    config_path = ROOT / EXPERIMENTS[exp_name].format(agg=agg)
    results_dir = f"results/baseline_{agg}/{exp_name}"

    reset_server(agg, results_dir)
    time.sleep(1)

    print(f"  Running {exp_name} with {agg}...")
    proc = subprocess.run(
        [
            PYTHON,
            str(ROOT / "scripts" / "run_label_private_splitfed.py"),
            "--experiment-config", str(config_path),
            "--max-batches", str(MAX_BATCHES),
            "--seed", "42",
        ],
        cwd=str(ROOT),
        capture_output=False,
    )
    if proc.returncode != 0:
        print(f"  WARNING: run exited with code {proc.returncode}")

    # Find the output CSV
    out_dir = ROOT / results_dir
    csvs = sorted(out_dir.rglob("experiment_summary_*.csv"),
                  key=lambda p: p.stat().st_mtime, reverse=True)
    if not csvs:
        return float("nan"), ""

    csv_path = csvs[0]
    rows = list(csv.DictReader(csv_path.open()))
    all_rounds = sorted({int(r["round"]) for r in rows})
    last3 = all_rounds[-3:] if len(all_rounds) >= 3 else all_rounds
    vals = [float(r["val_auroc"]) for r in rows
            if int(r["round"]) in last3 and r.get("val_auroc")]
    mean_auroc = sum(vals) / len(vals) if vals else float("nan")
    return mean_auroc, str(csv_path)


def main() -> None:
    print("=" * 60)
    print("Task 1 — Baseline Comparison (9 runs)")
    print("=" * 60)

    rows = []
    for agg in AGGREGATORS:
        print(f"\n--- Aggregator: {agg.upper()} ---")
        for exp_name in EXPERIMENTS:
            auroc, csv_path = run_one(agg, exp_name)
            snas_ref = SNAS_REFERENCE[exp_name]
            delta = auroc - snas_ref if auroc == auroc else float("nan")
            print(f"  {exp_name}: AUROC={auroc:.4f}  delta_vs_SNAS={delta:+.4f}")
            rows.append({
                "aggregator":            agg,
                "experiment":            exp_name,
                "mean_auroc_r8_10":      round(auroc, 6),
                "snas_auroc_reference":  snas_ref,
                "delta_vs_snas":         round(delta, 6),
                "result_csv":            csv_path,
            })

    # Write consolidated CSV
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with OUT_CSV.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved consolidated results to {OUT_CSV}")

    # Print summary table
    print("\n" + "=" * 60)
    print("BASELINE COMPARISON SUMMARY")
    print("=" * 60)
    print(f"{'Aggregator':<16} {'E1 AUROC':<12} {'E2 AUROC':<12} {'E4 AUROC'}")
    print(f"{'SNAS (reference)':<16} {SNAS_REFERENCE['e1_gradient_scaling']:.4f}       "
          f"{SNAS_REFERENCE['e2_free_rider']:.4f}       "
          f"{SNAS_REFERENCE['e4_attack_plus_dropout']:.4f}")
    for agg in AGGREGATORS:
        agg_rows = {r["experiment"]: r for r in rows if r["aggregator"] == agg}
        e1 = agg_rows.get("e1_gradient_scaling", {}).get("mean_auroc_r8_10", "N/A")
        e2 = agg_rows.get("e2_free_rider", {}).get("mean_auroc_r8_10", "N/A")
        e4 = agg_rows.get("e4_attack_plus_dropout", {}).get("mean_auroc_r8_10", "N/A")
        e1_s = f"{e1:.4f}" if isinstance(e1, float) else str(e1)
        e2_s = f"{e2:.4f}" if isinstance(e2, float) else str(e2)
        e4_s = f"{e4:.4f}" if isinstance(e4, float) else str(e4)
        print(f"{agg:<16} {e1_s:<12} {e2_s:<12} {e4_s}")


if __name__ == "__main__":
    main()
