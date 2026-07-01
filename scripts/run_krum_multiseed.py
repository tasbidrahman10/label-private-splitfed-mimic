"""
Run Krum baseline at 4 remaining seeds (7, 123, 2024, 31415) for E1/E2/E4.
Seed 42 already completed in Task 1 baseline comparison run.
After this completes, run run_significance_tests.py for paired Wilcoxon test.

Usage:
    Terminal 1: python scripts/start_server.py --config configs/server_baseline_krum.yaml
    Terminal 2: python scripts/run_krum_multiseed.py
"""
from __future__ import annotations

import csv
import glob
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import requests

ROOT = Path(__file__).resolve().parents[1]
VENV_PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
PYTHON = str(VENV_PYTHON) if VENV_PYTHON.exists() else sys.executable
SERVER_URL = "http://127.0.0.1:8000"
AUTH_TOKEN = "dev-sfl-token"
MAX_BATCHES = 200
REMAINING_SEEDS = [7, 123, 2024, 31415]   # seed 42 already done in Task 1

EXPERIMENTS = {
    "E1_gradient_scaling":  "configs/experiment_baseline_krum_e1_gradient_scaling.yaml",
    "E2_free_rider":        "configs/experiment_baseline_krum_e2_free_rider.yaml",
    "E4_attack_straggler":  "configs/experiment_baseline_krum_e4_attack_plus_dropout.yaml",
}


def reset_server(seed: int) -> None:
    resp = requests.post(
        f"{SERVER_URL}/admin/reset",
        json={"config_overrides": {"seed": seed, "aggregator_type": "krum"}},
        headers={"Authorization": f"Bearer {AUTH_TOKEN}"},
        timeout=30,
    )
    resp.raise_for_status()
    print(f"  Server reset: seed={seed} aggregator=krum")


def run_one(exp_name: str, config_path: str, seed: int) -> None:
    reset_server(seed)
    time.sleep(1)
    print(f"  Running {exp_name} seed={seed}...")
    subprocess.run(
        [PYTHON,
         str(ROOT / "scripts" / "run_label_private_splitfed.py"),
         "--experiment-config", str(ROOT / config_path),
         "--max-batches", str(MAX_BATCHES),
         "--seed", str(seed)],
        cwd=str(ROOT),
        check=True,
    )


def get_mean_auroc(folder: str, seed: int) -> float:
    matches = sorted(glob.glob(f"{folder}/*seed{seed}*.csv"))
    if not matches:
        return float("nan")
    rows = list(csv.DictReader(open(matches[-1])))
    all_rounds = sorted(set(int(r["round"]) for r in rows))
    last3 = all_rounds[-3:] if len(all_rounds) >= 3 else all_rounds
    vals = [float(r["val_auroc"]) for r in rows if int(r["round"]) in last3 and r.get("val_auroc")]
    return float(np.mean(vals)) if vals else float("nan")


def main() -> None:
    print(f"Running Krum at seeds {REMAINING_SEEDS} for E1/E2/E4 ({len(REMAINING_SEEDS)*3} runs)")

    for exp_name, cfg in EXPERIMENTS.items():
        print(f"\n--- {exp_name} ---")
        for seed in REMAINING_SEEDS:
            run_one(exp_name, cfg, seed)

    # Aggregate all 5 seeds per experiment
    RESULT_DIRS = {
        "E1_gradient_scaling": "results/baseline_krum/e1_gradient_scaling",
        "E2_free_rider":       "results/baseline_krum/e2_free_rider",
        "E4_attack_straggler": "results/baseline_krum/e4_attack_plus_dropout",
    }

    all_seeds = [42] + REMAINING_SEEDS
    print("\n=== KRUM 5-SEED SUMMARY ===")
    krum_results = {}
    for exp_name, folder in RESULT_DIRS.items():
        aurocs = [get_mean_auroc(folder, s) for s in all_seeds]
        valid = [v for v in aurocs if not np.isnan(v)]
        mean = float(np.mean(valid))
        std  = float(np.std(valid, ddof=1))
        krum_results[exp_name] = {"aurocs": aurocs, "mean": mean, "std": std, "seeds": all_seeds}
        print(f"{exp_name}: {[round(v,4) for v in aurocs]}")
        print(f"  mean={mean:.4f} +/-{std:.4f}")

    # Save Krum multiseed results
    out = []
    for exp_name, data in krum_results.items():
        out.append({
            "experiment": f"Krum_{exp_name}",
            "aggregator": "krum",
            "seeds": data["seeds"],
            "mean_auroc": round(data["mean"], 6),
            "std_auroc":  round(data["std"], 6),
            "raw_values": [round(v, 6) for v in data["aurocs"]],
        })
    Path("results/krum_multiseed_summary.json").write_text(json.dumps(out, indent=2))
    print("\nSaved results/krum_multiseed_summary.json")
    print("Now run: python scripts/run_significance_tests.py")


if __name__ == "__main__":
    main()
