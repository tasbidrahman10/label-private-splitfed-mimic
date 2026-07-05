"""
Gap-closing experiment batch (reviewer-anticipation fixes).

Two arms, both at the paper's 10 seeds, so every row lands with mean +/- std:

  ARM 1 "baseline_clean" (Gap #1 -- baseline variance):
      experiment_async_clean.yaml under the default SNAS server (gate on, clip on).
      Gives the async clean baseline AUROC +/- std, so the deltas reported elsewhere
      are no longer anchored to a single-seed 0.9649.

  ARM 2 "nodetect" (Gap #4 -- empirical async baseline AND gate ablation):
      E1/E2/E4/E6 under SNAS with the gate and norm-clipping DISABLED via server
      overrides (snas thresholds and fedavg_clip_ratio set to 1e9). This reduces the
      pipeline to a pure staleness-weighted asynchronous FedAvg with no anomaly
      detection -- the "does the gate actually help?" comparison.

Design notes:
  - Gate/clip are disabled purely through /admin/reset config_overrides
    (state.reset merges them into config and rebuilds the detector + aggregator).
  - Every run's [t_start, t_end] window is recorded so federation-entropy can later
    be sliced cleanly from the (freshly rotated) server log by timestamp.
  - Resumable: per-run checkpoint to results/gap_closing/progress.json; already-done
    (arm, exp, seed) triples are skipped. Retry with backoff on failure.

Usage (server must be running on a FRESH log -- see close_gaps_launch notes):
    python scripts/close_gaps_runner.py

Outputs:
    results/gap_closing/progress.json        (checkpoint: auroc + time window per run)
    results/gap_closing/baseline_clean.json  (Gap #1 aggregate)
    results/gap_closing/nodetect.json        (Gap #4 aggregate)
    results/gap_closing/run_windows.json     (for entropy slicing)
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

VENV_PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
PYTHON = str(VENV_PYTHON) if VENV_PYTHON.exists() else sys.executable

SERVER_URL  = "http://127.0.0.1:8000"
AUTH_TOKEN  = "dev-sfl-token"
MAX_BATCHES = 200
SEEDS = [42, 7, 123, 2024, 31415, 1001, 2002, 3003, 4004, 5005]

# Run order: baseline first, then the most dramatic no-detect arm (E2), then the
# rest; E6 (no attack) is least critical so it runs last (safe trim point).
BASELINE = {"async_clean": "configs/experiment_async_clean.yaml"}
NODETECT = {
    "E2_free_rider":       "configs/experiment_nogate_e2_free_rider.yaml",
    "E1_gradient_scaling": "configs/experiment_nogate_e1_gradient_scaling.yaml",
    "E4_attack_straggler": "configs/experiment_nogate_e4_attack_plus_dropout.yaml",
    "E6_honest_straggler": "configs/experiment_nogate_e6_dropout.yaml",
}

# Explicit override sets (both directions, to avoid state leakage between arms).
GATE_ON = {
    "aggregator_type": "snas",
    "snas_threshold_flag": 0.22,
    "snas_threshold_quarantine": 0.35,
    "fedavg_clip_ratio": 2.0,
}
GATE_OFF = {
    "aggregator_type": "snas",
    "snas_threshold_flag": 1e9,
    "snas_threshold_quarantine": 1e9,
    "fedavg_clip_ratio": 1e9,
}

MAX_ATTEMPTS = 4
RETRY_BACKOFF = [5, 10, 20]
SETTLE_SECONDS = 5
EXPECTED_ROUNDS = 10

OUT_DIR = ROOT / "results" / "gap_closing"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PROGRESS_PATH = OUT_DIR / "progress.json"


def _headers() -> dict:
    return {"Authorization": f"Bearer {AUTH_TOKEN}"}


def reset_server(seed: int, overrides: dict) -> None:
    resp = requests.post(
        f"{SERVER_URL}/admin/reset",
        json={"config_overrides": {"seed": seed, **overrides}},
        headers=_headers(),
        timeout=30,
    )
    resp.raise_for_status()


def _run_once(cfg: str, seed: int) -> tuple[float | None, float, float, str]:
    t_start = time.time()
    result = subprocess.run(
        [
            PYTHON,
            str(ROOT / "scripts" / "run_label_private_splitfed.py"),
            "--experiment-config", str(ROOT / cfg),
            "--max-batches", str(MAX_BATCHES),
            "--seed", str(seed),
        ],
        cwd=str(ROOT), capture_output=True, text=True,
    )
    t_end = time.time()
    if result.returncode != 0:
        tail = "\n".join((result.stdout + result.stderr).splitlines()[-12:])
        return None, t_start, t_end, f"exit {result.returncode}: {tail}"

    # newest seed CSV anywhere under results/ (runs are sequential)
    candidates = sorted(
        (ROOT / "results").rglob(f"*seed{seed}*.csv"),
        key=lambda p: p.stat().st_mtime, reverse=True,
    )
    if not candidates:
        return None, t_start, t_end, "no output CSV found"
    rows = list(csv.DictReader(candidates[0].open()))
    all_rounds = sorted({int(r["round"]) for r in rows})
    if len(all_rounds) < EXPECTED_ROUNDS:
        return None, t_start, t_end, f"only {len(all_rounds)}/{EXPECTED_ROUNDS} rounds"
    last3 = all_rounds[-3:]
    vals = [float(r["val_auroc"]) for r in rows if int(r["round"]) in last3 and r.get("val_auroc")]
    if not vals:
        return None, t_start, t_end, "no val_auroc values"
    return float(np.mean(vals)), t_start, t_end, ""


def run_group(arm: str, experiments: dict, overrides: dict, progress: dict) -> None:
    for exp, cfg in experiments.items():
        key = f"{arm}:{exp}"
        progress.setdefault(key, {})
        for seed in SEEDS:
            if str(seed) in progress[key]:
                print(f"  [skip] {key} seed={seed} -> {progress[key][str(seed)]['auroc']:.4f}")
                continue
            print(f"\n  [{arm}] {exp} seed={seed} ...")
            auroc = None
            for attempt in range(1, MAX_ATTEMPTS + 1):
                reset_server(seed, overrides)
                time.sleep(SETTLE_SECONDS)
                auroc, t0, t1, err = _run_once(cfg, seed)
                if auroc is not None:
                    break
                if attempt < MAX_ATTEMPTS:
                    wait = RETRY_BACKOFF[attempt - 1]
                    print(f"    ERROR (attempt {attempt}/{MAX_ATTEMPTS}): {err}. Retry in {wait}s")
                    time.sleep(wait)
                else:
                    print(f"    GAVE UP {key} seed={seed}: {err}")
            if auroc is None:
                continue
            progress[key][str(seed)] = {"auroc": auroc, "t_start": t0, "t_end": t1}
            PROGRESS_PATH.write_text(json.dumps(progress, indent=2))
            print(f"    -> AUROC(last3) = {auroc:.4f}")


def summarize(progress: dict, arm: str, experiments: dict) -> list[dict]:
    out = []
    for exp in experiments:
        done = progress.get(f"{arm}:{exp}", {})
        vals = [done[str(s)]["auroc"] for s in SEEDS if str(s) in done]
        if not vals:
            continue
        out.append({
            "arm": arm, "experiment": exp, "n_seeds": len(vals),
            "mean_auroc": float(np.mean(vals)),
            "std_auroc": float(np.std(vals, ddof=1)) if len(vals) > 1 else float("nan"),
            "min_auroc": float(np.min(vals)), "max_auroc": float(np.max(vals)),
            "raw": vals,
        })
    return out


def main() -> None:
    progress = json.loads(PROGRESS_PATH.read_text()) if PROGRESS_PATH.exists() else {}

    print("=" * 60)
    print("ARM 1/2: baseline_clean (Gap #1) -- async_clean x10, gate ON")
    print("=" * 60)
    run_group("baseline_clean", BASELINE, GATE_ON, progress)

    print("\n" + "=" * 60)
    print("ARM 2/2: nodetect (Gap #4) -- E1/E2/E4/E6 x10, gate+clip OFF")
    print("=" * 60)
    run_group("nodetect", NODETECT, GATE_OFF, progress)

    # aggregates
    baseline = summarize(progress, "baseline_clean", BASELINE)
    nodetect = summarize(progress, "nodetect", NODETECT)
    (OUT_DIR / "baseline_clean.json").write_text(json.dumps(baseline, indent=2))
    (OUT_DIR / "nodetect.json").write_text(json.dumps(nodetect, indent=2))

    # run windows for entropy slicing
    windows = []
    for key, runs in progress.items():
        arm, exp = key.split(":", 1)
        for seed, rec in runs.items():
            windows.append({"arm": arm, "experiment": exp, "seed": int(seed),
                            "t_start": rec["t_start"], "t_end": rec["t_end"]})
    (OUT_DIR / "run_windows.json").write_text(json.dumps(windows, indent=2))

    print("\n" + "=" * 60 + "\nDONE\n" + "=" * 60)
    for row in baseline + nodetect:
        s = row["std_auroc"]
        sstr = f"{s:.4f}" if s == s else "n/a"
        print(f"  {row['arm']:15s} {row['experiment']:22s} "
              f"{row['mean_auroc']:.4f} +/- {sstr} (n={row['n_seeds']})")


if __name__ == "__main__":
    main()
