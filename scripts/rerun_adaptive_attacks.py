"""
Regenerate the adaptive-adversary experiments (tab:adaptive) on 265-feature data.

Why
---
The `los` removal (P0-8) regenerated the data on 2026-08-17 and the whole main
grid with it, but the adaptive attacks behind Table `tab:adaptive` were last run
on 30 June - 1 July and the staleness-decay variants on 23 June. They are
therefore still on the pre-`los` 266-feature feature set, while Section 3.1 of
the manuscript states plainly that the feature count is 265. Nothing in the
paper flags the discrepancy. Since the attack-then-abstain result is the paper's
headline limitation, regenerating is safer than footnoting.

The grid
--------
    threshold_aware          adaptive attack A, exponential decay   10 rounds
    alternating_<decay>      adaptive attack B x 4 decay functions  10 rounds
    slow_ramp                adaptive attack C, exponential decay   15 rounds

Six runs, single-seed (42), matching how the original table was produced. The
four `alternating_*` cells share one experiment config differing only in
results_dir; the decay function itself is set per run through /admin/reset.

Interrupt safety
----------------
Progress is checkpointed to results/adaptive_rerun_progress.json after every
single run, written atomically (temp file + replace) so an interrupt mid-write
cannot corrupt it. Rerunning skips completed cells. This is not theoretical:
two mains failures on 2026-08-20 interrupted the P1 grid, and each cost exactly
one run because of this pattern.

The server must be running, and must not be shared with another experiment
driver -- /admin/reset would clobber state mid-round.

    python scripts/start_server.py          # terminal 1
    python scripts/rerun_adaptive_attacks.py

Flags:
    --status     print progress and exit (safe while running)
    --dry-run    list pending runs and exit
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

VENV_PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
PYTHON = str(VENV_PYTHON) if VENV_PYTHON.exists() else sys.executable

SERVER_URL = "http://127.0.0.1:8000"
AUTH_TOKEN = "dev-sfl-token"
MAX_BATCHES = 200
SEED = 42

PROGRESS_PATH = ROOT / "results" / "adaptive_rerun_progress.json"
SUMMARY_CSV = ROOT / "results" / "adaptive_rerun_summary.csv"

# cell -> (experiment config, staleness decay function)
GRID: dict[str, tuple[str, str]] = {
    "threshold_aware":        ("configs/experiment_adaptive_attack_a.yaml", "exponential"),
    "alternating_exponential": ("configs/experiment_adaptive_attack_b_exponential.yaml", "exponential"),
    "alternating_linear":      ("configs/experiment_adaptive_attack_b_linear.yaml", "linear"),
    "alternating_polynomial":  ("configs/experiment_adaptive_attack_b_polynomial.yaml", "polynomial"),
    "alternating_step":        ("configs/experiment_adaptive_attack_b_step.yaml", "step"),
    "slow_ramp":              ("configs/experiment_adaptive_attack_c.yaml", "exponential"),
}

# Gate parameters stated explicitly in both directions: /admin/reset MERGES
# overrides into the live config, so anything left unset persists from whichever
# run set it last. The P1 grid left the gate ON, but the sensitivity sweep that
# may have run before this one moves the thresholds around.
GATE = {
    "snas_threshold_flag": 0.22,
    "snas_threshold_quarantine": 0.35,
    "fedavg_clip_ratio": 2.0,
    "snas_beta": 0.35,
    "snas_gamma": 0.65,
    "snas_alpha": 0.0,
    "snas_anomaly_basis": "absolute",
    "aggregator_type": "snas",
}

MAX_ATTEMPTS = 3
RETRY_BACKOFF_SECONDS = [10, 30]
SETTLE_SECONDS = 3


def overrides_for(cell: str) -> dict:
    _, decay = GRID[cell]
    return {"seed": SEED, "staleness_decay": decay, **GATE}


def _headers() -> dict:
    return {"Authorization": f"Bearer {AUTH_TOKEN}"}


def wait_for_server(timeout: float = 120.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if requests.get(f"{SERVER_URL}/health", timeout=5).ok:
                return
        except requests.RequestException:
            pass
        time.sleep(5)
    raise RuntimeError(
        f"No server at {SERVER_URL} after {timeout:.0f}s. "
        f"Start it with: python scripts/start_server.py"
    )


def reset_server(cell: str) -> None:
    resp = requests.post(
        f"{SERVER_URL}/admin/reset",
        json={"config_overrides": overrides_for(cell)},
        headers=_headers(),
        timeout=60,
    )
    resp.raise_for_status()


def load_progress() -> dict:
    if PROGRESS_PATH.exists():
        try:
            return json.loads(PROGRESS_PATH.read_text())
        except json.JSONDecodeError:
            print(f"WARNING: {PROGRESS_PATH.name} unreadable — starting fresh.")
    return {}


def save_progress(progress: dict) -> None:
    """Atomic: an interrupt mid-write leaves the previous file intact."""
    PROGRESS_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = PROGRESS_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(progress, indent=2))
    tmp.replace(PROGRESS_PATH)


def metrics_from_csv(results_dir: Path, since: float) -> dict | None:
    """Mean AUROC over the last 3 rounds of the run that just finished."""
    candidates = [p for p in results_dir.rglob(f"*seed{SEED}*.csv")
                  if p.stat().st_mtime >= since]
    if not candidates:
        return None
    newest = max(candidates, key=lambda p: p.stat().st_mtime)
    rows = list(csv.DictReader(newest.open(encoding="utf-8")))
    if not rows:
        return None
    rounds = sorted({int(r["round"]) for r in rows})
    last3 = rounds[-3:] if len(rounds) >= 3 else rounds
    auroc = [float(r["val_auroc"]) for r in rows
             if int(r["round"]) in last3 and r.get("val_auroc")]
    auprc = [float(r["val_auprc"]) for r in rows
             if int(r["round"]) in last3 and r.get("val_auprc")]
    return {
        "mean_auroc": round(float(np.mean(auroc)), 6) if auroc else None,
        "mean_auprc": round(float(np.mean(auprc)), 6) if auprc else None,
        "n_rounds": len(rounds),
        "csv": newest.name,
    }


def run_one(cell: str) -> dict | None:
    config_rel, _decay = GRID[cell]
    import yaml
    results_dir = ROOT / yaml.safe_load(
        (ROOT / config_rel).read_text(encoding="utf-8"))["results_dir"]

    for attempt in range(MAX_ATTEMPTS):
        try:
            wait_for_server()
            reset_server(cell)
            time.sleep(SETTLE_SECONDS)
            started = time.time()
            subprocess.run(
                [
                    PYTHON,
                    str(ROOT / "scripts" / "run_label_private_splitfed.py"),
                    "--experiment-config", str(ROOT / config_rel),
                    "--max-batches", str(MAX_BATCHES),
                    "--seed", str(SEED),
                ],
                cwd=str(ROOT),
                check=True,
                capture_output=True,
            )
            metrics = metrics_from_csv(results_dir, started)
            if metrics and metrics["mean_auroc"] is not None:
                return metrics
            print(f"      no AUROC parsed (attempt {attempt + 1}/{MAX_ATTEMPTS})")
        except (subprocess.CalledProcessError, requests.RequestException, RuntimeError) as exc:
            detail = ""
            if isinstance(exc, subprocess.CalledProcessError) and exc.stderr:
                tail = exc.stderr.decode(errors="replace").strip().splitlines()[-1:]
                detail = f" — {tail[0]}" if tail else ""
            print(f"      attempt {attempt + 1}/{MAX_ATTEMPTS} failed: "
                  f"{type(exc).__name__}{detail}")
        if attempt < len(RETRY_BACKOFF_SECONDS):
            time.sleep(RETRY_BACKOFF_SECONDS[attempt])
    return None


def write_summary(progress: dict) -> None:
    rows = []
    for cell, (config_rel, decay) in GRID.items():
        entry = progress.get(cell)
        rows.append({
            "cell": cell,
            "staleness_decay": decay,
            "config": config_rel,
            "mean_auroc": (entry or {}).get("mean_auroc"),
            "mean_auprc": (entry or {}).get("mean_auprc"),
            "n_rounds": (entry or {}).get("n_rounds"),
            "source_csv": (entry or {}).get("csv"),
        })
    with SUMMARY_CSV.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def print_status() -> None:
    progress = load_progress()
    print(f"{'cell':26s} {'decay':12s} {'AUROC':>9s} {'AUPRC':>9s} {'rounds':>7s}")
    print("-" * 68)
    done = 0
    for cell, (_cfg, decay) in GRID.items():
        entry = progress.get(cell)
        if entry:
            done += 1
            print(f"{cell:26s} {decay:12s} {entry['mean_auroc']:9.4f} "
                  f"{entry['mean_auprc']:9.4f} {entry['n_rounds']:>7d}")
        else:
            print(f"{cell:26s} {decay:12s} {'--':>9s} {'--':>9s} {'--':>7s}")
    print("-" * 68)
    print(f"{done}/{len(GRID)} runs complete"
          + ("  — FINISHED" if done == len(GRID) else ""))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.status:
        print_status()
        return

    progress = load_progress()
    todo = [c for c in GRID if c not in progress]
    print(f"Grid: {len(GRID)} runs | done: {len(GRID) - len(todo)} | pending: {len(todo)}")
    if args.dry_run:
        for cell in todo:
            print(f"  {cell}  (decay={GRID[cell][1]})")
        return
    if not todo:
        print("Nothing to do — all runs complete.")
        write_summary(progress)
        return

    for i, cell in enumerate(todo, 1):
        print(f"[{i}/{len(todo)}] {cell}  decay={GRID[cell][1]}  ({datetime.now():%H:%M:%S})")
        started = time.time()
        metrics = run_one(cell)
        elapsed = time.time() - started
        if metrics is None:
            print(f"      -> FAILED  ({elapsed / 60:.1f} min)")
            continue
        progress[cell] = metrics
        save_progress(progress)
        write_summary(progress)
        print(f"      -> AUROC {metrics['mean_auroc']:.4f}  "
              f"AUPRC {metrics['mean_auprc']:.4f}  ({elapsed / 60:.1f} min)")

    print(f"\nDone. Summary: {SUMMARY_CSV}")
    print_status()


if __name__ == "__main__":
    main()
