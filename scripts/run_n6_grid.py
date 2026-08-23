"""
n=6 supplementary grid (P2 — N >= 5 clients).

Answers the review's biggest ask. At n=3 Krum sits outside its own N >= 2f+3
condition: k clamps to 1, honest scores tie bit-for-bit, and `min()` breaks the
tie by insertion order — Client 0 won 300/300 rounds on the corrected data. At
n=6, k = n - f - 2 = 3 with no clamp, and trimmed mean trims 1 per tail instead
of 0 (it was plain FedAvg at n=3). Both baselines are validly configured here
for the first time.

Grid: E1 gradient scaling x {snas, krum, trimmed_mean, median, fltrust} x 5 seeds
      = 25 runs at ~13 min each (~5.5 h). Clients train sequentially, so an n=6
      run costs roughly 2x an n=3 run.

Treated as a SUPPLEMENTARY table per N6_MIGRATION.md Phase 8: n=3 stays the main
result. Converting the whole paper to n=6 would invalidate every number in it.

The server must already be running on the n=6 profile:
    $env:SFL_CLIENT_CONFIG_DIR = "configs/clients_n6"
    python scripts/start_server.py --config configs/server_n6.yaml
    python scripts/run_n6_grid.py

Interrupt-safe: checkpointed to results/n6_grid_progress.json after every run,
written temp-then-rename, so an interrupt mid-write cannot corrupt it. Five mains
failures during the P1 grid cost one run each; this driver inherits that design.
Failed runs are recorded as null and retried on the next invocation, not skipped.

Flags:
    --only <aggregator>   restrict to one aggregator
    --seeds <n...>        restrict to specific seeds
    --dry-run             list pending work, then exit
    --status              read-only progress report
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from datetime import timedelta
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
SEEDS = [42, 7, 123, 2024, 31415]
CLIENT_CONFIG_DIR = "configs/clients_n6"
EXPECTED_CLIENTS = 6

PROGRESS_PATH = ROOT / "results" / "n6_grid_progress.json"
SUMMARY_PATH = ROOT / "results" / "n6_grid_summary.json"

EXPERIMENT = "E1_gradient_scaling"

# aggregator -> experiment config. Each config carries its own results_dir;
# verified distinct, because four adaptive-attack configs previously shared one
# and silently overwrote each other.
GRID: dict[str, str] = {
    "snas":         "configs/experiment_e1_gradient_scaling_n6.yaml",
    "krum":         "configs/experiment_baseline_krum_e1_gradient_scaling_n6.yaml",
    "trimmed_mean": "configs/experiment_baseline_trimmed_mean_e1_gradient_scaling_n6.yaml",
    "median":       "configs/experiment_baseline_median_e1_gradient_scaling_n6.yaml",
    "fltrust":      "configs/experiment_baseline_fltrust_e1_gradient_scaling_n6.yaml",
}

# /admin/reset MERGES overrides into the live config: a value left unset persists
# from whichever run set it last. Every gate and clip parameter is therefore
# stated explicitly so one arm's settings cannot silently carry into the next.
GATE = {"snas_threshold_flag": 0.22, "snas_threshold_quarantine": 0.35}
CLIP = {"fedavg_clip_ratio": 2.0}

MAX_ATTEMPTS = 3
RETRY_BACKOFF_SECONDS = [10, 30]
SETTLE_SECONDS = 3


def overrides_for(aggregator: str, seed: int) -> dict:
    return {
        "seed": seed,
        "aggregator_type": aggregator,
        "snas_anomaly_basis": "absolute",
        **GATE,
        **CLIP,
    }


# --- server ------------------------------------------------------------------

def _headers() -> dict:
    return {"Authorization": f"Bearer {AUTH_TOKEN}"}


def wait_for_server(timeout: float = 120.0) -> None:
    """Block until the server answers /health, so a restart mid-grid is survivable."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if requests.get(f"{SERVER_URL}/health", timeout=5).ok:
                return
        except requests.RequestException:
            pass
        time.sleep(5)
    raise RuntimeError(
        f"No server at {SERVER_URL} after {timeout:.0f}s. Start it with "
        f"SFL_CLIENT_CONFIG_DIR={CLIENT_CONFIG_DIR} and configs/server_n6.yaml."
    )


def assert_n6_server() -> None:
    """Refuse to run against an n=3 server — six clients is the entire point."""
    resp = requests.get(f"{SERVER_URL}/health", timeout=10)
    resp.raise_for_status()
    n = resp.json().get("registered_clients")
    if n not in (0, EXPECTED_CLIENTS):
        raise RuntimeError(
            f"Server reports {n} registered clients, expected {EXPECTED_CLIENTS}. "
            f"It is probably running the n=3 profile. Restart it with "
            f"SFL_CLIENT_CONFIG_DIR={CLIENT_CONFIG_DIR} and configs/server_n6.yaml."
        )


def reset_server(aggregator: str, seed: int) -> None:
    resp = requests.post(
        f"{SERVER_URL}/admin/reset",
        json={"config_overrides": overrides_for(aggregator, seed)},
        headers=_headers(),
        timeout=60,
    )
    resp.raise_for_status()


# --- progress ----------------------------------------------------------------

def load_progress() -> dict:
    if PROGRESS_PATH.exists():
        try:
            return json.loads(PROGRESS_PATH.read_text())
        except json.JSONDecodeError:
            print(f"WARNING: {PROGRESS_PATH.name} unreadable — starting fresh.")
    return {}


def save_progress(progress: dict) -> None:
    """Atomic write: an interrupt mid-write leaves the previous file intact."""
    PROGRESS_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = PROGRESS_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(progress, indent=2))
    tmp.replace(PROGRESS_PATH)


# --- one run -----------------------------------------------------------------

def mean_auroc_from_csv(seed: int, since: float) -> float:
    """Mean AUROC over the last 3 rounds of the run that just finished."""
    candidates = [
        p for p in (ROOT / "results").rglob(f"*seed{seed}*.csv")
        if p.stat().st_mtime >= since
    ]
    if not candidates:
        return float("nan")
    newest = max(candidates, key=lambda p: p.stat().st_mtime)
    rows = list(csv.DictReader(newest.open()))
    if not rows:
        return float("nan")
    all_rounds = sorted({int(r["round"]) for r in rows})
    last3 = all_rounds[-3:] if len(all_rounds) >= 3 else all_rounds
    values = [
        float(r["val_auroc"]) for r in rows
        if int(r["round"]) in last3 and r.get("val_auroc")
    ]
    return float(np.mean(values)) if values else float("nan")


def run_one(config_path: str, aggregator: str, seed: int) -> float:
    """Run one (aggregator, seed) with retries. Returns mean AUROC, or NaN."""
    env = dict(os.environ, SFL_CLIENT_CONFIG_DIR=CLIENT_CONFIG_DIR)
    for attempt in range(MAX_ATTEMPTS):
        try:
            wait_for_server()
            reset_server(aggregator, seed)
            time.sleep(SETTLE_SECONDS)
            started = time.time()
            subprocess.run(
                [
                    PYTHON,
                    str(ROOT / "scripts" / "run_label_private_splitfed.py"),
                    "--experiment-config", str(ROOT / config_path),
                    "--max-batches", str(MAX_BATCHES),
                    "--seed", str(seed),
                ],
                cwd=str(ROOT),
                env=env,
                check=True,
                capture_output=True,
            )
            auroc = mean_auroc_from_csv(seed, started)
            if not np.isnan(auroc):
                return auroc
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
    return float("nan")


# --- reporting ---------------------------------------------------------------

def write_summary(progress: dict) -> None:
    summary: dict = {}
    for agg in GRID:
        cell = progress.get(f"{agg}:{EXPERIMENT}", {})
        ok = {s: v for s, v in cell.items() if v is not None}
        if not ok:
            continue
        vals = list(ok.values())
        summary[f"{agg}:{EXPERIMENT}"] = {
            "n_seeds": len(vals),
            "mean_auroc": round(float(np.mean(vals)), 6),
            "std_auroc": round(float(np.std(vals)), 6),
            "per_seed": ok,
        }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2))


def print_status(progress: dict) -> bool:
    print(f"{'aggregator':<16}{'seeds':>9}{'mean':>10}{'std':>10}")
    print("-" * 45)
    total_done = 0
    for agg in GRID:
        cell = progress.get(f"{agg}:{EXPERIMENT}", {})
        ok = [v for v in cell.values() if v is not None]
        bad = [s for s, v in cell.items() if v is None]
        total_done += len(ok)
        mean = f"{np.mean(ok):.4f}" if ok else "--"
        std = f"{np.std(ok):.4f}" if ok else "--"
        flag = f"  ({len(bad)} failed)" if bad else ""
        print(f"{agg:<16}{len(ok):>5}/{len(SEEDS):<3}{mean:>10}{std:>10}{flag}")
    total = len(GRID) * len(SEEDS)
    print("-" * 45)
    print(f"{total_done}/{total} runs complete ({100.0 * total_done / total:.1f}%)")
    if total_done < total:
        left = total - total_done
        print(f"Not finished. ~{left} runs left "
              f"(~{timedelta(seconds=int(left * 13 * 60))} at 13 min/run).")
        return False
    print("\nFINISHED — every cell complete.")
    return True


# --- main --------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="n=6 supplementary grid (P2).")
    ap.add_argument("--only", choices=sorted(GRID), help="Run only this aggregator.")
    ap.add_argument("--seeds", type=int, nargs="+", help="Restrict to these seeds.")
    ap.add_argument("--dry-run", action="store_true", help="List pending work, then exit.")
    ap.add_argument("--status", action="store_true", help="Read-only progress report.")
    args = ap.parse_args()

    progress = load_progress()

    if args.status:
        print_status(progress)
        return

    grid = {args.only: GRID[args.only]} if args.only else GRID
    seeds = args.seeds or SEEDS

    todo = []
    for agg, cfg in grid.items():
        done = progress.get(f"{agg}:{EXPERIMENT}", {})
        for seed in seeds:
            # Failed runs stored as null are retried, not skipped.
            if done.get(str(seed)) is None:
                todo.append((agg, cfg, seed))

    if not todo:
        print("Nothing to do — every requested cell is already complete.")
        print_status(progress)
        return

    print(f"{len(todo)} run(s) pending "
          f"(~{timedelta(seconds=int(len(todo) * 13 * 60))} at 13 min/run).")
    if args.dry_run:
        for agg, _, seed in todo:
            print(f"  {agg}:{EXPERIMENT} seed={seed}")
        return

    assert_n6_server()

    durations: list[float] = []
    for i, (agg, cfg, seed) in enumerate(todo, 1):
        key = f"{agg}:{EXPERIMENT}"
        eta = ""
        if durations:
            rem = (len(todo) - i + 1) * float(np.mean(durations))
            eta = f", ETA {timedelta(seconds=int(rem))}"
        print(f"[{i}/{len(todo)}] {key} seed={seed} "
              f"({time.strftime('%H:%M:%S')}{eta})", flush=True)
        started = time.time()
        auroc = run_one(cfg, agg, seed)
        durations.append(time.time() - started)

        progress.setdefault(key, {})[str(seed)] = None if np.isnan(auroc) else auroc
        save_progress(progress)
        write_summary(progress)
        shown = "FAILED (will retry next run)" if np.isnan(auroc) else f"{auroc:.4f}"
        print(f"      -> {shown}  [{timedelta(seconds=int(durations[-1]))}]", flush=True)

    print()
    print_status(progress)


if __name__ == "__main__":
    main()
