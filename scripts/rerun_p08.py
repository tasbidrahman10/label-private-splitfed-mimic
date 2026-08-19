"""
Full n=3 regeneration after the P0-8 `los` removal.

Dropping `los` changes the feature set (266 -> 265), so every number in the
paper's main tables was produced on data that no longer exists. This driver
reruns the complete grid on the corrected data:

    SNAS      x {clean, E1, E2, E4, E6}  x 10 seeds   = 50 runs
    Krum      x {E1, E2, E4}             x 10 seeds   = 30 runs
    Trimmed   x {E1, E2, E4}             x 10 seeds   = 30 runs
    FLTrust   x {E1, E2, E4}             x 10 seeds   = 30 runs
                                          total       = 140 runs (~18 h)

FLTrust is included at the full ten seeds deliberately: it previously
degenerated to a plain mean because of the state-lifecycle and
gradient-layout bugs, so its published row was never a real FLTrust result
and one seed would not be enough to replace it.

Interrupt-safe. Progress is checkpointed to results/p08_rerun_progress.json
after every single run; rerunning skips completed (aggregator, experiment,
seed) triples. The write is atomic (temp file + replace), so an interrupt
mid-write cannot corrupt the progress file.

The server must be running:
    Terminal 1: python scripts/start_server.py
    Terminal 2: python scripts/rerun_p08.py

Useful flags:
    --only snas          run just one aggregator group
    --seeds 42 7         restrict to specific seeds
    --dry-run            list what would run, then exit
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
SEEDS = [42, 7, 123, 2024, 31415, 1001, 2002, 3003, 4004, 5005]

PROGRESS_PATH = ROOT / "results" / "p08_rerun_progress.json"
SUMMARY_PATH = ROOT / "results" / "p08_rerun_summary.json"

# aggregator -> {experiment label: experiment config}
GRID: dict[str, dict[str, str]] = {
    "snas": {
        "clean":               "configs/experiment_async_clean.yaml",
        "E1_gradient_scaling": "configs/experiment_e1_gradient_scaling.yaml",
        "E2_free_rider":       "configs/experiment_e2_free_rider.yaml",
        "E4_attack_straggler": "configs/experiment_e4_attack_plus_dropout.yaml",
        "E6_honest_straggler": "configs/experiment_e6_dropout.yaml",
    },
    "krum": {
        "E1_gradient_scaling": "configs/experiment_baseline_krum_e1_gradient_scaling.yaml",
        "E2_free_rider":       "configs/experiment_baseline_krum_e2_free_rider.yaml",
        "E4_attack_straggler": "configs/experiment_baseline_krum_e4_attack_plus_dropout.yaml",
    },
    "trimmed_mean": {
        "E1_gradient_scaling": "configs/experiment_baseline_trimmed_mean_e1_gradient_scaling.yaml",
        "E2_free_rider":       "configs/experiment_baseline_trimmed_mean_e2_free_rider.yaml",
        "E4_attack_straggler": "configs/experiment_baseline_trimmed_mean_e4_attack_plus_dropout.yaml",
    },
    "fltrust": {
        "E1_gradient_scaling": "configs/experiment_baseline_fltrust_e1_gradient_scaling.yaml",
        "E2_free_rider":       "configs/experiment_baseline_fltrust_e2_free_rider.yaml",
        "E4_attack_straggler": "configs/experiment_baseline_fltrust_e4_attack_plus_dropout.yaml",
    },
    # Gate ablation: SNAS with the anomaly gate and norm clipping disabled,
    # reducing the pipeline to pure staleness-weighted async FedAvg. Paired
    # against the "snas" rows above to quantify what the gate contributes.
    "nodetect": {
        "E1_gradient_scaling": "configs/experiment_nogate_e1_gradient_scaling.yaml",
        "E2_free_rider":       "configs/experiment_nogate_e2_free_rider.yaml",
        "E4_attack_straggler": "configs/experiment_nogate_e4_attack_plus_dropout.yaml",
        "E6_honest_straggler": "configs/experiment_nogate_e6_dropout.yaml",
    },
}

# /admin/reset MERGES overrides into the live config rather than replacing it, so
# a value set by one group persists until another group sets it back. Every group
# therefore states the gate parameters explicitly in both directions — otherwise
# the nodetect arm's 1e9 thresholds would leak into every run after it and
# silently disable the gate for the aggregators that are supposed to have it on.
GATE_ON = {
    "snas_threshold_flag": 0.22,
    "snas_threshold_quarantine": 0.35,
    "fedavg_clip_ratio": 2.0,
}
GATE_OFF = {
    "snas_threshold_flag": 1e9,
    "snas_threshold_quarantine": 1e9,
    "fedavg_clip_ratio": 1e9,
}


def overrides_for(group: str, seed: int) -> dict:
    """config_overrides posted to /admin/reset for one run."""
    aggregator = "snas" if group == "nodetect" else group
    gate = GATE_OFF if group == "nodetect" else GATE_ON
    return {"seed": seed, "aggregator_type": aggregator, **gate}

MAX_ATTEMPTS = 3
RETRY_BACKOFF_SECONDS = [10, 30]
SETTLE_SECONDS = 3


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
        f"No server at {SERVER_URL} after {timeout:.0f}s. "
        f"Start it with: python scripts/start_server.py"
    )


def reset_server(seed: int, group: str) -> None:
    resp = requests.post(
        f"{SERVER_URL}/admin/reset",
        json={"config_overrides": overrides_for(group, seed)},
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


def run_one(config_path: str, seed: int, aggregator: str) -> float:
    """Run one (config, seed) with retries. Returns mean AUROC, or NaN on failure."""
    for attempt in range(MAX_ATTEMPTS):
        try:
            wait_for_server()
            reset_server(seed, aggregator)
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
                detail = exc.stderr.decode(errors="replace").strip().splitlines()[-1:]
                detail = f" — {detail[0]}" if detail else ""
            print(f"      attempt {attempt + 1}/{MAX_ATTEMPTS} failed: "
                  f"{type(exc).__name__}{detail}")
        if attempt < len(RETRY_BACKOFF_SECONDS):
            time.sleep(RETRY_BACKOFF_SECONDS[attempt])
    return float("nan")


# --- driver ------------------------------------------------------------------

def pending(grid: dict, seeds: list[int], progress: dict) -> list[tuple[str, str, str, int]]:
    """Runs still to do. A recorded null means the run failed after all retries,
    so it stays pending and is attempted again on the next invocation rather
    than being silently treated as complete."""
    todo = []
    for aggregator, experiments in grid.items():
        for label, cfg in experiments.items():
            done = progress.get(f"{aggregator}:{label}", {})
            for seed in seeds:
                if done.get(str(seed)) is None:
                    todo.append((aggregator, label, cfg, seed))
    return todo


def write_summary(progress: dict) -> None:
    summary = {}
    for key, seed_map in sorted(progress.items()):
        values = [v for v in seed_map.values() if v is not None and not np.isnan(v)]
        summary[key] = {
            "n_seeds": len(values),
            "mean_auroc": round(float(np.mean(values)), 6) if values else None,
            "std_auroc": round(float(np.std(values, ddof=1)), 6) if len(values) > 1 else None,
            "per_seed": seed_map,
        }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2))


def print_status() -> None:
    """Read-only progress report. Safe to call while the grid is running."""
    progress = load_progress()
    total = sum(len(e) for e in GRID.values()) * len(SEEDS)
    done = failed = 0
    rows = []
    for aggregator, experiments in GRID.items():
        for label in experiments:
            seen = progress.get(f"{aggregator}:{label}", {})
            ok = [v for v in seen.values() if v is not None]
            bad = [s for s, v in seen.items() if v is None]
            done += len(ok)
            failed += len(bad)
            mean = f"{np.mean(ok):.4f}" if ok else "--"
            std = f"{np.std(ok, ddof=1):.4f}" if len(ok) > 1 else "--"
            rows.append((f"{aggregator}:{label}", len(ok), len(SEEDS), mean, std, bad))

    print(f"{'cell':40s} {'seeds':>7s} {'mean':>8s} {'std':>8s}")
    print("-" * 68)
    for name, n, want, mean, std, bad in rows:
        flag = f"  FAILED seeds: {','.join(bad)}" if bad else ""
        print(f"{name:40s} {n:>3d}/{want:<3d} {mean:>8s} {std:>8s}{flag}")
    print("-" * 68)

    pct = 100.0 * done / total if total else 0.0
    print(f"{done}/{total} runs complete ({pct:.1f}%)"
          + (f", {failed} failed and will be retried" if failed else ""))
    if done >= total:
        print("\nFINISHED — every cell complete.")
    else:
        eta = timedelta(seconds=int((total - done) * 7.5 * 60))
        print(f"Not finished. ~{total - done} runs left (~{eta} at 7.5 min/run).")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", choices=sorted(GRID), help="Run only this aggregator group.")
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS, help="Restrict to these seeds.")
    ap.add_argument("--dry-run", action="store_true", help="List pending runs and exit.")
    ap.add_argument("--status", action="store_true",
                    help="Print progress and exit. Safe to run while the grid is going.")
    args = ap.parse_args()

    if args.status:
        print_status()
        return

    grid = {args.only: GRID[args.only]} if args.only else GRID
    progress = load_progress()
    todo = pending(grid, args.seeds, progress)

    total = sum(len(e) for e in grid.values()) * len(args.seeds)
    print(f"Grid: {total} runs | already done: {total - len(todo)} | pending: {len(todo)}")
    if args.dry_run:
        for a, label, _, seed in todo:
            print(f"  {a}:{label} seed={seed}")
        return
    if not todo:
        print("Nothing to do — all runs complete.")
        write_summary(progress)
        return

    print(f"Estimated remaining: ~{timedelta(seconds=int(len(todo) * 8 * 60))} "
          f"(at ~8 min/run)\nProgress file: {PROGRESS_PATH}\n")

    durations: list[float] = []
    for i, (aggregator, label, cfg, seed) in enumerate(todo, 1):
        key = f"{aggregator}:{label}"
        eta = ""
        if durations:
            remaining = (len(todo) - i + 1) * float(np.mean(durations))
            eta = f" | ETA {timedelta(seconds=int(remaining))}"
        print(f"[{i}/{len(todo)}] {key} seed={seed}"
              f" ({datetime.now():%H:%M:%S}){eta}")

        started = time.time()
        auroc = run_one(cfg, seed, aggregator)
        durations.append(time.time() - started)

        progress.setdefault(key, {})[str(seed)] = None if np.isnan(auroc) else auroc
        save_progress(progress)
        write_summary(progress)

        shown = "FAILED" if np.isnan(auroc) else f"{auroc:.4f}"
        print(f"      -> {shown}  ({durations[-1] / 60:.1f} min)")

    print(f"\nDone. Summary: {SUMMARY_PATH}")
    for key, s in json.loads(SUMMARY_PATH.read_text()).items():
        if s["mean_auroc"] is not None:
            std = f"{s['std_auroc']:.4f}" if s["std_auroc"] is not None else "n/a"
            print(f"  {key:38s} {s['mean_auroc']:.4f} +/- {std}  (n={s['n_seeds']})")


if __name__ == "__main__":
    main()
