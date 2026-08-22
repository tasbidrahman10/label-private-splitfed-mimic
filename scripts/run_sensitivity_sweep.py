"""
Hyperparameter sensitivity sweep (Task 4).

Runs three sweeps per the professor's specification:

  Sweep 1 (9 points):   vary beta from 0.10 to 0.90, gamma = 1 - beta
                        Fixed: thresholds at defaults (0.22 / 0.35)
                        Experiments: E1 (gradient scaling) + E2 (free rider)

  Sweep 2 (<=121 pts):  vary threshold_flag (0.10-0.30) x threshold_quarantine (0.25-0.50)
                        Skip invalid combos where flag >= quarantine
                        Fixed: beta/gamma at defaults (0.35 / 0.65)
                        Experiment: E4 (attack + straggler — hardest case)

  Sweep 3 (6 points):   vary clip_ratio from 1.5 to 4.0
                        Fixed: all others at defaults
                        Experiment: E1 (focused on warm-up round 1 AUROC drop)

All sweeps use seed=42 (single seed). After running, re-run the best and
default points at 5 seeds for final reported numbers (done in run_multiseed.py).

Usage (server must be running):
    python scripts/run_sensitivity_sweep.py

Results saved to results/sensitivity_sweep.csv
"""
from __future__ import annotations

import csv
import itertools
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
WARMUP_ROUNDS = 1   # must match anomaly_detector.WARMUP_ROUNDS
OUT_CSV     = ROOT / "results" / "sensitivity_sweep.csv"

# Default SNAS parameters (confirmed operating point)
DEFAULTS = {
    "snas_beta":               0.35,
    "snas_gamma":              0.65,
    "snas_alpha":              0.0,
    "snas_threshold_flag":     0.22,
    "snas_threshold_quarantine": 0.35,
    "fedavg_clip_ratio":       2.0,
}

EXPERIMENT_CONFIGS = {
    "E1": "configs/experiment_e1_gradient_scaling.yaml",
    "E2": "configs/experiment_e2_free_rider.yaml",
    "E4": "configs/experiment_e4_attack_plus_dropout.yaml",
}


def _headers() -> dict:
    return {"Authorization": f"Bearer {AUTH_TOKEN}"}


def reset_server(overrides: dict) -> None:
    resp = requests.post(
        f"{SERVER_URL}/admin/reset",
        json={"config_overrides": {"seed": 42, **overrides}},
        headers=_headers(),
        timeout=30,
    )
    resp.raise_for_status()


def run_experiment(experiment: str, server_overrides: dict) -> dict:
    """Run one experiment with given server config overrides. Returns metrics dict."""
    reset_server(server_overrides)
    time.sleep(0.5)
    # Timestamp the run so its gate records can be selected by time rather than
    # by a fixed tail length that assumed a particular client count.
    run_started = time.time()

    cfg_path = ROOT / EXPERIMENT_CONFIGS[experiment]
    # Unique results dir per override to avoid CSV collision
    override_tag = "_".join(f"{k.split('_')[-1]}{v:.3f}" for k, v in sorted(server_overrides.items()))
    results_dir = ROOT / "results" / "sensitivity" / experiment / override_tag

    result = subprocess.run(
        [
            PYTHON,
            str(ROOT / "scripts" / "run_label_private_splitfed.py"),
            "--experiment-config", str(cfg_path),
            "--max-batches", str(MAX_BATCHES),
            "--seed", "42",
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )

    # Parse AUROC from stdout (rounds 8-10 mean)
    auroc_values = []
    for line in result.stdout.splitlines():
        if "mean AUROC=" in line:
            try:
                val = float(line.split("mean AUROC=")[1].split(",")[0])
                auroc_values.append(val)
            except (ValueError, IndexError):
                pass

    final_aurocs = auroc_values[-3:] if len(auroc_values) >= 3 else auroc_values
    mean_auroc = float(np.mean(final_aurocs)) if final_aurocs else float("nan")

    # Detection rate and FPR from the gate records this run produced
    detection_rate = _extract_detection_rate(experiment, run_started)
    fpr = _extract_fpr(experiment, run_started)

    return {
        "mean_auroc_r8_10":  round(mean_auroc, 6),
        "detection_rate":     detection_rate,
        "false_positive_rate": fpr,
        "round1_auroc_drop":  round(auroc_values[0] - auroc_values[1], 6) if len(auroc_values) >= 2 else "nan",
    }


def _attackers_for(experiment: str) -> set[int]:
    """Attacker client ids, read from the experiment config rather than assumed.

    The previous version hardcoded `attacker_cid = 1`, which is right for E1 and
    E4 but WRONG for E2, whose free rider is client 2. Every E2 row in
    sensitivity_sweep.csv produced before 2026-08-22 therefore measured an
    honest client and reported 0% detection, contradicting the 100% that
    Table 2 reports for the same setting. No published claim rested on that
    column, but it must not be quoted.
    """
    import yaml
    cfg = yaml.safe_load((ROOT / EXPERIMENT_CONFIGS[experiment]).read_text(encoding="utf-8"))
    return {int(k) for k in (cfg.get("attack_config") or {})}


def _gate_records_for_run(since: float) -> list[dict]:
    """Gate records written since `since`, i.e. belonging to the run just done.

    Replaces a fixed `snas_lines[-20:]` tail whose comment assumed "10 rounds x
    2 submitted clients in E4". E1 and E2 keep all three clients inside the
    window and log 30 records per run, so the tail silently dropped their
    earliest rounds -- the same bug class Nihal fixed in the attack-magnitude
    sweep but which survived here.
    """
    metrics_file = (ROOT / "results" / "label_private_splitfed" / "server"
                    / "server_metrics.jsonl")
    if not metrics_file.exists():
        return []
    out = []
    with metrics_file.open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if '"gate_decision"' not in line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("timestamp", 0) >= since:
                out.append(rec)
    return out


def _extract_detection_rate(experiment: str, since: float) -> float:
    """Detection rate over post-warm-up attacker rounds of the run just finished."""
    try:
        recent = _gate_records_for_run(since)
        attackers = _attackers_for(experiment)
        total = [e for e in recent
                 if e["client_id"] in attackers and e["round"] > WARMUP_ROUNDS]
        detected = [e for e in total if e["gate_decision"] in ("flag", "quarantine")]
        return round(len(detected) / len(total), 4) if total else float("nan")
    except Exception:
        return float("nan")


def _extract_fpr(experiment: str, since: float) -> float:
    """False-positive rate over post-warm-up honest client-rounds of that run.

    Also previously assumed client 1 was the only attacker, so on E2 it counted
    the real attacker (client 2) as honest and the honest client 1 as the
    attacker -- inverting both metrics for that experiment.
    """
    try:
        recent = _gate_records_for_run(since)
        attackers = _attackers_for(experiment)
        honest = [e for e in recent
                  if e["client_id"] not in attackers and e["round"] > WARMUP_ROUNDS]
        false_pos = [e for e in honest if e["gate_decision"] in ("flag", "quarantine")]
        return round(len(false_pos) / len(honest), 4) if honest else float("nan")
    except Exception:
        return float("nan")


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------
# The sweep is ~139 runs at ~6.5 min each, so a full pass is most of a day. The
# original version accumulated every result in memory and wrote the CSV only
# after the last point, which meant any interruption -- a mains failure, a
# reboot -- discarded the entire run. Two outages during the P1 grid on
# 2026-08-20 made that a real risk rather than a theoretical one.
#
# Each point is now recorded to a progress file the moment it completes, using
# the same atomic temp-file-and-replace as rerun_p08.py so an interrupt mid-write
# cannot corrupt it, and the CSV is rewritten after every point. Re-running skips
# whatever is already recorded, so a resume costs at most the one point that was
# in flight.
PROGRESS_PATH = ROOT / "results" / "sensitivity_sweep_progress.json"

KEY_FIELDS = ("sweep_id", "experiment", "beta", "gamma",
              "threshold_flag", "threshold_quarantine", "clip_ratio")


def _load_progress() -> dict:
    if PROGRESS_PATH.exists():
        try:
            return json.loads(PROGRESS_PATH.read_text())
        except json.JSONDecodeError:
            print(f"WARNING: {PROGRESS_PATH.name} unreadable — starting fresh.")
    return {}


def _save_progress(progress: dict) -> None:
    PROGRESS_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = PROGRESS_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(progress, indent=2))
    tmp.replace(PROGRESS_PATH)


def _point_key(row_meta: dict) -> str:
    return "|".join(f"{row_meta[k]}" for k in KEY_FIELDS)


def _write_csv(progress: dict) -> None:
    rows = list(progress.values())
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with OUT_CSV.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


PROGRESS = _load_progress()


def run_point(row_meta: dict, overrides: dict) -> dict:
    """run_experiment for one grid point, skipping it if already recorded."""
    key = _point_key(row_meta)
    if key in PROGRESS:
        return PROGRESS[key]
    metrics = run_experiment(row_meta["experiment"], overrides)
    row = {**row_meta, **metrics}
    PROGRESS[key] = row
    _save_progress(PROGRESS)
    _write_csv(PROGRESS)
    return row


def sweep1_beta(betas: list[float] | None = None) -> list[dict]:
    """Vary beta from 0.10 to 0.90, evaluate on E1 and E2."""
    if betas is None:
        betas = list(np.round(np.arange(0.10, 0.91, 0.10), 2))
    results = []
    for beta in betas:
        gamma = round(1.0 - beta, 4)
        overrides = {**DEFAULTS, "snas_beta": beta, "snas_gamma": gamma}
        print(f"  Sweep1 beta={beta:.2f} gamma={gamma:.2f}")
        for exp in ["E1", "E2"]:
            results.append(run_point({
                "sweep_id": "sweep1_beta",
                "experiment": exp,
                "beta": beta,
                "gamma": gamma,
                "threshold_flag": DEFAULTS["snas_threshold_flag"],
                "threshold_quarantine": DEFAULTS["snas_threshold_quarantine"],
                "clip_ratio": DEFAULTS["fedavg_clip_ratio"],
            }, overrides))
    return results


def sweep2_thresholds(
    flags: list[float] | None = None,
    quarantines: list[float] | None = None,
) -> list[dict]:
    """Vary threshold_flag x threshold_quarantine, evaluate on E4."""
    if flags is None:
        flags = list(np.round(np.arange(0.10, 0.31, 0.02), 3))
    if quarantines is None:
        quarantines = list(np.round(np.arange(0.25, 0.51, 0.025), 4))
    results = []
    combos = [(tf, tq) for tf, tq in itertools.product(flags, quarantines) if tf < tq]
    print(f"  Sweep2: {len(combos)} valid (threshold_flag, threshold_quarantine) combos on E4")
    for tf, tq in combos:
        overrides = {
            **DEFAULTS,
            "snas_threshold_flag": tf,
            "snas_threshold_quarantine": tq,
        }
        print(f"    flag={tf:.3f} quarantine={tq:.4f}")
        results.append(run_point({
            "sweep_id": "sweep2_thresholds",
            "experiment": "E4",
            "beta": DEFAULTS["snas_beta"],
            "gamma": DEFAULTS["snas_gamma"],
            "threshold_flag": tf,
            "threshold_quarantine": tq,
            "clip_ratio": DEFAULTS["fedavg_clip_ratio"],
        }, overrides))
    return results


def sweep3_clip_ratio(ratios: list[float] | None = None) -> list[dict]:
    """Vary clip_ratio from 1.5 to 4.0, evaluate on E1 (warm-up protection)."""
    if ratios is None:
        ratios = list(np.round(np.arange(1.5, 4.01, 0.5), 2))
    results = []
    for cr in ratios:
        overrides = {**DEFAULTS, "fedavg_clip_ratio": cr}
        print(f"  Sweep3 clip_ratio={cr:.1f}")
        results.append(run_point({
            "sweep_id": "sweep3_clip_ratio",
            "experiment": "E1",
            "beta": DEFAULTS["snas_beta"],
            "gamma": DEFAULTS["snas_gamma"],
            "threshold_flag": DEFAULTS["snas_threshold_flag"],
            "threshold_quarantine": DEFAULTS["snas_threshold_quarantine"],
            "clip_ratio": cr,
        }, overrides))
    return results


def main() -> None:
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    all_results = []

    print("\n=== Sweep 1: beta/gamma ===")
    all_results.extend(sweep1_beta())

    print("\n=== Sweep 2: thresholds (E4 — longest sweep) ===")
    all_results.extend(sweep2_thresholds())

    print("\n=== Sweep 3: clip_ratio ===")
    all_results.extend(sweep3_clip_ratio())

    if all_results:
        fieldnames = list(all_results[0].keys())
        with OUT_CSV.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(all_results)
        print(f"\nSaved {len(all_results)} sweep results to {OUT_CSV}")

        # Summary statistics
        print("\nSweep 1 (beta) — E1 detection rate range:")
        s1 = [r for r in all_results if r["sweep_id"] == "sweep1_beta" and r["experiment"] == "E1"]
        for r in s1:
            print(f"  beta={r['beta']:.2f}: auroc={r['mean_auroc_r8_10']} dr={r['detection_rate']}")

        s2 = [r for r in all_results if r["sweep_id"] == "sweep2_thresholds"]
        zero_fpr = [r for r in s2 if r["false_positive_rate"] == 0.0]
        print(f"\nSweep 2 (thresholds) — zero-FPR region: {len(zero_fpr)}/{len(s2)} grid points")


if __name__ == "__main__":
    main()
