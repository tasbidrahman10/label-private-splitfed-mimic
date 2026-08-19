"""
Attack-magnitude sweep (addresses the "are you sure about 100% detection?"
reviewer question).

Every existing detection-rate figure in the paper (E1, Attack A, sensitivity
sweeps) was measured at ONE fixed, strong attack magnitude (10x gradient
scaling, or free rider's fully-random weights). This sweep varies the
gradient_scaling attack's `scale` parameter from mild to extreme on the E1
setup (Client 1 attacker, Medical/Cardiac honest, no straggler) to produce a
detection-rate CURVE instead of a single point -- direct evidence for whether
100% detection holds because the gate is genuinely discriminative, or only
because we never tried a weak/subtle attack.

Does not touch any existing E1 results (scale=10 point already on record).
Single seed=42, matching the convention used for Task 4's other sweeps.

Usage (two terminals):
    Terminal 1: python scripts/start_server.py
    Terminal 2: python scripts/run_attack_magnitude_sweep.py

Results saved to results/attack_magnitude_sweep.csv
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

SERVER_URL   = "http://127.0.0.1:8000"
AUTH_TOKEN   = "dev-sfl-token"
MAX_BATCHES  = 200
SEED         = 42
ATTACKER_CID = 1

E1_CONFIG = "configs/experiment_e1_gradient_scaling.yaml"

# Sub-unit -> extreme. Scales below 1.0 shrink the submitted encoder instead of
# inflating it, which is the evasion direction a reviewer raised: SNAS's norm
# term uses |log(||theta_i|| / ||theta^g||)|, so it is symmetric about 1.0 in
# theory and a 0.1x downscale should score the same as a 10x upscale. That
# symmetry had never been tested empirically -- the sweep started at 1.2 -- so
# the band below 1.0 is included here to confirm it holds rather than asserting
# it. 10.0 remains the sanity-check anchor against the on-record E1 result.
SCALES = [0.1, 0.3, 0.5, 0.8, 1.2, 1.5, 2.0, 3.0, 5.0, 7.0, 10.0]

DEFAULTS = {
    "snas_beta":                 0.35,
    "snas_gamma":                0.65,
    "snas_alpha":                0.0,
    "snas_threshold_flag":       0.22,
    "snas_threshold_quarantine": 0.35,
    "fedavg_clip_ratio":         2.0,
}

OUT_CSV = ROOT / "results" / "attack_magnitude_sweep.csv"
METRICS_FILE = ROOT / "results" / "label_private_splitfed" / "server" / "server_metrics.jsonl"


def _headers() -> dict:
    return {"Authorization": f"Bearer {AUTH_TOKEN}"}


def reset_server(overrides: dict) -> None:
    resp = requests.post(
        f"{SERVER_URL}/admin/reset",
        json={"config_overrides": {"seed": SEED, "aggregator_type": "snas", **overrides}},
        headers=_headers(),
        timeout=30,
    )
    resp.raise_for_status()


def extract_gate_metrics(since: float) -> dict:
    """Detection rate + first-detection round for the attacker in the run that
    just finished.

    Records are selected by timestamp rather than by a fixed tail of the file.
    The previous fixed 20-line tail assumed "10 rounds x 2 clients", but E1 keeps
    all three clients inside the submission window and therefore logs 30 gate
    decisions per run, so the tail silently truncated the earliest rounds and
    computed the detection rate over a subset of the attacker's active rounds.
    """
    if not METRICS_FILE.exists():
        return {"detection_rate": float("nan"), "first_detection_round": None, "ever_quarantined": False}

    gate_lines = []
    with METRICS_FILE.open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if "gate_decision" not in line:
                continue
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            if e.get("timestamp", 0) >= since:
                gate_lines.append(e)

    attacker_events = sorted(
        [e for e in gate_lines if e["client_id"] == ATTACKER_CID and e["round"] > 1],
        key=lambda e: e["round"],
    )
    if not attacker_events:
        return {"detection_rate": float("nan"), "first_detection_round": None, "ever_quarantined": False}
    detected = [e for e in attacker_events if e["gate_decision"] in ("flag", "quarantine")]
    first_round = next((e["round"] for e in attacker_events if e["gate_decision"] in ("flag", "quarantine")), None)
    ever_quarantined = any(e["gate_decision"] == "quarantine" for e in attacker_events)
    return {
        "detection_rate": round(len(detected) / len(attacker_events), 4),
        "first_detection_round": first_round,
        "ever_quarantined": ever_quarantined,
        "n_active_rounds": len(attacker_events),
    }


MAX_ATTEMPTS = 4
RETRY_BACKOFF_SECONDS = [5, 10, 20]
SETTLE_SECONDS = 5   # let /admin/reset fully settle before the run starts --
                      # SNAS aggregation has a known race (robust_fedavg.py
                      # KeyError on sample_counts) if a round starts before
                      # client re-registration has landed.
EXPECTED_ROUNDS = 10


def _attempt_once(scale: float) -> tuple[dict | None, str]:
    reset_server(DEFAULTS)
    time.sleep(SETTLE_SECONDS)
    run_started = time.time()

    attack_config = json.dumps({str(ATTACKER_CID): {"type": "gradient_scaling", "params": {"scale": scale}}})

    result = subprocess.run(
        [
            PYTHON,
            str(ROOT / "scripts" / "run_label_private_splitfed.py"),
            "--experiment-config", str(ROOT / E1_CONFIG),
            "--max-batches", str(MAX_BATCHES),
            "--seed", str(SEED),
            "--attack-config", attack_config,
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )

    auroc_values = []
    for line in result.stdout.splitlines():
        if "mean AUROC=" in line:
            try:
                auroc_values.append(float(line.split("mean AUROC=")[1].split(",")[0]))
            except (ValueError, IndexError):
                pass

    if result.returncode != 0:
        tail = "\n".join(result.stdout.splitlines()[-15:] + result.stderr.splitlines()[-15:])
        return None, f"exit code {result.returncode}. Tail:\n{tail}"

    if len(auroc_values) < EXPECTED_ROUNDS:
        return None, f"only {len(auroc_values)}/{EXPECTED_ROUNDS} rounds completed (client-side error mid-run?)"

    final = auroc_values[-3:]
    mean_auroc = float(sum(final) / len(final))
    round1_drop = auroc_values[0] - auroc_values[1]
    gate = extract_gate_metrics(run_started)

    return {
        "scale": scale,
        "mean_auroc_r8_10": round(mean_auroc, 6),
        "round1_auroc_drop": round(round1_drop, 6),
        "detection_rate": gate["detection_rate"],
        "first_detection_round": gate["first_detection_round"],
        "ever_quarantined": gate["ever_quarantined"],
        "n_active_rounds": gate.get("n_active_rounds", "nan"),
    }, ""


def run_one(scale: float) -> dict:
    for attempt in range(1, MAX_ATTEMPTS + 1):
        row, err = _attempt_once(scale)
        if row is not None:
            return row
        if attempt < MAX_ATTEMPTS:
            wait = RETRY_BACKOFF_SECONDS[attempt - 1]
            print(f"  ERROR (attempt {attempt}/{MAX_ATTEMPTS}) at scale={scale}: {err}. Retrying in {wait}s...")
            time.sleep(wait)
        else:
            print(f"  GAVE UP at scale={scale} after {MAX_ATTEMPTS} attempts: {err}")
    return {
        "scale": scale,
        "mean_auroc_r8_10": "nan",
        "round1_auroc_drop": "nan",
        "detection_rate": "nan",
        "first_detection_round": None,
        "ever_quarantined": False,
        "n_active_rounds": "nan",
    }


def load_completed() -> list[dict]:
    """Points already in the output CSV, so an interrupted sweep resumes."""
    if not OUT_CSV.exists():
        return []
    with OUT_CSV.open(encoding="utf-8") as fh:
        return [r for r in csv.DictReader(fh) if r.get("scale")]


def main() -> None:
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    rows = load_completed()
    done = {float(r["scale"]) for r in rows}
    todo = [s for s in SCALES if s not in done]
    if done:
        print(f"Resuming: {len(done)} scale(s) already recorded "
              f"({sorted(done)}), {len(todo)} remaining.")
    if not todo:
        print("All scales already complete. Delete the CSV to force a rerun.")
        return

    for scale in todo:
        print(f"\n{'='*60}\nscale={scale}x\n{'='*60}")
        row = run_one(scale)
        rows.append(row)
        rows.sort(key=lambda r: float(r["scale"]))
        print(f"  AUROC(r8-10)={row['mean_auroc_r8_10']}  DR={row['detection_rate']}  "
              f"first_detect_round={row['first_detection_round']}  quarantined={row['ever_quarantined']}")

        # checkpoint after every point
        with OUT_CSV.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    print(f"\nSaved {len(rows)} points to {OUT_CSV}")
    print("\nscale  | AUROC   | DR     | first_detect | quarantined")
    for r in rows:
        print(f"{r['scale']:<6} | {r['mean_auroc_r8_10']:<7} | {r['detection_rate']:<6} | "
              f"{str(r['first_detection_round']):<12} | {r['ever_quarantined']}")


if __name__ == "__main__":
    main()
