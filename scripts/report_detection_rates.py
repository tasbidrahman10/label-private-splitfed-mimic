"""
Detection and false-positive rates per experiment, from the server gate log.

P1-2 needs DR/FPR for the two newly-executed attacks (label-flip, backdoor),
and the same numbers for the existing attacks are worth recomputing from the
same code path so every row of the paper's Table 2 comes from one definition.

Which client is the attacker is read from each experiment's `attack_config`,
never assumed, so this stays correct if an experiment is repointed.

Definitions
-----------
Detection is reported over **post-warm-up attacker submissions**, matching the
wording P0-7 settled on. `WARMUP_ROUNDS = 1` in anomaly_detector.py and the
gate forces `accept` while `round_idx <= WARMUP_ROUNDS`, so rounds 0 and 1 are
excluded from both numerator and denominator; the count of excluded rounds is
reported alongside rather than hidden.

  detection rate   = attacker rounds with decision in {flag, quarantine}
                     / post-warm-up attacker rounds
  quarantine rate  = attacker rounds with decision == quarantine
                     / post-warm-up attacker rounds
  false positive   = honest rounds with decision in {flag, quarantine}
                     / post-warm-up honest client-rounds

Detection and quarantine are reported separately on purpose. A flagged client
is weight-penalised but still aggregated, so "detected" does not mean
"excluded" -- the distinction is what the P0-1 magnitude sweep turned on, and
it is what separates the two new attacks from gradient scaling.

Mapping log records to runs
---------------------------
The server log is cumulative (~134 MB, months of runs), so records are never
read wholesale. Gate records are grouped into runs structurally, by the round
counter restarting at 0 on `/admin/reset` -- the same rule
compute_federation_entropy.py uses -- and each block is then attributed to the
run whose `experiment_summary_*_seed*.csv` bears the closest preceding
timestamp. Anything that cannot be attributed within `--tolerance` minutes is
counted and reported, never silently folded into a neighbouring cell.

Usage
-----
    python scripts/report_detection_rates.py --since "2026-08-20 12:00"
    python scripts/report_detection_rates.py --cells snas:E3_label_flip
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import re
from datetime import datetime
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SERVER_LOG = ROOT / "results" / "label_private_splitfed" / "server" / "server_metrics.jsonl"
OUT_CSV = ROOT / "results" / "detection_rates.csv"
OUT_JSON = ROOT / "results" / "detection_rates.json"

PROGRESS_PATH = ROOT / "results" / "p08_rerun_progress.json"
MATCH_TOLERANCE = 1e-9

WARMUP_ROUNDS = 1          # must match anomaly_detector.WARMUP_ROUNDS
DETECTED = {"flag", "quarantine"}
RUN_ID_RE = re.compile(r"_(\d{8}_\d{6})_seed(\d+)\.csv$")


def load_grid():
    spec = importlib.util.spec_from_file_location(
        "rerun_p08", ROOT / "scripts" / "rerun_p08.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.GRID


def experiment_meta(config_rel: str) -> tuple[Path, set[int]]:
    """(results dir, attacker client ids) for one experiment config."""
    cfg = yaml.safe_load((ROOT / config_rel).read_text(encoding="utf-8"))
    attackers = {int(k) for k in (cfg.get("attack_config") or {})}
    return ROOT / cfg["results_dir"], attackers


def last3_mean_auroc(csv_path: Path) -> float:
    """Client-averaged AUROC over the last 3 rounds — rerun_p08's definition."""
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    if not rows:
        return float("nan")
    rounds = sorted({int(r["round"]) for r in rows})[-3:]
    values = [float(r["val_auroc"]) for r in rows
              if int(r["round"]) in rounds and r.get("val_auroc")]
    return sum(values) / len(values) if values else float("nan")


def discover_runs(grid, since: datetime | None):
    """Grid runs only: (start_time, cell, seed, attackers).

    Deliberately NOT filtered by --cells. Blocks are attributed to the run
    whose start most closely precedes them, so the candidate pool has to
    contain every run that could own a block. Filtering the pool instead would
    let a block belonging to an excluded cell be attributed to whichever
    included cell happened to run before it -- silently mixing, say, a Krum run
    into a SNAS row. --cells filters the report, not the attribution.

    A run counts only if `p08_rerun_progress.json` records it AND the CSV
    reproduces the recorded mean AUROC to 1e-9. Filename matching alone is not
    enough: several scripts write into the same results_dir, so
    `results/experiment_e1_gradient_scaling` holds 24 CSVs since 17 August for
    ten grid seeds. Eleven of them are the P0-1 attack-magnitude sweep, whose
    sub-2x points evade the gate *by design* -- folding those in dropped
    snas:E1 detection from 100% to 69% and would have read as the paper
    overclaiming. The rest are smoke and timing runs.
    """
    progress = json.loads(PROGRESS_PATH.read_text()) if PROGRESS_PATH.exists() else {}
    runs = []
    for aggregator, experiments in grid.items():
        for label, config_rel in experiments.items():
            cell = f"{aggregator}:{label}"
            directory, attackers = experiment_meta(config_rel)
            if not directory.exists():
                continue
            recorded = progress.get(cell, {})
            by_seed: dict[int, list] = {}
            for path in directory.glob("experiment_summary_*_seed*.csv"):
                match = RUN_ID_RE.search(path.name)
                if not match:
                    continue
                started = datetime.strptime(match.group(1), "%Y%m%d_%H%M%S")
                if since and started < since:
                    continue
                seed = int(match.group(2))
                want = recorded.get(str(seed))
                if want is None:
                    continue
                if abs(last3_mean_auroc(path) - want) > MATCH_TOLERANCE:
                    continue
                by_seed.setdefault(seed, []).append(started)
            # Deterministic repeats of one config can both match; take the
            # newest, matching rerun_p08.mean_auroc_from_csv's own rule.
            for seed, starts in by_seed.items():
                runs.append((max(starts), cell, seed, attackers))
    return sorted(runs)


def read_gate_blocks(since: datetime | None, logs: list[Path] | None = None):
    """Gate-decision records grouped into runs by the round counter resetting.

    Several logs can be passed: the P0 grid ran on a different machine and its
    server log is gitignored, so Nihal's copy arrives as a separate file rather
    than merged into the local one (which holds this machine's runs and must
    not be overwritten). Records from all logs are pooled and re-sorted by
    timestamp before being split into runs, so interleaving is handled and the
    round-restart rule still applies.
    """
    sources = logs or [SERVER_LOG]
    missing = [s for s in sources if not s.exists()]
    if missing:
        raise SystemExit("Log(s) not found: " + ", ".join(str(m) for m in missing))
    cutoff = since.timestamp() if since else None
    records = []
    # Logs from two machines share the history from before they diverged -- the
    # local and P0 logs hold the same 1,300 records from 2026-07-05. Pooling
    # without deduplication would count those rounds twice and corrupt the
    # round-restart grouping, so records are keyed on the tuple that uniquely
    # identifies one gate decision.
    seen: set[tuple] = set()
    duplicates = 0
    # Streamed line by line: these files are ~100-160 MB, mostly per-batch rows.
    for source in sources:
        with source.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if '"gate_decision"' not in line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if cutoff and record.get("timestamp", 0) < cutoff:
                    continue
                key = (record.get("timestamp"), record.get("client_id"), record.get("round"))
                if key in seen:
                    duplicates += 1
                    continue
                seen.add(key)
                records.append(record)
    if duplicates:
        print(f"note: {duplicates} duplicate gate records across logs, kept once each")
    # Pooled logs are not globally ordered; runs are derived from round
    # restarts, which only means anything in timestamp order.
    records.sort(key=lambda r: r.get("timestamp", 0.0))
    blocks, current, previous_round = [], [], None
    for record in records:
        if previous_round is not None and record["round"] < previous_round:
            blocks.append(current)
            current = []
        current.append(record)
        previous_round = record["round"]
    if current:
        blocks.append(current)
    return blocks


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--since", help='Only runs at/after this local time, '
                                        '"YYYY-MM-DD HH:MM".')
    parser.add_argument("--cells", nargs="+", help="Restrict to these grid cells.")
    # A run's first aggregation lands ~0.8 min after its start (measured across
    # every grid cell), and consecutive runs are ~6.5 min apart. A block more
    # than a few minutes late therefore belongs to something else -- typically a
    # partial or aborted run whose own CSV is not a grid run, so it falls
    # through to the nearest preceding grid run and contaminates that cell.
    # At the old 12-minute default this pulled six such blocks into snas:clean
    # and manufactured a 3.4% false-quarantine rate on runs that have no
    # attacker; the genuine blocks all peak at SNAS ~0.11, far below the 0.22
    # flag threshold. 4 minutes keeps every genuine block and excludes those.
    parser.add_argument("--tolerance", type=float, default=4.0,
                        help="Max minutes between a run's start and its gate block.")
    parser.add_argument("--log", nargs="+", type=Path,
                        help="Server metrics log(s) to read. Defaults to the local "
                             "one; pass both to include a copy from another machine, "
                             "e.g. --log results/label_private_splitfed/server/"
                             "server_metrics.jsonl results/label_private_splitfed/"
                             "server/server_metrics_nihal_p08.jsonl")
    args = parser.parse_args()

    since = datetime.strptime(args.since, "%Y-%m-%d %H:%M") if args.since else None
    grid = load_grid()
    runs = discover_runs(grid, since)
    if not runs:
        raise SystemExit("No runs found in that window.")
    wanted = set(args.cells) if args.cells else None

    blocks = read_gate_blocks(since or runs[0][0], args.log)

    # Attribute each block to the run whose start most closely precedes it.
    per_cell: dict[str, dict] = {}
    unattributed = 0
    for block in blocks:
        block_start = datetime.fromtimestamp(block[0]["timestamp"])
        candidates = [r for r in runs if r[0] <= block_start]
        if not candidates:
            unattributed += 1
            continue
        started, cell, seed, attackers = candidates[-1]
        if (block_start - started).total_seconds() / 60.0 > args.tolerance:
            unattributed += 1
            continue
        bucket = per_cell.setdefault(cell, {
            "attackers": sorted(attackers), "seeds": set(),
            "atk_rounds": 0, "atk_detected": 0, "atk_quarantined": 0,
            "honest_rounds": 0, "honest_detected": 0,
            "warmup_atk_rounds": 0, "snas_attacker": [], "snas_honest": [],
        })
        bucket["seeds"].add(seed)
        for record in block:
            is_attacker = record["client_id"] in attackers
            snas = record.get("snas")
            if record["round"] <= WARMUP_ROUNDS:
                if is_attacker:
                    bucket["warmup_atk_rounds"] += 1
                continue
            detected = record.get("gate_decision") in DETECTED
            if is_attacker:
                bucket["atk_rounds"] += 1
                bucket["atk_detected"] += detected
                bucket["atk_quarantined"] += record.get("gate_decision") == "quarantine"
                if snas is not None:
                    bucket["snas_attacker"].append(snas)
            else:
                bucket["honest_rounds"] += 1
                bucket["honest_detected"] += detected
                if snas is not None:
                    bucket["snas_honest"].append(snas)

    report, rows = {}, []
    for cell, b in sorted(per_cell.items()):
        if not b["attackers"]:
            continue    # clean / honest-straggler cells have no attacker to detect
        if wanted and cell not in wanted:
            continue    # report filter only; attribution already used every run
        mean = lambda xs: round(sum(xs) / len(xs), 4) if xs else None
        entry = {
            "attacker_clients": b["attackers"],
            "n_seeds": len(b["seeds"]),
            "attacker_rounds_post_warmup": b["atk_rounds"],
            "attacker_rounds_in_warmup_excluded": b["warmup_atk_rounds"],
            "detection_rate": round(b["atk_detected"] / b["atk_rounds"], 4) if b["atk_rounds"] else None,
            "quarantine_rate": round(b["atk_quarantined"] / b["atk_rounds"], 4) if b["atk_rounds"] else None,
            "honest_client_rounds": b["honest_rounds"],
            "false_positive_rate": round(b["honest_detected"] / b["honest_rounds"], 4) if b["honest_rounds"] else None,
            "mean_snas_attacker": mean(b["snas_attacker"]),
            "mean_snas_honest": mean(b["snas_honest"]),
        }
        report[cell] = entry
        rows.append({"cell": cell, **entry, "attacker_clients": ",".join(map(str, b["attackers"]))})

    OUT_JSON.write_text(json.dumps({
        "warmup_rounds_excluded": WARMUP_ROUNDS,
        "detected_decisions": sorted(DETECTED),
        "unattributed_gate_blocks": unattributed,
        "cells": report,
    }, indent=2))
    if rows:
        with OUT_CSV.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    print(f"{'cell':30s} {'seeds':>5s} {'DR':>7s} {'quar':>7s} {'FPR':>7s} "
          f"{'SNAS atk':>9s} {'SNAS hon':>9s}")
    print("-" * 82)
    for cell, e in report.items():
        print(f"{cell:30s} {e['n_seeds']:>5d} "
              f"{e['detection_rate']:>7.1%} {e['quarantine_rate']:>7.1%} "
              f"{e['false_positive_rate']:>7.1%} "
              f"{e['mean_snas_attacker']:>9.4f} {e['mean_snas_honest']:>9.4f}")
    print("-" * 82)
    print(f"gate blocks that could not be attributed to a run: {unattributed}")
    print(f"\nWrote {OUT_CSV}\n      {OUT_JSON}")


if __name__ == "__main__":
    main()
