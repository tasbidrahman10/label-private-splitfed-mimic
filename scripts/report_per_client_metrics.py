"""
P1-4 — per-client AUROC and AUPRC for every cell of the regenerated grid.

Analysis only: no experiment is rerun. Every number here comes from the
`experiment_summary_option_b_*_seed*.csv` files that the 180-run P0-8
regeneration already wrote, plus whatever P1 arms have completed since.

Why this exists
---------------
The paper reports a single AUROC per cell, averaged over clients. With client
prevalence spanning 6.9%-16.1% that hides two things a clinical reviewer will
ask about:

  * AUPRC, which is sensitive to prevalence in a way AUROC is not. A 0.96 AUROC
    on a 6.9%-prevalence partition is a much weaker statement than the same
    number at 16.1%.
  * Whether the aggregate is carried by one partition. Krum's federation
    collapse is already reported via contribution entropy; per-client accuracy
    is the complementary view.

Identifying which runs count
----------------------------
The results directories are cumulative and still hold pre-P0-8 runs on the old
266-feature data -- `results/experiment_e1_gradient_scaling` alone has 58 CSVs,
most from June and July. Globbing by seed would silently average stale
266-feature runs into the corrected numbers, which is exactly the class of bug
that produced the wrong federation-entropy table.

So the run set is not inferred from filenames. `results/p08_rerun_progress.json`
records the mean AUROC of every run in the grid, and a CSV is accepted as that
run only if its own last-3-round mean AUROC reproduces the recorded value to
1e-9. A cell that cannot be matched is reported as missing rather than guessed.

Where several CSVs match one (cell, seed), the runs are deterministic repeats
of the same configuration -- verified identical to eight decimal places on the
one case that occurs, snas:E1_gradient_scaling seed 42 -- and the newest is
taken, matching `rerun_p08.mean_auroc_from_csv`'s own newest-by-mtime rule.

Rounds averaged
---------------
Last 3 rounds of each run, identical to `rerun_p08.mean_auroc_from_csv`, so the
AUROC column here reproduces the paper's Table 2/3 values exactly and the AUPRC
column is directly comparable to it.

Usage
-----
    python scripts/report_per_client_metrics.py
    python scripts/report_per_client_metrics.py --only snas krum
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
PROGRESS_PATH = ROOT / "results" / "p08_rerun_progress.json"
OUT_CSV = ROOT / "results" / "p1_4_per_client_metrics.csv"
OUT_JSON = ROOT / "results" / "p1_4_per_client_metrics.json"

MATCH_TOLERANCE = 1e-9
LAST_N_ROUNDS = 3

# client_id -> (config file, human label). Used only for prevalence context and
# readable output; the metric values are keyed by the client_id in the CSVs.
CLIENT_CONFIGS = ROOT / "configs" / "clients"


def load_grid():
    """Import rerun_p08's GRID so this report can never drift from the runner."""
    spec = importlib.util.spec_from_file_location(
        "rerun_p08", ROOT / "scripts" / "rerun_p08.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.GRID, module.SEEDS


def results_dir_for(config_rel: str) -> Path:
    cfg = yaml.safe_load((ROOT / config_rel).read_text(encoding="utf-8"))
    return ROOT / cfg["results_dir"]


def read_run(csv_path: Path) -> dict[int, dict[str, float]] | None:
    """Per-client mean AUROC/AUPRC/F1 over the last LAST_N_ROUNDS rounds."""
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    if not rows:
        return None
    rounds = sorted({int(r["round"]) for r in rows})[-LAST_N_ROUNDS:]
    per_client: dict[int, dict[str, list[float]]] = {}
    for r in rows:
        if int(r["round"]) not in rounds:
            continue
        bucket = per_client.setdefault(int(r["client_id"]), {"auroc": [], "auprc": [], "f1": []})
        for name, col in (("auroc", "val_auroc"), ("auprc", "val_auprc"), ("f1", "val_f1")):
            if r.get(col):
                bucket[name].append(float(r[col]))
    return {
        cid: {k: float(np.mean(v)) for k, v in vals.items() if v}
        for cid, vals in per_client.items()
    }


def run_mean_auroc(per_client: dict[int, dict[str, float]]) -> float:
    """Client-averaged AUROC — the quantity rerun_p08 checkpoints."""
    vals = [c["auroc"] for c in per_client.values() if "auroc" in c]
    return float(np.mean(vals)) if vals else float("nan")


def locate_run(directory: Path, seed: str, recorded_auroc: float):
    """The CSV for one (cell, seed), identified by reproducing its AUROC.

    Returns (per_client_metrics, csv_path, n_matches) or (None, None, 0).
    """
    matches = []
    for path in directory.glob(f"*seed{seed}.csv"):
        per_client = read_run(path)
        if per_client and abs(run_mean_auroc(per_client) - recorded_auroc) < MATCH_TOLERANCE:
            matches.append((path, per_client))
    if not matches:
        return None, None, 0
    path, per_client = max(matches, key=lambda pc: pc[0].stat().st_mtime)
    return per_client, path, len(matches)


def client_prevalence() -> dict[int, dict[str, float]]:
    """Full-cohort mortality rate per client, read straight from the CSVs.

    Reported as context for AUPRC. This is the whole-partition prevalence, not
    the validation-split prevalence the AUPRC is actually computed against, and
    is labelled as such in the output.
    """
    import pandas as pd

    out = {}
    for cfg_path in sorted(CLIENT_CONFIGS.glob("*.yaml")):
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        data = ROOT / cfg["data_path"]
        if not data.exists():
            continue
        labels = pd.read_csv(data, usecols=["mortality"])["mortality"]
        out[int(cfg["client_id"])] = {
            "name": cfg_path.stem,
            "n_stays": int(len(labels)),
            "prevalence": round(float(labels.mean()), 5),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="+", help="Restrict to these aggregator groups.")
    args = ap.parse_args()

    grid, _seeds = load_grid()
    if args.only:
        unknown = set(args.only) - set(grid)
        if unknown:
            ap.error(f"unknown group(s): {sorted(unknown)}. Known: {sorted(grid)}")
        grid = {g: grid[g] for g in args.only}

    if not PROGRESS_PATH.exists():
        raise SystemExit(f"{PROGRESS_PATH} not found — nothing to report on.")
    progress = json.loads(PROGRESS_PATH.read_text())

    prevalence = client_prevalence()
    report: dict[str, dict] = {}
    rows: list[dict] = []
    missing: list[str] = []
    duplicates: list[str] = []

    for aggregator, experiments in grid.items():
        for label, config_rel in experiments.items():
            key = f"{aggregator}:{label}"
            directory = results_dir_for(config_rel)
            seed_map = progress.get(key, {})

            # seed -> {client_id -> metrics}
            per_seed: dict[str, dict[int, dict[str, float]]] = {}
            for seed, recorded in seed_map.items():
                if recorded is None:
                    continue
                found, path, n = locate_run(directory, seed, recorded)
                if found is None:
                    missing.append(f"{key} seed{seed}")
                    continue
                if n > 1:
                    duplicates.append(f"{key} seed{seed} ({n} identical runs)")
                per_seed[seed] = found

            if not per_seed:
                continue

            client_ids = sorted({c for v in per_seed.values() for c in v})
            cell: dict[str, dict] = {"n_seeds": len(per_seed), "clients": {}}

            for cid in client_ids:
                stats = {}
                for metric in ("auroc", "auprc", "f1"):
                    vals = [v[cid][metric] for v in per_seed.values()
                            if cid in v and metric in v[cid]]
                    if not vals:
                        continue
                    stats[f"{metric}_mean"] = round(float(np.mean(vals)), 6)
                    stats[f"{metric}_std"] = (
                        round(float(np.std(vals, ddof=1)), 6) if len(vals) > 1 else None
                    )
                stats["n_seeds"] = len(
                    [v for v in per_seed.values() if cid in v]
                )
                cell["clients"][str(cid)] = stats

                info = prevalence.get(cid, {})
                rows.append({
                    "cell": key,
                    "aggregator": aggregator,
                    "experiment": label,
                    "client_id": cid,
                    "client_name": info.get("name", ""),
                    "client_prevalence": info.get("prevalence", ""),
                    "n_seeds": stats["n_seeds"],
                    "auroc_mean": stats.get("auroc_mean"),
                    "auroc_std": stats.get("auroc_std"),
                    "auprc_mean": stats.get("auprc_mean"),
                    "auprc_std": stats.get("auprc_std"),
                    "f1_mean": stats.get("f1_mean"),
                    "f1_std": stats.get("f1_std"),
                })

            # Client-averaged values: AUROC reproduces the paper's headline
            # number, AUPRC is the new companion metric for the same runs.
            for metric in ("auroc", "auprc", "f1"):
                per_run = [
                    float(np.mean([c[metric] for c in v.values() if metric in c]))
                    for v in per_seed.values()
                    if any(metric in c for c in v.values())
                ]
                if per_run:
                    cell[f"macro_{metric}_mean"] = round(float(np.mean(per_run)), 6)
                    cell[f"macro_{metric}_std"] = (
                        round(float(np.std(per_run, ddof=1)), 6) if len(per_run) > 1 else None
                    )
            report[key] = cell

    payload = {
        "description": (
            "P1-4: per-client AUROC/AUPRC/F1 over the last 3 rounds, averaged "
            "across seeds. Runs identified by reproducing the mean AUROC "
            "recorded in p08_rerun_progress.json, so pre-P0-8 266-feature runs "
            "in the same directories are excluded."
        ),
        "rounds_averaged": LAST_N_ROUNDS,
        "client_prevalence_note": (
            "Full-partition mortality rate, for context. AUPRC is computed on "
            "the validation split, whose prevalence may differ slightly."
        ),
        "client_prevalence": {str(k): v for k, v in prevalence.items()},
        "cells": report,
        "unmatched_runs": missing,
        "duplicate_matches_resolved_to_newest": duplicates,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2))

    fieldnames = list(rows[0].keys()) if rows else []
    with OUT_CSV.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # --- console summary ---
    print("Client prevalence (full partition):")
    for cid, info in sorted(prevalence.items()):
        print(f"  client{cid} {info['name']:22s} n={info['n_stays']:6d}  "
              f"prevalence={info['prevalence']:.3%}")

    print(f"\n{'cell':38s} {'AUROC':>16s} {'AUPRC':>16s}  per-client AUPRC")
    print("-" * 100)
    for key, cell in report.items():
        au = f"{cell.get('macro_auroc_mean', float('nan')):.4f}"
        aus = cell.get("macro_auroc_std")
        ap_ = f"{cell.get('macro_auprc_mean', float('nan')):.4f}"
        aps = cell.get("macro_auprc_std")
        per = "  ".join(
            f"c{cid}={c.get('auprc_mean', float('nan')):.4f}"
            for cid, c in sorted(cell["clients"].items())
        )
        print(f"{key:38s} {au}+/-{aus if aus is not None else 0:.4f} "
              f"{ap_}+/-{aps if aps is not None else 0:.4f}  {per}")
    print("-" * 100)
    print(f"cells reported: {len(report)}")
    if duplicates:
        print(f"duplicate (cell,seed) matches resolved to newest: {len(duplicates)}")
    if missing:
        print(f"UNMATCHED runs (excluded): {len(missing)}")
        for entry in missing[:10]:
            print(f"   {entry}")
    print(f"\nWrote {OUT_CSV}\n      {OUT_JSON}")


if __name__ == "__main__":
    main()
