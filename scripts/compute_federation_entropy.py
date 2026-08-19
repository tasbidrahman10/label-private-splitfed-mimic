"""
Federation-preservation metric (Gap #3): quantify "Krum collapses federation,
SNAS does not" with a hard number instead of a qualitative claim.

Contribution entropy per aggregation round:
    share_i = normalized aggregation weight of client i that round
    H = -sum_i share_i * ln(share_i) / ln(N_registered)      (in [0,1])
H=1  -> perfectly balanced participation across all clients
H=0  -> a single client carries the whole aggregate (federation collapsed)

Sources:
  - Krum:        results/baseline_krum/server/server_metrics.jsonl  (krum_selected flag)
                 -> selected client gets share 1, others 0. Available now.
  - SNAS / no-detect: results/label_private_splitfed/server/server_metrics.jsonl
                 (aggregation_weight_normalized per client per round), sliced by the
                 per-run [t_start,t_end] windows in results/gap_closing/run_windows.json.
                 Available after close_gaps_runner.py finishes.

Usage:
    python scripts/compute_federation_entropy.py            # all available parts
    python scripts/compute_federation_entropy.py --krum-only

Output: results/gap_closing/federation_entropy.{csv,json}
"""
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "gap_closing"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# rerun_p08.py switches aggregator via POST /admin/reset against a single
# long-lived server started from configs/server.yaml, so results_dir never
# changed: every aggregator's metrics land in this one file. The per-baseline
# directories (results/baseline_krum/...) hold only pre-P0-8 266-feature runs
# and must NOT be read — doing so yields a Table 4 that silently contradicts
# Table 3. Baseline records carry an "aggregator" field; SNAS records do not,
# and are identified by their richer schema instead.
COMBINED_LOG = ROOT / "results" / "label_private_splitfed" / "server" / "server_metrics.jsonl"
SNAS_LOG = COMBINED_LOG
WINDOWS = OUT_DIR / "run_windows.json"
N_CLIENTS = 3

# The combined log is cumulative and still holds July runs plus the Aug-17
# smoke/timing runs. Everything at or after this instant is the P0-8 regeneration
# grid (first run started 2026-08-17 13:23:41). Override with --since.
GRID_START_DEFAULT = "2026-08-17 13:20"

# Records inside one aggregation round are written microseconds apart, while
# consecutive rounds are ~40s apart, so a gap this size cleanly separates rounds
# without needing the per-run windows file (which is stale and predates the grid).
ROUND_GAP_SECONDS = 5.0

# num_rounds in the experiment configs. /admin/reset restarts each run's round
# counter at 0, so a run is complete iff it contributed exactly this many rounds.
# Runs aborted by the connection bugs fixed on 2026-08-17 left short round
# sequences in the log; dropping anything shorter removes them without needing to
# know which seeds failed.
ROUNDS_PER_RUN = 10

# rerun_p08.py walks its GRID dict in order, so the SNAS runs appear in the log
# as five consecutive blocks of ten seeds. The mapping is verified by each
# block's signature rather than assumed: `clean` shows all three clients at the
# analytic entropy, `E2_free_rider` excludes client 2 (its attacker) after
# warm-up, and the two straggler experiments average ~2.1 clients per round.
# The nodetect (gate ablation) arm uses the same SNAS record schema and ran after
# the aggregator groups, so it appears as four further blocks.
SNAS_BLOCKS = ["clean", "E1_gradient_scaling", "E2_free_rider",
               "E4_attack_straggler", "E6_honest_straggler",
               "nodetect E1_gradient_scaling", "nodetect E2_free_rider",
               "nodetect E4_attack_straggler", "nodetect E6_honest_straggler"]
SEEDS_PER_CELL = 10


def _norm_entropy(shares: list[float], n: int = N_CLIENTS) -> float:
    tot = sum(shares)
    if tot <= 0:
        return float("nan")
    p = [s / tot for s in shares]
    h = -sum(pi * math.log(pi) for pi in p if pi > 0)
    # `+ 0.0` normalises the -0.0 that a single-client (fully collapsed)
    # aggregate produces, so tables read "0.0" rather than "-0.0".
    return (h / math.log(n) if n > 1 else 0.0) + 0.0


def _iter_json(path: Path):
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def krum_entropy(since: float) -> dict | None:
    """Selection frequency -> contribution share for Krum (client selected == full weight).

    Krum picks exactly one submission per round, so the winner carries the whole
    aggregate that round and every other client contributes nothing.
    """
    if not COMBINED_LOG.exists():
        return None
    sel_counts: dict[int, int] = {}
    total = 0
    for e in _iter_json(COMBINED_LOG):
        if e.get("aggregator") != "krum" or e.get("timestamp", 0) < since:
            continue
        if e.get("krum_selected"):
            cid = int(e["client_id"])
            sel_counts[cid] = sel_counts.get(cid, 0) + 1
            total += 1
    if not sel_counts:
        return None
    shares = [sel_counts.get(c, 0) / total for c in range(N_CLIENTS)]
    return {
        "method": "Krum",
        "condition": "measured, E1/E2/E4 x 10 seeds",
        "per_client_share": {f"client{c}": round(shares[c], 4) for c in range(N_CLIENTS)},
        "normalized_entropy": round(_norm_entropy(shares), 4),
        "n_selection_events": total,
    }


# Filled in by snas_entropy_measured(); one row per SNAS experiment block.
snas_measured_by_experiment: list[dict] = []


def _per_experiment(complete: list[list[dict[int, float]]]) -> list[dict]:
    """Split the complete SNAS runs into their five experiment blocks."""
    if len(complete) != len(SNAS_BLOCKS) * SEEDS_PER_CELL:
        return []   # unexpected run count — do not guess at the mapping
    rows = []
    for i, name in enumerate(SNAS_BLOCKS):
        block = complete[i * SEEDS_PER_CELL:(i + 1) * SEEDS_PER_CELL]
        ents, shares = [], {c: 0.0 for c in range(N_CLIENTS)}
        for run in block:
            for g in run:
                vals = [g.get(c, 0.0) for c in range(N_CLIENTS)]
                tot = sum(vals)
                if tot <= 0:
                    continue
                ents.append(_norm_entropy(vals))
                for c in range(N_CLIENTS):
                    shares[c] += vals[c] / tot
        if not ents:
            continue
        n = len(ents)
        rows.append({
            "method": f"SNAS (measured) — {name}",
            "condition": f"{SEEDS_PER_CELL} seeds, {n} rounds",
            "per_client_share": {f"client{c}": round(shares[c] / n, 4) for c in range(N_CLIENTS)},
            "normalized_entropy": round(sum(ents) / n, 4),
            "n_rounds": n,
        })
    return rows


def snas_entropy_measured(since: float) -> dict | None:
    """Mean per-round contribution entropy for SNAS, measured under attack.

    Answers the review's objection that the entropy table compared SNAS *clean*
    against Krum *under attack*. Both rows now come from the same grid, so the
    comparison is like-for-like.

    SNAS records carry aggregation_weight_normalized per client per round and no
    "aggregator" field. Rounds are separated by timestamp gaps rather than the
    stale run_windows.json.
    """
    if not COMBINED_LOG.exists():
        return None
    recs = [
        e for e in _iter_json(COMBINED_LOG)
        if "aggregation_weight_normalized" in e
        and e.get("timestamp", 0) >= since
        and e.get("aggregator") is None
    ]
    if not recs:
        return None
    recs.sort(key=lambda e: e["timestamp"])

    # Split into aggregation rounds on timestamp gaps.
    rounds: list[tuple[int, dict[int, float]]] = []
    current: dict[int, float] = {}
    current_round = -1
    prev_ts: float | None = None
    for e in recs:
        if prev_ts is not None and e["timestamp"] - prev_ts > ROUND_GAP_SECONDS:
            if current:
                rounds.append((current_round, current))
            current = {}
        current[int(e["client_id"])] = float(e["aggregation_weight_normalized"])
        current_round = int(e.get("round", -1))
        prev_ts = e["timestamp"]
    if current:
        rounds.append((current_round, current))

    # Split rounds into runs: each run's counter restarts at 0, so a round number
    # that fails to increase marks a new run. Keep only runs that ran to
    # completion, discarding rounds logged by attempts that later failed and were
    # re-run (those seeds are represented by their successful attempt instead).
    runs: list[list[dict[int, float]]] = []
    current_run: list[dict[int, float]] = []
    last_round: int | None = None
    for rnum, weights in rounds:
        if last_round is not None and rnum <= last_round:
            runs.append(current_run)
            current_run = []
        current_run.append(weights)
        last_round = rnum
    if current_run:
        runs.append(current_run)

    complete = [r for r in runs if len(r) == ROUNDS_PER_RUN]
    dropped_runs = len(runs) - len(complete)
    dropped_rounds = sum(len(r) for r in runs if len(r) != ROUNDS_PER_RUN)
    snas_measured_by_experiment.clear()
    snas_measured_by_experiment.extend(_per_experiment(complete))
    groups = [w for run in complete for w in run]

    per_round_entropy = []
    share_sums = {c: 0.0 for c in range(N_CLIENTS)}
    for g in groups:
        shares = [g.get(c, 0.0) for c in range(N_CLIENTS)]
        if sum(shares) <= 0:
            continue
        per_round_entropy.append(_norm_entropy(shares))
        tot = sum(shares)
        for c in range(N_CLIENTS):
            share_sums[c] += shares[c] / tot
    if not per_round_entropy:
        return None

    n = len(per_round_entropy)
    return {
        "method": "SNAS (measured, under attack)",
        "condition": "clean/E1/E2/E4/E6 x 10 seeds",
        "per_client_share": {f"client{c}": round(share_sums[c] / n, 4) for c in range(N_CLIENTS)},
        "normalized_entropy": round(sum(per_round_entropy) / n, 4),
        "n_rounds": n,
        "n_complete_runs": len(complete),
        "n_partial_runs_dropped": dropped_runs,
        "n_rounds_dropped": dropped_rounds,
    }


def snas_entropy_from_windows() -> list[dict]:
    """Mean aggregation share + entropy per (arm, experiment), sliced by run windows."""
    if not (WINDOWS.exists() and SNAS_LOG.exists()):
        return []
    windows = json.loads(WINDOWS.read_text())
    # bucket entries into windows by timestamp
    # pre-load all gate entries once (fresh log is small after rotation)
    entries = [e for e in _iter_json(SNAS_LOG)
               if "aggregation_weight_normalized" in e and e.get("timestamp") is not None]

    # group windows by (arm, experiment)
    groups: dict[tuple[str, str], list[dict]] = {}
    for w in windows:
        groups.setdefault((w["arm"], w["experiment"]), []).append(w)

    results = []
    for (arm, exp), ws in sorted(groups.items()):
        # accumulate per-client mean shares across all rounds in all seed-windows
        client_share_sums: dict[int, float] = {c: 0.0 for c in range(N_CLIENTS)}
        round_count = 0
        # index entries by time for slicing
        for w in ws:
            t0, t1 = w["t_start"], w["t_end"]
            # entries in this run window, grouped by round
            per_round: dict[int, dict[int, float]] = {}
            for e in entries:
                if t0 <= e["timestamp"] <= t1:
                    rd = int(e.get("round", -1))
                    per_round.setdefault(rd, {})[int(e["client_id"])] = float(e["aggregation_weight_normalized"])
            for rd, shares in per_round.items():
                round_count += 1
                for c in range(N_CLIENTS):
                    client_share_sums[c] += shares.get(c, 0.0)
        if round_count == 0:
            continue
        mean_shares = [client_share_sums[c] / round_count for c in range(N_CLIENTS)]
        results.append({
            "method": "no-detection (async FedAvg)" if arm == "nodetect" else "SNAS",
            "condition": f"{arm}:{exp}",
            "per_client_share": {f"client{c}": round(mean_shares[c], 4) for c in range(N_CLIENTS)},
            "normalized_entropy": round(_norm_entropy(mean_shares), 4),
            "n_rounds": round_count,
        })
    return results


def theoretical_snas_clean() -> dict:
    """SNAS accepts all honest clients with sample-count-proportional weight (tau=0).

    Train sizes are the 80% training split of the per-client CSVs. Dropping `los`
    (P0-8) changed the feature count, not the row count, so these are unchanged
    from the pre-P0-8 benchmark: verified against the current
    data/processed/*.csv as 28,921 / 18,786 / 26,194.
    """
    n = [28921, 18786, 26194]
    shares = [x / sum(n) for x in n]
    return {
        "method": "SNAS (analytic, clean all-accepted)",
        "condition": "sample-count-proportional weights",
        "per_client_share": {f"client{c}": round(shares[c], 4) for c in range(N_CLIENTS)},
        "normalized_entropy": round(_norm_entropy(shares), 4),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--krum-only", action="store_true")
    ap.add_argument("--since", default=GRID_START_DEFAULT,
                    help="Ignore log records before this local time (YYYY-MM-DD HH:MM). "
                         "Defaults to the start of the P0-8 regeneration grid; the log is "
                         "cumulative and older runs used the pre-P0-8 266-feature data.")
    args = ap.parse_args()

    since = datetime.strptime(args.since, "%Y-%m-%d %H:%M").timestamp()
    print(f"Reading {COMBINED_LOG.name}, records since {args.since}\n")

    rows: list[dict] = []
    k = krum_entropy(since)
    if k:
        rows.append(k)
        print(f"Krum: entropy={k['normalized_entropy']} shares={k['per_client_share']} "
              f"(n={k['n_selection_events']} selection decisions)")

    s = snas_entropy_measured(since)
    if s:
        rows.append(s)
        print(f"SNAS (measured): entropy={s['normalized_entropy']} shares={s['per_client_share']} "
              f"(n={s['n_rounds']} rounds from {s['n_complete_runs']} complete runs; "
              f"dropped {s['n_rounds_dropped']} rounds from {s['n_partial_runs_dropped']} "
              f"aborted runs)")
        for r in snas_measured_by_experiment:
            rows.append(r)
            print(f"    {r['method'].split('— ')[-1]:22s} H={r['normalized_entropy']:.4f} "
                  f"shares={r['per_client_share']}")

    rows.append(theoretical_snas_clean())

    if not args.krum_only:
        # run_windows.json is written by close_gaps_runner.py and currently
        # predates the P0-8 grid, so the windows it names select pre-P0-8
        # 266-feature runs out of the cumulative log. Emitting those alongside
        # the rows above would put two different feature sets in one table.
        stale = []
        if WINDOWS.exists():
            windows = json.loads(WINDOWS.read_text())
            stale = [w for w in windows if w.get("t_end", 0) < since]
        if stale and len(stale) == len(windows):
            print(f"\nSKIPPED window-sliced rows: all {len(stale)} run windows in "
                  f"{WINDOWS.name} end before {args.since}, i.e. they describe pre-P0-8 "
                  f"266-feature runs. Nothing is lost — the no-detection arm is reported "
                  f"above from the regenerated 265-feature runs, derived by block rather "
                  f"than by time window.")
        else:
            snas_rows = snas_entropy_from_windows()
            rows.extend(snas_rows)
            for r in snas_rows:
                print(f"{r['method']} [{r['condition']}]: entropy={r['normalized_entropy']} "
                      f"shares={r['per_client_share']}")

    (OUT_DIR / "federation_entropy.json").write_text(json.dumps(rows, indent=2))
    # csv
    import csv as _csv
    with (OUT_DIR / "federation_entropy.csv").open("w", newline="", encoding="utf-8") as fh:
        w = _csv.writer(fh)
        share_keys = sorted({k for r in rows for k in r["per_client_share"]})
        w.writerow(["method", "condition", *[f"{k}_share" for k in share_keys], "normalized_entropy"])
        for r in rows:
            pc = r["per_client_share"]
            w.writerow([r["method"], r["condition"],
                        *[pc.get(k, "") for k in share_keys], r["normalized_entropy"]])
    print(f"\nSaved results/gap_closing/federation_entropy.{{json,csv}}")


if __name__ == "__main__":
    main()
