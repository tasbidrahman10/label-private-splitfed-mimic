"""
Krum's selection distribution at N=6, from the n=6 server log.

Table `tab:n6-krum` reports which client Krum selected in each aggregation
round of the n=6 E1 grid. Every other table in the paper is backed by a
committed artifact under results/ (Table 4 by gap_closing/federation_entropy.csv,
Tables 2/3/5 by p08_rerun_summary.json); this one was not, because its only
source is a `server_metrics*.jsonl` and `.gitignore` excludes those as large
append-only logs. This script derives the counts so the CSV/JSON can be
committed in the log's place.

The smoke-test contamination
----------------------------
Phase 8's two-round smoke test wrote into the *same* log as the grid, so a
naive count returns 52 selection records, not the 50 the grid produced
(5 seeds x 10 rounds). Both extras are client 2. They are separated here by
the largest inter-record time gap rather than by a hardcoded timestamp, so the
rule still holds if the log is regenerated: the smoke test and the grid are
separated by ~36 minutes, while consecutive grid rounds are seconds apart.
The split is reported rather than applied silently.

Usage
-----
    python scripts/report_n6_krum_selection.py
    python scripts/report_n6_krum_selection.py --log path/to/server_metrics.jsonl

Writes results/n6_krum_selection.csv and results/n6_krum_selection.json.
"""
from __future__ import annotations

import argparse
import collections
import csv
import datetime as dt
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# The n=6 server config writes to server_n6/; the log has also been handed over
# as a copy under server/. Try both, newest first, so this works either way.
CANDIDATE_LOGS = [
    ROOT / "results" / "label_private_splitfed" / "server_n6" / "server_metrics.jsonl",
    ROOT / "results" / "label_private_splitfed" / "server" / "server_metrics_nihal_p08(1).jsonl",
]

OUT_CSV = ROOT / "results" / "n6_krum_selection.csv"
OUT_JSON = ROOT / "results" / "n6_krum_selection.json"

N_CLIENTS = 6
ATTACKER = 1          # configs/experiment_baseline_krum_e1_gradient_scaling_n6.yaml
EXPECTED_ROUNDS = 50  # 5 seeds x 10 rounds


def find_log(explicit: str | None) -> Path:
    if explicit:
        p = Path(explicit)
        if not p.is_absolute():
            p = ROOT / p
        if not p.exists():
            sys.exit(f"log not found: {p}")
        return p
    for p in CANDIDATE_LOGS:
        if p.exists():
            return p
    sys.exit(
        "No n=6 server log found. Looked for:\n  "
        + "\n  ".join(str(p) for p in CANDIDATE_LOGS)
        + "\nPass one with --log."
    )


def load_selections(path: Path) -> list[dict]:
    """Krum selection records, oldest first.

    Filters on the raw line before parsing: the log is tens of MB and only a
    few hundred records carry krum_selected.
    """
    recs = []
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if "krum_selected" not in line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("aggregator") == "krum" and r.get("krum_selected") is True:
                recs.append(r)
    recs.sort(key=lambda r: r["timestamp"])
    return recs


def split_smoke_from_grid(recs: list[dict]) -> tuple[list[dict], list[dict], float]:
    """Split at the largest time gap. Returns (smoke, grid, gap_seconds)."""
    if len(recs) < 2:
        return [], recs, 0.0
    gap, idx = max(
        (recs[i + 1]["timestamp"] - recs[i]["timestamp"], i)
        for i in range(len(recs) - 1)
    )
    return recs[: idx + 1], recs[idx + 1:], gap


def stamp(ts: float) -> str:
    return dt.datetime.fromtimestamp(ts).isoformat(timespec="seconds")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--log", default=None, help="path to the n=6 server_metrics jsonl")
    args = ap.parse_args()

    log = find_log(args.log)
    recs = load_selections(log)
    if not recs:
        sys.exit(f"no krum selection records in {log}")

    smoke, grid, gap = split_smoke_from_grid(recs)
    counts = collections.Counter(r["client_id"] for r in grid)
    n = len(grid)

    rows = [
        {
            "client_id": c,
            "role": "attacker" if c == ATTACKER else "honest",
            "selections": counts.get(c, 0),
            "share": round(counts.get(c, 0) / n, 4),
        }
        for c in range(N_CLIENTS)
    ]

    payload = {
        "source_log": str(log.relative_to(ROOT)).replace("\\", "/"),
        "experiment": "E1_gradient_scaling",
        "n_clients": N_CLIENTS,
        "attacker_client": ATTACKER,
        "seeds": [42, 7, 123, 2024, 31415],
        "total_selection_records_in_log": len(recs),
        "smoke_test_records_excluded": len(smoke),
        "split_gap_seconds": round(gap, 1),
        "grid_window": {"start": stamp(grid[0]["timestamp"]),
                        "end": stamp(grid[-1]["timestamp"])},
        "grid_selection_decisions": n,
        "distinct_clients_selected": len({r["client_id"] for r in grid}),
        "attacker_ever_selected": any(r["client_id"] == ATTACKER for r in grid),
        "per_client": {str(r["client_id"]): r["selections"] for r in rows},
    }

    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with OUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"source log      : {payload['source_log']}")
    print(f"records in log  : {len(recs)}")
    print(f"smoke excluded  : {len(smoke)} (split at a {gap / 60:.1f} min gap)")
    print(f"grid decisions  : {n}"
          + ("" if n == EXPECTED_ROUNDS else f"  [expected {EXPECTED_ROUNDS}]"))
    print()
    print(f"{'client':>7s} {'role':>9s} {'selections':>11s} {'share':>7s}")
    print("-" * 38)
    for r in rows:
        print(f"{r['client_id']:>7d} {r['role']:>9s} {r['selections']:>11d} {r['share']:>7.1%}")
    print("-" * 38)
    print(f"distinct clients selected : {payload['distinct_clients_selected']}")
    print(f"attacker ever selected    : {payload['attacker_ever_selected']}")
    print(f"\nWrote {OUT_CSV.relative_to(ROOT)}\n      {OUT_JSON.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
