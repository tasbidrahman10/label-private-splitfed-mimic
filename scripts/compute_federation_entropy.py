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
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "gap_closing"
OUT_DIR.mkdir(parents=True, exist_ok=True)

KRUM_LOG = ROOT / "results" / "baseline_krum" / "server" / "server_metrics.jsonl"
SNAS_LOG = ROOT / "results" / "label_private_splitfed" / "server" / "server_metrics.jsonl"
WINDOWS = OUT_DIR / "run_windows.json"
N_CLIENTS = 3


def _norm_entropy(shares: list[float], n: int = N_CLIENTS) -> float:
    tot = sum(shares)
    if tot <= 0:
        return float("nan")
    p = [s / tot for s in shares]
    h = -sum(pi * math.log(pi) for pi in p if pi > 0)
    return h / math.log(n) if n > 1 else 0.0


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


def krum_entropy() -> dict | None:
    """Selection frequency -> contribution share for Krum (client selected == full weight)."""
    if not KRUM_LOG.exists():
        return None
    sel_counts: dict[int, int] = {}
    rounds = set()
    for e in _iter_json(KRUM_LOG):
        if "krum_selected" not in e:
            continue
        rounds.add((e.get("round"), e.get("timestamp")))
        if e.get("krum_selected"):
            cid = int(e["client_id"])
            sel_counts[cid] = sel_counts.get(cid, 0) + 1
    if not sel_counts:
        return None
    total = sum(sel_counts.values())
    shares = [sel_counts.get(c, 0) / total for c in range(N_CLIENTS)]
    return {
        "method": "Krum",
        "condition": "all (E1/E2/E4, 10 seeds)",
        "per_client_share": {f"client{c}": round(shares[c], 4) for c in range(N_CLIENTS)},
        "normalized_entropy": round(_norm_entropy(shares), 4),
        "n_selection_events": total,
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
    Train sizes from the dataset summary: Medical 28921, Surgical 18786, Cardiac 26194."""
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
    args = ap.parse_args()

    rows: list[dict] = []
    k = krum_entropy()
    if k:
        rows.append(k)
        print(f"Krum: entropy={k['normalized_entropy']} shares={k['per_client_share']} "
              f"(n={k['n_selection_events']} selections)")
    rows.append(theoretical_snas_clean())

    if not args.krum_only:
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
        w.writerow(["method", "condition", "client0_share", "client1_share", "client2_share", "normalized_entropy"])
        for r in rows:
            pc = r["per_client_share"]
            w.writerow([r["method"], r["condition"], pc["client0"], pc["client1"], pc["client2"], r["normalized_entropy"]])
    print(f"\nSaved results/gap_closing/federation_entropy.{{json,csv}}")


if __name__ == "__main__":
    main()
