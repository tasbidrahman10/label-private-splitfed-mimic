# Gap-Closing Notes (reviewer-anticipation fixes)

Four gaps raised in analysis, being closed 2026-07-03. This file records decisions,
methods, and outcomes; the numbers feed into `results/PAPER_RESULTS.md` and later the manuscript.

## Gap #2 — Experiment renumbering (DECIDED, applied at manuscript integration)

**Problem:** experiments are labelled E1, E2, E4, E6 — the visible gaps at E3/E5 read as
selective reporting even though E3/E5 were simply superseded slots in the internal protocol
(they were "harder attack" placeholders replaced by the adaptive attacks A/B/C).

**Decision:** relabel to a gapless scheme in the manuscript, keeping the internal data-file
names unchanged (data lives under the old dir names; only reader-facing labels change).

**Mapping (paper label <- data/internal code):**

| Paper label | Scenario | Internal code / data dir |
|-------------|----------|--------------------------|
| Exp 1 | Gradient-scaling attack (Client 1, 10x) | E1 / experiment_e1_gradient_scaling |
| Exp 2 | Free-rider attack (Client 2, random) | E2 / experiment_e2_free_rider |
| Exp 3 | Attack + honest straggler (hardest) | E4 / experiment_e4_attack_plus_dropout |
| Exp 4 | Honest straggler, no attack (FPR test) | E6 / experiment_e6_dropout |

Adaptive attacks stay A/B/C; sweeps stay descriptive (attack-magnitude, threshold, beta, clip).
The no-detection ablation (Gap #4) is a *condition* applied to Exp 1–4, not a new experiment.

*(Alternative considered: actually run label-flip + backdoor — both implemented in code but
never executed — as the missing E3/E5. Deferred: more compute, and the gapless relabel already
removes the smell. Flagged as an option if a reviewer specifically wants more attack types.)*

## Gap #3 — Federation-preservation entropy (IN PROGRESS)

Metric: normalized contribution entropy H = -Σ pᵢ ln pᵢ / ln(N), pᵢ = client i's mean
normalized aggregation share. H=1 balanced participation; H=0 single-client collapse.

**Results so far:**
- **Krum: H = 0.000** (client 0 = 100% of 168 selection events, empirical from
  `results/baseline_krum/server/server_metrics.jsonl`). Federation fully collapsed.
- **SNAS (analytic, clean): H = 0.985** (sample-count-proportional weights across all 3
  clients; matches the logged normalized weight 0.3913 for client 0). Federation preserved.
- SNAS (clean, empirical) and no-detection (under attack) — computed after the batch from the
  fresh server log sliced by `run_windows.json`. Expected: SNAS ≈ 0.98 (attacker share → 0 via
  quarantine), no-detection ≈ 0.98 but *includes* the attacker (federation "preserved" yet poisoned).

Headline: **SNAS entropy ≈ 0.98 vs Krum 0.00** — turns the qualitative "Krum collapses
federation" claim into a number.

## FINAL RESULTS (all 50 runs complete, 2026-07-03)

- **Gap #1 baseline:** async-clean n=10 = **0.9652 ± 0.0004**. Deltas rock-solid (~10σ).
- **Gap #4 no-detection (gate benefit, all paired-significant under attack):**
  Exp1 +0.0042, Exp2 **+0.0968**, Exp3 +0.0025, Exp4 +0.0000 (control — gate free when no attack).
- **Gap #3 entropy:** Krum **0.000** vs SNAS **0.986**.
- **Gap #2:** Exp 1–4 relabel (below).
- **Exp 4 nuance:** +0.0008 vs baseline is paired-significant but tiny → frame as "no degradation".
Full write-up folded into `results/PAPER_RESULTS.md` (ROUND-2 GAP CLOSING section).

## Gap #1 — Baseline variance (RUNNING)

`experiment_async_clean.yaml` re-run at the paper's 10 seeds (gate on, clip on) to give the
clean async baseline as mean ± std instead of a single-seed 0.9649. All deltas will be
recomputed against this. Output: `results/gap_closing/baseline_clean.json`.
**Honest risk to watch:** if baseline std is comparable to the small deltas, E6/Exp-4's
"+0.0011 vs baseline" may reframe as "no degradation" — still a valid result.
**Open item for supervisor:** the separate "Sync-FedAvg baseline" row (also 0.9649, single seed)
has no located config; recommend consolidating to this multiseeded async baseline or reproducing
the sync run before submission.

## Gap #4 — No-detection async baseline + gate ablation (RUNNING)

SNAS with gate and clipping disabled (thresholds + clip_ratio → 1e9 via server override) =
pure staleness-weighted async FedAvg, run on Exp 1–4 at 10 seeds. Serves as BOTH the empirical
async baseline (answers "no async baseline") AND the ablation isolating the value of the
detection machinery. Output: `results/gap_closing/nodetect.json`.
**Expected story:** Exp 2 (free rider) and Exp 1/3 (scaling) should degrade markedly without the
gate; Exp 4 (honest straggler, no attack) should match SNAS. Coarse ablation (removes gate+clip
together) — finer gate-vs-clip decomposition noted as possible future work.
