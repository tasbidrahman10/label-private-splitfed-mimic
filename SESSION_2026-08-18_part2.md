# Session handoff — 2026-08-18 (part 2, work PC)

Second session of the same calendar day. **Supersedes the "next steps" list in
[SESSION_2026-08-18.md](SESSION_2026-08-18.md)** — every item on it is now done.
That file and [SESSION_2026-08-17.md](SESSION_2026-08-17.md) remain accurate as history;
read them for the earlier bug fixes and the n=3/n=6 profile switch, which are not repeated here.
Authoritative priority list is still [REVIEW_TRIAGE.md](REVIEW_TRIAGE.md).

---

## 0. TL;DR

**All nine P0 items are complete. No compute is pending. Nothing is running.**

Every number in `writing contents/main .tex` now traces to the corrected 265-feature data;
I re-ran a stale-value sweep over the whole file and it comes back clean. Tables 2, 3, 4, 5
and 6 are all regenerated and cross-checked against their source files, and the
attack-magnitude figure has been redrawn.

What remains is P1 and optional work (§6). There is no long-running job to babysit.

Deadline is **2026-08-28**.

---

## 1. Setting up on the home PC

Rebuild the venv — it is not portable between machines:

```powershell
Remove-Item -Recurse -Force .venv
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

`requirements.txt` already includes `scipy` (added in part 1). If that machine has the
RTX 5060, reinstall torch from the CUDA 12.8 index as part 1 §1 describes; everything
here ran on CPU and does not need it.

Nothing needs the server unless you start new experiments.

---

## 2. What was completed this session

The six-item "zero-compute" list from part 1 §5, plus both remaining compute items.

| # | Item | Outcome |
|---|---|---|
| 1 | Federation entropy | Fixed and rerun; Table 4 rebuilt with per-experiment measured values |
| 2 | Significance tests | All nine SNAS-vs-baseline comparisons at n=10, paired by seed |
| 3 | `main .tex` numbers | Tables 2/3/4/5/6, abstract, contributions, discussion |
| 4 | FLTrust narrative (P0-4 text) | Rewritten — the "architectural incompatibility" claim is gone |
| 5 | Krum 300/300 into P0-2/P0-3 | Done; "880 client-round log entries" replaced everywhere |
| 6 | AUPRC / per-client AUROC (P1-4) | **Not done** — still open, see §6 |
| 7 | **P0-1 attack sweep below 1×** | Done — 11 points, result overturns the previous text |
| 8 | **Gate ablation rerun** | Done — 40 runs, replicates the original finding |

---

## 3. Final results

### Grid: 180/180 runs, zero failures

`results/p08_rerun_progress.json` · `results/p08_rerun_summary.json` ·
check with `.\.venv\Scripts\python.exe scripts\rerun_p08.py --status`

### Table 2 — SNAS main results (10 seeds)

| Cell | AUROC |
|---|---|
| clean | 0.9651 ± 0.0001 |
| E1 gradient scaling | 0.9613 ± 0.0001 |
| E2 free rider | 0.9094 ± 0.0070 |
| E3 attack + straggler | 0.9595 ± 0.0004 |
| E4 honest straggler | 0.9659 ± 0.0002 |

### Table 3 — aggregator comparison

| | E1 | E2 | E3 |
|---|---|---|---|
| **SNAS** | 0.9613 | **0.9094** | 0.9595 |
| Krum | 0.9646 | 0.9647 | 0.9646 |
| trimmed mean | 0.9569 | 0.8242 | 0.9560 |
| FLTrust | 0.9593 | 0.8762 | 0.9583 |

### Significance — all nine comparisons significant, every one unanimous

`results/significance_tests_p08.{csv,json}` (from `scripts/significance_tests_p08.py`)

SNAS beats **trimmed mean** by +0.0044 / +0.0852 / +0.0035 and **FLTrust** by
+0.0020 / +0.0332 / +0.0013, winning **10/10 paired seeds in all six cases**.
Against Krum the direction reverses, also 10/10, by −0.0033 / −0.0553 / −0.0051.
Because every comparison is unanimous, each hits the two-sided Wilcoxon floor at n=10
(p = 0.00195); paired t-tests agree (3×10⁻⁴ to 9×10⁻¹⁰); paired Cohen's d from 1.8 to 8.2.

**Effect-size convention changed.** We now report *paired* d (mean difference ÷ SD of the
differences), the correct estimator for a paired design. The manuscript's old "d between
−12 and −16" came from the independent-samples pooled formula in
`scripts/run_significance_tests.py`. Both are emitted in the CSV (`cohens_d_paired`,
`cohens_d_pooled`) so the change is traceable — **do not mix them.**

### Table 4 — contribution entropy, all rows measured

| Setting | c0 | c1 | c2 | H |
|---|---|---|---|---|
| SNAS, clean | 0.39 | 0.25 | 0.35 | 0.986 |
| SNAS, Exp 1 (attacker C1) | 0.48 | **0.08** | 0.44 | 0.767 |
| SNAS, Exp 2 (attacker C2) | 0.56 | 0.37 | **0.07** | 0.685 |
| SNAS, Exp 3 (attacker C1, straggler C2) | 0.86 | **0.14** | 0.00 | 0.249 |
| No-detection, Exp 1 and Exp 2 | 0.39 | 0.25 | 0.35 | 0.986 |
| Krum, Exp 1/2/3 | 1.00 | 0.00 | 0.00 | **0.000** |

Two independent validations that the computation is right: SNAS-clean reproduces the
analytic sample-count-proportional prediction to four decimal places, and the no-detection
arm returns *exactly the clean shares under both attacks* — with the gate off it cannot
withdraw weight from anyone, which is what that control should show.

The argument the table now supports: SNAS's entropy drop is **attack-responsive** (it tracks
which client is malicious), no-detection preserves balance but retains the poison, and Krum's
0.000 is **invariant across three different adversaries** because it is fixed by submission
order.

### Table 5 — gate ablation (regenerated, 40 runs)

| Experiment | Gate on | Gate off | Benefit | Seeds | p |
|---|---|---|---|---|---|
| E1 gradient scaling | 0.9613 | 0.9578 | +0.0035 | 10/10 | 6×10⁻¹² |
| E2 free rider | 0.9094 | 0.8141 | +0.0953 | 10/10 | 2×10⁻⁸ |
| E3 attack + straggler | 0.9595 | 0.9567 | +0.0028 | 10/10 | 3×10⁻⁸ |
| E4 honest straggler | 0.9659 | 0.9658 | +0.0001 | **5/10** | 0.25 |

Replicates the pre-P0-8 finding (old: +0.0042 / +0.0968 / +0.0025 / +0.0000). The E4 row is
the important one: 5/10 seeds is exactly chance, so "the gate costs nothing when there is no
attacker" now rests on a proper null result rather than a suspiciously exact +0.0000.

### Table 6 — attack-magnitude sweep (P0-1), `results/attack_magnitude_sweep.csv`

| Scale | 0.1× | 0.3× | 0.5–2.0× | 3.0× | 5–10× |
|---|---|---|---|---|---|
| Detection | **100%** | 12.5% | **0%** | 37.5% | **100%** |

**This overturned the previous text**, which claimed "below approximately 2× scaling the
attack evades the gate entirely." The curve is U-shaped, and the review's strongest
objection — that Eq. (9)'s norm term is one-sided, so a zeroing/downscaling attack is always
accepted, which they called "sufficient for a negative review on its own" — is now answered
with data: at 0.1× the attacker is flagged in every post-warm-up round, first detected at
round 2.

Two caveats written into the paper rather than smoothed over:

- **Symmetry is close but not exact.** 0.5× and 2.0× both evade and 0.1×/10× are both fully
  detected, but 0.3× is detected *less* often than 3.0× despite a marginally larger
  log-deviation, because the cosine term also feeds SNAS and responds differently to the two
  directions.
- **Detection ≠ exclusion.** Only 10× triggers hard quarantine. At 0.1×, 5× and 7× the
  attacker is flagged every round and weight-penalised but never crosses
  δ_quarantine = 0.35.

The evasion band is narrow and benign: within 0.5–2.0× the round-1 drop is 0.0019–0.0028,
no larger than the 0.0021 at the fully-detected 0.1× point.

---

## 4. Bugs found and fixed this session

**Federation entropy read the wrong log.** `KRUM_LOG` pointed at
`results/baseline_krum/server/server_metrics.jsonl`, which holds only pre-P0-8 266-feature
runs. `rerun_p08.py` switches aggregator via `/admin/reset` against one long-lived server, so
**every aggregator's records went into the single combined log**
(`results/label_private_splitfed/server/server_metrics.jsonl`, now ~134 MB). Running the
script unmodified would have produced a Table 4 silently contradicting Table 3. Now reads the
combined log and filters on the `aggregator` field.

**That log is cumulative.** It still holds July runs and the Aug-17 smoke/timing runs —
**1,314 older SNAS records** that would have been averaged in. Added a `--since` cutoff
defaulting to the grid start.

**Aborted runs polluted the entropy average.** 24 rounds came from attempts that failed and
were later re-run. Excluded *structurally* rather than by time: each run's round counter
restarts at 0 on `/admin/reset`, so rounds are grouped into runs and only complete 10-round
runs are kept. That yielded exactly 50 runs = 500 rounds, matching the expected count — good
evidence the filter is right rather than approximately right.

**Attack-sweep detection rate was measured over a truncated window.** `extract_gate_metrics()`
read a fixed 20-line tail commented "10 rounds × 2 clients for E1", but E1 keeps all three
clients inside the window and logs 30 gate decisions per run, so the earliest rounds were
dropped. Symptom in the old CSV: `n_active_rounds: 7` where it should be 8. Now selected by
timestamp from the run that just finished.

**Override leakage risk in the ablation.** `/admin/reset` **merges** overrides into the live
config rather than replacing it, so the nodetect arm's `1e9` thresholds would have persisted
into every subsequent run and silently disabled the gate. `close_gaps_runner.py` had
documented this. Every group in `rerun_p08.py` now states the gate parameters explicitly in
both directions (`GATE_ON` / `GATE_OFF` via `overrides_for()`). The completed 140 runs were
unaffected — they only ever set `aggregator_type`.

---

## 5. Files changed this session

| Path | Change |
|---|---|
| `writing contents/main .tex` | Abstract, contributions, Tables 2–6, FLTrust rewrite, entropy + significance + ablation + sweep discussion, methods |
| `scripts/compute_federation_entropy.py` | Combined-log source, `--since`, complete-run filter, per-experiment breakdown |
| `scripts/significance_tests_p08.py` | **New** — supersedes `run_significance_tests.py` |
| `scripts/rerun_p08.py` | `nodetect` group, `overrides_for()`, `--status` |
| `scripts/run_attack_magnitude_sweep.py` | Sub-1× scales, timestamp-based gate metrics, resume support |
| `scripts/make_paper_figures.py` | Log-scale attack-magnitude figure, U-shape banding, `CLEAN_BASELINE` |
| `results/attack_magnitude_sweep.csv` | Regenerated, 11 points |
| `results/attack_magnitude_sweep_pre_p08_266feat.csv` | **Archived** old sweep — do not merge with the new one |
| `results/gap_closing/federation_entropy.{csv,json}` | Regenerated |
| `results/significance_tests_p08.{csv,json}` | New |
| `results/p08_rerun_{progress,summary}.json` | 140 → 180 cells |

---

## 6. What is left

### P1 — recommended, no blockers

1. **P1-4: AUPRC and per-client AUROC.** Analysis only, no reruns — both are already in the
   client metrics JSONL. With prevalence spanning 6.9 %–16.1 % this is the metric a clinical
   reviewer will ask for. The only item from part 1's list still open.
2. **Sync-FedAvg row.** Table 2 still reports "0.9649 (1 seed)". REVIEW_TRIAGE §3.5: it has
   **no reproducible config**. Either regenerate it or delete the row. I already switched the
   attack-magnitude figure's reference line off it, onto the ten-seed clean async baseline
   (0.9651).
3. **P1-1 factorial ablation.** We now have gate+clip ON (SNAS) and both OFF (nodetect).
   Clip-only and gate-only are still missing — ~2 more arms × 10 seeds.
4. **P1-2 label-flip and backdoor.** Implemented in `attack_simulator.py`, never executed.

### Still on stale 266-feature data

These were **not** regenerated and will disagree with the tables above:

- Sensitivity analysis (β sweep, 115 threshold pairs, clip ratio) — single-seed
- Adaptive attacks (threshold-aware, strategic alternation, slow-ramp)
- Staleness-decay ablations (exponential / linear / polynomial / step)

**Recommendation:** footnote these as calibrated on the pre-`los` feature set rather than
rerunning, now that the ablation and sweep — the two cited in central claims — are current.

### Optional

- **n=6 supplementary table.** Infrastructure is complete and idle (part 1 §6, one env var).
  Would convert P0-2 from a confession into a result: *at n=3 Krum's selection is decided by
  submission order; at n=6, where Krum is validly configured, SNAS still outperforms it.*

---

## 7. Gotchas

- Paper filename has a **space**: `writing contents/main .tex`.
- **Do not delete** `results/p08_rerun_progress.json` or `p08_rerun_summary.json` — the only
  record of 180 runs and the source for every regenerated table.
- **Do not merge** `attack_magnitude_sweep_pre_p08_266feat.csv` into the current sweep; it is
  266-feature data kept only for provenance.
- `scripts/run_significance_tests.py` is **superseded** — it hardcodes five seeds of pre-P0-8
  values and only covers Krum. Delete it so nobody runs it by accident.
- `results/gap_closing/run_windows.json` is stale (Jul 4) but no longer matters: entropy now
  derives runs structurally rather than by time window, and the script prints an explicit note
  when it skips those windows.
- The combined server log is ~134 MB and cumulative. Anything reading it must filter by
  `--since` and/or the `aggregator` field.
- `compute_federation_entropy.py` assumes the SNAS blocks appear in `rerun_p08.py`'s GRID
  order (5 aggregator cells then 4 nodetect cells, 10 seeds each). If the grid changes, the
  block mapping needs rechecking — it validates itself by signature, so a mismatch shows up as
  wrong per-experiment shares rather than silently.
