# Submission readiness — problems, evidence, and the decision to make

**Target:** Springer *Machine Learning* · **Deadline:** 2026-08-28 (5 days from today, 2026-08-23)
**Manuscript:** [writing contents/main.tex](writing%20contents/main.tex) (840 lines)
**Companion to:** [REVIEW_TRIAGE.md](REVIEW_TRIAGE.md) (P0/P1/P2 list, all P0+P1 closed)

This file exists so you can decide the framing question before any more text is written.
Nothing here needs new experiments.

---

## 0. TL;DR

An external review raised eight concrete problems. **I verified every one of them against the
code, the artifacts, and the manuscript. All eight are correct** — including one where my own
earlier report to you was wrong (§1.1).

The problems split cleanly:

| Tier | What | Count | Cost to fix | Needs your decision? |
|---|---|---|---|---|
| **A** | Framing — could cause rejection | 4 | 1–4 h | **Yes** |
| **B** | Concrete errors — unambiguous | 7 | ~2 h total | No |
| **C** | Known limitations, already disclosed | 5 | 0 | No |

**The single most dangerous item is A1** (abstract oversells relative to the body). It is also
one of the cheapest to fix. Everything in Tier B I can fix without further input.

---

## 1. Verification of the review's claims

Every claim was checked against a specific artifact, not accepted on assertion.

| # | Claim | Verdict | Evidence |
|---|---|---|---|
| 1 | Author contribution statement commented out | **CORRECT** | line 751 begins `% \bmhead{Author contribution}` |
| 2 | "Round 2" contradicts the rounds 1–2 warm-up | **CORRECT** | 5 table cells: lines 563, 628, 636, 637, 638 |
| 3 | Contribution entropy doesn't recompute | **CORRECT** | shares give H = 0.833 / 0.800 / 0.369; table states 0.767 / 0.685 / 0.249 |
| 4 | Intro overclaims the gate's benefit | **CORRECT** | line 67 says "gate recovers up to 0.097"; that is gate **+ clip**. Gate alone = +0.0685 |
| 5 | Line 379 AUROC/AUPRC baseline ambiguity | **CORRECT** | "0.7008 AUROC and 0.2459 AUPRC against a clean baseline of 0.8475" — 0.8475 is the AUPRC baseline |
| 6 | AFLGuard / FedAsync / FedBuff never compared | **CORRECT** | mentioned 4 / 2 / 3 times in prose, **0 times** in `tab:baselines` |
| 7 | Abstract too long | **CORRECT — worse than stated** | ~**521 words** (review estimated ~450; Springer norm is ≤250) |
| 8 | Unverifiable citations | **CORRECT** | see §3.7 |

### 1.1 A correction to my earlier report

I previously told you all six Springer declarations were present. **That was wrong.** My grep
matched the declaration text *inside* a comment. `Author contribution` is commented out, and
Springer requires it. Corrected here so the record is accurate.

---

## 2. Tier A — the four framing problems

These are what determine accept/reject. Three are writing; one is a scope decision.

### A1. The abstract oversells relative to the body — **highest risk in the package**

The abstract claims SNAS is *"significantly better than both trimmed mean and FLTrust on every
attack setting."* Both halves are a problem:

- **Coordinate median is never mentioned** — and §5.3 states plainly that it beats SNAS on 2 of
  3 attacks, 10/10 seeds, paired *d* = −13.3.
- **Exp 5 / Exp 6 are never mentioned** — the backdoor is the most damaging attack in the study
  (0.7008 AUROC / 0.2459 AUPRC).
- **Trimmed mean at N=3 *is* FedAvg** by the paper's own analysis (`floor(0.2 × 3) = 0`), so the
  claim reduces to "we beat FedAvg."

**Why this is the dangerous one:** the body is scrupulously honest. But a reviewer reads the
abstract first. When they reach §5.3 and find the losing results that the abstract omitted, the
honesty of the body starts reading as *"the abstract was hiding something."* That converts a
fixable paper into a reject.

**Fix cost:** ~1 h. Rewrite the abstract to match the body and cut it to ~250 words.

### A2. The results refute the headline claim

Taken together, the paper's own evidence says:

| Finding | Where |
|---|---|
| Coordinate median beats SNAS on 2 of 3 attacks, 10/10 seeds | §5.3 |
| Backdoor destroys accuracy (0.7008) *despite* 100 % detection | Exp 6 |
| Alternating attacker evades 100 % of the time | `tab:adaptive` |
| At N=6 — the only correctly-specified configuration — SNAS is 3rd of 5 | §5.4 |

What survives as a clean SNAS win is: **+0.0037 AUROC on the one experiment where a client
straggles.** A reviewer will write *"the proposed method is outperformed by a one-line
baseline."*

This is not a data problem — the data is fine and the analysis is right. It is a **framing**
problem, and it is the subject of the decision in §5.

### A3. Novelty is thin for this venue

SNAS is cosine + norm anomaly, weighted, divided by (1+τ), with two thresholds. There is no
convergence analysis, no breakdown-point result, no guarantee. *Machine Learning* generally
expects either strong theory or overwhelming empirics. The empirics here are careful but narrow
(one dataset, three clients, one task).

**Not fixable in five days.** It is a fit problem, and it is the strongest argument for the
reframe in §5.

### A4. AFLGuard is not in the baseline table

AFLGuard is the closest prior work — Byzantine-robust **asynchronous** FL — and §2.4 discusses it
at length. The empirical comparison is then against three **synchronous** aggregators. Same for
FedAsync and FedBuff. This is the first thing a reviewer will grep for.

Two ways out:

| Option | Cost | Notes |
|---|---|---|
| Justify the absence in text | ~40 min | AFLGuard/FedBuff need the server to validate a full client model against a locally-trained reference. A SplitFed server holds only encoder + head, so no such reference exists — the same constraint that forced FLTrust's adaptation (§5.2). The argument is already half-written in the paper |
| Implement it as a real baseline | ~1 day + ~10 h runs | Strongest answer; competes directly with the reframe for your remaining time |

---

## 3. Tier B — concrete errors (no decision needed, ~2 h total)

| # | Problem | Location | Fix |
|---|---|---|---|
| B1 | Author contribution commented out | line 751 | Uncomment |
| B2 | "Round 2" contradicts rounds 1–2 warm-up | lines 563, 628, 636–638 | → "Round 3" |
| B3 | Entropy doesn't recompute from displayed shares | `tab:entropy` | Add footnote: H is the **mean of per-round H**, the shares column is the **mean of per-round shares**; H is concave, so by Jensen the two differ. Without this, a reviewer writes "the authors' numbers don't reproduce" |
| B4 | Intro credits +0.097 to the gate | line 67 | It is gate **+ clip**. Gate main effect is +0.0685. §5.5 is already stricter than the intro |
| B5 | AUROC/AUPRC baseline ambiguity | line 379 | Name the metric explicitly |
| B6 | Abstract ~521 words | line 43 | Cut to ~250 (folded into A1) |
| B7 | Citations unverified | bibliography | see §3.7 |

### 3.7 Citation problems in detail

| Key | Problem |
|---|---|
| `peng2026belisa` | Key says "belisa" but the entry is *Byzantine-Robust Asynchronous FL via Feature Fingerprinting* (Shen, Peng et al.). **Key/title mismatch, and no backing PDF** in `Reference papers/` |
| `tpavsl2026` | *Stealthy Targeted Poisoning … Vertical Split Learning*. **No backing PDF** |
| `yang2025gas` | Key says 2025, entry says 2024 |
| `dou2026securesplit` | Backed by a PDF ✓, but uses "et al." in the entry — poor form for Springer |

`writing contents/CITATION_VERIFICATION.md` **does not cover any of these four.** They were never
verified.

**One unverifiable citation is disproportionately damaging** — it invites the reviewer to
question everything else. These must be confirmed against the real papers or removed.

---

## 4. Tier C — known limitations, already disclosed (no action)

These are real weaknesses, but the paper already states each one plainly. Leave them.

- **Benchmark is clinically invalid** — no observation window, row-level split with 14–17 %
  duplicate lab blocks, no held-out test set. Hence the implausible 0.9651 AUROC. Disclosed in
  Limitations item 2. Fixing it means a ~55 h regeneration of all 510 runs — not a 5-day job.
- **"Asynchronous" is sequential clients with `sleep()`** on one machine. Now disclosed in Setup.
- **FPR rests on 16 honest client-rounds** — a small denominator.
- **Adaptive experiments are single-seed.**
- **ALIE / Fang attacks missing** — now named in Limitations item 8.

---

## 5. The decision: how far to reframe

The review's central recommendation is that your strongest genuine finding **is not SNAS**. It is
this:

> Standard Byzantine-robust aggregators break in specific, diagnosable ways under low-client
> asynchronous SplitFed:
> - **Krum** collapses on a tie-break artifact (300/300 → 1/50 at N=6)
> - **Trimmed mean** degenerates to the plain mean at N=3
> - **Coordinate median**'s breakdown point hits zero at n=2 under dropout (90/300 rounds)
> - **FLTrust** silently becomes uniform averaging without update-basis + previous-round reference
> - **Update-basis detection** is provably uninformative at n=2 (symmetry proof)

Under that framing, SNAS becomes a **probe** that demonstrates the anchored-vs-quorum-dependent
distinction — and every result that currently looks like a defeat becomes supporting evidence.

### The three options

| | Option A — Full reframe | Option B — Fix abstract only | Option C — Errors only |
|---|---|---|---|
| **Work** | Retitle; rewrite abstract, contributions, intro framing, conclusion | Rewrite abstract to match body | Tier B only |
| **Time** | ~3–4 h | ~1 h | ~1 h |
| **New experiments** | None | None | None |
| **Fixes A1?** | Yes | Yes | **No** |
| **Fixes A2?** | Yes — defeats become findings | No | No |
| **Fixes A3?** | Partly — diagnostic framing is more novel than the method | No | No |
| **Risk left** | Novelty still modest; benchmark still invalid | "Outperformed by a one-line baseline" | Abstract reads as concealment |

### What does not change under any option

- No new experiments; all 510 runs stand
- Every number in the paper stays (33/33 currently reproduce exactly against artifacts)
- §5.3 (median), §5.4 (N=6), §6.1 (update basis) are already written honestly and need no edits

---

## 6. Recommendation

**Option A + justify AFLGuard in text + all of Tier B.** Roughly 5–6 hours total, no compute,
comfortably inside five days.

Reasoning:

1. **A1 must be fixed under any option** — it is the one failure mode that turns a fixable paper
   into a reject, and it costs an hour.
2. **The reframe is nearly free** because the evidence already exists and is already written up
   honestly. You are re-pointing the claims at results you already have, not producing new ones.
3. **It converts your three biggest liabilities into assets.** The median beating SNAS, the N=6
   parity, and the update-basis failure are currently three separate admissions of weakness.
   Under the diagnostic framing they are three independent confirmations of the same thesis.
4. **Novelty improves** — "here is a method" is a crowded claim; "here is how the standard
   toolbox fails in this regime, with mechanisms" is not.

### Honest expectation either way

Submitting is worth it. This is a hybrid journal, the deadline is real, and a rejection costs
only time. But go in expecting **major revision**, with novelty and benchmark validity as the two
pressure points. The reframe improves your odds; it does not make this a safe accept at
*Machine Learning*.

If you want better odds on fit alone, *Journal of Biomedical Informatics*, *IEEE JBHI*, or
Springer's *Applied Intelligence* are more natural homes for this contribution.

### Also worth knowing

`\documentclass[pdflatex,sn-mathphys-num]{sn-jnl}` produces **numbered** citations. Springer
*Machine Learning* uses **author–year**. Verify against the journal's author instructions — it is
a one-word change, but a desk-reject trigger if wrong.

---

## 7. What I have already done this session (for context)

All verified, all committed to nothing yet:

- Wrote the N=6 results into the paper (§5.4, two tables, three cross-references)
- Fixed 6 code-vs-paper contradictions: warm-up length, sticky quarantine, round indexing,
  FLTrust in the conclusion, Figure 1 plotting stale June data, 40 undisclosed excluded scores
- Added three REVIEW_TRIAGE disclosures: τ is participation not model-version, sequential
  simulation, ALIE/Fang omission
- Relocated the "split-learning-native" claim to the aggregation-object level (§3.3)
- Fixed `make_paper_figures.py` to reconcile runs against the progress file and regenerated

**Current state: 33/33 numeric claims reproduce, 28/28 citations resolve, 11/11 tables and 5/5
figures cited, LaTeX validates clean.** Everything is uncommitted and the git index is stale.
