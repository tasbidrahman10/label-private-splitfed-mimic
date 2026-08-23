# Reframing option: lead with the characterization result

**Status: a proposal, not applied.** `main.tex` is untouched by this file. Decide, then
say the word and it becomes four edits: abstract, contribution list, one intro paragraph,
and optionally the title.

---

## The problem this solves

The paper is currently structured to sell SNAS. After the N=6 corrections, the evidence
does not support that sale:

| Comparison | Result |
|---|---|
| vs FLTrust | SNAS wins, N=3 **and** N=6. Real and durable. |
| vs trimmed mean | Wins at N=3 only because trimmed mean trims zero values. **Retracted at N=6.** |
| vs coordinate median | Loses on both pure attacks. Wins **only** on Exp 3, by +0.0037. |
| vs Krum | Loses on raw AUROC at N=3; Krum's win is a tie-break artifact. |
| at N=6 | 3rd of 5 on E1, 4th of 5 on E2. |

A reviewer reading this writes: *"the authors' own evidence shows the proposed method does
not outperform a coordinate-wise median except in one narrow configuration."* That is a
fair reading of the paper as currently framed, and it is close to fatal.

**But the same evidence supports a different, stronger paper.** The durable findings are
not about SNAS winning. They are about *how robust aggregation is evaluated at small N*:

1. Krum at N=3, f=1 violates N ≥ 2f+3; k clamps to 1; honest scores tie **bit-for-bit**;
   the winner is decided by submission order. Its apparent advantage is an artifact, and
   it vanishes at N=6 (300/300 → 1/50).
2. Trimmed mean at ratio 0.2 and N=3 trims ⌊0.6⌋ = 0 values and **is plain averaging**.
   Published as a "robust baseline" it is nothing of the kind. Correcting it moves the
   free-rider result by +0.134 AUROC and reverses the comparison.
3. FLTrust in SplitFed **silently degenerates to uniform averaging** unless the reference
   gradient is routed through the server head *and* taken from a retained previous-round
   encoder. Missing either half produces a plausible-looking number that is not FLTrust.
4. Update-space detection is **provably blind at n = 2**: with one peer,
   A_cos_i = 1 − cos(Δi, Δj) = A_cos_j. Attacker and honest client receive identical
   scores. Confirmed byte-identical in the logs.
5. The distinction that survives all of it: **quorum-dependent robustness degrades under
   asynchrony; anchor-referenced per-submission detection does not.** The median loses its
   guarantee in the 30% of rounds that aggregate two clients. That is a property of the
   estimator class, not of our method.

Findings 1–3 mean that low-client robust-aggregation comparisons in the literature are
likely wrong in the same way. That is a contribution independent of whether SNAS wins.

---

## Proposed abstract (~300 words)

> Split federated learning (SplitFed) pairs the label locality of split learning with
> federated aggregation, and inherits from federated learning a standard set of
> Byzantine-robust aggregation baselines. We show that in the low-client cross-silo regime
> these baselines are routinely evaluated **outside their own validity conditions**, and
> that correcting this reverses conclusions — including two of our own.
>
> We build SENTINEL-SplitFed, a server-side extension that replaces blocking aggregation
> with timed submission windows, tracks per-client staleness, and gates each encoder
> submission by a Staleness-Normalized Anomaly Score (SNAS), and use it as an instrument to
> evaluate Krum, trimmed mean, coordinate-wise median and FLTrust on a non-IID MIMIC-IV
> mortality benchmark at N = 3 and N = 6.
>
> Three baselines are mis-specified at N = 3. Krum violates N ≥ 2f+3, its neighbour count
> clamps to one, honest scores tie bit-for-bit, and the winner is fixed by submission
> order — it selects one client in 300 of 300 rounds, and at N = 6 spreads across four
> honest clients while never selecting the adversary. Trimmed mean at ratio 0.2 discards
> ⌊0.6⌋ = 0 values and is plain averaging; validly configured at N = 6 it gains 0.134 AUROC
> on the free-rider attack and **overtakes our own method**, which we retract accordingly.
> FLTrust degenerates silently to uniform averaging unless its reference gradient is both
> routed through the server head and taken from a retained previous-round encoder.
>
> What survives correction is a narrower claim about estimator class rather than about our
> method: quorum-dependent robustness degrades under asynchrony while anchor-referenced
> per-submission detection does not. The coordinate median, strongest under full
> participation, loses Byzantine tolerance entirely in the 30% of rounds that aggregate two
> clients, and update-space detection is provably blind at n = 2. SNAS detects all four
> non-adaptive attacks (8/8, Wilson 95% CI [0.68, 1.00]) at no observed false positives
> (0/16, [0.00, 0.19]), but is fourth of five at N = 6 and we say so.

---

## Proposed contribution list

Replace the current seven bullets with five:

- **A validity audit of robust-aggregation baselines at low client counts.** Krum, trimmed
  mean and FLTrust are each mis-specified at N = 3 in a way that produces publishable-looking
  but meaningless numbers. We give the arithmetic, the corrected measurements at N = 6, and
  the magnitude of the resulting error (up to 0.134 AUROC).
- **A negative result on our own method, reported in full.** SNAS's largest claimed
  advantage does not survive a valid trimmed-mean configuration and is retracted at N ≥ 6.
  We report the reversal, its mechanism (SNAS pays a two-round warm-up cost that order
  statistics do not), and the remedy we did not implement.
- **A distinction that survives correction**: quorum-dependent robustness degrades under
  asynchrony while anchor-referenced per-submission detection does not — with the median's
  breakdown measured directly (90 of 300 rounds carry no tolerance) and a proof that
  update-space detection is blind at n = 2.
- **SNAS itself**, a staleness-normalized anomaly gate for the SplitFed encoder-aggregation
  boundary, with calibration separated from evaluation, Wilson intervals on every rate, a
  2×2 factorial ablation, and family-wise corrected significance.
- **A characterization of where it fails**: attack-then-abstain evasion, no single quarantine
  threshold separating direction- from magnitude-driven attacks, and a disabled
  activation channel that bounds the split-learning-native claim.

---

## Optional title change

Current:
> SENTINEL-SplitFed: Staleness-Aware Encoder Anomaly Detection for Intrusion-Resilient
> Asynchronous Split Federated Learning in Critical-Care Prediction

Alternative:
> When Robust Aggregation Isn't: Validity Conditions, Asynchrony, and Encoder Anomaly
> Detection in Low-Client Split Federated Learning

The alternative matches the evidence better and is more likely to interest a reviewer.
It is also a bigger change — co-authors, and possibly the submission record. **Recommend
keeping the current title and reframing only the abstract and contributions**, which
captures most of the benefit at none of the coordination cost.

---

## What does NOT change

Every number, table, figure and section of Results and Discussion stays exactly as it is.
This is a framing change only — the paper already contains all the evidence the reframe
describes. Sections 6.3, 7.2 and 7.4 in particular already argue it; the abstract and
introduction simply do not currently lead with it.
