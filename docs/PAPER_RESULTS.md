# Async-SplitFed-IR — Paper Results Log
## Springer Nature Machine Learning · Special Issue: Federated Learning for Critical Applications
## Deadline: July 15, 2026 | Seed: 42 | Batches/round: 200 | Window: 25s

---

## DATASET SUMMARY (from notebooks — exact figures)
Source: `notebooks/client{0,1,2}_model.ipynb`

| Client | Partition | Total samples | Train | Validation | Mortality rate | pos_weight |
|--------|-----------|--------------|-------|------------|---------------|------------|
| 0 | Medical ICU | 36,152 | 28,921 | 7,231 | **16.1%** | 5.0 (capped) |
| 1 | Surgical ICU | 23,483 | 18,786 | 4,697 | **11.5%** | 5.0 (capped) |
| 2 | Cardiac ICU | 32,743 | 26,194 | 6,549 | **6.9%** | 5.0 (capped) |
| **Total** | All ICU | **92,378** | **73,901** | **18,477** | 11.8% avg | — |

Input features: 266 (261 continuous clinical features scaled to [−5, 5] + 5 vasopressor binary indicators)
Train/val split: 80/20 stratified, seed=42
Non-IID characteristic: mortality rates differ substantially across partitions (16.1% vs 11.5% vs 6.9%), reflecting genuine clinical heterogeneity.

Val batches per client (batch_size=64): Medical=113, Surgical=74, Cardiac=103

Communication cost per client per round: 1,638,400 bytes (activations) + 1,638,400 bytes (gradients) = 3,276,800 bytes ≈ 3.13 MB

---

## TABLE 0A — Option A: Non-Label-Private Baseline (5 rounds, Colab run)
Source: `results/option_a/experiment_summary_option_a_20260506_175659.csv`
Protocol: Server computes loss using client-provided labels (NOT label-private).
Purpose: Comparison baseline to demonstrate privacy-utility trade-off.

| Round | Medical AUROC | Surgical AUROC | Cardiac AUROC | **Mean AUROC** |
|-------|--------------|----------------|---------------|----------------|
| 1 | 0.9546 | 0.9651 | 0.9749 | **0.9648** |
| 2 | 0.9454 | 0.9685 | 0.9753 | **0.9630** |
| 3 | 0.9454 | 0.9682 | 0.9782 | **0.9639** |
| 4 | 0.9468 | 0.9679 | 0.9784 | **0.9644** |
| 5 | 0.9472 | 0.9684 | 0.9784 | **0.9647** |

Round 5 detailed: Medical AUPRC=0.8290 F1=0.6705 | Surgical AUPRC=0.8538 F1=0.7506 | Cardiac AUPRC=0.8345 F1=0.6880

## TABLE 0B — Option B: Label-Private SplitFed (5 rounds, Colab run) — Phase 1 Result
Source: `results/option_b/` client JSONL files + `results/option_b/experiment_summary_option_b_20260506_161324.csv`
Protocol: Label-private — server never receives labels, client computes loss locally.
Server: Google Colab | Clients: Local | Communication: Cloudflare tunnel

| Round | Medical AUROC | Surgical AUROC | Cardiac AUROC | **Mean AUROC** |
|-------|--------------|----------------|---------------|----------------|
| 1 | 0.9515 | 0.9636 | 0.9749 | **0.9633** |
| 2 | 0.9448 | 0.9667 | 0.9749 | **0.9621** |
| 3 | 0.9449 | 0.9669 | 0.9775 | **0.9631** |
| 4 | 0.9467 | 0.9674 | 0.9774 | **0.9638** |
| 5 | 0.9476 | 0.9687 | 0.9781 | **0.9648** |

Per-client training loss (round 1 → round 5):
- Medical:  1.1354 → 0.5507 (loss reduced 51.5%)
- Surgical: 0.7171 → 0.3903 (loss reduced 45.6%)
- Cardiac:  0.6767 → 0.3147 (loss reduced 53.5%)

FedAvg round timing (Colab distributed): 6.34–6.86 seconds per round

### KEY PAPER FINDING — Privacy-Utility Trade-off:
Option A (non-label-private) Round 5 mean AUROC: **0.9647**
Option B (label-private)     Round 5 mean AUROC: **0.9648**
**Difference: +0.0001 in favour of label-private (within noise)**

Label-private SplitFed achieves essentially identical performance to non-label-private SplitFed
while preserving the privacy of mortality labels. This validates the privacy mechanism has
negligible impact on model quality — a core paper claim for Section 3.

## TABLE 1 — Sync-FedAvg Baseline (10 rounds, all 3 clients, synchronous)
Source: `results/label_private_splitfed/experiment_summary_option_b_20260621_142224.csv`

| Round | Medical AUROC | Surgical AUROC | Cardiac AUROC | **Mean AUROC** | Mean AUPRC |
|-------|--------------|----------------|---------------|----------------|------------|
| 1 | 0.9490 | 0.9636 | 0.9743 | **0.9623** | 0.8403 |
| 2 | 0.9430 | 0.9670 | 0.9739 | **0.9613** | 0.8275 |
| 3 | 0.9433 | 0.9666 | 0.9767 | **0.9622** | 0.8316 |
| 4 | 0.9450 | 0.9662 | 0.9762 | **0.9625** | 0.8337 |
| 5 | 0.9455 | 0.9680 | 0.9769 | **0.9635** | 0.8385 |
| 6 | 0.9460 | 0.9672 | 0.9776 | **0.9636** | 0.9380 |
| 7 | 0.9462 | 0.9684 | 0.9789 | **0.9645** | 0.8404 |
| 8 | 0.9470 | 0.9684 | 0.9779 | **0.9644** | 0.8419 |
| 9 | 0.9472 | 0.9688 | 0.9788 | **0.9649** | 0.8441 |
| 10 | 0.9476 | 0.9695 | 0.9789 | **0.9653** | 0.8438 |

**Peak mean AUROC: 0.9653 (round 10)**
Trend: Monotonic increase with minor noise. Plateau emerging rounds 8–10 (gains < 0.001/round).

---

## TABLE 2 — Async-SplitFed-IR Clean Run (10 rounds, mild lag: 0/5/10s delays)
Source: `results/async_clean/experiment_summary_option_b_20260622_162946.csv`

| Round | Medical AUROC | Surgical AUROC | Cardiac AUROC | **Mean AUROC** |
|-------|--------------|----------------|---------------|----------------|
| 1 | 0.9490 | 0.9636 | 0.9743 | **0.9623** |
| 2 | 0.9430 | 0.9670 | 0.9739 | **0.9613** |
| 3 | 0.9433 | 0.9666 | 0.9767 | **0.9622** |
| 4 | 0.9450 | 0.9662 | 0.9762 | **0.9625** |
| 5 | 0.9455 | 0.9680 | 0.9769 | **0.9635** |
| 6 | 0.9460 | 0.9672 | 0.9776 | **0.9636** |
| 7 | 0.9462 | 0.9684 | 0.9789 | **0.9645** |
| 8 | 0.9470 | 0.9684 | 0.9779 | **0.9644** |
| 9 | 0.9472 | 0.9688 | 0.9788 | **0.9649** |
| 10 | 0.9476 | 0.9695 | 0.9789 | **0.9653** |

**Key finding: Async-SplitFed-IR with mild lag achieves IDENTICAL performance to Sync-FedAvg.**
All 3 clients submitted within the 25s window every round. SNAS all null (warm-up + no reference yet).
NOTE: SNAS will be populated in re-run after fixing prev_global_encoder bug (see code fix 2026-06-22).

---

## TABLE 3 — E6: Honest Straggler (Client 2 dropout, 30s delay, no attack) ✅ COMPLETE
Source: `results/experiment_e6_dropout/experiment_summary_option_b_20260622_165332.csv`

| Round | Medical AUROC | Surgical AUROC | Cardiac AUROC* | **Mean AUROC** | FedAvg clients |
|-------|--------------|----------------|----------------|----------------|----------------|
| 1 | 0.9490 | 0.9636 | 0.9743 | **0.9623** | all (round 1 warmup) |
| 2 | 0.9471 | 0.9654 | 0.9753 | **0.9626** | 0+1 only |
| 3 | 0.9482 | 0.9672 | 0.9757 | **0.9637** | 0+1 only |
| 4 | 0.9494 | 0.9672 | 0.9753 | **0.9640** | 0+1 only |
| 5 | 0.9498 | 0.9687 | 0.9764 | **0.9650** | 0+1 only |
| 6 | 0.9511 | 0.9677 | 0.9778 | **0.9655** | 0+1 only |
| 7 | 0.9507 | 0.9692 | 0.9786 | **0.9662** | 0+1 only |
| 8 | 0.9514 | 0.9687 | 0.9771 | **0.9657** | 0+1 only |
| 9 | 0.9506 | 0.9693 | 0.9785 | **0.9661** | 0+1 only |
| 10 | 0.9511 | 0.9688 | 0.9784 | **0.9661** | 0+1 only |

*Client 2 validates locally (protocol continues) but weights NOT included in FedAvg.
Aggregation weights (2-client): Medical=0.6062, Surgical=0.3938

### E6 Key Findings:
1. **AUROC maintained and exceeds sync baseline**: E6 peak 0.9662 vs sync peak 0.9653 (Δ=+0.001)
   - 2-client FedAvg gives Medical higher weight (0.606 vs 0.391), producing marginally better global encoder
   - Difference is small (~0.001) but consistently positive across all rounds 2–10
2. **Client 2 local AUROC trend**: 0.9743 → 0.9784 (continues improving despite missing FedAvg)
   - Label-private protocol still functions: Client 2 still trains via server forward/backward
   - Client 2 loads the 2-client global encoder each round as its starting point
3. **False positive rate = 0 (architectural guarantee)**:
   - Client 2 SNAS entries in E6: ZERO (gate never evaluated for late clients)
   - Honest stragglers are excluded at the window controller level, not adversarial detection level
   - Paper claim: "timing-based exclusion is architecturally separate from SNAS detection, guaranteeing zero false quarantine risk for stragglers regardless of SNAS threshold choice"
4. **Staleness tracking**: Client 2 τ grows 1→10 across rounds (confirmed in async_controller logs)

---

## TABLE 4 — E1: Gradient Scaling Attack on Client 1 (scale=10) ✅ COMPLETE (2nd run)
Source: `results/experiment_e1_gradient_scaling/experiment_summary_option_b_20260622_174400.csv`
Config: `configs/experiment_e1_gradient_scaling.yaml`

### SNAS Design Finding (critical for paper Section 4):
Activation divergence (`alpha` component) is NON-DISCRIMINATIVE for gradient scaling:
- Gradient scaling shifts the global encoder's scale → ALL clients' activations drift equally
- act_div (round 1 of failed run): Honest=7.33, Attacker=4.72 → not separating the attacker
- Final SNAS formula: **alpha=0.0, beta=0.35, gamma=0.65** (weight-space signals only)
- Paper argument: "gradient scaling is a global-scope attack — it perturbs all clients'
  activation distributions equally. Weight-space signals (norm + cosine) are the only
  discriminative features. We set alpha=0 as an empirical finding and log activation
  divergence for analysis without using it in gating decisions."

### SNAS Configuration for all experiments:
- alpha=0.0, beta=0.35, gamma=0.65
- threshold_flag=0.22, threshold_quarantine=0.35
- WARMUP_ROUNDS=1 (round 0 only), FedAvg clip_ratio=2.0

### E1 AUROC per round ✅
Source: `results/experiment_e1_gradient_scaling/experiment_summary_option_b_20260622_180022.csv`

| Round | Medical | Surgical (ATK) | Cardiac | **Mean** | Client 1 gate |
|-------|---------|----------------|---------|----------|---------------|
| 1 | 0.9490 | 0.9636 | 0.9743 | **0.9623** | accept (warm-up) |
| 2 | 0.9304 | 0.9650 | 0.9702 | **0.9552** | **quarantine** ← first detection |
| 3 | 0.9323 | 0.9664 | 0.9712 | **0.9566** | quarantine |
| 4 | 0.9338 | 0.9677 | 0.9717 | **0.9577** | quarantine |
| 5 | 0.9341 | 0.9675 | 0.9726 | **0.9581** | quarantine |
| 6 | 0.9360 | 0.9675 | 0.9731 | **0.9589** | quarantine |
| 7 | 0.9362 | 0.9676 | 0.9740 | **0.9593** | quarantine |
| 8 | 0.9364 | 0.9678 | 0.9737 | **0.9593** | quarantine |
| 9 | 0.9364 | 0.9683 | 0.9744 | **0.9597** | quarantine |
| 10 | 0.9370 | 0.9683 | 0.9747 | **0.9600** | quarantine |

### E1 SNAS signal analysis (from failed run, round 1 data — before quarantine cascade):
| Client | act_div | norm_anomaly | cos_anomaly | SNAS(new formula) | Gate(predicted) |
|--------|---------|-------------|-------------|-------------------|-----------------|
| 0 (honest) | 7.33 | 0.146 | 0.245 | 0.181 | accept |
| 1 (attacker) | 4.72 | 0.509 | 0.171 | 0.391 | **quarantine** |
| 2 (honest) | 6.39 | 0.141 | 0.180 | 0.155 | accept |

### E1 Detection Metrics ✅
| Metric | Value |
|--------|-------|
| Detection rate | **100%** (8/8 detection rounds) |
| First detection round | **2** (immediately after warm-up) |
| False positive rate | **0%** (0/16 honest-client-rounds) |
| AUROC degradation (rounds 8-10 mean) | **Δ = −0.0056** (0.9597 vs baseline 0.9653) |
| AUROC recovery | Yes — 0.9552 (round 2) → 0.9600 (round 10) as quarantine removes poison |

### E1 SNAS per round (key values):
| Round | C0 SNAS (honest) | C1 SNAS (attacker) | C2 SNAS (honest) |
|-------|-----------------|-------------------|------------------|
| 1 | 0.1805 | **0.3909** | 0.1542 |
| 2 | 0.1361 | **0.4080** | 0.1474 |
| 5 | 0.1139 | **0.4231** | 0.0904 |
| 10 | — | — | — |

norm_anomaly for C1: 0.51–0.54 consistently; for C0/C2: 0.06–0.15.
Clear separation enables quarantine without false positives.

---

## TABLE 5 — E2: Free Rider Attack on Client 2 ✅ COMPLETE
Source: `results/experiment_e2_free_rider/experiment_summary_option_b_20260622_210315.csv`

| Round | Medical | Surgical | Cardiac (ATK) | **Mean** | C2 gate |
|-------|---------|----------|---------------|----------|---------|
| 1 | 0.9490 | 0.9636 | 0.9743 | **0.9623** | accept (warm-up) |
| 2 | 0.7940 | 0.8151 | 0.8297 | **0.8129** | **quarantine** ← severe initial drop |
| 3 | 0.7881 | 0.8112 | 0.8467 | **0.8154** | quarantine |
| 4 | 0.8373 | 0.8554 | 0.8750 | **0.8559** | quarantine |
| 5 | 0.8578 | 0.8816 | 0.8910 | **0.8768** | quarantine |
| 6 | 0.8704 | 0.9021 | 0.9021 | **0.8915** | quarantine |
| 7 | 0.8790 | 0.9121 | 0.9083 | **0.8998** | quarantine |
| 8 | 0.8858 | 0.9218 | 0.9127 | **0.9068** | quarantine |
| 9 | 0.8897 | 0.9283 | 0.9152 | **0.9111** | quarantine |
| 10 | 0.8930 | 0.9318 | 0.9177 | **0.9142** | quarantine |

### E2 Detection Metrics ✅
| Metric | Value |
|--------|-------|
| Detection rate | **100%** (8/8 detection rounds) |
| First detection round | **2** |
| False positive rate | **0%** |
| C2 SNAS range | 0.40–0.47 (consistently above quarantine=0.35) |
| C0/C1 SNAS range | 0.006–0.17 (well below flag=0.22) |
| Mean AUROC rounds 8–10 | **0.9107** |
| AUROC degradation | **Δ = −0.0546** (vs baseline 0.9653) |
| Recovery trend | 0.8129 (round 2) → 0.9142 (round 10) — gradual, 8+ rounds |

### E2 Key Findings:
1. **Severe initial corruption**: Free rider submits random weights in warm-up round 0. All clients
   load this corrupted encoder → AUROC crashes to 0.8129. Far worse than E1 (Δ=−0.005 vs Δ=−0.055).
   Reason: random weights corrupt both direction AND magnitude of the global encoder.
2. **Recovery confirmed**: Clean 2-client FedAvg (Medical + Surgical) gradually restores performance.
   System is self-healing after attack is detected and quarantined.
3. **Early detectability**: C2 SNAS at round 1 (warm-up, still accepted) = **0.4275** — already
   above quarantine threshold. With WARMUP_ROUNDS=0, detection would be immediate (round 1).
   Paper ablation: vary warm-up rounds {0, 1, 2} to show detection-vs-stability trade-off.
4. **E1 vs E2 degradation contrast** (paper Section 5 discussion):
   - E1 (gradient scaling): Δ=−0.006 because norm clipping limits corruption
   - E2 (free rider): Δ=−0.055 because random weights corrupt unconditionally during warm-up
   - Suggests: tighter warm-up (or warm-up screening) would limit free rider damage

---

## TABLE 6 — E4: Gradient Scaling (Client 1) + Dropout (Client 2) ✅ COMPLETE — CORE RESULT
Source: `results/experiment_e4_attack_plus_dropout/experiment_summary_option_b_20260622_211353.csv`

| Round | Medical | Surgical (ATK) | Cardiac (STRAG) | **Mean** | C1 gate | C2 in SNAS? |
|-------|---------|----------------|-----------------|----------|---------|-------------|
| 1 | 0.9490 | 0.9636 | 0.9743 | **0.9623** | accept (warm-up) | No ✅ |
| 2 | 0.9291 | 0.9652 | 0.9684 | **0.9543** | **quarantine** | No ✅ |
| 3 | 0.9309 | 0.9662 | 0.9689 | **0.9554** | quarantine | No ✅ |
| 4 | 0.9327 | 0.9670 | 0.9690 | **0.9563** | quarantine | No ✅ |
| 5 | 0.9333 | 0.9671 | 0.9698 | **0.9568** | quarantine | No ✅ |
| 6 | 0.9350 | 0.9674 | 0.9697 | **0.9574** | flag | No ✅ |
| 7 | 0.9355 | 0.9674 | 0.9701 | **0.9577** | quarantine | No ✅ |
| 8 | 0.9356 | 0.9673 | 0.9692 | **0.9574** | quarantine | No ✅ |
| 9 | 0.9355 | 0.9673 | 0.9693 | **0.9574** | quarantine | No ✅ |
| 10 | 0.9359 | 0.9675 | 0.9694 | **0.9576** | quarantine | No ✅ |

### E4 Detection Metrics ✅ — THE CORE PAPER RESULT
| Metric | Value |
|--------|-------|
| Client 1 detection rate | **100%** (quarantined 7/8 rounds, flagged 1/8) |
| Client 1 first detection | **Round 2** |
| Client 0 false positive rate | **0%** |
| Client 2 false positive rate | **0% — architectural guarantee** (0 SNAS entries — never evaluated) |
| Mean AUROC rounds 8–10 | **0.9574** |
| AUROC degradation | **Δ = −0.0079** (vs baseline 0.9653) |

### E4 Key Findings:
1. **Simultaneous attacker + straggler handled correctly.**
   Client 1 (attacker, on-time) → quarantined by SNAS.
   Client 2 (straggler, 30s late) → excluded by window controller before SNAS even activates.
   The two mechanisms are architecturally orthogonal — zero interference.

2. **Only Client 0 in FedAvg from round 2 onward** (C1 quarantined, C2 missed window).
   Despite single-client aggregation, AUROC degrades only Δ=−0.0079 — near-baseline performance.

3. **SNAS separation in E4:**
   C0 SNAS: 0.069–0.212 (all below flag=0.22)
   C1 SNAS: 0.345–0.374 (all above flag=0.22, mostly above quarantine=0.35)
   C2 SNAS: N/A — never submitted within window.

4. **Paper claim validated:** "In the hardest scenario — simultaneous adversarial client and
   legitimate straggler — Async-SplitFed-IR correctly identifies the attacker (DR=100%)
   while guaranteeing zero false quarantine for the straggler (FPR=0%), with only Δ=0.008
   AUROC degradation vs. the synchronous clean baseline."

---

## MASTER RESULTS TABLE — Section 5 of Paper
| Experiment | Attack | Straggler | DR | FPR | AUROC (R8–10) | Δ vs baseline |
|-----------|--------|-----------|-----|-----|--------------|--------------|
| Sync-FedAvg baseline | None | None | N/A | N/A | 0.9649 | — |
| Async clean (mild lag) | None | None | N/A | N/A | 0.9649 | 0.0000 |
| E6 — honest straggler | None | C2 (30s) | N/A | **0%** | 0.9660 | **+0.001** |
| E1 — gradient scaling | C1 (×10) | None | **100%** | **0%** | 0.9597 | −0.006 |
| E4 — attack + straggler | C1 (×10) | C2 (30s) | **100%** | **0%** | 0.9574 | −0.008 |
| E2 — free rider | C2 (random) | None | **100%** | **0%** | 0.9107 | −0.055 |

DR = Detection Rate (fraction of active rounds client is flagged/quarantined).
FPR = False Positive Rate (honest clients incorrectly quarantined).

## TABLE 7 — Decay Function Ablation ✅ COMPLETE (Empirical)
Scenario: Rotating straggler — Client 2 submits in odd rounds (10s delay, τ=1) and
misses even rounds (30s delay). No attack. 4 runs with different staleness_decay.
Sources: `results/ablation_{linear,exponential,polynomial,step}/`

### Mean AUROC per round across 4 decay functions:
| Round | Linear | Exponential | Polynomial | Step | Async Baseline |
|-------|--------|------------|------------|------|----------------|
| 1 | 0.9623 | 0.9623 | 0.9623 | 0.9623 | 0.9649 |
| 2 | 0.9613 | 0.9613 | 0.9613 | 0.9613 | 0.9649 |
| 3 | 0.9624 | 0.9624 | 0.9624 | 0.9624 | 0.9649 |
| 4 | 0.9630 | 0.9631 | 0.9630 | 0.9632 | 0.9649 |
| 5 | 0.9637 | 0.9637 | 0.9637 | 0.9639 | 0.9649 |
| 6 | 0.9638 | 0.9638 | 0.9638 | 0.9637 | 0.9649 |
| 7 | 0.9645 | 0.9646 | 0.9645 | 0.9644 | 0.9649 |
| 8 | 0.9643 | 0.9646 | 0.9643 | 0.9643 | 0.9649 |
| 9 | 0.9651 | 0.9652 | 0.9651 | 0.9651 | 0.9649 |
| **10** | **0.9650** | **0.9651** | **0.9650** | **0.9652** | **0.9649** |

### Aggregation weight for Client 2 (τ=1, odd-round submissions):
| Decay | Theoretical w(1) | Actual mean agg weight | Behaviour |
|-------|-----------------|----------------------|-----------|
| Linear | 0.5000 | 0.2154 | Halves trust after 1 miss |
| Exponential | 0.6065 | 0.2498 | Smooth reduction ← **our default** |
| Polynomial | 0.5000 | 0.2154 | Same as linear at τ=1, faster at τ≥2 |
| Step (c=3) | **1.0000** | **0.3544** | Full trust until cutoff — same as fresh client |

### Client 2 SNAS values (τ=1, honest straggler):
All 4 decay functions: SNAS = 0.030–0.044 (well below flag=0.22) → gate=**accept** in all runs.
Detection of honest stragglers: **0% false positive rate across all 4 functions** ✅

### Theoretical decay weights w(τ) for all values:
| τ | Linear | Exponential | Polynomial | Step (c=3) |
|---|--------|------------|------------|------------|
| 0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 1 | 0.5000 | 0.6065 | 0.5000 | 1.0000 |
| 2 | 0.3333 | 0.3679 | 0.2000 | 1.0000 |
| 3 | 0.2500 | 0.2231 | 0.1000 | **1.0000** |
| 4 | 0.2000 | 0.1353 | 0.0588 | **0.0000** |
| 5 | 0.1667 | 0.0821 | 0.0385 | 0.0000 |

### Ablation Key Findings:
1. **AUROC robust across all decay functions**: Max difference at round 10 = 0.0002.
   All 4 functions maintain performance near the async clean baseline (0.9649).
   Decay function choice does not significantly impact model accuracy.

2. **Meaningful weight differences**: Step function gives Client 2 full weight (same
   as a fresh client), while linear/polynomial halve its contribution and exponential
   gives 60.7%. In adversarial settings, step's generosity could be exploited by an
   attacker who deliberately alternates between attacking and abstaining.

3. **Exponential justified as default**:
   - Smooth, monotonic: no discontinuity at cutoff (unlike step)
   - Faster decay than linear beyond τ=2 (stricter on persistent stragglers)
   - Mathematically tractable: w(τ) = e^(-α·τ) has closed-form analysis
   - Standard in async FL literature (FedAsync, ASGD-FL)

4. **FPR = 0% for all functions**: SNAS correctly identifies Client 2 as honest
   (SNAS 0.030–0.044) regardless of decay choice. The gate is robust to this choice.

---

## SNAS CALIBRATION LOG
**Status: COMPLETE — 2026-06-22, after prev_global_encoder bug fix**
Source: `results/label_private_splitfed/server/server_metrics.jsonl` (last 9 SNAS entries)

### Clean baseline SNAS values (no attack, mild lag 0/5/10s):
| Round | Client | SNAS | Act. Divergence | Cosine Anomaly | Gate |
|-------|--------|------|----------------|----------------|------|
| 0 | 0,1,2 | null | null | null | accept (no prev_encoder yet) |
| 1 | 0 | 0.1114 | 0.0680 | 0.1765 | accept |
| 1 | 1 | 0.1330 | 0.0752 | 0.2196 | accept |
| 1 | 2 | 0.1149 | 0.0713 | 0.1804 | accept |
| 2 | 0 | 0.1011 | 0.0455 | 0.1845 | accept |
| 2 | 1 | 0.1248 | 0.0590 | 0.2235 | accept |
| 2 | 2 | 0.1067 | 0.0654 | 0.1687 | accept |

### Calibration summary:
- Clean SNAS range: **0.101–0.133** (mean ≈ 0.116)
- threshold_flag = 0.4 → **3.4× clean mean** ✅ adequate headroom
- threshold_quarantine = 0.8 → **6.9× clean mean** ✅ very conservative
- Thresholds confirmed as appropriate — no recalibration needed
- Warm-up: rounds 1–4 always accept (state.current_round 0–3 ≤ WARMUP_ROUNDS=3)
- Real detection starts: round 5 (1-indexed) in a 10-round experiment → 6 detection rounds

### Expected SNAS under attacks (predicted, to be confirmed by experiments):
- Gradient scaling (×10): cosine_anomaly → ~0.9+ (directions flipped), SNAS >> 0.4
- Free rider: cosine_anomaly → ~1.0 (random weights), SNAS >> 0.8
- Honest straggler (τ=5): SNAS dampened by 1/(1+5)=0.167 → stays below 0.4

---

## COMMUNICATION OVERHEAD SUMMARY
| Metric | Value |
|--------|-------|
| Activation payload per batch | 64 samples × 32 dims × 4 bytes = 8,192 bytes |
| Gradient payload per batch | 8,192 bytes |
| Per client per round (200 batches) | 3,276,800 bytes (3.13 MB) |
| All 3 clients per round | 9,830,400 bytes (9.38 MB) |
| Total over 10 rounds (3 clients) | 98,304,000 bytes (93.8 MB) |
| Encoder FedAvg payload per client | ~77 KB (19,360 params × 4 bytes) |

---

## AGGREGATION WEIGHT LOG (Async Clean, rounds 0–9)
Source: `results/label_private_splitfed/server/server_metrics.jsonl` (SNAS lines)

All clients accepted, staleness_tau=0, staleness_weight=1.0 for all rounds.
Normalised aggregation weights (proportional to dataset size):
- Client 0 (Medical): 0.3913
- Client 1 (Surgical): 0.2542
- Client 2 (Cardiac): 0.3544

---

## NOTES FOR PAPER WRITING

### Section 3 (System Model) claims supported by results:
- Label-private protocol validated: server never accesses raw features or labels
- 3-client non-IID MIMIC-IV partition with distinct patient populations
- FedAvg aggregation weighted by dataset size

### Section 4 (Async-SplitFed-IR) claims to support with results:
- SNAS_i = (α·div_i + β·anom_i)/(1+τ_i): all components logged per round
- Honest stale client: SNAS dampened by (1+τ) → not quarantined (E6 validates)
- Adversarial client: SNAS super-linear despite staleness (E1, E2, E4 validate)
- Warm-up rounds 1–3: gate always "accept" regardless of SNAS value

### Section 5 (Experiments) tables needed:
1. Table 1: Sync-FedAvg baseline ✅
2. Table 2: Async clean (identical to baseline) ✅
3. Table 3: E6 straggler — FPR = 0 for Client 2 (PENDING full 10-round)
4. Table 4: E1 detection — TPR, detection round, AUROC degradation (PENDING)
5. Table 5: E2 detection (PENDING)
6. Table 6: E4 combined — most important (PENDING)
7. Table 7: Decay ablation (PENDING)
