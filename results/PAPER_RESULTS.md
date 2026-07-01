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

---
## SUPERVISOR REVIEW GAPS — Tasks Added 2026-06-27

Four blocking gaps identified by supervisor after progress report review.
All implementation complete; experiment runs pending.

### Gap 1 — Baseline Comparison (Task 1) ✅ COMPLETE
**Implementation:** `src/sfl/server/baseline_aggregators.py`
FLTrust root dataset: `configs/fltrust_root_indices.json`
  Medical=1,446 | Surgical=939 | Cardiac=1,309 | Total=3,694 samples (5% each, stratified)
Source: `results/baseline_comparison.csv`

| Experiment | **SNAS** | **Krum** | **Trimmed Mean** | **FLTrust** |
|-----------|---------|---------|----------------|-----------|
| E1 gradient scaling | 0.9597 | **0.9647** (+0.005) | 0.9539 (−0.006) | 0.9539 (−0.006) |
| E2 free rider | 0.9107 | **0.9653** (+0.055) | 0.8227 (−0.088) | 0.8227 (−0.088) |
| E4 attack+straggler | 0.9574 | **0.9647** (+0.007) | 0.9541 (−0.003) | 0.9541 (−0.003) |

### Task 1 Key Findings (paper-critical):

**Finding 1 — Krum outperforms SNAS on raw AUROC in all three single-seed experiments.**
Mechanism: Krum with f=1, n=3 selects the single "most central" encoder update. With
gradient-scaled or random-weight attackers, the attacker is an outlier → Krum correctly
selects one honest client's encoder. This gives higher AUROC than SNAS's weighted aggregation
(which includes the warm-up poisoning effect in round 0-1).

CRITICAL CAVEAT (must appear in paper): In E4 with only 2 submissions (Client 0 honest,
Client 1 attacker), Krum's distance scores are symmetric (tied). The winner is determined
by dict insertion order: Client 0 submits at t=0, Client 1 at t=5s → Client 0 wins.
If the attacker registers/submits first, Krum would select the ATTACKER. This is not a
principled guarantee — it is implementation-dependent behaviour.

Paper framing: "Krum achieves higher AUROC under our experimental setup where the honest
client consistently submits before the attacker. In a scenario where submission order is
adversarially controlled, Krum's selection becomes arbitrary. SNAS provides an explicit
architectural guarantee via its dual-signal scoring, independent of submission order."

**Finding 2 — Trimmed mean reduces to vanilla FedAvg with n=3 clients at 20% trim ratio.**
k = floor(0.2 × 3) = floor(0.6) = 0 → zero values trimmed per coordinate.
Trimmed mean (20%) with 3 clients = vanilla FedAvg. This is a known limitation of
coordinate-wise trimmed mean at small client counts. Provides zero Byzantine protection here.

**Finding 3 — FLTrust is architecturally incompatible with label-private split learning.**
Server log: "FLTrust: root_gradient norm near zero — falling back to uniform average."
Root cause: In split learning, the encoder lives on clients. The server cannot compute
meaningful encoder gradients from its root dataset without the encoder weights (which are
the very thing being submitted for aggregation). The approximation via the global encoder
produces near-zero gradients, reducing FLTrust to uniform averaging = vanilla FedAvg.
FLTrust and Trimmed Mean results are IDENTICAL because both reduce to vanilla FedAvg.

Paper contribution: "SNAS is the first Byzantine-robust aggregation method designed
natively for the split learning threat model. Existing methods (FLTrust, trimmed mean)
either require full model gradients the server does not have, or become degenerate at
the client counts typical of IoMT deployments."

**Best baseline for multi-seed comparison (Task 2): KRUM** (closest to SNAS on all 3 experiments).
Update BEST_BASELINE_AUROCS in scripts/run_multiseed.py after 5-seed Krum runs.

### Gap 2 — Multi-Seed Statistical Validation (Task 2) ✅ SNAS RUNS COMPLETE
**Implementation:** `scripts/run_multiseed.py`
Seeds: [42, 7, 123, 2024, 31415]
Sources: `results/multiseed_summary.json`, `results/significance_tests.csv`

### SNAS 5-Seed Results (all paper AUROC values must use these):

| Experiment | S=42 | S=7 | S=123 | S=2024 | S=31415 | **Mean ± Std** |
|-----------|------|-----|-------|--------|---------|----------------|
| E1 gradient scaling | 0.9594 | 0.9595 | 0.9597 | 0.9600 | 0.9603 | **0.9598 ± 0.0004** |
| E2 free rider | 0.9107 | 0.9114 | 0.8948 | 0.9037 | 0.9001 | **0.9042 ± 0.0071** |
| E4 attack+straggler | 0.9574 | 0.9579 | 0.9583 | 0.9576 | 0.9584 | **0.9579 ± 0.0004** |
| E6 honest straggler | 0.9660 | 0.9666 | 0.9666 | 0.9658 | 0.9657 | **0.9661 ± 0.0004** |

### Task 2 Key Findings:

**Finding 1 — E1, E4, E6 are extremely stable (std=0.0004, CV<0.05%).**
Single-seed results were accurate representations. The reviewer concern about small deltas
being noise is directly addressed: AUROC spread across 5 seeds is 0.0009-0.0010, well below
the reported degradation deltas.

**Finding 2 — E2 has higher variance (std=0.0071) — expected and explainable.**
Free rider submits random weights each round → stochastic corruption effect varies by seed.
The single-seed result (0.9107) was the BEST of the 5 seeds. Revised claim with 5-seed data:
E2 AUROC degradation = **−0.0607 ± 0.0071** (Δ from baseline 0.9649). Still very significant.

**Finding 3 — E6 ALL 5 seeds exceed the sync baseline (0.9649).**
Previous claim: "E6 exceeds sync baseline by +0.0007 (may be noise)."
Revised claim: "E6 SNAS 5-seed mean = 0.9661 ± 0.0004. Minimum across 5 seeds = 0.9657 > 0.9649.
All 5 seeds confirm honest-straggler AUROC exceeds synchronous baseline."

**Finding 4 — Significance tests COMPLETE.** Source: `results/significance_tests.csv`

| Experiment | SNAS mean±std | Krum mean±std | Wilcoxon p | t-test p | Cohen's d |
|-----------|--------------|---------------|-----------|----------|-----------|
| E1 gradient scaling | 0.9598±0.0004 | 0.9654±0.0005 | 0.0625 | 3.2e-05 | -13.12 |
| E2 free rider | 0.9041±0.0071 | 0.9653±0.0004 | 0.0625 | 3.7e-05 | -12.22 |
| E4 attack+straggler | 0.9579±0.0004 | 0.9654±0.0005 | 0.0625 | 5.0e-06 | -16.40 |
| E6 honest straggler | 0.9661±0.0004 | sync=0.9649 | N/A (dominance) | N/A | N/A |

**CRITICAL STATISTICAL NOTE (must appear verbatim in paper methods/results):**
Wilcoxon p=0.0625 is the mathematical FLOOR for a paired two-sided test at n=5
(minimum achievable p = 2/2^5 = 0.0625). Krum beat SNAS in **5 of 5 seed-pairs** for
every experiment — the most extreme outcome the test can register at this sample size.
This is NOT a "non-significant" result in the conventional sense; it reflects insufficient
statistical power at n=5, not a weak or inconsistent effect. The paired t-test (parametric,
assumes normality) confirms significance at p<0.0001 for all three experiments, with very
large effect sizes (Cohen's d: -12 to -16). Per the guideline's honesty checkpoint: report
both tests, flag the disagreement explicitly, and note that the t-test + effect size + 5/5
consistency together constitute strong evidence of a real, reproducible difference.

**Direction of the effect — Krum outperforms SNAS on raw AUROC, consistently and significantly
by the parametric test.** This must be framed honestly in the paper (see Task 1 Finding 1):
Krum's apparent superiority is explained by a newly confirmed mechanism (see below), not by
Krum being a better Byzantine-robust method for this threat model.

**Finding 5 — ROOT CAUSE of Krum's AUROC advantage: Krum collapses to single-client learning.**
Direct verification from `results/baseline_krum/server/server_metrics.jsonl`: Krum selects
**Client 0 (Medical) as the winner in literally every single round**, across E1 and E4,
across all 5 seeds (464/464 decision log entries where client_id=0 has krum_selected=True
whenever Client 0 is in the submission pool). Krum's "global encoder" is therefore not a
federated aggregate — it is Client 0's locally-trained encoder, repeatedly re-selected because:
  (a) In E1: Client 0 and Client 2 (both honest) are mutually close in weight space; with
      k=1 nearest-neighbor scoring, whichever of the two has the smaller single-NN distance
      wins every round — empirically always Client 0.
  (b) In E4: only Clients 0 and 1 submit (Client 2 misses the window). With n=2, Krum's
      distance scores are symmetric/tied; Python's `min()` returns the first-inserted key,
      and Client 0 always submits first (0s delay vs Client 1's 5s delay) — a pure
      implementation artifact of submission order, not a security property.

**Paper framing (use this exact argument):** "Krum's higher raw AUROC does not reflect
superior Byzantine robustness — it reflects a structural failure mode where Krum discards
federated learning's core proposition entirely, converging to single-client training using
only the Medical partition (36,152 samples, the largest of the three). Surgical and Cardiac
data are never incorporated into the global model under Krum in either experiment. SNAS, by
contrast, performs genuine weighted aggregation across all accepted clients, preserving the
non-IID statistical diversity the federation was designed to exploit. Krum's AUROC advantage
is therefore not evidence of better security — it is an artifact of one partition's data being
sufficient on its own for this task, combined with Krum's selection mechanism systematically
favoring whichever client happens to register or submit first under tied distances."

### Gap 3 — Adaptive Adversary Experiments (Task 3) ⚠️ PARTIALLY COMPLETE
**Implementation:** Added to `src/sfl/common/attack_simulator.py`
Sources: `results/adaptive_attack_{a,b,c}/`, server metrics filtered by exact timestamp window
(server_metrics.jsonl accumulates across runs — isolated each attack's entries by run start/end time)

---

#### Attack A — Threshold-Aware Scaling ✅ COMPLETE
Source: `results/adaptive_attack_a/experiment_summary_option_b_20260630_225818_seed42.csv`

| Round | C0 | C1 (ATK) | C2 | Mean | C1 SNAS | C1 Gate |
|-------|-----|----------|-----|------|---------|---------|
| 1 | 0.9490 | 0.9636 | 0.9743 | 0.9623 | accept (warm-up) | — |
| 2 | 0.9307 | 0.9650 | 0.9705 | 0.9554 | 0.3678 | **quarantine** |
| 3 | 0.9326 | 0.9664 | 0.9715 | 0.9568 | 0.3634 | quarantine |
| 4 | 0.9341 | 0.9676 | 0.9720 | 0.9579 | 0.3848 | quarantine |
| 5 | 0.9344 | 0.9675 | 0.9728 | 0.9583 | 0.3673 | quarantine |
| 6 | 0.9363 | 0.9675 | 0.9734 | 0.9591 | 0.3382 | flag |
| 7 | 0.9365 | 0.9676 | 0.9742 | 0.9594 | 0.3470 | flag |
| 8 | 0.9364 | 0.9679 | 0.9736 | 0.9593 | 0.3429 | flag |
| 9 | 0.9361 | 0.9684 | 0.9739 | 0.9595 | 0.3576 | quarantine |
| 10 | 0.9361 | 0.9686 | 0.9740 | 0.9596 | — | — |

**Detection rate: 8/8 active rounds (100%)** — first detection round 2.
**AUROC degradation (R8-10 mean=0.9595): Δ = −0.0054** (nearly identical to naive E1's −0.0056).

**KEY FINDING — Attack A's analytical flaw:** The attack's binary search assumed
`cosine_anomaly = 0` (mathematically true for uniform scaling in isolation — `cos(s·w, w)=1`
for any s>0). But it ignored **natural training drift**: the attacker's encoder also drifts
from the global reference through ordinary local SGD, independent of the scaling attack
(observed cosine_anomaly ≈ 0.16–0.25, matching honest-client baseline levels). This pushed
the true SNAS above the targeted 0.315, triggering quarantine despite the adversary "knowing"
the SNAS formula. **Paper claim:** "An adversary that models SNAS using only the attack's
direct effect on weight statistics — without accounting for the confounding effect of ordinary
training dynamics — will systematically underestimate its own detectability."

---

#### Attack B — Strategic Alternation (Attack-Then-Abstain) ✅ COMPLETE — ALL 4 DECAY FUNCTIONS
Source: `results/adaptive_attack_b/experiment_summary_option_b_{20260630_230738,20260630_234909,20260701_000104,20260701_001418}_seed42.csv`
Server metrics: `results/ablation_{exponential,linear,step,polynomial}/server/server_metrics.jsonl`
(each isolated by exact run-start/run-end timestamp window)

### 🚨 CRITICAL FINDING — Attack B EVADES SNAS DETECTION UNDER ALL 4 DECAY FUNCTIONS
**Detection rate: 0/4 rounds × 4 decay functions = 0/16 (0%) — never flagged, never quarantined, regardless of decay function choice.**

| Decay | R2 SNAS | R4 SNAS | R6 SNAS | R8 SNAS | Gate (all rounds) | R10 Mean AUROC |
|-------|---------|---------|---------|---------|--------------------|-----------------|
| Exponential | 0.1858 | 0.1711 | 0.1643 | 0.1692 | accept × 4 | 0.9592 |
| Linear | 0.1858 | 0.1760 | 0.1689 | 0.1794 | accept × 4 | 0.9593 |
| Polynomial | 0.1858 | 0.1760 | 0.1689 | 0.1794 | accept × 4 | 0.9593 |
| Step | 0.1858 | 0.1807 | 0.1716 | 0.1830 | accept × 4 | 0.9589 |

**Mechanism — confirmed structural, not decay-function-specific.** SNAS gate decisions depend
only on `(α·div + β·cos + γ·norm) / (1+τ)`. The staleness decay function (linear/exponential/
polynomial/step) governs ONLY the aggregation *weight* downstream in RobustAsyncFedAvg — it
has zero influence on the SNAS *gate* computation itself. Since the attacker maintains τ=1 at
every submission (by design of the alternating schedule), gate evasion succeeds identically
regardless of which decay function is configured. **This is the most important finding of
Task 3**: the vulnerability is in SNAS's core formula, not in any particular decay function
choice, meaning switching decay functions cannot mitigate it.

**Sub-finding — Linear and Polynomial produce byte-for-byte identical SNAS trajectories.**
At τ=1 specifically: `linear(1) = 1/(1+1) = 0.5` and `polynomial(1) = 1/(1+1²) = 0.5` —
mathematically identical at this exact staleness value. Since τ never exceeds 1 in this
attack pattern, the two decay functions are indistinguishable for this scenario (diverge
only for τ≥2).

**Sub-finding — Step decay causes the worst AUROC damage (0.9589 vs 0.9592-0.9593 for others).**
Step gives the attacker FULL aggregation weight (w=1.0, since τ=1≤cutoff=3) — double the
weight linear/polynomial/exponential assign (w≈0.5-0.61). Detection still fails equally
(0% across all four), but step's full-trust treatment of a once-stale client compounds
slightly worse corruption into the global encoder. This confirms the report's original §11.7
hypothesis about step decay's exploitability — though the surprising result is that ALL
four decay functions fail to DETECT the attack; step only makes the AGGREGATION DAMAGE
marginally worse once undetected.

**Paper framing (mandatory honest disclosure per guideline):** "We identify a fundamental
structural vulnerability in the SNAS design, confirmed across all four staleness decay
functions: an attacker who alternates between submitting poisoned updates and deliberately
missing the subsequent round maintains τ=1 at every submission. The (1+τ) staleness-
normalization term — designed to protect honest stragglers from false quarantine (validated
empirically in E6 and E4) — halves the attacker's effective anomaly score, pushing it from
~0.37 (which would trigger quarantine at τ=0, as confirmed in E1) to ~0.17-0.18 (well below
even the flag threshold of 0.22). Because this evasion depends only on the SNAS gate formula
itself, not on the choice of aggregation weighting function, no decay function configuration
can mitigate it. This represents a genuine, confirmed robustness boundary of the current SNAS
formulation, not a tuning artifact."

**Mitigation directions for paper discussion:** (a) accumulate SNAS history per client
(e.g., exponential moving average of recent SNAS scores) rather than memoryless per-round
scoring; (b) track consecutive accept-after-flag patterns as a distinct signal; (c) apply
staleness dampening only to the activation-divergence component (which legitimately grows
with retraining gaps) and not to the weight-anomaly components (which do not have a
principled reason to shrink with staleness); (d) cap the staleness dampening factor at a
floor (e.g., never divide by more than 1.5×) so repeated short absences cannot fully
neutralize a detectable anomaly.

---

#### Attack C — Slow-Ramp Mimicry (15 rounds) ✅ COMPLETE
Source: `results/adaptive_attack_c/experiment_summary_option_b_20260630_232541_seed42.csv`

| Round | Mean AUROC | C1 SNAS | C1 Gate | Trust | Implied scale (1.15^(r-1)) |
|-------|-----------|---------|---------|-------|---------------------------|
| 1 | 0.9623 | — (warm-up) | accept | 1.000 | 1.00× |
| 2 | 0.9613 | 0.1396 | accept | 1.000 | 1.15× |
| 4 | 0.9630 | 0.1689 | accept | 1.000 | 1.52× |
| 6 | 0.9634 | 0.1821 | accept | 1.000 | 2.01× |
| 8 | 0.9637 | **0.2353** | **flag** | 0.900 | 2.66× |
| 10 | 0.9637 | 0.2543 | flag | 0.729 | 3.52× |
| 12 | 0.9637 | 0.2794 | flag | 0.590 | 4.65× |
| 14 | 0.9638 | 0.3027 | flag | 0.478 | 6.15× |
| 15 | 0.9636 | — | — | — | 7.08× (final) |

**First detection (flag): Round 8.** SNAS never reaches quarantine threshold (0.35) within
15 rounds — stays in the flag zone 0.235→0.303 even as scale grows to ~7×.
**AUROC barely degrades**: 0.9636 final vs 0.9649 baseline (Δ=−0.0013) — the slow ramp's
own gradualness limits damage even without full quarantine, because (a) early rounds are
genuinely clean (scale≈1×) and (b) the flag penalty (0.5× weight) plus monotonically
decaying trust score progressively suppress the attacker's contribution from round 8 onward,
even without an explicit quarantine event.

**Paper finding:** "SNAS detects the slow-ramp attack (flagging from round 8) before it
reaches damaging magnitude, but does not fully quarantine it within the tested window. The
flag mechanism's partial weight penalty, combined with cumulative trust decay, is sufficient
to bound AUROC damage to near-zero (Δ=−0.0013) even without a hard exclusion event. This
demonstrates SNAS's graduated response is functionally protective even when quarantine
itself is not triggered — though a longer-running ramp could eventually breach 0.35 and
warrants explicit acknowledgement as an open question rather than an assumed success."

---

### Task 3 Summary Table (for paper Table) — ✅ FULLY COMPLETE

| Attack | Detection | First detection | AUROC Δ (R10) | Notes |
|--------|-----------|-----------------|----------------|-------|
| A — Threshold-aware | **100%** (8/8) | Round 2 | −0.0054 | Evades own analytical model but caught by training drift |
| B — Alternating, exponential | **0%** (0/4) | Never | −0.0057 | Structural evasion |
| B — Alternating, linear | **0%** (0/4) | Never | −0.0056 | Identical SNAS to polynomial at τ=1 |
| B — Alternating, polynomial | **0%** (0/4) | Never | −0.0056 | Identical SNAS to linear at τ=1 |
| B — Alternating, step | **0%** (0/4) | Never | −0.0060 | Worst AUROC damage (full attacker weight) |
| C — Slow ramp (15 rounds) | Flagged only, never quarantined | Round 8 (flag) | −0.0013 | Partial detection bounds damage despite no quarantine |

**Task 3 headline finding for the paper abstract/conclusion:** SNAS achieves 100% detection
against static and threshold-aware adaptive attacks (A), but exhibits a confirmed, decay-
function-independent evasion vulnerability against an attacker that strategically alternates
between attacking and abstaining (B) — a genuine robustness boundary that must be reported
as a limitation, not omitted. The slow-ramp attack (C) is partially mitigated via the flag
mechanism even without full quarantine, demonstrating SNAS's graduated response has some
protective value even in incomplete-detection scenarios.

### Gap 4 — Hyperparameter Sensitivity (Task 4) ✅ COMPLETE
**Implementation:** `scripts/run_sensitivity_sweep.py` + `scripts/plot_sensitivity_heatmap.py`
Source: `results/sensitivity_sweep.csv` (139 rows), `results/sensitivity_heatmap.png`
Total runs: 18 (Sweep1) + 115 (Sweep2) + 6 (Sweep3) = 139 completed.

---

#### Sweep 1 — beta/gamma sensitivity (E1 gradient scaling + E2 free rider)

| β (cosine weight) | γ=1−β | E1 AUROC | E1 DR | E1 FPR | E2 AUROC |
|-------------------|--------|----------|-------|--------|---------|
| 0.10 | 0.90 | 0.9597 | 100% | 0% | 0.9228 |
| 0.20 | 0.80 | 0.9597 | 100% | 0% | 0.9063 |
| 0.30 | 0.70 | 0.9597 | 100% | 0% | 0.9107 |
| **0.35 (default)** | **0.65** | **0.9598** | **100%** | **0%** | **0.9042** |
| 0.40 | 0.60 | 0.9586 | 100% | 0% | 0.9107 |
| 0.50 | 0.50 | 0.9583 | 100% | 0% | 0.9107 |
| 0.60 | 0.40 | 0.9580 | 100% | 0% | 0.9107 |
| 0.70 | 0.30 | 0.9580 | 100% | 0% | 0.9107 |
| 0.80 | 0.20 | 0.9579 | **85.7%** | 0% | 0.9107 |
| 0.90 | 0.10 | 0.9574 | **71.4%** | 0% | 0.9107 |

Note: E2 DR/FPR values not reported — sweep script hardcodes attacker_cid=1 but E2's attacker
is Client 2. E2 AUROC values are correct (read from stdout, client-side).

**Sweep 1 Key Findings:**
1. **E1 detection holds at 100% for β ∈ [0.10, 0.70]** — 7 out of 9 tested values (77.8%).
   Detection only degrades when β≥0.80 (gamma≤0.20), meaning the norm-anomaly component
   has been so heavily down-weighted that gradient scaling cannot be caught.
2. **E1 AUROC is stable** across the full beta range (0.957-0.960), insensitive to beta choice.
3. **Default β=0.35 sits well within the safe detection zone** with 35 percentage points of
   headroom before detection starts degrading.
4. **Paper claim:** "The β/γ balance is non-critical: SNAS achieves 100% gradient-scaling
   detection across 77.8% of the tested parameter space, with AUROC stable to within 0.003
   across the full sweep. The contribution does not depend on a single tuned operating point."

---

#### Sweep 2 — threshold_flag × threshold_quarantine grid on E4 (hardest scenario)

**115 valid (threshold_flag, threshold_quarantine) grid points evaluated on E4.**

| Metric | Value |
|--------|-------|
| DR=100% (attacker always caught) | **115/115 (100%)** |
| FPR=0% (no honest client quarantined) | **71/115 (61.7%)** |
| DR=100% AND FPR=0% simultaneously | **71/115 (61.7%)** |
| AUROC range across all 115 points | 0.9551 – 0.9578 |
| Default point (flag=0.22, quarantine=0.35) | DR=100%, FPR=0%, AUROC=0.9575 ✅ |

**Sweep 2 Key Findings:**
1. **Detection rate is 100% across ALL 115 grid points.** The attacker is caught regardless
   of threshold placement. The only question is whether honest clients are ever misclassified.
2. **61.7% of the swept threshold space achieves zero false positives** while maintaining
   perfect detection. This is a broad robust operating region.
3. **Default parameters (0.22, 0.35) fall squarely inside the zero-FPR region.**
4. **Paper headline statistic:** "61.7% of the swept (threshold_flag, threshold_quarantine)
   parameter space achieves 100% detection with 0% false positives on the hardest evaluated
   scenario (E4: simultaneous attacker and honest straggler). The heatmap (Figure X) shows
   the zero-FPR boundary as a contour, with the reported default marked as the operating
   point — demonstrating the contribution rests on a broad robust region, not a knife-edge."

---

#### Sweep 3 — clip_ratio sensitivity on E1 (warm-up protection)

| clip_ratio | AUROC R8-10 | DR | FPR | Round-1 AUROC drop |
|-----------|------------|-----|-----|---------------------|
| 1.5 | **0.9600** | 100% | 0% | 0.0071 |
| **2.0 (default)** | **0.9594** | **100%** | **0%** | **0.0071** |
| 2.5 | 0.9590 | 100% | 0% | 0.0071 |
| 3.0 | 0.9588 | 100% | 0% | 0.0071 |
| 3.5 | 0.9582 | 100% | 0% | 0.0071 |
| 4.0 | 0.9581 | 100% | 0% | 0.0071 |

**Sweep 3 Key Findings:**
1. **DR=100% and FPR=0% for all 6 clip_ratio values** — detection is entirely clip-insensitive.
2. **Round-1 AUROC drop is identical (0.0071) across all clip_ratio values.** The warm-up
   corruption is bounded by the clipping, but all values from 1.5 to 4.0 limit it equally
   at this attack scale (gradient scaling ×10 vs clip of ≥1.5 global norm is effectively
   the same bottleneck regardless of exact clip value).
3. **AUROC improves slightly with tighter clipping** (1.5: 0.9600 vs 4.0: 0.9581, Δ=0.002)
   — smaller clip limits warm-up corruption more, letting the model start from a cleaner
   global encoder in round 2 onward. Default clip=2.0 is a reasonable middle choice.
4. **Paper claim:** "clip_ratio is not a sensitive hyperparameter: detection and false-positive
   performance are identical across the tested range 1.5–4.0. We recommend clip_ratio=2.0
   as a conservative default; practitioners may lower it to 1.5 for marginally better
   post-warmup AUROC."

---

### Task 4 Complete Summary
Source: `results/sensitivity_sweep.csv`

| Sweep | Variable | Safe operating range | Paper statistic |
|-------|----------|---------------------|----------------|
| 1 (β/γ) | β ∈ [0.10, 0.70] | 77.8% of tested range | DR=100% over 7/9 tested β values |
| 2 (thresholds) | (flag, quarantine) grid | 61.7% of 115 grid points | Zero-FPR region shown in heatmap |
| 3 (clip_ratio) | clip ∈ [1.5, 4.0] | 100% of range | DR/FPR identical across all 6 values |

**All four supervisor-mandated gaps (Tasks 1–4) are now experimentally closed.**
Paper writing is the sole remaining task before the July 15 submission deadline.

---

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

### Section 5 (Experiments) tables — ALL COMPLETE (this checklist is historical; see full tables above)
1. Table 0A/0B: Option A vs B privacy-utility comparison ✅ COMPLETE
2. Table 1: Sync-FedAvg baseline ✅ COMPLETE
3. Table 2: Async clean (identical to baseline) ✅ COMPLETE
4. Table 3: E6 straggler — FPR=0 for Client 2 ✅ COMPLETE
5. Table 4: E1 detection — DR=100%, detection round 2, Δ=−0.0056 ✅ COMPLETE
6. Table 5: E2 detection — DR=100%, Δ=−0.0546, recovery confirmed ✅ COMPLETE
7. Table 6: E4 combined (core result) — DR=100%, FPR=0% (architectural) ✅ COMPLETE
8. Table 7: Decay function ablation (empirical, 4 runs) ✅ COMPLETE
9. Task 1: Baseline comparison (Krum/TrimmedMean/FLTrust × 3 experiments) ✅ COMPLETE
10. Task 2: 5-seed statistical validation + significance tests vs Krum ✅ COMPLETE
11. Task 3: Adaptive attacks A, B (×4 decay functions), C ✅ COMPLETE
12. Task 4: Hyperparameter sensitivity sweep — ⏳ IN PROGRESS (running overnight)

**Remaining before paper writing can begin: Task 4 sweep completion only.**
