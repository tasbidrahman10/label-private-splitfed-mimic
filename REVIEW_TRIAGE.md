# Triage of the ChatGPT "reviewer" report against the actual codebase

**Scope of this check:** every claim in [chatgpt_limitations.txt](chatgpt_limitations.txt) was verified against
[main.tex](main.tex), the implementation in [src/sfl/](src/sfl/), the experiment configs in [configs/](configs/),
the raw logs in [results/](results/), and the internal notes in [docs/](docs/) and [plan/](plan/).
Where a claim could be settled numerically, it was (Krum tie-breaking was re-simulated; FLTrust's
code path was traced end-to-end; the data files were opened).

**Headline:** the review is roughly **60% right, 25% right-for-the-wrong-reason, 15% wrong**.
Its single most damaging claim (Equation 9 is asymmetric) is **false** — the absolute value is
there in both the paper and the code. But three of its concerns are worse in reality than it
knew, and my own scan turned up **two problems it never saw that are more serious than anything
in its list** — chiefly target leakage in the dataset.

---

## Verdict table

### The review's eight "major concerns"

| # | Concern | Verdict | Evidence |
|---|---------|---------|----------|
| 1 | Eq. (9) norm score is asymmetric → zeroing attack always accepted | **WRONG** — but a real untested gap sits underneath | [main.tex:176](main.tex#L176) has `\left|\log(...)\right|`; [anomaly_detector.py:92](src/sfl/server/anomaly_detector.py#L92) has `abs(np.log(...))` |
| 2 | Attack is θ×10, not update scaling; detector compares absolute weights | **VALID** | [attack_simulator.py:25-33](src/sfl/common/attack_simulator.py#L25-L33), [anomaly_detector.py:45-94](src/sfl/server/anomaly_detector.py#L45-L94) |
| 3a | Krum invalid at n=3 (needs n ≥ 2f+3); Client-0 result is an artifact | **VALID — and worse than the review knew** | [baseline_aggregators.py:73-86](src/sfl/server/baseline_aggregators.py#L73-L86); reproduced numerically below |
| 3b | "880" is unexplained | **VALID** | 464 log rows = 170 decisions; 880 counts client-round rows, not decisions |
| 3c | Trimmed mean at ratio 0.2 / n=3 trims nothing | **VALID — already admitted in the paper** | [baseline_aggregators.py:115](src/sfl/server/baseline_aggregators.py#L115), [main.tex:359](main.tex#L359) |
| 3d | Reported "FLTrust" is uniform averaging, so it isn't FLTrust | **VALID — and the paper's explanation for why is wrong** | [app.py:427](src/sfl/server/app.py#L427) + [app.py:188](src/sfl/server/app.py#L188) → root gradient is always `zeros(1)` |
| 3e | Baselines not on the same seeds | **VALID** | [baseline_comparison.csv](results/baseline_comparison.csv) is seed-42 only for trimmed mean / FLTrust |
| 4a | Staleness = missed rounds ≠ model staleness | **VALID — and worse than the review knew** | [multi_runner.py:252-253](src/sfl/client/multi_runner.py#L252-L253): every client reloads the newest encoder each round |
| 4b | Algorithm 1 resets τ before aggregation | **VALID for the paper, not for the code** | Paper [main.tex:285](main.tex#L285) vs code [app.py:273](src/sfl/server/app.py#L273) then [app.py:341](src/sfl/server/app.py#L341) |
| 4c | It's buffered/windowed semi-async; no timing metrics reported | **VALID** | Clients run sequentially in one process; "delay" is `time.sleep` ([multi_runner.py:227](src/sfl/client/multi_runner.py#L227)) |
| 5 | Defense covers only encoder submission, not the server head | **VALID** | Attack is applied to the submitted state dict then reverted ([multi_runner.py:239-245](src/sfl/client/multi_runner.py#L239-L245)) — `g_φ` is never poisoned in any experiment |
| 6 | One task, one centre, three clients, no test set | **VALID** | [dataset.py:40-46](src/sfl/client/dataset.py#L40-L46) is an 80/20 train/val split; nothing is held out |
| 7 | Ablation disables gate **and** clipping, so the gain isn't attributable to SNAS | **VALID — self-admitted** | [close_gaps_runner.py:77-78](scripts/close_gaps_runner.py#L77-L78); [GAP_CLOSING_NOTES.md](results/gap_closing/GAP_CLOSING_NOTES.md) calls it "coarse" |
| 8 | 100% detection excludes warm-up rounds | **VALID** | "8/8 detection rounds" in [PAPER_RESULTS.md](results/PAPER_RESULTS.md); abstract says 100% unqualified |

### The review's "additional issues"

| Claim | Verdict |
|-------|---------|
| "Label-private" is misleading because the server evaluates the loss | **VALID against the paper, INVALID against the code** — see [§2.5](#25-the-paper-describes-the-wrong-protocol) |
| Entropy table compares SNAS clean vs Krum under attack | **VALID** — [main.tex:397-399](main.tex#L397-L399); the SNAS row is also analytic, not measured under attack |
| Table 9 mislabels d(τ) | **INVALID** — the draft has 8 tables; there is no Table 9 |
| Threshold selection reuses the evaluation setting | **VALID** — δ_q = 0.35 sits ~10% under the observed attacker SNAS of 0.39–0.42 |
| Preprocessing description inadequate | **VALID — and hiding a real problem** (see [§3.1](#31-target-leakage-the-most-serious-problem-in-the-project)) |
| Quantum FL/SL paragraphs are off-topic | Judgment call — defensible either way |
| Ref 1 (FedAvg) is bibliographically wrong | **INVALID** — [main.tex:566](main.tex#L566) already cites AISTATS 2017 correctly |
| FLTrust should be cited as NDSS 2021 | **INVALID** — [main.tex:581](main.tex#L581) already does |

The two bibliography complaints are simply wrong about the current file, which suggests the
reviewer was working from a stale copy or hallucinated them. Don't spend time on them.

---

## Part 1 — Where the review is wrong

### 1.1 Equation (9) is symmetric. The attack it describes does not work.

This is the claim the review calls "sufficient for a negative review on its own", and it is false.
Both the paper and the code take the **absolute value** of the log ratio:

```latex
A^{norm}_i = (1/K) Σ_k min( |log(‖θ_{i,k}‖ / (‖θ^g_k‖+ε))| / log(100), 1 )
```

```python
log_ratio = abs(np.log(max(ratio, 1e-8)))          # anomaly_detector.py:92
scores.append(min(log_ratio / LOG_CAP, 1.0))
```

The reviewer's worked example (θ_i = 0.1·θ_g) yields **+0.325**, not −0.325. With δ_flag = 0.22
that is already a **flag**, and once you add the ordinary cosine drift of a locally trained
encoder (0.17–0.22 observed in the logs) the score reaches ≈0.395 > δ_q = 0.35 → **quarantine**.

**But do not simply rebut this.** The reviewer stumbled onto a genuine hole by accident: the
attack-magnitude sweep in [main.tex:499-506](main.tex#L499-L506) only sweeps scale ≥ 1.2. The
symmetric evasion band below 1 (roughly 0.4×–1.0×) has never been tested. Closing it costs one
line — see P0-1.

### 1.2 The Krum "zero neighbours" mechanism is wrong; the conclusion is right anyway

The review guesses that k = n−f−2 = 0 makes every score zero. The code clamps
`k = max(1, n-2) = 1` ([baseline_aggregators.py:74-75](src/sfl/server/baseline_aggregators.py#L74-L75)),
so scores are not zero. **The outcome the reviewer predicted still happens, by a different route** —
see §2.1.

---

## Part 2 — Where the review is right, and worse than it knew

### 2.1 Krum's "federation collapse" is a tie-break artifact, and your own notes say so

With k = 1 and three clients (one attacker), the score of each honest client is its distance to
the *other* honest client — the **same number for both, bit-identical**. `min()` then returns
whichever key was inserted first, i.e. whichever client submitted first. Client 0 has delay 0s.

I reproduced this directly:

```
k = 1
scores: {0: 0.2002136670334604, 1: 77454.05971162449, 2: 0.2002136670334604}
score[0] == score[2]:  True
winner: 0
winner with reversed insertion order: 2
```

In Exp 3 it is starker: only two clients submit, so *both* scores are tied and the winner is
purely submission order — **if the attacker submitted first, Krum would select the attacker.**

Your own [PAPER_RESULTS.md:455-470](results/PAPER_RESULTS.md#L455) already states this:
*"a pure implementation artifact of submission order, not a security property"*, and even drafts
the honest framing. **The manuscript does not contain that disclosure.** [main.tex:357](main.tex#L357)
presents 880/880 Client-0 selection as a property of Krum, and the entropy claim H = 0 in
Table 4 rests on it.

This is the most serious *integrity* exposure in the submission: an internal document says
"artifact", the paper says "finding". A reviewer with code access finds this in ten minutes.

Two further details: "880 of 880 decision entries" counts **client-round log rows**, not
aggregation decisions (464 rows = 170 decisions in the on-disk log; the entropy script actually
used 168 selection events). And Krum at n=3, f=1 violates its own n ≥ 2f+3 condition, so the
configuration is outside its guarantee regardless.

### 2.2 The FLTrust result is a bug, not an architectural incompatibility

The paper claims FLTrust "reduces to uniform averaging because the server cannot compute
meaningful encoder gradients" ([main.tex:359](main.tex#L359)) and turns that into a contribution
("SNAS is the first Byzantine-robust method designed natively for split learning").

The actual cause is a state-lifecycle bug:

1. `/fedavg/open_round` sets `state.global_encoder = None` so clients get a 404 during the window ([app.py:427](src/sfl/server/app.py#L427)).
2. Aggregation runs later, while it is still `None`.
3. `_compute_fltrust_root_gradient` short-circuits on `state.global_encoder is None` and returns `np.zeros(1)` ([app.py:188-195](src/sfl/server/app.py#L188-L195)).
4. `fltrust_aggregate` sees `root_norm ≈ 0` and falls back to a plain mean ([baseline_aggregators.py:146-149](src/sfl/server/baseline_aggregators.py#L146-L149)).

The confirmation is in your own numbers: FLTrust and trimmed mean are **bit-identical** across
all three experiments (0.953913 / 0.822653 / 0.954101 in
[baseline_comparison.csv](results/baseline_comparison.csv)) because both are plain averaging.

The rest of `_compute_fltrust_root_gradient` is *correct* — it instantiates the global encoder,
runs root data through the server head, and backprops to encoder gradients. Swap
`state.global_encoder` → `state.prev_global_encoder` and FLTrust actually runs.

Two caveats once it does run: `fltrust_aggregate` compares **absolute weight vectors** against a
**gradient**, which is a category error (real FLTrust compares updates Δ against the server
update), and it rescales client weights to the gradient's norm, which will produce a near-zero
encoder. Both need fixing together with the null check.

### 2.3 Staleness is a counter, not staleness

Worse than the review realised. In [multi_runner.py:252-253](src/sfl/client/multi_runner.py#L252-L253)
**every trainer reloads the global encoder every round**, including one that just "missed" the
window. A "stale" client therefore trains from the *newest* model and submits an update that is
not stale at all — τ is pure bookkeeping, fully decoupled from model version.

That has a direct consequence for the paper's central claim. The honest-straggler experiment
(Exp 4) shows "no degradation" partly because there is no actual staleness to degrade anything.
And the attack-then-abstain evasion is characterised as a deep structural property of memoryless
scoring when it is at least partly a property of this simulation, where abstaining costs the
attacker nothing.

Also: clients run **sequentially in one process**, and delays are `time.sleep`. There is no
concurrency, so no wall-clock benefit exists to measure — which is why the paper reports none.
"Asynchronous" is accurate only as *windowed/buffered aggregation*.

### 2.4 The attack never touches the server head

`AttackSimulator` is applied to the submitted state dict and immediately reverted
([multi_runner.py:239-245](src/sfl/client/multi_runner.py#L239-L245)); the malicious client trains
honestly through the split forward/backward. So `g_φ` is never poisoned in any experiment. The
title and abstract say "intrusion-resilient SplitFed"; what is demonstrated is
**aggregation-time encoder replacement detection**. The review is right that this is a scope
overclaim.

### 2.5 The paper describes the wrong protocol

[main.tex:115](main.tex#L115) says *"The server computes predictions ŷ = g_φ(a_i), **evaluates the
supervised loss**, and returns ∂L/∂a_i to the client."* That is not what the code does. In
[trainer.py:142-151](src/sfl/client/trainer.py#L142-L151) the **client** computes the BCE loss
from returned logits and sends back the logit gradient; labels never leave the client. This is
U-shaped / label-private split learning (your "Option B").

So the reviewer's "label-private is misleading" objection is valid *against the text* — and the
fix strengthens the paper. Rewrite §3.1 to match the implementation and the label-privacy claim
becomes defensible rather than contradicted. Two sentences.

### 2.6 Detection rate denominator

"8/8 detection rounds" excludes warm-up. Over all submissions the rate is 8/10 = 80%. The
abstract's unqualified "100% detection" is not supportable as written; "100% of post-warm-up
attacker submissions (8/8), 80% of all attacker submissions" is, and costs nothing.

---

## Part 3 — What the review missed

### 3.1 Target leakage — the most serious problem in the project

The review complained that preprocessing is under-described. It is, and the description would
expose this:

- **`los` (length of stay) is a model input.** [data/processed/client0_medical.csv](data/processed/) column 1. LOS is only known at discharge and is strongly coupled to in-hospital death.
- **Labs are aggregated over the entire admission**, vitals over the entire ICU stay — min / max / mean / std / count with **no observation window** (no first-24h cut anywhere in [MIMIC_IV_SPLIT.ipynb](MIMIC_IV_SPLIT.ipynb)). The label is `hospital_expire_flag`. So the features include measurements taken during the terminal deterioration and, for non-survivors, right up to death.

This is almost certainly why AUROC is **0.965**. Published MIMIC-IV in-hospital mortality models
using first-24h features land at **0.85–0.90**. A 0.965 will be read as leakage by any clinical
ML reviewer, and they will be right.

None of the security conclusions are invalidated by this — the relative comparisons (SNAS vs
Krum vs no-detection) all hold on the same substrate. But the clinical framing is not defensible
and one reviewer question ("what is your observation window?") ends the discussion.

**Cheapest honest fix in 13 days:** drop `los`, state plainly that features are whole-stay
aggregates, and reframe the task as a *benchmark substrate for the security study* rather than a
clinical mortality model. Report the resulting AUROC honestly, whatever it is.

### 3.2 Admission-level rows leak across the train/val split

Rows are ICU stays (`stay_id`) but labs are joined at `hadm_id` and the label is admission-level.
Measured directly: **5,365 / 36,152 Medical rows share an identical lab feature block** (3,903 /
23,483 Surgical; 4,603 / 32,743 Cardiac) — i.e. multiple ICU stays from the same admission carry
identical features and identical labels. The 80/20 `train_test_split` is row-level and stratified
only on the label ([dataset.py:40-46](src/sfl/client/dataset.py#L40-L46)), so ~15% of rows have a
near-twin on the other side of the split.

Fix: group-aware split on `hadm_id` (ideally `subject_id`). The processed CSVs no longer carry
the IDs, so this requires re-exporting from the notebook — half a day, and it also gives you the
held-out test set the review asks for.

### 3.3 The "split-learning-native" claim rests on a disabled component

[plan/RA_Guideline_AsyncSplitFed.md:56](plan/RA_Guideline_AsyncSplitFed.md#L56) frames the novelty
as the **dual signal** — weights *and* activations, the latter being available only because
SplitFed transmits activations. But `snas_alpha: 0.0` ([configs/server.yaml:33](configs/server.yaml#L33))
disables the activation channel, on the empirically sound grounds that it doesn't discriminate
gradient scaling.

What remains — cosine + norm on submitted weights — is available in *any* FL setting and is not
split-learning-native at all. The activation divergence is implemented and logged
([anomaly_detector.py:20-42](src/sfl/server/anomaly_detector.py#L20-L42)); it is simply unused.

This is where the review's "Novelty: Moderate" comes from. Two options: run α > 0 against
free-rider / label-flip / backdoor (attacks where activation drift *should* discriminate, unlike
uniform scaling) and report it as a conditional signal; or drop the "split-learning-native"
framing and claim the SplitFed contribution at the *aggregation-object* level (the encoder) only.
The first is a real experiment; the second is an honest edit.

### 3.4 Two attacks are implemented but never run

`label_flip_proxy` and `backdoor` exist in [attack_simulator.py:36-69](src/sfl/common/attack_simulator.py#L36-L69)
and have never been executed — [GAP_CLOSING_NOTES.md](results/gap_closing/GAP_CLOSING_NOTES.md)
confirms this was deferred, and the E3/E5 numbering gap is why the experiments were relabelled.
The review's "add more attacks" is therefore much cheaper to satisfy than it looks: two configs
and a run.

### 3.5 The Sync-FedAvg baseline row has no reproducible config

Table 2 reports "Sync-FedAvg baseline 0.9649 (1 seed)". Your own notes flag it:
*"has no located config; recommend consolidating to this multiseeded async baseline or reproducing
the sync run before submission."* It is a single-seed number in a table of ten-seed numbers, and
you cannot currently regenerate it. Either reproduce it or delete the row.

---

## Part 4 — What to do, in priority order

Today is **2026-08-15**; the deadline is **2026-08-28**. Thirteen days. The plan below is sized
for that, and everything in P0 is text or a small code change, not new science.

### P0 — Blocking. Do not submit without these. (~3–4 days)

| # | Action | Effort | Why |
|---|--------|--------|-----|
| P0-1 | **Extend the attack-magnitude sweep below 1×.** Change `SCALES` in [run_attack_magnitude_sweep.py:50](scripts/run_attack_magnitude_sweep.py#L50) to `[0.1, 0.3, 0.5, 0.8, 1.2, 1.5, 2.0, 3.0, 5.0, 7.0, 10.0]` and rerun. | 1 line + ~2h compute | Pre-empts the review's zeroing/downscaling attack with data instead of a rebuttal. Add a rebuttal sentence noting Eq. (9) is already symmetric. |
| P0-2 | **Disclose the Krum tie-break.** Add the artifact caveat from [PAPER_RESULTS.md:455-470](results/PAPER_RESULTS.md#L455) to §5.2 and Table 4. State that at n=3, f=1 Krum is outside its n ≥ 2f+3 condition, that honest-client scores tie exactly, and that the winner is submission order — so with adversarial ordering Krum would select the attacker. | 1–2 h | The claim as written is unsupportable and your own notes contradict it. This *strengthens* the paper: "Krum is not merely collapsing federation, its selection is arbitrary here." |
| P0-3 | **Fix the "880".** Report Krum decisions, not log rows: "Client 0 selected in every aggregation decision (N = 3xx across E1/E2/E3 × 10 seeds; 880 client-round log entries)." | 30 min | The number is currently arithmetically unexplainable, exactly as the review says. |
| P0-4 | **Fix or withdraw FLTrust.** Preferred: change `state.global_encoder` → `state.prev_global_encoder` at [app.py:188](src/sfl/server/app.py#L188), switch `fltrust_aggregate` to compare **updates** Δ_i = θ_i − θ^g against the root gradient direction, rerun 3 experiments × 10 seeds. Fallback if time runs out: **remove FLTrust from Table 3** and state it was not evaluated because the reference-gradient adaptation was not validated. | 4–6 h + compute, or 30 min to withdraw | Publishing a plain mean as "FLTrust" is the kind of thing that gets a paper retracted, and the stated reason for it is provably not the actual reason. |
| P0-5 | **Correct §3.1 to describe the real protocol** — client computes the loss, labels never transmitted. | 20 min | Removes the "label-private is misleading" objection entirely and makes the privacy claim true. |
| P0-6 | **Fix Algorithm 1.** Split `arrival_staleness` (used for SNAS and aggregation weight) from the post-round registry update, matching [app.py:273](src/sfl/server/app.py#L273) / [app.py:341](src/sfl/server/app.py#L341). | 30 min | The code is correct; the pseudocode says the opposite and invites a correctness challenge. |
| P0-7 | **Qualify every detection claim.** Abstract and §5: "100% of post-warm-up attacker submissions (8/8); 80% including the warm-up round" and "0% observed FPR (0/16 honest client-rounds)". | 30 min | Cheap, and the review will otherwise call it overclaiming. |
| P0-8 | **Drop `los` from the feature set and describe preprocessing fully** — cohort, label definition (`hospital_expire_flag`), aggregation scope (whole-admission labs, whole-stay vitals, **no observation window**), imputation, scaling, per-client n and prevalence (36,152 / 16.1%; 23,483 / 11.5%; 32,743 / 6.9%). Reframe the benchmark as a substrate for the security study. | ~1 day incl. rerun | §3.1 above. Leaving 0.965 unexplained next to a silent whole-stay feature window is the biggest single reviewer risk after Krum. |
| P0-9 | **Narrow the title and threat model** to encoder-submission attacks. State explicitly that server-head poisoning during split training is out of scope and not defended. | 1 h | §2.4. Cheaper and more honest than pretending otherwise. |

### P1 — Strongly recommended; each removes a named reviewer objection. (~3–4 days)

| # | Action | Effort |
|---|--------|--------|
| P1-1 | **Factorial ablation.** You already have gate+clip ON (SNAS) and both OFF (nodetect). Add **clip-only** (`snas_threshold_* = 1e9`, `clip_ratio = 2.0`) and **gate-only** (`clip_ratio = 1e9`) via `/admin/reset` overrides in [close_gaps_runner.py:67-78](scripts/close_gaps_runner.py#L67-L78). Four cells × 10 seeds. | ~1 day compute, ~1 h code |
| P1-2 | **Run label-flip and backdoor** — already implemented, never executed. Two configs, mirroring [experiment_e1_gradient_scaling.yaml](configs/experiment_e1_gradient_scaling.yaml). Answers "only two attack types" and fills the E3/E5 gap for real. | ~4 h |
| P1-3 | **Run trimmed mean on all 10 seeds**, and swap trim ratio 0.2 → **coordinate-wise median**, which is the meaningful robust baseline at n=3. | ~4 h |
| P1-4 | **Report AUPRC and per-client AUROC.** Both are already logged in the client metrics JSONL — this is analysis only, no reruns. With prevalence spanning 6.9%–16.1%, AUPRC is the metric a clinical reviewer will ask for. | ~3 h |
| P1-5 | **Redefine the detector over updates** Δ_i = θ_i − θ_i^{base} rather than absolute weights, and report whether detection holds. This is the review's deepest technical point: cosine between absolute models is dominated by the shared base. If it holds, say so; if it doesn't, that is itself a finding worth reporting. | ~1 day |
| P1-6 | **Measure entropy for SNAS under attack**, not analytically-clean-vs-Krum-under-attack. `snas_entropy_from_windows()` in [compute_federation_entropy.py](scripts/compute_federation_entropy.py) already does this — just point it at the attack windows. | ~2 h |
| P1-7 | **Separate threshold calibration from evaluation.** Calibrate (δ_flag, δ_q) on the clean run plus one held-out attack magnitude, then evaluate on the rest. Say which data set the thresholds came from. | ~3 h |

### P2 — Real science, will not fit before 28 August. Put in Limitations / Future Work. (weeks)

- **N ≥ 5 clients** so Krum is validly configured, plus fraction-of-malicious sweeps. This is the review's biggest ask and the one you cannot satisfy in 13 days. Say so explicitly in Limitations rather than hoping it goes unnoticed.
- **Version-based staleness** τ_i = r − v_i with clients actually training from the model version they fetched. Requires reworking the client loop so a "stale" client does *not* reload the newest encoder ([multi_runner.py:252](src/sfl/client/multi_runner.py#L252)). Without this the async claims stay weak.
- **Genuine async timing metrics** — time-to-target AUROC, wall-clock, per-round wait. Requires actual concurrent clients, not sequential `time.sleep`.
- **Group-aware (patient-level) splits and a held-out test set** — needs re-export from the notebook with `subject_id`/`hadm_id` retained (§3.2).
- **Second dataset (eICU)** for multi-centre validation.
- **Server-head defense or rollback**, or accept the narrowed scope permanently.
- **Standard attack suite**: ALIE / min-max, Fang-style, adaptive-against-SNAS.

---

## Part 5 — Two viable strategies

**Strategy A — Submit on 28 August with a narrowed claim.** Do all of P0 and as much of P1 as
fits (P1-1, P1-2, P1-4 are the highest value per hour). Retitle around
*staleness-aware detection of encoder-submission attacks in windowed-asynchronous SplitFed*.
Move N ≥ 5, version-based staleness, real async timing, and the second dataset into an explicit
Limitations section. The paper becomes a smaller, correct contribution instead of a larger,
challengeable one. **This is achievable and is what I'd recommend.**

**Strategy B — Skip this deadline.** Do P0 + P1 + the group-aware split and N = 5/10 clients, and
target the next suitable venue. The work is genuinely better with a valid Krum configuration and
a leakage-free benchmark, and Krum's tie-break disclosure hurts far less when n ≥ 5 makes the
comparison legitimate.

What is **not** viable: submitting as-is. Not because the review is right about Equation (9) — it
isn't — but because of P0-2 (an internal document calls a headline result an artifact), P0-4 (a
plain mean is reported as FLTrust, with a stated cause that is not the actual cause), and P0-8
(a 0.965 AUROC standing on whole-admission features plus a length-of-stay input).

---

## Appendix — the three code fixes, concretely

**A. FLTrust null check** — [app.py:188](src/sfl/server/app.py#L188)
```python
# was: if state.fltrust_root_data is None or state.global_encoder is None:
ref_encoder = state.global_encoder or state.prev_global_encoder
if state.fltrust_root_data is None or ref_encoder is None:
```
…and use `ref_encoder` in `encoder.load_state_dict(...)` at [app.py:207](src/sfl/server/app.py#L207).
Then change `fltrust_aggregate` to operate on Δ_i = θ_i − θ^g rather than on θ_i.

**B. Sub-unit attack scales** — [run_attack_magnitude_sweep.py:50](scripts/run_attack_magnitude_sweep.py#L50)
```python
SCALES = [0.1, 0.3, 0.5, 0.8, 1.2, 1.5, 2.0, 3.0, 5.0, 7.0, 10.0]
```

**C. Factorial ablation arms** — [close_gaps_runner.py:67-78](scripts/close_gaps_runner.py#L67-L78)
```python
GATE_ON_CLIP_ON   = {"snas_threshold_flag": 0.22, "snas_threshold_quarantine": 0.35, "fedavg_clip_ratio": 2.0}
GATE_OFF_CLIP_ON  = {"snas_threshold_flag": 1e9,  "snas_threshold_quarantine": 1e9,  "fedavg_clip_ratio": 2.0}
GATE_ON_CLIP_OFF  = {"snas_threshold_flag": 0.22, "snas_threshold_quarantine": 0.35, "fedavg_clip_ratio": 1e9}
GATE_OFF_CLIP_OFF = {"snas_threshold_flag": 1e9,  "snas_threshold_quarantine": 1e9,  "fedavg_clip_ratio": 1e9}
```
