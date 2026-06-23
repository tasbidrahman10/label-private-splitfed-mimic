# RA Research Guideline
## Asynchronous Split Federated Learning with Staleness-Aware Aggregation and Malicious Client Detection on MIMIC-IV

**Supervisor:** Dr. Sumaiya Tabassum Nimi  
**Framework Base:** Label-Private SplitFed on MIMIC-IV (Student Implementation — December 2025)  
**Extension Target:** Async-SplitFed-IR (Intrusion-Resilient Asynchronous Split Federated Learning)

---

## Part 0 — What You Have Already Built

Before describing what needs to be done next, it is important to recognize exactly what your current implementation achieves and why it is a strong foundation.

Your submitted report establishes a working **label-private Split Federated Learning system** with the following verified components:

| Component | Status | Location in Your Code |
|---|---|---|
| 3-client non-IID MIMIC-IV partition (Medical, Surgical, Cardiac) | ✅ Complete | `src/sfl/client/dataset.py` |
| ClientEncoder (266 → 64 → 32 dims) | ✅ Complete | `src/sfl/common/models.py` |
| ServerTopModel (32 → 64 → 1 logit) | ✅ Complete | `src/sfl/common/models.py` |
| Label-private forward/backward protocol | ✅ Complete | `src/sfl/client/trainer.py` |
| FastAPI server with activation/gradient endpoints | ✅ Complete | `src/sfl/server/app.py` |
| Synchronous FedAvg over client encoders | ✅ Complete | `src/sfl/server/fedavg.py` |
| AUROC/AUPRC/F1 local validation | ✅ Complete | `src/sfl/common/metrics.py` |
| Cloudflare tunnel for distributed testing | ✅ Complete | Deployment setup |
| 5-round experiment, mean AUROC 0.9644 | ✅ Verified | Experiment summary CSV |

**The critical limitation you yourself identified in Section 14 of your report:**

> *"The framework does not claim that activations are cryptographically secure. Additional security mechanisms such as activation perturbation, differential privacy, secure aggregation, or quantum-inspired feature scrambling can be layered on top of this substrate in later research extensions."*

This extension directly addresses that limitation by adding **asynchrony resilience** and **adversarial robustness** on top of your existing substrate. You will not rewrite your system — you will extend it.

---

## Part 1 — Research Problem Statement

### 1.1 The Gap in Your Current System

Your current FedAvg implementation in `src/sfl/server/fedavg.py` operates synchronously:

```
Round opens → Wait for ALL 3 clients → Aggregate → Distribute global encoder → Repeat
```

This is unrealistic for two reasons:

**Reason 1 — Availability heterogeneity.** In real IoMT deployments, a cardiac ICU monitor, a wearable device, and a surgical suite system will not complete their local training in the same time window. Network lag, compute differences, and patient load variation mean clients naturally arrive at different times.

**Reason 2 — Security blindness.** Your current system accepts any encoder submission from a registered client. A malicious actor who compromises one of the three client nodes can submit arbitrarily poisoned encoder weights and they will be averaged into the global model without any check.

### 1.2 The Research Question

> *In an asynchronous Split Federated Learning system where clients submit updates at different times, how can the server distinguish between a legitimately slow client (whose update is merely stale) and a malicious client (whose update is adversarially crafted), and aggregate updates robustly under both conditions?*

This question has not been formally studied in the split learning literature. Most Byzantine-robust FL work operates on weight space only. Your system gives you an additional signal — the **activation space** — because clients transmit activations during the forward pass. This dual-signal setting (weights + activations) is what makes this contribution novel.

---

## Part 2 — System Architecture for the Extension

The extension adds three new modules to your existing server without modifying the client training logic or the label-private protocol.

### 2.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    EXISTING CLIENT SIDE (unchanged)                 │
│  ClientEncoder → activation → POST /option-b/forward               │
│  ← logit ← BCELoss ← dL/dz → POST /option-b/backward ← dL/dA ←   │
│  POST /fedavg/submit_encoder    GET /fedavg/global_encoder          │
└─────────────────────────────────────────────────────────────────────┘
                              ↕ HTTP (FastAPI, unchanged endpoints)
┌─────────────────────────────────────────────────────────────────────┐
│                    EXTENDED SERVER SIDE                             │
│                                                                     │
│  ┌─────────────────────┐                                            │
│  │  Time Window        │  NEW: Replaces synchronous wait            │
│  │  Controller         │  Opens window T, aggregates on deadline    │
│  └──────────┬──────────┘                                            │
│             ↓                                                        │
│  ┌─────────────────────┐                                            │
│  │  Staleness Registry │  NEW: Tracks τ_i per client per round      │
│  │                     │  Computes decay weight w_i = f(τ_i)        │
│  └──────────┬──────────┘                                            │
│             ↓                                                        │
│  ┌─────────────────────┐                                            │
│  │  Malicious Client   │  NEW: Computes SNAS_i per submission       │
│  │  Detection Gate     │  Quarantines or flags clients              │
│  └──────────┬──────────┘                                            │
│             ↓                                                        │
│  ┌─────────────────────┐                                            │
│  │  Robust Async FedAvg│  MODIFIED: Weighted average using          │
│  │  Aggregator         │  staleness-decayed, anomaly-gated weights  │
│  └─────────────────────┘                                            │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.2 New Files to Create

Add the following to your existing `src/sfl/server/` directory:

```
src/sfl/server/
    async_controller.py      ← Time window management
    staleness.py             ← Staleness registry and decay functions
    anomaly_detector.py      ← SNAS computation and client gating
    robust_fedavg.py         ← Weighted aggregation (replaces fedavg.py)
    trust_state.py           ← Per-client trust scores and quarantine state
src/sfl/common/
    attack_simulator.py      ← Inject attacks for evaluation
```

---

## Part 3 — Algorithmic Details

### 3.1 Module 1 — Time Window Controller (`async_controller.py`)

#### Concept

Instead of blocking until all 3 clients submit, the server opens a timed window per round. When the window closes, aggregation proceeds with whoever has submitted.

#### Algorithm

```
ASYNC_ROUND_CONTROLLER:

Input:  registered_clients C = {c_0, c_1, c_2}
        base_window_duration T_base (seconds)
        max_consecutive_misses K_max

State:  submission_buffer: dict[client_id → encoder_weights]
        arrival_timestamps: dict[client_id → float]
        round_number: int
        consecutive_misses: dict[client_id → int]

On round start:
    1. Broadcast round_start signal to all clients
    2. Open submission window:
           T_adaptive = T_base × (1 + 0.1 × mean(staleness_counts))
           deadline = current_time + T_adaptive
    3. Accept submissions until deadline:
           while current_time < deadline:
               if submission received from client c_i:
                   submission_buffer[c_i] = weights
                   arrival_timestamps[c_i] = current_time
    4. On deadline:
           submitted = keys(submission_buffer)
           missing = C \ submitted
           for c_i in missing:
               consecutive_misses[c_i] += 1
               if consecutive_misses[c_i] >= K_max:
                   flag c_i for deep inspection
           for c_i in submitted:
               consecutive_misses[c_i] = 0
    5. Pass submission_buffer to StalenessRegistry
    6. Clear submission_buffer for next round
```

#### Implementation Notes

In your existing `app.py`, the `/fedavg/submit_encoder` endpoint collects submissions and the server waits for `len(submissions) == num_clients`. Replace this blocking check with a background task that runs the window controller. Use Python's `asyncio` since FastAPI is already async:

```python
# In async_controller.py
import asyncio
from datetime import datetime

class AsyncRoundController:
    def __init__(self, client_ids, base_window_seconds=60, max_misses=3):
        self.client_ids = set(client_ids)
        self.T_base = base_window_seconds
        self.K_max = max_misses
        self.submission_buffer = {}
        self.consecutive_misses = {c: 0 for c in client_ids}
        self.round_number = 0
        self._round_open = False

    async def open_round(self, staleness_counts: dict) -> dict:
        """Opens async window, returns submissions received before deadline."""
        self._round_open = True
        self.submission_buffer = {}
        mean_staleness = sum(staleness_counts.values()) / len(staleness_counts)
        T_adaptive = self.T_base * (1 + 0.1 * mean_staleness)
        await asyncio.sleep(T_adaptive)
        self._round_open = False
        self._update_miss_counts()
        self.round_number += 1
        return dict(self.submission_buffer)

    def receive_submission(self, client_id, weights):
        if self._round_open and client_id in self.client_ids:
            self.submission_buffer[client_id] = weights

    def _update_miss_counts(self):
        for c in self.client_ids:
            if c in self.submission_buffer:
                self.consecutive_misses[c] = 0
            else:
                self.consecutive_misses[c] += 1
```

#### Simulation Mode

For experiments without real network delay, simulate asynchrony by adding configurable artificial delays per client before submission. Add a `delay_config` parameter to `multi_runner.py`:

```python
# In multi_runner.py — add before each client's submit_encoder call
delay_seconds = delay_config.get(client_id, 0)
time.sleep(delay_seconds)
client.submit_encoder(round_num)
```

This lets you simulate a slow Cardiac client (high delay) or a missing Surgical client (delay > window) without needing three separate machines.

---

### 3.2 Module 2 — Staleness Registry (`staleness.py`)

#### Concept

For every client, track how many rounds have passed since it last successfully contributed to aggregation. Use this staleness counter τ_i to compute a downweighting factor.

#### Staleness Counter Definition

```
τ_i(r) = number of consecutive rounds before round r in which client i did NOT submit
```

So if Client 1 submitted in rounds 1, 2, missed round 3, missed round 4, and submitted in round 5:
- τ_1(1) = 0, τ_1(2) = 0, τ_1(5) = 2

#### Decay Functions

Implement all four as selectable options. You will compare them in ablation experiments.

```python
# In staleness.py
import math

class StalenessDecay:

    @staticmethod
    def linear(tau: int, alpha: float = 1.0) -> float:
        """w = 1 / (1 + alpha * tau)"""
        return 1.0 / (1.0 + alpha * tau)

    @staticmethod
    def exponential(tau: int, alpha: float = 0.5) -> float:
        """w = exp(-alpha * tau)"""
        return math.exp(-alpha * tau)

    @staticmethod
    def polynomial(tau: int, beta: float = 2.0) -> float:
        """w = 1 / (1 + tau^beta)"""
        return 1.0 / (1.0 + tau ** beta)

    @staticmethod
    def step(tau: int, cutoff: int = 3) -> float:
        """w = 1 if tau <= cutoff else 0"""
        return 1.0 if tau <= cutoff else 0.0
```

#### Final Aggregation Weight

The base weight for each client is its dataset size (as you already do in FedAvg). The staleness-aware weight multiplies this:

```
w_i_final = (n_i / N_submitted) × decay(τ_i)
```

Where n_i is client i's sample count and N_submitted is the total samples across submitted clients only (not all registered clients). Renormalize so weights sum to 1.

---

### 3.3 Module 3 — Malicious Client Detection Gate (`anomaly_detector.py`)

This is the most novel component. It computes the **Staleness-Normalized Anomaly Score (SNAS)** for each submitted client.

#### Two Detection Signals

**Signal A — Activation Distribution Divergence**

During the forward pass (which already happens in your `/option-b/forward` endpoint), the server receives a batch of 32-dim activations from each client. Maintain a rolling reference distribution per client built from previous rounds when the client was verified honest.

The divergence is computed as the **symmetric KL divergence** between the current activation batch distribution and the client's reference:

```python
import numpy as np
from scipy.stats import entropy

def activation_divergence(current_activations: np.ndarray,
                           reference_mean: np.ndarray,
                           reference_std: np.ndarray) -> float:
    """
    current_activations: shape (batch_size, 32)
    Computes feature-wise KL divergence averaged across dimensions.
    """
    curr_mean = current_activations.mean(axis=0)
    curr_std  = current_activations.std(axis=0) + 1e-8
    ref_std   = reference_std + 1e-8

    # KL(current || reference) approximated via Gaussian assumption
    kl = (np.log(ref_std / curr_std)
          + (curr_std**2 + (curr_mean - reference_mean)**2) / (2 * ref_std**2)
          - 0.5)
    return float(kl.mean())
```

Update the reference after each clean round using an exponential moving average:

```python
alpha_ema = 0.1
ref_mean = (1 - alpha_ema) * ref_mean + alpha_ema * curr_mean
ref_std  = (1 - alpha_ema) * ref_std  + alpha_ema * curr_std
```

**Signal B — Encoder Weight Cosine Anomaly**

After a client submits encoder weights, compute cosine similarity against the previous global encoder:

```python
import torch
import torch.nn.functional as F

def weight_cosine_anomaly(submitted_weights: dict,
                           global_weights: dict) -> float:
    """Returns 1 - cosine_similarity, averaged across all parameter tensors."""
    scores = []
    for key in submitted_weights:
        w_sub = submitted_weights[key].flatten().float()
        w_glb = global_weights[key].flatten().float()
        sim = F.cosine_similarity(w_sub.unsqueeze(0),
                                   w_glb.unsqueeze(0)).item()
        scores.append(1.0 - sim)
    return float(np.mean(scores))
```

#### SNAS — The Novel Detection Statistic

```
SNAS_i = (α × divergence_i + β × cosine_anomaly_i) / (1 + τ_i)
```

Where α and β are weighting coefficients (default α = 0.6, β = 0.4, tunable). The denominator normalizes the anomaly by staleness: a stale but honest client naturally drifts more, so its raw anomaly score is dampened.

```python
# In anomaly_detector.py
class MaliciousClientGate:

    def __init__(self, threshold_quarantine=0.8, threshold_flag=0.4,
                 alpha=0.6, beta=0.4):
        self.threshold_q = threshold_quarantine
        self.threshold_f = threshold_flag
        self.alpha = alpha
        self.beta = beta
        # Per-client reference state
        self.ref_means = {}   # client_id → np.ndarray (32,)
        self.ref_stds  = {}   # client_id → np.ndarray (32,)

    def compute_snas(self, client_id, current_activations,
                      submitted_weights, global_weights, staleness) -> float:
        div  = activation_divergence(current_activations,
                                      self.ref_means[client_id],
                                      self.ref_stds[client_id])
        anom = weight_cosine_anomaly(submitted_weights, global_weights)
        snas = (self.alpha * div + self.beta * anom) / (1 + staleness)
        return snas

    def gate(self, client_id, snas) -> str:
        """Returns 'accept', 'flag', or 'quarantine'."""
        if snas > self.threshold_q:
            return 'quarantine'
        elif snas > self.threshold_f:
            return 'flag'
        else:
            return 'accept'

    def update_reference(self, client_id, activations, ema=0.1):
        curr_mean = activations.mean(axis=0)
        curr_std  = activations.std(axis=0) + 1e-8
        if client_id not in self.ref_means:
            self.ref_means[client_id] = curr_mean
            self.ref_stds[client_id]  = curr_std
        else:
            self.ref_means[client_id] = (1-ema)*self.ref_means[client_id] + ema*curr_mean
            self.ref_stds[client_id]  = (1-ema)*self.ref_stds[client_id]  + ema*curr_std
```

#### Trust Score Evolution

Maintain a per-client trust score T_i initialized to 1.0 and updated based on gate decisions:

```
If 'accept':      T_i ← min(1.0,  T_i + 0.05)
If 'flag':        T_i ← T_i × 0.9
If 'quarantine':  T_i ← T_i × 0.5,  exclude from aggregation this round
```

A quarantined client can re-enter after K_rehab consecutive clean rounds. Store this in `trust_state.py`.

---

### 3.4 Module 4 — Robust Async FedAvg (`robust_fedavg.py`)

Replace your current `fedavg.py` with a version that incorporates staleness decay and anomaly gating.

#### Full Aggregation Algorithm

```
ROBUST_ASYNC_FEDAVG(submissions, staleness_counts, gate_decisions, sample_counts):

Input:
    submissions: dict[client_id → encoder_state_dict]  (only submitted clients)
    staleness_counts: dict[client_id → τ_i]
    gate_decisions: dict[client_id → 'accept'|'flag'|'quarantine']
    sample_counts: dict[client_id → n_i]
    decay_fn: one of {linear, exponential, polynomial, step}

Step 1 — Filter quarantined clients:
    accepted = {c: w for c, w in submissions.items()
                if gate_decisions[c] != 'quarantine'}

Step 2 — Compute raw weights:
    for each c_i in accepted:
        staleness_weight = decay_fn(staleness_counts[c_i])
        # Apply extra penalty for flagged clients
        flag_penalty = 0.5 if gate_decisions[c_i] == 'flag' else 1.0
        raw_weight[c_i] = sample_counts[c_i] × staleness_weight × flag_penalty

Step 3 — Normalize:
    total = sum(raw_weight.values())
    norm_weight[c_i] = raw_weight[c_i] / total

Step 4 — Weighted average:
    global_encoder = {}
    for key in encoder_keys:
        global_encoder[key] = sum(
            norm_weight[c_i] × accepted[c_i][key]
            for c_i in accepted
        )

Step 5 — Distribute global_encoder to all non-quarantined clients

Return global_encoder, norm_weight (for logging)
```

```python
# In robust_fedavg.py
import torch

class RobustAsyncFedAvg:
    def __init__(self, decay_fn=StalenessDecay.exponential):
        self.decay_fn = decay_fn

    def aggregate(self, submissions, staleness_counts,
                  gate_decisions, sample_counts) -> dict:
        accepted = {c: w for c, w in submissions.items()
                    if gate_decisions.get(c, 'accept') != 'quarantine'}
        if not accepted:
            return None  # No valid updates this round

        raw_weights = {}
        for c in accepted:
            s = self.decay_fn(staleness_counts.get(c, 0))
            p = 0.5 if gate_decisions.get(c) == 'flag' else 1.0
            raw_weights[c] = sample_counts[c] * s * p

        total = sum(raw_weights.values())
        norm = {c: raw_weights[c] / total for c in accepted}

        global_state = {}
        keys = list(next(iter(accepted.values())).keys())
        for key in keys:
            global_state[key] = sum(
                norm[c] * accepted[c][key].float() for c in accepted
            )
        return global_state
```

---

## Part 4 — Attack Simulator (`attack_simulator.py`)

You need to inject attacks to evaluate your detection. Add an `AttackSimulator` class that wraps a client's encoder submission.

### Attack Implementations

```python
# In src/sfl/common/attack_simulator.py
import torch
import copy

class AttackSimulator:

    @staticmethod
    def gradient_scaling(weights: dict, scale: float = 10.0) -> dict:
        """Multiply all weight tensors by a large scalar."""
        return {k: v * scale for k, v in weights.items()}

    @staticmethod
    def label_flip_proxy(weights: dict, noise_std: float = 0.3) -> dict:
        """Approximate label-flip effect: add large structured noise."""
        return {k: v + torch.randn_like(v) * noise_std for k, v in weights.items()}

    @staticmethod
    def free_rider(weights: dict) -> dict:
        """Submit random weights of the same shape."""
        return {k: torch.randn_like(v) for k, v in weights.items()}

    @staticmethod
    def backdoor(weights: dict, trigger_scale: float = 5.0,
                  trigger_fraction: float = 0.1) -> dict:
        """Perturb a fraction of weights by a large amount."""
        poisoned = copy.deepcopy(weights)
        for k, v in poisoned.items():
            mask = (torch.rand_like(v) < trigger_fraction).float()
            poisoned[k] = v + mask * trigger_scale
        return poisoned

    @staticmethod
    def slow_poisoner(weights: dict, delay_rounds: int = 2,
                       scale: float = 5.0) -> dict:
        """
        Simulates a client that waits (becomes stale) then submits
        a scaled poisoned update. Pass τ=delay_rounds to staleness registry
        alongside this submission to simulate the timing.
        """
        return AttackSimulator.gradient_scaling(weights, scale)
```

In `multi_runner.py`, add an `attack_config` parameter:

```python
attack_config = {
    'client_1': {'type': 'gradient_scaling', 'params': {'scale': 10.0}},
    # 'client_2': {'type': 'free_rider'},
}
```

Apply the attack before the client's encoder submission call.

---

## Part 5 — Integration: What Changes in Existing Files

### `src/sfl/server/app.py`

Add two new behaviors:

1. **Store activations per client per round** in the forward endpoint for the anomaly detector. Extend the existing activation context storage (Section 9 of your report) to also pass to `MaliciousClientGate.update_reference()` after each backward step.

2. **Change FedAvg trigger** from `if len(submissions) == num_clients` to triggering via the async controller's window deadline.

Specifically, in your current `app.py`, find the FedAvg trigger logic and replace:

```python
# CURRENT (synchronous — from your report Section 7)
if len(fedavg_state.submissions) == len(registered_clients):
    global_encoder = fedavg(fedavg_state.submissions, sample_counts)
    fedavg_state.clear()
```

Replace with:

```python
# NEW (async — trigger from controller's deadline callback)
async def on_round_deadline(submitted_clients: dict):
    gate_decisions = {}
    for c_id, weights in submitted_clients.items():
        activations = activation_buffer.get(c_id)
        staleness = staleness_registry.get_staleness(c_id)
        snas = gate.compute_snas(c_id, activations, weights,
                                  global_encoder_state, staleness)
        gate_decisions[c_id] = gate.gate(c_id, snas)
        trust_state.update(c_id, gate_decisions[c_id])
        gate.update_reference(c_id, activations)
    
    new_global = robust_fedavg.aggregate(
        submitted_clients, staleness_registry.counts,
        gate_decisions, sample_counts
    )
    staleness_registry.update_after_round(set(submitted_clients.keys()))
    log_round_metrics(gate_decisions, staleness_registry.counts, new_global)
```

### `src/sfl/server/state.py`

Add fields for:
- `activation_buffer`: stores the most recent activation batch per client within a round
- `trust_scores`: per-client trust score
- `quarantine_list`: set of currently quarantined client IDs
- `staleness_counts`: per-client τ values

### `src/sfl/client/trainer.py`

No changes needed. The label-private protocol is unchanged. All new logic lives server-side.

### `src/sfl/client/multi_runner.py`

Add:
- `delay_config` dictionary for simulating async arrival
- `attack_config` dictionary for injecting attacks before submission

---

## Part 6 — Experimental Protocol

### 6.1 Baseline Experiments (reproduce first)

Run your existing synchronous system on the same 3 clients for 10 rounds to get a stable performance baseline. Record per-client AUROC per round. This is your **Sync-FedAvg** baseline.

### 6.2 Async Experiments (no attacks)

Simulate three delay scenarios using `delay_config` in `multi_runner.py`:

| Scenario | Client 0 Delay | Client 1 Delay | Client 2 Delay |
|---|---|---|---|
| **Mild lag** | 0s | 5s | 10s |
| **One dropout** | 0s | 0s | > window T |
| **Rotating dropout** | Alternates by round | Alternates | Alternates |

Measure: AUROC per client, global AUROC, rounds where aggregation happened with < 3 clients, effect of each decay function.

### 6.3 Attack Experiments (the key contribution)

Run each attack type individually and then in combination. For each, measure:

- **Detection rate**: fraction of malicious rounds where SNAS > threshold_quarantine
- **False positive rate**: fraction of honest-stale rounds incorrectly quarantined
- **AUROC degradation**: global AUROC under attack vs. no-attack baseline
- **Recovery speed**: number of rounds for AUROC to return within 1% of clean baseline after attack stops

| Experiment | Attack Type | Attacker | Honest Stale Client |
|---|---|---|---|
| E1 | Gradient scaling (×10) | Client 1 | None |
| E2 | Free rider | Client 2 | None |
| E3 | Slow poisoner (τ=2) | Client 1 | None |
| E4 | Gradient scaling | Client 1 | Client 2 (dropout) |
| E5 | Backdoor | Client 0 | Client 2 (lag) |
| E6 | No attack | None | Client 2 (chronic lag) |

E4 and E5 are the hardest cases — they require your system to simultaneously handle a real attacker and a legitimate straggler, which is the core problem your SNAS statistic is designed to solve.

### 6.4 Ablation Studies

| Ablation | Variable | Fixed |
|---|---|---|
| Decay function | {linear, exponential, polynomial, step} | α=0.5, thresholds default |
| SNAS threshold | threshold_quarantine ∈ {0.4, 0.6, 0.8, 1.0} | exponential decay |
| α/β balance | α ∈ {0.2, 0.4, 0.6, 0.8}, β = 1-α | exponential decay |
| Window size T | T ∈ {30s, 60s, 120s, ∞} | exponential decay, default thresholds |

---

## Part 7 — Metrics and Logging

### 7.1 New Metrics to Log Per Round

Extend your existing `client_metrics.jsonl` and `server_metrics.jsonl` to include:

```json
{
  "round": 3,
  "client_id": "client_1",
  "staleness_tau": 2,
  "staleness_weight": 0.135,
  "activation_divergence": 0.42,
  "cosine_anomaly": 0.31,
  "snas": 0.243,
  "gate_decision": "flag",
  "trust_score": 0.81,
  "included_in_aggregation": true,
  "aggregation_weight_normalized": 0.287
}
```

### 7.2 Summary Statistics for the Paper

For each experimental condition, report:

- Mean AUROC ± std across rounds (per client and global)
- Detection rate (True Positive Rate for malicious clients)
- False Positive Rate (stale-but-honest clients flagged)
- Mean rounds to first detection
- Mean rounds to AUROC recovery after attack cessation

---

## Part 8 — Paper Structure (Target Contribution)

Based on what you will implement, the paper structure should be:

**Title:** *Staleness-Normalized Anomaly Detection for Asynchronous Split Federated Learning in Clinical IoMT*

**Section 1 — Introduction:** Motivate async IoMT, gap between async FL and split FL literature, the stale-vs-adversarial problem.

**Section 2 — Related Work:** Async FL (FedAsync, ASGD-based FL), Byzantine-robust FL (Krum, FLAME, FLTrust), Split FL (Thapa et al.), healthcare federated learning. Show none combine all three.

**Section 3 — System Model:** Your existing SplitFed architecture (cite your current implementation), extend with formal async and threat model definitions.

**Section 4 — Async-SplitFed-IR:** Present the four modules. Formally define SNAS. Prove that for an honest stale client, SNAS grows sub-linearly with τ, while for an adversarial client it grows super-linearly (this is the key theoretical claim — you will need to sketch an informal proof).

**Section 5 — Experiments:** MIMIC-IV setup (cite your current partitioning), attack scenarios, results.

**Section 6 — Conclusion**

---

## Part 9 — Implementation Timeline

| Week | Task | Deliverable |
|---|---|---|
| 1 | Implement `staleness.py` and `async_controller.py`; simulation mode in `multi_runner.py` | Async system running, no attacks |
| 2 | Implement `anomaly_detector.py` and `trust_state.py`; integrate activation buffer into `state.py` | SNAS computed and logged per round |
| 3 | Implement `robust_fedavg.py`; integrate into `app.py` | Full pipeline end-to-end |
| 4 | Implement `attack_simulator.py`; run baseline and mild attack experiments | E1, E2, E6 results |
| 5 | Run E3, E4, E5 (harder attack scenarios) | Core results table |
| 6 | Ablation studies | Ablation tables |
| 7 | Write Sections 3 and 4 of paper | Draft |
| 8 | Write Sections 1, 2, 5, 6; revise | Full draft for supervisor review |

---

## Part 10 — Common Pitfalls to Avoid

**On the activation buffer:** Your server currently discards the activation context after the backward step (Section 9 of your report says *"The context is removed after the backward step"*). You need to save a copy of the activation *before* discarding the context, for use in the anomaly detector. Do not change the backward logic — add a copy operation before context removal.

**On normalizing SNAS thresholds:** SNAS values depend on the scale of your 32-dim activations, which in turn depend on your BatchNorm layers. Calibrate your thresholds on clean data first (rounds 1-3 with no attack) before running attack experiments. Log all raw SNAS values in early runs to understand the natural distribution.

**On the slow poisoner experiment:** When simulating the slow poisoner, you must both (a) apply the attack to the weights and (b) register the staleness τ=2 in the staleness registry. If you forget (b), the system will correctly penalize the update for staleness but will not face the core challenge of the slow poisoner pattern.

**On FedAvg with only one client submitting:** Your `robust_fedavg.py` must handle the edge case where only one client passes the gate. In this case the aggregation weight is 1.0 for that client and the global encoder simply becomes that client's encoder. This is correct behavior — do not skip aggregation.

**On reference distribution initialization:** For the first 2-3 rounds, the anomaly detector has no reference to compare against. Initialize the reference with the first submission from each client and flag these early rounds as warm-up. Do not make gate decisions during warm-up.

---

*Guideline version 1.0 — for internal RA use. To be updated after Week 4 experimental results.*
