# SENTINEL-SplitFed

**Staleness-Aware Encoder Anomaly Detection for Intrusion-Resilient Asynchronous Split Federated Learning in Critical-Care Prediction**

![Status](https://img.shields.io/badge/status-research-blue) ![Python](https://img.shields.io/badge/python-3.10%2B-blue) ![Task](https://img.shields.io/badge/task-MIMIC--IV%20mortality-purple)

SENTINEL-SplitFed is a server-side extension for **asynchronous split federated learning** that adds **intrusion resilience** to label-private clinical prediction. It replaces synchronous, trust-everyone aggregation with timed asynchronous windows, per-client staleness tracking, and a **Staleness-Normalized Anomaly Score (SNAS)** that separates *honest-but-late* clients from *malicious* ones before aggregation — a distinction prior split-learning systems do not make.

This repository contains the full implementation, experiment configurations, results, and figure-generation code accompanying the paper.

---

## Paper

> **SENTINEL-SplitFed: Staleness-Aware Encoder Anomaly Detection for Intrusion-Resilient Asynchronous Split Federated Learning in Critical-Care Prediction.**
> *Under review, Springer Nature Machine Learning — Special Issue on Federated Learning for Critical Applications (2026).*

A BibTeX entry will be added here upon acceptance. See [Citation](#citation).

---

## Why this exists

Collaborative clinical ML must satisfy three competing goals at once: **predictive performance**, **privacy**, and **operational robustness**. Split Federated Learning (SplitFed) keeps raw features and labels on the client and shares only intermediate activations — good for privacy. But standard SplitFed pipelines are:

1. **Synchronous** — the server waits for every client, so one slow hospital stalls the whole round; and
2. **Trust-blind** — any registered client's encoder is accepted, so a compromised client can poison the global model.

The core difficulty is that a *legitimately stale* client and a *malicious* client both look "different" from the current global encoder. SENTINEL-SplitFed is, to our knowledge, the first **split-learning-native** mechanism to resolve that ambiguity before aggregation.

---

## Key contributions

- **SNAS** — a server-side anomaly gate combining directional (cosine) and magnitude (norm) encoder anomalies, normalized by client staleness `τ` so that benign delay is not mistaken for malice.
- **Asynchronous, intrusion-resilient aggregation** — timed submission windows, a staleness registry, accept/flag/quarantine gating, and staleness-decayed robust averaging with norm clipping. **The client-side label-private protocol is unchanged.**
- **Honest evaluation** — Byzantine-robust baselines (Krum, trimmed mean, FLTrust), a no-detection async ablation, adaptive adversaries, and hyperparameter/attack-magnitude sweeps, all across 10 seeds.
- **A characterized robustness boundary** — an attack-then-abstain adversary can evade the memoryless gate; we disclose this openly and show it is *self-limiting* (the same staleness that aids evasion also collapses the attacker's aggregation weight).

---

## Headline results (MIMIC-IV, 3 non-IID ICU clients, 10 seeds)

| Experiment | Attack | Straggler | AUROC (R8–10) | Detection | FPR |
|---|---|---|---|---|---|
| Sync-FedAvg baseline | — | — | 0.9649 | — | — |
| Async clean | — | mild lag | **0.9652 ± 0.0004** | — | — |
| Exp 1 — gradient scaling | Client 1, ×10 | — | 0.9598 ± 0.0004 | **100%** | **0%** |
| Exp 2 — free rider | Client 2, random | — | 0.9050 ± 0.0071 | **100%** | **0%** |
| Exp 3 — attack + straggler | Client 1, ×10 | Client 2 | 0.9581 ± 0.0005 | **100%** | **0%** |
| Exp 4 — honest straggler | — | Client 2 | 0.9660 ± 0.0004 | — | **0%** |

Additional findings:

- **Detection tracks attack severity.** An attack-magnitude sweep shows 0% detection below ~2× scaling (where damage is negligible), a transition near 3×, and **100% detection from 5× onward** — so "100%" is not an artifact of testing only extreme attacks.
- **Federation preservation, quantified.** Contribution entropy is **0.986 for SNAS** (balanced participation) versus **0.000 for Krum**, which collapses to a single client (880/880 selections) despite its higher raw AUROC.
- **The gate earns its place.** Ablating it (pure staleness-weighted async FedAvg) drops the free-rider case from 0.905 → 0.808, while costing nothing when clients are honest (0.9660 either way).

*(Paper labels Exp 1–4 map to the config names `e1`, `e2`, `e4`, `e6` respectively — see the table under [Reproducing the experiments](#reproducing-the-experiments).)*

---

## How it works

**Label-private split protocol (client-side, unchanged):**

```text
Client                                           Server
------                                           ------
ClientEncoder: 266 features -> 32-dim activation
        | activation -------------------------->  ServerTopModel: 32 -> logits
        |                                          (never sees features or labels)
        | <----------------------------- logits
Compute BCE loss LOCALLY with private labels
        | dL/dlogits ------------------------->   Backprop server model
        | <-------------------------- activation gradient
Backprop encoder, optimizer step

After each round: client encoders -> robust async aggregation -> global encoder -> clients
```

**Server-side SENTINEL-SplitFed modules (the contribution):**

| Module | File | Role |
|---|---|---|
| Async round controller | `src/sfl/server/async_controller.py` | Aggregates at window close instead of waiting for all clients |
| Staleness registry | `src/sfl/server/staleness.py` | Tracks consecutive missed rounds `τ`; four decay functions |
| SNAS anomaly gate | `src/sfl/server/anomaly_detector.py` | `SNAS = (β·cos + γ·norm) / (1+τ)` → accept / flag / quarantine |
| Robust aggregation | `src/sfl/server/robust_fedavg.py` | Staleness-decayed, gate-weighted averaging with norm clipping |
| Trust state | `src/sfl/server/trust_state.py` | Per-client trust decay across rounds |
| Attack simulation | `src/sfl/common/attack_simulator.py` | Gradient scaling, free rider, adaptive attacks, etc. |
| Byzantine baselines | `src/sfl/server/baseline_aggregators.py` | Krum, trimmed mean, FLTrust (drop-in aggregators) |

---

## Repository structure

```text
src/sfl/
  common/        models.py, attack_simulator.py, serialization.py, metrics.py, ...
  client/        dataset.py, api.py, trainer.py, multi_runner.py   (label-private protocol)
  server/        app.py (FastAPI), async_controller.py, staleness.py,
                 anomaly_detector.py, robust_fedavg.py, trust_state.py,
                 baseline_aggregators.py, fedavg.py, state.py, trainer.py

configs/         server*.yaml + experiment_*.yaml (main, baselines, ablations,
                 adaptive attacks, async, no-detection)

scripts/         start_server.py, run_label_private_splitfed.py,
                 run_multiseed.py / extend_multiseed_to_10.py,
                 run_baseline_comparison.py, run_krum_multiseed.py,
                 run_sensitivity_sweep.py, run_attack_magnitude_sweep.py,
                 close_gaps_runner.py, compute_federation_entropy.py,
                 run_significance_tests.py, make_paper_figures.py,
                 plot_sensitivity_heatmap.py

results/         per-experiment metrics (CSV/JSONL), multiseed & significance
                 summaries, gap_closing/ (baseline variance, no-detection,
                 federation entropy), figures/, PAPER_RESULTS.md

notebooks/       client{0,1,2}_model.ipynb, server training notebook
docs/            supplementary reports and notes
```

Raw MIMIC-IV data, virtual environments, large logs, and model binaries are excluded via `.gitignore`.

---

## Installation

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

(Use `source .venv/bin/activate` on Linux/macOS.)

---

## Data

Experiments use **MIMIC-IV** ICU records (92,378 stays across Medical, Surgical, and Cardiac ICU partitions; 266 features; binary in-hospital mortality). MIMIC-IV is available from [PhysioNet](https://physionet.org/content/mimiciv/) under a credentialed data-use agreement. **No patient data is included in this repository** — only derived metrics and configuration.

Client assets are expected locally (not committed):

```text
data/processed/client{0_medical,1_surgical,2_cardiac}.csv
models/client{0_medical,1_surgical,2_cardiac}/scaler.pkl
```

---

## Reproducing the experiments

All experiments run a local FastAPI server plus a client runner. **Terminal 1** starts the server; **Terminal 2** runs an experiment.

```powershell
# Terminal 1 — start the server (SNAS aggregator by default)
python scripts/start_server.py --config configs/server.yaml

# Terminal 2 — run a single experiment at a given seed
python scripts/run_label_private_splitfed.py --experiment-config configs/experiment_e1_gradient_scaling.yaml --seed 42
```

**Paper experiment ↔ config map:**

| Paper | Scenario | Experiment config |
|---|---|---|
| Exp 1 | Gradient scaling (Client 1, ×10) | `configs/experiment_e1_gradient_scaling.yaml` |
| Exp 2 | Free rider (Client 2, random) | `configs/experiment_e2_free_rider.yaml` |
| Exp 3 | Attack + honest straggler | `configs/experiment_e4_attack_plus_dropout.yaml` |
| Exp 4 | Honest straggler, no attack | `configs/experiment_e6_dropout.yaml` |

**Full result suite (each uses the local server + `/admin/reset` between runs):**

```powershell
python scripts/run_multiseed.py               # SNAS across 10 seeds (Exp 1-4)
python scripts/run_baseline_comparison.py     # Krum / trimmed mean / FLTrust
python scripts/run_krum_multiseed.py          # Krum across 10 seeds
python scripts/run_sensitivity_sweep.py       # beta/gamma, thresholds, clip ratio
python scripts/run_attack_magnitude_sweep.py  # detection vs attack strength
python scripts/close_gaps_runner.py           # async-clean baseline + no-detection ablation
python scripts/compute_federation_entropy.py  # SNAS vs Krum contribution entropy
python scripts/run_significance_tests.py      # Wilcoxon / paired t-test / Cohen's d
python scripts/make_paper_figures.py          # regenerate all manuscript figures
```

Aggregated numbers and the full experimental record are in
[`results/PAPER_RESULTS.md`](results/PAPER_RESULTS.md).

The server also supports a remote deployment (e.g., Colab GPU + Cloudflare tunnel) via
`scripts/start_server_colab.py` and the `--server-url` flag on the client runner.

---

## Limitations

We report one genuine robustness boundary honestly: a memoryless per-round gate can be evaded by an **attack-then-abstain** adversary that alternates attacking and skipping rounds to hold `τ=1`. We show this is **self-limiting** — the same staleness that lowers the SNAS score also lowers the attacker's aggregation weight, so effective poisoning power collapses as `τ` grows (worst measured damage is only −0.006 AUROC). Temporal SNAS accumulation and a capped staleness denominator are discussed as mitigations. The benchmark also uses three clients (a realistic low-client cross-silo setting) and single-seed hyperparameter sweeps; larger-scale and multi-seed sweeps are future work.

---

## Citation

```bibtex
@article{sentinelsplitfed2026,
  title   = {SENTINEL-SplitFed: Staleness-Aware Encoder Anomaly Detection for
             Intrusion-Resilient Asynchronous Split Federated Learning in
             Critical-Care Prediction},
  author  = {[Authors]},
  journal = {Machine Learning (Springer Nature), Special Issue on Federated
             Learning for Critical Applications},
  year    = {2026},
  note    = {Under review}
}
```

---

## Acknowledgements

Built on MIMIC-IV (PhysioNet) and the SplitFed learning paradigm. This work extends a label-private SplitFed foundation toward intrusion-resilient asynchronous federation for critical-care AI.
