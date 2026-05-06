# Progress Report: Label-Private Split Federated Learning Framework

## 1. Project Overview

This report summarizes the current progress of the Split Federated Learning framework developed for privacy-preserving healthcare AI using non-IID MIMIC-IV client partitions.

The main goal of this phase was to move beyond notebook-based simulation and implement a real client-server SplitFed workflow where:

- patient data remains with local clients,
- labels remain with local clients,
- the server model runs remotely on Google Colab,
- clients send only intermediate activations,
- gradients are returned from the server to update client encoders,
- client encoders are aggregated through FedAvg.

This phase has been successfully completed and tested.

## 2. Research Motivation

Healthcare AI models often require collaboration across multiple hospitals, units, or clinical data sources. However, direct centralization of patient data creates privacy, regulatory, and institutional barriers.

The implemented framework addresses this by allowing multiple clinical clients to collaboratively train a mortality prediction model without sharing raw patient features or labels with the server.

This forms the practical SplitFed foundation for the broader research direction toward privacy-preserving and security-aware collaborative healthcare AI.

## 3. Dataset And Client Partitioning

The MIMIC-IV dataset was split into three non-IID client groups based on clinical unit type:

| Client | Partition | Description |
|---|---|---|
| Client 0 | Medical | Medical ICU-style patient partition |
| Client 1 | Surgical | Surgical ICU-style patient partition |
| Client 2 | Cardiac | Cardiac ICU-style patient partition |

This non-IID setup reflects realistic healthcare environments where different hospitals or departments have different patient distributions, mortality rates, and clinical characteristics.

## 4. Implemented Framework

The implemented method is a label-private Split Federated Learning framework.

The framework has two main model components:

| Component | Location | Role |
|---|---|---|
| ClientEncoder | Local client side | Converts private ICU features into intermediate activations |
| ServerTopModel | Remote server side | Receives activations and produces mortality prediction logits |

The local client keeps:

- raw patient records,
- mortality labels,
- scaler,
- client encoder,
- local dataloader.

The server keeps:

- server-side prediction model,
- temporary activation-gradient state,
- FedAvg aggregation state,
- server-side logs and checkpoint state.

## 5. Model Architecture

### Client-Side Encoder

The client model receives 266 ICU features and converts them into a 32-dimensional activation vector.

```text
Input: 266 clinical features
Linear(266 -> 64)
BatchNorm
ReLU
Dropout(0.5)
Linear(64 -> 32)
BatchNorm
ReLU
Output: 32-dimensional activation
```

### Server-Side Model

The server model receives the 32-dimensional activation and produces a mortality prediction logit.

```text
Input: 32-dimensional activation
Linear(32 -> 64)
BatchNorm
ReLU
Dropout(0.3)
Linear(64 -> 1)
Output: mortality logit
```

| Model Part | Parameter Count |
|---|---:|
| ClientEncoder | 19,360 |
| ServerTopModel | 2,305 |

## 6. Label-Private Activation-Gradient Flow

The final framework uses the label-private SplitFed protocol.

The training flow is:

```text
1. Client processes local patient batch.
2. ClientEncoder generates intermediate activation.
3. Client sends activation to the remote server.
4. ServerTopModel returns prediction logits.
5. Client computes loss locally using private mortality labels.
6. Client sends only the gradient signal back to the server.
7. Server backpropagates through the server-side model.
8. Server returns activation gradients to the client.
9. Client updates its encoder locally.
10. After each round, client encoders are aggregated using FedAvg.
```

Privacy boundary:

```text
Raw patient data: never leaves the client
Labels: never leave the client
Server receives: activations and gradient signals only
```

## 7. System Validation

The framework was tested with a real distributed setup:

| Component | Runtime Location |
|---|---|
| Clients | Local machine |
| Server | Google Colab |
| Communication | Public tunnel through Cloudflare |
| Training protocol | Label-private SplitFed |
| Aggregation | FedAvg across 3 clients |

The Colab server successfully received client requests for:

- client registration,
- activation forward pass,
- gradient backward pass,
- validation prediction,
- FedAvg submission,
- global encoder retrieval.

All tested API requests returned successful `200 OK` responses during the medium-scale runs.

## 8. Current Experimental Results

A 5-round label-private SplitFed experiment was completed using:

```text
3 clients
200 training batches per client per round
limited validation batches
remote Colab server
local client-side data and labels
FedAvg after every round
```

### AUROC Progress

| Round | Medical Client AUROC | Surgical Client AUROC | Cardiac Client AUROC | Mean AUROC |
|---|---:|---:|---:|---:|
| 1 | 0.9515 | 0.9636 | 0.9749 | 0.9633 |
| 2 | 0.9448 | 0.9670 | 0.9749 | 0.9622 |
| 3 | 0.9449 | 0.9666 | 0.9775 | 0.9630 |
| 4 | 0.9467 | 0.9662 | 0.9774 | 0.9634 |
| 5 | 0.9476 | 0.9677 | 0.9781 | 0.9644 |

### Round 5 Detailed Metrics

| Client | Loss | AUROC | AUPRC | F1 |
|---|---:|---:|---:|---:|
| Medical | 0.5507 | 0.9476 | 0.8297 | 0.6838 |
| Surgical | 0.3903 | 0.9677 | 0.8543 | 0.7432 |
| Cardiac | 0.3147 | 0.9781 | 0.8363 | 0.6859 |

### Observations

- The framework trained stably across all 3 non-IID clients.
- FedAvg completed successfully after every round.
- Client losses decreased over training.
- Mean AUROC remained strong and stable, reaching approximately `0.9644` by round 5.
- The label-private protocol preserved the intended privacy boundary.

## 9. Communication Cost Estimate

For the medium-scale 5-round run, each client exchanged activation and gradient tensors with the server.

Per client per round:

| Item | Bytes |
|---|---:|
| Activations sent to server | 1,638,400 |
| Activation gradients returned to client | 1,638,400 |
| Total per client per round | 3,276,800 |

For all 3 clients:

| Item | Bytes |
|---|---:|
| Total communication per round | 9,830,400 |
| Total over 5 rounds | 49,152,000 |

This does not include small metadata, validation prediction requests, or model-weight transfer during FedAvg. It provides a core estimate for activation-gradient communication.

## 10. Current Status

The core SplitFed implementation phase is complete.

Completed:

- non-IID MIMIC-IV client setup,
- client-side encoders,
- remote server-side model,
- real client-server communication,
- label-private activation-gradient flow,
- FedAvg aggregation,
- multi-round training validation,
- experiment metric logging,
- collaborator-facing documentation.

## 11. Remaining Work

The remaining work is mainly paper preparation and final experimentation:

- run full no-limit training experiments,
- generate final plots and tables,
- compare with local-only and previous simulation baselines,
- write the methodology section,
- formalize the privacy assumptions,
- discuss communication cost and runtime,
- extend later toward security-aware or QR-SecaaS modules.

## 12. Summary

The project has successfully implemented a real label-private Split Federated Learning framework for MIMIC-IV mortality prediction. The framework now supports distributed training between local clients and a remote Colab server while keeping both raw patient data and labels on the client side.

The current results demonstrate stable multi-client training, effective FedAvg aggregation, and strong validation AUROC across non-IID medical, surgical, and cardiac client partitions. This completes the core engineering milestone for the SplitFed phase and provides a strong foundation for the next research-paper stage.

