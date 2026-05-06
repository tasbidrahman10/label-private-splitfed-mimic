# SplitFed Implementation Report

## 1. Purpose

This report summarizes the Split Federated Learning framework implemented in this repository. The goal of this phase was to move from a notebook-based simulation to a real client-server training flow where local clients communicate with a server model hosted remotely on Google Colab.

The final workflow selected for the research framework is label-private SplitFed. In this mode, raw patient features and labels remain on the local client side.

## 2. Research Context

The project plan describes a broader Hybrid Classical-Quantum Split Federated Learning for Intrusion Resilience framework and a later QR-SecaaS product layer. That future layer targets quantum-resilient security for healthcare AI pipelines, including protection against poisoning, insider threats, manipulated IoMT devices, and model-update attacks.

This repository currently implements the core SplitFed foundation needed before those later security layers:

- non-IID healthcare client partitions,
- client-side encoders,
- server-side top model,
- activation-gradient exchange,
- FedAvg aggregation,
- privacy boundary enforcement.

## 3. Dataset And Client Setup

The MIMIC-IV data is partitioned into three non-IID client groups:

| Client | Clinical partition | Local data file |
|---|---|---|
| Client 0 | Medical | `client0_medical.csv` |
| Client 1 | Surgical | `client1_surgical.csv` |
| Client 2 | Cardiac | `client2_cardiac.csv` |

Each client keeps:

- its private CSV,
- its local scaler,
- its pretrained encoder,
- its local labels,
- its local dataloader.

The server keeps:

- only the server-side classifier,
- request state for activation-gradient exchange,
- FedAvg aggregation state,
- server-side logs/checkpoints.

## 4. Model Architecture

Client encoder:

```text
Input: 266 ICU features
Linear(266 -> 64)
BatchNorm
ReLU
Dropout(0.5)
Linear(64 -> 32)
BatchNorm
ReLU
Output: 32-dimensional activation
```

Server model:

```text
Input: 32-dimensional activation
Linear(32 -> 64)
BatchNorm
ReLU
Dropout(0.3)
Linear(64 -> 1)
Output: mortality logit
```

Parameter counts:

| Component | Parameters |
|---|---:|
| ClientEncoder | 19,360 |
| ServerTopModel | 2,305 |

## 5. Label-Private SplitFed Flow

The implemented primary flow is:

```text
1. Client loads private batch X and label y.
2. Client computes activation A = ClientEncoder(X).
3. Client sends A only to server.
4. Server computes logit z = ServerTopModel(A).
5. Server returns z to client.
6. Client computes BCEWithLogitsLoss(z, y) locally.
7. Client computes gradient dz.
8. Client sends dz to server.
9. Server backprops through ServerTopModel and returns dA.
10. Client runs A.backward(dA) and updates ClientEncoder.
11. After a round, all clients submit encoder weights.
12. Server performs FedAvg and returns the global encoder to clients.
```

Important privacy property:

```text
Server receives activations and gradients, but not raw ICU features or labels.
```

## 6. API Endpoints

| Endpoint | Description |
|---|---|
| `GET /health` | Confirms server is live |
| `POST /register_client` | Registers client metadata |
| `POST /option-b/forward` | Receives activation and returns logits |
| `POST /option-b/backward` | Receives logit gradient and returns activation gradient |
| `POST /predict` | Returns validation logits from activations |
| `POST /fedavg/submit_encoder` | Receives client encoder weights |
| `GET /fedavg/global_encoder` | Returns FedAvg encoder |
| `POST /checkpoint/server` | Saves server model checkpoint |

## 7. How To Run

### Local debug

Terminal 1:

```powershell
.\.venv\Scripts\python.exe scripts\start_server.py --config configs\server.yaml
```

Terminal 2:

```powershell
.\.venv\Scripts\python.exe scripts\run_label_private_splitfed.py --rounds 1 --max-batches 5 --max-val-batches 5
```

### Colab server run

Upload only these to Google Drive folder `mimic-sfl-server`:

```text
configs/
src/
scripts/
requirements.txt
```

In Colab:

```python
from google.colab import drive
drive.mount('/content/drive')
```

```python
%cd /content/drive/MyDrive/mimic-sfl-server
!pip install -r requirements.txt
```

```python
!nohup python scripts/start_server.py --config configs/server.yaml > server.log 2>&1 &
!sleep 5
!curl http://127.0.0.1:8000/health
```

Expose with Cloudflare:

```python
!wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O cloudflared
!chmod +x cloudflared
!nohup ./cloudflared tunnel --url http://127.0.0.1:8000 > cloudflared.log 2>&1 &
!sleep 5
!cat cloudflared.log
```

Then run clients locally:

```powershell
.\.venv\Scripts\python.exe scripts\run_label_private_splitfed.py --server-url https://YOUR-TUNNEL-URL --rounds 5 --max-batches 200 --max-val-batches 100
```

## 8. Current Metrics

The following medium-scale label-private run completed with:

```text
rounds = 5
training batches per client per round = 200
validation batches limited
server = Colab via Cloudflare tunnel
clients = local machine
```

| Round | Client 0 AUROC | Client 1 AUROC | Client 2 AUROC | Mean AUROC |
|---|---:|---:|---:|---:|
| 1 | 0.9515 | 0.9636 | 0.9749 | 0.9633 |
| 2 | 0.9448 | 0.9670 | 0.9749 | 0.9622 |
| 3 | 0.9449 | 0.9666 | 0.9775 | 0.9630 |
| 4 | 0.9467 | 0.9662 | 0.9774 | 0.9634 |
| 5 | 0.9476 | 0.9677 | 0.9781 | 0.9644 |

Round 5 detailed metrics:

| Client | Loss | AUROC | AUPRC | F1 |
|---|---:|---:|---:|---:|
| Medical | 0.5507 | 0.9476 | 0.8297 | 0.6838 |
| Surgical | 0.3903 | 0.9677 | 0.8543 | 0.7432 |
| Cardiac | 0.3147 | 0.9781 | 0.8363 | 0.6859 |

Communication estimate for each client per round in the medium run:

```text
activation bytes = 1,638,400
gradient bytes   = 1,638,400
total per client = 3,276,800 bytes
total for 3 clients per round = 9,830,400 bytes
```

## 9. What Has Been Proven

This phase proves:

- the server can run remotely on Colab,
- local clients can connect through a public tunnel,
- clients send activations to the server,
- server returns logits and activation gradients,
- labels remain on clients,
- local encoders update from returned gradients,
- FedAvg aggregates client encoders,
- multi-round training is stable.

## 10. Remaining Paper Work

The SplitFed implementation phase is complete. Remaining work is research packaging:

- run full no-limit experiments,
- generate final plots,
- prepare method diagram,
- write experimental setup,
- document privacy assumptions,
- position QR-SecaaS as future security/product extension,
- add quantum/security modules later.

