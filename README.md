# Label-Private Split Federated Learning for MIMIC-IV Mortality Prediction

This repository contains a research framework for label-private Split Federated Learning (SplitFed) over non-IID MIMIC-IV ICU client partitions. The current implementation connects local client models to a remote server model hosted on Google Colab and performs real activation-gradient exchange without sending raw patient records or labels to the server.

The broader research vision is inspired by the project plan for Hybrid Classical-Quantum Split Federated Learning for Intrusion Resilience (HCQ-SFL-IR) and QR-SecaaS. In that plan, QR-SecaaS is a later privacy/security product layer for collaborative healthcare AI. This repository focuses on the current paper-oriented foundation: a working label-private SplitFed training pipeline for distributed clinical prediction.

## Research Goal

Modern healthcare AI often requires multiple hospitals, wards, or device groups to collaborate without centralizing sensitive patient data. This project studies a privacy-preserving setup where ICU data remains local to each client while a shared server-side model learns from intermediate activations.

Current task:

- Use MIMIC-IV ICU data split into non-IID clients.
- Train client encoders locally.
- Host the server-side model remotely, for example on Colab GPU.
- Exchange activations, logits, gradients, and encoder weights through an API.
- Keep raw features and labels on the client side.
- Aggregate client encoders with FedAvg.

## Current Client Split

The current non-IID setup uses three clinical-unit-based clients:

| Client | Partition | Local data file |
|---|---|---|
| Client 0 | Medical ICU-style partition | `client0_medical.csv` |
| Client 1 | Surgical ICU-style partition | `client1_surgical.csv` |
| Client 2 | Cardiac ICU-style partition | `client2_cardiac.csv` |

Raw data is intentionally excluded from GitHub through `.gitignore`.

## Implemented Method: Label-Private SplitFed

The final workflow keeps labels private. The server never receives patient features or labels.

```text
Client side                                    Server side
-----------                                    -----------
Load local CSV + scaler
Run ClientEncoder
Generate activation/smashed data
        |
        | activation only
        v
                                          ServerTopModel forward
                                          Return logits
        ^
        | logits
Client computes BCE loss locally
using private labels
Compute gradient wrt logits
        |
        | logit gradient
        v
                                          Backprop server model
                                          Return activation gradient
        ^
        | activation gradient
Client backprops into encoder
Client optimizer step

After each round:
client encoders -> FedAvg server -> global encoder -> clients
```

## What Is Implemented

- Real FastAPI server for the remote server-side model.
- Local client runners for all three MIMIC-IV partitions.
- Label-private activation-gradient flow.
- FedAvg across client encoders.
- Per-client validation through `/predict` while labels stay local.
- Local experiment summaries in CSV and JSONL.
- Colab-compatible server launch workflow.

## Repository Structure

```text
configs/
  clients/
    client0_medical.yaml
    client1_surgical.yaml
    client2_cardiac.yaml
  server.yaml
  experiment_label_private_splitfed.yaml

src/sfl/
  common/
    models.py            # ClientEncoder and ServerTopModel
    serialization.py     # tensor and state_dict serialization
    metrics.py           # AUROC, AUPRC, F1 helpers
    config.py            # config/path helpers
  client/
    dataset.py           # local client CSV/scaler preprocessing
    api.py               # API client for server calls
    trainer.py           # label-private SplitFed client logic
    multi_runner.py      # all-client experiment runner
  server/
    app.py               # FastAPI endpoints
    trainer.py           # server-side forward/backward logic
    fedavg.py            # FedAvg aggregation
    state.py             # server state and checkpointing

scripts/
  start_server.py
  start_server_colab.py
  run_label_private_splitfed.py
  run_all_clients.py
  run_client.py

notebooks/
  client0_model.ipynb
  client1_model.ipynb
  client2_model.ipynb
  server_sfl_training.ipynb

plan/
  QR-SecaaS.pdf

docs/
  SPLITFED_REPORT.md
```

## Setup

Create and activate a virtual environment, then install dependencies:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

The local machine must contain the private client assets:

```text
data/processed/client0_medical.csv
data/processed/client1_surgical.csv
data/processed/client2_cardiac.csv

models/client0_medical/scaler.pkl
models/client0_medical/encoder.pth
models/client1_surgical/scaler.pkl
models/client1_surgical/encoder.pth
models/client2_cardiac/scaler.pkl
models/client2_cardiac/encoder.pth
```

Do not upload or commit MIMIC-IV data.

## Local Debug Run

Terminal 1, start server:

```powershell
.\.venv\Scripts\python.exe scripts\start_server.py --config configs\server.yaml
```

Terminal 2, run label-private SplitFed:

```powershell
.\.venv\Scripts\python.exe scripts\run_label_private_splitfed.py --rounds 1 --max-batches 5 --max-val-batches 5
```

The server terminal should show requests such as:

```text
POST /register_client
POST /option-b/forward
POST /option-b/backward
POST /predict
POST /fedavg/submit_encoder
GET  /fedavg/global_encoder
```

## Colab Server Workflow

Upload only these repository parts to Google Drive folder `mimic-sfl-server`:

```text
configs/
src/
scripts/
requirements.txt
```

Do not upload:

```text
data/
models/
results/
notebooks/
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
```

```python
!sleep 5
!curl http://127.0.0.1:8000/health
```

Expose server:

```python
!wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O cloudflared
!chmod +x cloudflared
!nohup ./cloudflared tunnel --url http://127.0.0.1:8000 > cloudflared.log 2>&1 &
```

```python
!sleep 5
!cat cloudflared.log
```

Copy the `https://...trycloudflare.com` URL.

Run clients locally from VS Code PowerShell:

```powershell
.\.venv\Scripts\python.exe scripts\run_label_private_splitfed.py --server-url https://YOUR-TUNNEL-URL --rounds 5 --max-batches 200 --max-val-batches 100
```

For a full run, remove the batch limits:

```powershell
.\.venv\Scripts\python.exe scripts\run_label_private_splitfed.py --server-url https://YOUR-TUNNEL-URL --rounds 10
```

## API Endpoints

| Endpoint | Purpose |
|---|---|
| `GET /health` | Check server status |
| `POST /register_client` | Register local clients |
| `POST /option-b/forward` | Server forward pass from activation to logits |
| `POST /option-b/backward` | Server backward pass from logit gradient to activation gradient |
| `POST /predict` | Validation prediction from activation |
| `POST /fedavg/submit_encoder` | Submit client encoder weights |
| `GET /fedavg/global_encoder` | Download aggregated encoder |
| `POST /checkpoint/server` | Save server checkpoint |

The root URL `/` intentionally has no dashboard yet, so `{"detail":"Not Found"}` is normal.

## Current Experiment Evidence

A 5-round label-private SplitFed run with 200 training batches per client per round and limited validation completed successfully:

| Round | Client 0 AUROC | Client 1 AUROC | Client 2 AUROC | Mean AUROC |
|---|---:|---:|---:|---:|
| 1 | 0.9515 | 0.9636 | 0.9749 | 0.9633 |
| 2 | 0.9448 | 0.9670 | 0.9749 | 0.9622 |
| 3 | 0.9449 | 0.9666 | 0.9775 | 0.9630 |
| 4 | 0.9467 | 0.9662 | 0.9774 | 0.9634 |
| 5 | 0.9476 | 0.9677 | 0.9781 | 0.9644 |

These are medium-scale framework validation results, not yet final paper results.

## Research Roadmap

Current phase:

- Label-private SplitFed implementation and real Colab-local client-server flow.

Next paper-oriented phase:

- Full no-limit training runs.
- Paper-ready tables and plots.
- Communication-cost analysis.
- Runtime analysis.
- Comparison with local-only and prior simulation baselines.
- Privacy/security discussion.

Later QR-SecaaS phase:

- Quantum feature scrambling.
- Quantum anomaly scoring.
- Trust-aware aggregation.
- Poisoning/backdoor/gradient-manipulation attack evaluation.
- Security-as-a-Service dashboard or API layer.

## Suggested Citation/Framing

This repository implements the label-private SplitFed foundation for privacy-preserving collaborative healthcare AI. QR-SecaaS is treated as a later product/security layer built on top of the SplitFed training substrate.
