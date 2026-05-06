# SplitFederated Framework Implementation Plan

This repository now separates exploratory notebooks from the real client-server
research framework.

## Current Protocol: Label-Private SplitFed

The implemented protocol keeps labels private.

```text
client -> server: activation only
server -> client: logit
client: computes loss locally with private label
client -> server: gradient with respect to logit
server -> client: gradient with respect to activation
client: activation.backward(gradient), then encoder_optimizer.step()
```

This is the primary privacy-preserving method for the paper.

## Important Files

```text
configs/
  server.yaml
  experiment_label_private_splitfed.yaml
  experiment_option_b.yaml
  clients/client0_medical.yaml
  clients/client1_surgical.yaml
  clients/client2_cardiac.yaml

src/sfl/common/
  models.py
  serialization.py
  config.py
  metrics.py

src/sfl/server/
  app.py
  trainer.py
  state.py
  fedavg.py

src/sfl/client/
  dataset.py
  api.py
  trainer.py
  runner.py

scripts/
  start_server.py
  start_server_colab.py
  run_client.py
  run_all_clients.py
  run_label_private_splitfed.py
```

## Local Smoke Test

Terminal 1:

```powershell
.\.venv\Scripts\python.exe scripts\start_server.py --config configs\server.yaml
```

Terminal 2:

```powershell
.\.venv\Scripts\python.exe scripts\run_client.py --client-config configs\clients\client0_medical.yaml --rounds 1 --max-batches 2 --skip-fedavg
```

All three clients, one tiny debug round with FedAvg:

```powershell
.\.venv\Scripts\python.exe scripts\run_label_private_splitfed.py --rounds 1 --max-batches 2 --max-val-batches 2
```

Label-private SplitFed tiny debug round:

```powershell
.\.venv\Scripts\python.exe scripts\run_label_private_splitfed.py --rounds 1 --max-batches 2 --max-val-batches 2
```

Each all-client run writes local summaries under the experiment results folder:

```text
results/label_private_splitfed/experiment_summary_option_b_<run_id>.csv
results/label_private_splitfed/experiment_summary_option_b_<run_id>.jsonl
```

If the server code is updated, the runner also asks the server to save:

```text
results/label_private_splitfed/server/server_model_<mode>_<run_id>.pth
```

When using Colab, re-upload `src/`, `scripts/`, `configs/`, and
`requirements.txt` after code changes, restart the Colab server, then rerun the
local client command.

For Colab, start `scripts/start_server_colab.py`, expose port `8000` with a
tunnel such as ngrok or cloudflared, then pass the public URL with
`--server-url`.
