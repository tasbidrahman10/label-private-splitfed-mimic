# Migrating from n=3 to n=6 clients

Step-by-step, from running [MIMIC_IV_SPLIT.ipynb](MIMIC_IV_SPLIT.ipynb) in Colab through to a
federated run with six silos. Companion to [REVIEW_TRIAGE.md](REVIEW_TRIAGE.md) — this covers the
first P2 item (N≥5 so Krum is validly configured) and folds in P0-8 (`los` removal) because both
need the same notebook run.

**Two things happen automatically at n=6 and are the whole point of the exercise:**

| | n=3 (now) | n=6 |
|---|---|---|
| Krum `k = n − f − 2` ([baseline_aggregators.py:73](src/sfl/server/baseline_aggregators.py#L73)) | k=0 → clamped to 1 → honest scores tie **bit-identically** → winner = submission order | k=3, clamp never fires, no tie, **valid regime** (n ≥ 2f+3) |
| Trimmed mean `k = floor(0.2·n)` ([baseline_aggregators.py:115](src/sfl/server/baseline_aggregators.py#L115)) | k=0 → trims nothing | k=1 → actually trims |

Neither needs a code change. That is why this is the cheapest P2 item.

---

## Phase 0 — Protect the n=3 artifacts (15 min)

`data/` is gitignored but `models/` is tracked, so the n=3 **encoders** are recoverable from git and
the n=3 **CSVs** are not. Back them up before anything else.

```powershell
Copy-Item -Recurse data/processed data/processed_n3
Copy-Item -Recurse models models_n3
git add MIMIC_IV_SPLIT.ipynb scripts/pretrain_client.py N6_MIGRATION.md
git commit -m "Parameterize MIMIC split for n=6; add config-driven client pretraining"
```

You will keep the n=3 results as the paper's main table, so these must survive.

---

## Phase 1 — Colab (~40 min, mostly unattended)

1. Upload [MIMIC_IV_SPLIT.ipynb](MIMIC_IV_SPLIT.ipynb) to Colab and mount Drive. `BASE_PATH` in
   cell 0 is unchanged.
2. **Run cells 0–5.** This is the ~30 min labevents/chartevents streaming.
3. **Run cell 6** (new). Writes `master_df.parquet` next to the dataset. Everything below is now
   cheap to re-run, and on any later visit you can skip cells 1–5 entirely by uncommenting the two
   loader lines in cell 6.
4. **Cell 7** already has `SPLIT_SCHEME = 'six'`, `DROP_LOS = True`, `KEEP_ID_COLS = False`. Run it
   and check the printout against this:

   | Client | Care unit | Expected stays |
   |---|---|---|
   | client0_micu | MICU | 20,703 |
   | client1_medsurg | MICU/SICU | 15,449 |
   | client2_sicu | SICU | 13,009 |
   | client3_tsicu | TSICU | 10,474 |
   | client4_ccu | CCU | 10,775 |
   | client5_cvicu | CVICU | 14,771 |

   85,181 retained. The 7,197 dropped rows are Neuro Intermediate + Neuro Stepdown, which the old
   3-way map folded into the "cardiac" client.
5. **Run cells 8–9.** Cell 9 prints `expected_num_clients`, `input_dim`, and all six client YAMLs
   ready to paste. **`input_dim` will be 265, not 266** — `los` is one of the six demographic
   features. Copy that printout somewhere; Phase 3 needs it.
6. Download the six CSVs plus `split_summary.csv` from
   `.../fl_clients_n6/` on Drive.

> **Optional but recommended:** flip `SPLIT_SCHEME` back to `'three'` (keeping `DROP_LOS = True`)
> and re-run cells 7–9. That regenerates the 3-client data without the leaking `los` feature, so
> your main results and the n=6 arm rest on the same feature set and remain comparable. It costs
> one minute now and takes P0-8 off the list. Output lands in `fl_clients/`, so back up the
> originals first if you want them.

---

## Phase 2 — Data into the repo (5 min)

```
data/processed/
    client0_micu.csv
    client1_medsurg.csv
    client2_sicu.csv
    client3_tsicu.csv
    client4_ccu.csv
    client5_cvicu.csv
```

Delete or move aside `client{0,1,2}_{medical,surgical,cardiac}.csv` — several code paths glob or
scan this directory and will pick up strays.

---

## Phase 3 — Configs (20 min)

**3a. Six client YAMLs.** Paste from the cell 9 printout into
`configs/clients/client{0..5}_{name}.yaml`. Delete the old three. Each is the same 13-line shape as
[client0_medical.yaml](configs/clients/client0_medical.yaml) with `input_dim: 265`.

**3b. Server configs — 8 files.** `input_dim: 266 → 265` and `expected_num_clients: 3 → 6` in:

```
configs/server.yaml
configs/server_ablation_{exponential,linear,polynomial,step}.yaml
configs/server_baseline_{krum,trimmed_mean,fltrust}.yaml
```

Check both keys in every file — some only carry `input_dim`.

---

## Phase 4 — Code changes

Five edits. Everything else adapts on its own.

### 4a. `src/sfl/client/multi_runner.py:22-26` — hardcoded client list

```python
DEFAULT_CLIENT_CONFIGS = [
    "configs/clients/client0_medical.yaml",
    "configs/clients/client1_surgical.yaml",
    "configs/clients/client2_cardiac.yaml",
]
```
→
```python
DEFAULT_CLIENT_CONFIGS = sorted(
    str(p.relative_to(repo_root())).replace("\\", "/")
    for p in (repo_root() / "configs" / "clients").glob("*.yaml")
)
```
Add `repo_root` to the existing `from sfl.common.config import ...` line. (Lexical sort is correct
for fewer than 10 clients; beyond that, sort by the YAML's `client_id`.)

### 4b. `src/sfl/server/state.py:139` **and** `:271` — FLTrust roster, duplicated

The same block appears in `_load_fltrust_root_data` and again in the `reset()` path. Both need it.

```python
client_cfgs = [(0, "medical"), (1, "surgical"), (2, "cardiac")]

self.fltrust_root_data = {}
self.fltrust_root_labels = {}

for cid, cname in client_cfgs:
    cfg = _load_yaml(str(root / "configs" / "clients" / f"client{cid}_{cname}.yaml"))
    indices = meta["clients"][str(cid)]
```
→
```python
client_cfg_paths = sorted((root / "configs" / "clients").glob("*.yaml"))

self.fltrust_root_data = {}
self.fltrust_root_labels = {}

for cfg_path in client_cfg_paths:
    cfg = _load_yaml(str(cfg_path))
    cid = int(cfg["client_id"])
    indices = meta["clients"][str(cid)]
```
The rest of each loop body is unchanged.

### 4c. `src/sfl/server/app.py:201-206` — pins client0's YAML

Only used to read `input_dim` / `encoder_hidden_dim`, which are identical for all clients and
already present in the server config. Drop the dependency:

```python
cfg_path = repo_root() / "configs" / "clients" / "client0_medical.yaml"
cfg = load_yaml(str(cfg_path))
encoder = ClientEncoder(
    input_dim=int(cfg.get("input_dim", 266)),
    hidden_dim=int(cfg.get("encoder_hidden_dim", 64)),
).to(state.device)
```
→
```python
encoder = ClientEncoder(
    input_dim=int(state.config.get("input_dim", 265)),
    hidden_dim=int(state.config.get("encoder_hidden_dim", 64)),
).to(state.device)
```

### 4d. `scripts/create_fltrust_root.py:30-34`

```python
CLIENT_CONFIGS = sorted(
    str(p) for p in (ROOT / "configs" / "clients").glob("*.yaml")
)
```

### 4e. `scripts/compute_federation_entropy.py:173-176` — three fixed CSV columns

```python
share_keys = sorted({k for r in rows for k in r["per_client_share"]})
w.writerow(["method", "condition", *[f"{k}_share" for k in share_keys], "normalized_entropy"])
for r in rows:
    pc = r["per_client_share"]
    w.writerow([r["method"], r["condition"],
                *[pc.get(k, "") for k in share_keys], r["normalized_entropy"]])
```

### Needs attention only for the malicious-fraction sweep

[app.py:137](src/sfl/server/app.py#L137) and [app.py:141](src/sfl/server/app.py#L141) hardcode
`f_byzantine=1` and `trim_ratio=0.2`. Fine for the f=1 arm. For an f=2 run (2 of 6 malicious),
read both from `state.config` so Krum's `k` and the trim fraction track the actual adversary count.

### What does *not* change

[state.py:92](src/sfl/server/state.py#L92) / [state.py:216](src/sfl/server/state.py#L216) derive
`client_ids = list(range(n))` from `expected_num_clients`, so `StalenessRegistry`,
`AsyncRoundController`, `TrustState` and `MaliciousClientGate` all follow automatically. The SNAS
detector, `RobustAsyncFedAvg`, `dataset.py`, and both baseline aggregators are already n-agnostic.

---

## Phase 5 — Pretrain the six encoders (~2–4 h, unattended)

Each client needs `scaler.pkl` + `encoder.pth` in its `model_dir`
([dataset.py:53](src/sfl/client/dataset.py#L53), [dataset.py:89](src/sfl/client/dataset.py#L89)).
These came from three duplicated notebooks; [scripts/pretrain_client.py](scripts/pretrain_client.py)
is a config-driven port of them (same architecture, Adam @ 1e-4, `ReduceLROnPlateau`, 80 epochs,
early stop at patience 8, `pos_weight` capped at 5). It fits the scaler first and then builds
loaders through `load_client_dataloaders`, so pre-training sees exactly the tensors federated
training will.

**The `.venv` in this repo will not launch on this machine.** It was created elsewhere — its
`pyvenv.cfg` records `home = C:\Users\USER\AppData\Local\Python\pythoncore-3.14-64` and
`command = ... -m venv E:\project\mimic-split-learning\.venv`, neither of which exists here, so
`.venv\Scripts\python.exe` exits immediately. The interpreter actually available is Python 3.11 at
`C:\Users\HP\AppData\Local\Programs\Python\Python311\python.exe`. Rebuild before Phase 5:

```powershell
python -m venv .venv --clear
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Any script in `scripts/` that shells out via `VENV_PYTHON` (e.g.
[run_multiseed.py:38](scripts/run_multiseed.py#L38)) is hitting the same wall, so this is worth
fixing regardless of the n=6 work.

Then:

```powershell
Get-ChildItem configs/clients/*.yaml | ForEach-Object {
    .\.venv\Scripts\python.exe scripts/pretrain_client.py --config $_.FullName
}
```

Note the per-client val AUROC it prints. With `los` dropped, expect a **drop from ~0.965** toward
the 0.85–0.90 band that published MIMIC-IV mortality models occupy. That is the leakage coming out,
not a regression — see [REVIEW_TRIAGE.md §3.1](REVIEW_TRIAGE.md).

> **Actual result (2026-08-17, n=6 run):** the drop did not happen. All six clients landed at
> 0.9481–0.9755 val AUROC (micu 0.9514, medsurg 0.9481, sicu 0.9651, tsicu 0.9602, ccu 0.9679,
> cvicu 0.9755) — barely below the original 0.965. `los` was confirmed absent from the input
> (266 columns → 265 features + `mortality`, verified directly on `client0_micu.csv`), so this
> isn't a leftover-`los` bug; it's evidence that `los` was never the dominant leak. The two
> mechanisms REVIEW_TRIAGE.md §3.1/§3.2 name as the deeper causes — whole-admission/whole-stay
> feature aggregation with no observation window, and row-level (not admission-grouped) train/val
> splitting — are still present and are the more likely explanation. Neither is addressed by this
> migration; both remain open items if the paper needs a leakage-free AUROC number rather than a
> security-comparison substrate.

---

## Phase 6 — Experiment configs (30 min)

`delay_config` and `attack_config` are keyed by client id and currently stop at 2. For each
experiment you intend to rerun, e.g. [experiment_e1_gradient_scaling.yaml](configs/experiment_e1_gradient_scaling.yaml):

```yaml
delay_config:
  0: 0
  1: 3
  2: 6
  3: 9
  4: 12
  5: 15

attack_config:
  1:
    type: gradient_scaling
    params:
      scale: 10.0
```

Keep every delay under `async_window_seconds: 25.0` unless you specifically want a client to miss
the window. One attacker out of six is f≈16.7%; for the fraction sweep add a second entry (e.g.
client 4) and set `f_byzantine` accordingly per 4e above.

You do not need to convert all ~20 experiment configs — only the ones in the n=6 table.

---

## Phase 7 — FLTrust root

```powershell
.\.venv\Scripts\python.exe scripts/create_fltrust_root.py
```

Regenerates `configs/fltrust_root_indices.json` for six clients. Row indices point into the new
CSVs, so the stale file **must** be replaced or the server will slice the wrong rows.

Independently of n, FLTrust is still the P0-4 bug (a plain mean reported as FLTrust). Fix that in
the same pass — you are already editing both call sites.

---

## Phase 8 — Smoke test, then run

```powershell
# terminal 1
.\.venv\Scripts\python.exe scripts/start_server.py

# terminal 2 — 2 rounds, few batches, just checking the plumbing
.\.venv\Scripts\python.exe scripts/run_all_clients.py `
    --experiment-config configs/experiment_e1_gradient_scaling.yaml `
    --rounds 2 --max-batches 20
```

Confirm before launching the long runs:

- six `Registered client N (name)` lines,
- Krum logs a **non-zero, non-tied** score set with a winner that is not always client 0,
- trimmed mean logs `trimmed 1 from each tail` rather than 0,
- no `expected 265 features, found 266` from [dataset.py:35](src/sfl/client/dataset.py#L35).

Then the real runs: multi-seed SNAS plus the three baselines on the same seeds.

---

## Rollback

Everything is reversible. Set `SPLIT_SCHEME = 'three'` in the notebook, restore
`data/processed_n3` and `models_n3`, and revert the Phase 3–4 edits — all five are small and none
touch the SNAS detector.

## What this does and does not buy you

**Answers:** Krum's invalid configuration, the tie-break artifact behind the 880/880 claim
(P0-2/P0-3), the meaningless trim ratio (P1-3), "only three constructed clients", and — via the
same notebook run — `los` leakage (P0-8).

**Does not answer:** version-based staleness, genuine async timing, patient-level splits and a
held-out test set, a second dataset, or server-head defense. Those stay in Limitations.

**Sequencing:** if you are on the 28 Aug deadline, treat this as a *supplementary* table rather
than a rewrite. Keep n=3 as the main results and add one n=6 comparison — SNAS vs Krum vs
coordinate-median vs FLTrust on E1, 3–5 seeds. That converts P0-2 from a confession into a result:
*at n=3 Krum's selection is decided by submission order; at n=6, where Krum is validly configured,
SNAS still outperforms it.* Converting the whole paper to n=6 invalidates every number you have
and is the higher-risk path with under two weeks left.
