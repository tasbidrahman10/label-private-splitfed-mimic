from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import torch
import requests

from sfl.client.api import SFLServerClient
from sfl.client.dataset import load_client_dataloaders, load_encoder_weights_path
from sfl.client.trainer import OptionBClientTrainer
from sfl.common.config import load_yaml
from sfl.common.config import ensure_dir
from sfl.common.logging_utils import choose_device, set_seed
from sfl.common.models import ClientEncoder


DEFAULT_CLIENT_CONFIGS = [
    "configs/clients/client0_medical.yaml",
    "configs/clients/client1_surgical.yaml",
    "configs/clients/client2_cardiac.yaml",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run all SFL clients against one API server.")
    parser.add_argument("--experiment-config", default="configs/experiment_label_private_splitfed.yaml")
    parser.add_argument("--client-config", action="append", dest="client_configs")
    parser.add_argument("--server-url", default=None)
    parser.add_argument("--rounds", type=int, default=None)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument("--skip-fedavg", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--mode", choices=["option_b"], default=None)
    return parser


def build_trainer(client_config: dict, experiment_config: dict, api: SFLServerClient, device: torch.device):
    if experiment_config.get("results_dir"):
        client_config["results_dir"] = str(
            Path(experiment_config["results_dir"])
            / f"client{client_config['client_id']}_{client_config['client_name']}"
        )
    train_loader, val_loader, pos_weight = load_client_dataloaders(client_config, device)
    encoder = ClientEncoder(
        input_dim=int(client_config.get("input_dim", 266)),
        hidden_dim=int(client_config.get("encoder_hidden_dim", 64)),
    ).to(device)
    encoder.load_state_dict(torch.load(load_encoder_weights_path(client_config), map_location=device))

    client_id = int(client_config["client_id"])
    client_name = str(client_config["client_name"])
    api.register_client(
        client_id=client_id,
        client_name=client_name,
        train_size=len(train_loader.dataset),
        pos_weight=pos_weight,
    )
    print(f"Registered client {client_id} ({client_name}) with {len(train_loader.dataset):,} samples")

    mode = str(experiment_config.get("mode", "option_b"))
    if mode == "option_b":
        trainer = OptionBClientTrainer(
            client_config,
            encoder,
            train_loader,
            val_loader,
            api,
            device,
            pos_weight=pos_weight,
        )
        save_name = "encoder_option_b_latest.pth"
    else:
        raise ValueError(f"Unsupported mode: {mode}")
    return trainer, save_name


def run_all_clients(args: argparse.Namespace) -> None:
    experiment_config = load_yaml(args.experiment_config)
    if args.mode:
        experiment_config["mode"] = args.mode
    set_seed(42)
    device = choose_device(args.device)
    server_url = args.server_url or experiment_config["server_url"]
    api = SFLServerClient(server_url, auth_token=str(experiment_config.get("auth_token", "")))
    print(f"Server health: {api.health()}")
    mode = str(experiment_config.get("mode", "option_b"))
    results_dir = ensure_dir(experiment_config.get("results_dir", f"results/{mode}"))
    run_id = time.strftime("%Y%m%d_%H%M%S")
    summary_jsonl = results_dir / f"experiment_summary_{mode}_{run_id}.jsonl"
    summary_csv = results_dir / f"experiment_summary_{mode}_{run_id}.csv"
    summary_rows: list[dict] = []

    client_paths = args.client_configs or DEFAULT_CLIENT_CONFIGS
    trainers = [
        build_trainer(load_yaml(path), experiment_config, api, device)
        for path in client_paths
    ]

    rounds = args.rounds or int(experiment_config.get("num_rounds", 1))
    fedavg_enabled = bool(experiment_config.get("fedavg_every_round", True)) and not args.skip_fedavg

    for round_idx in range(1, rounds + 1):
        print(f"\n=== Round {round_idx} ===")
        round_started = time.perf_counter()
        round_rows: list[dict] = []
        for trainer, _save_name in trainers:
            client_started = time.perf_counter()
            train_metrics = trainer.train_round(round_idx, max_batches=args.max_batches)
            train_duration = time.perf_counter() - client_started
            val_started = time.perf_counter()
            loss_value = train_metrics.get("avg_server_loss", train_metrics.get("avg_local_loss", 0.0))
            val_metrics = trainer.validate(round_idx, max_batches=args.max_val_batches)
            val_duration = time.perf_counter() - val_started
            row = {
                "run_id": run_id,
                "mode": mode,
                "round": round_idx,
                "client_id": trainer.client_id,
                "train_batches": train_metrics["batches"],
                "train_loss": loss_value,
                "val_batches": val_metrics.get("val_batches", 0),
                "val_auroc": val_metrics.get("auroc"),
                "val_auprc": val_metrics.get("auprc"),
                "val_f1": val_metrics.get("f1"),
                "activation_bytes": train_metrics["activation_bytes"],
                "gradient_bytes": train_metrics["gradient_bytes"],
                "train_duration_sec": train_duration,
                "val_duration_sec": val_duration,
            }
            round_rows.append(row)
            summary_rows.append(row)
            with summary_jsonl.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(row) + "\n")
            print(
                f"Client {trainer.client_id}: loss={loss_value:.4f}, "
                f"batches={train_metrics['batches']}, "
                f"val_auroc={val_metrics.get('auroc', float('nan')):.4f}"
            )

        if fedavg_enabled:
            fedavg_started = time.perf_counter()
            statuses = [trainer.submit_encoder_for_fedavg() for trainer, _save_name in trainers]
            print(f"FedAvg statuses: {statuses}")
            for trainer, _save_name in trainers:
                trainer.load_global_encoder(max_wait_seconds=30.0)
            print("Loaded global encoder into all clients")
            fedavg_duration = time.perf_counter() - fedavg_started
            fedavg_row = {
                "run_id": run_id,
                "mode": mode,
                "round": round_idx,
                "event": "fedavg",
                "duration_sec": fedavg_duration,
                "statuses": statuses,
            }
            with summary_jsonl.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(fedavg_row) + "\n")

        round_duration = time.perf_counter() - round_started
        if round_rows:
            mean_auroc = sum(float(row["val_auroc"]) for row in round_rows if row["val_auroc"] is not None) / len(round_rows)
            print(f"Round {round_idx} mean AUROC={mean_auroc:.4f}, duration={round_duration:.1f}s")

    for trainer, save_name in trainers:
        saved_path = trainer.save_encoder(save_name)
        print(f"Saved client {trainer.client_id} encoder to {saved_path}")
    try:
        checkpoint = api.save_server_checkpoint(f"server_model_{mode}_{run_id}.pth")
        print(f"Saved server checkpoint: {checkpoint}")
    except requests.HTTPError as exc:
        if exc.response is not None and exc.response.status_code == 404:
            print("Server checkpoint endpoint not available; skipping server checkpoint save.")
        else:
            raise

    if summary_rows:
        with summary_csv.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"Saved experiment summary CSV to {summary_csv}")
        print(f"Saved experiment summary JSONL to {summary_jsonl}")


def main() -> None:
    run_all_clients(build_parser().parse_args())


if __name__ == "__main__":
    main()
