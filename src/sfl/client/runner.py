from __future__ import annotations

import argparse
from pathlib import Path

import torch

from sfl.client.api import SFLServerClient
from sfl.client.dataset import load_client_dataloaders, load_encoder_weights_path
from sfl.client.trainer import OptionBClientTrainer
from sfl.common.config import load_yaml
from sfl.common.logging_utils import choose_device, set_seed
from sfl.common.models import ClientEncoder


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run an SFL client.")
    parser.add_argument("--client-config", required=True)
    parser.add_argument("--experiment-config", default="configs/experiment_label_private_splitfed.yaml")
    parser.add_argument("--server-url", default=None)
    parser.add_argument("--rounds", type=int, default=None)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument("--skip-fedavg", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--mode", choices=["option_b"], default=None)
    return parser


def run_client(args: argparse.Namespace) -> None:
    client_config = load_yaml(args.client_config)
    experiment_config = load_yaml(args.experiment_config)
    if experiment_config.get("results_dir"):
        client_config["results_dir"] = str(
            Path(experiment_config["results_dir"])
            / f"client{client_config['client_id']}_{client_config['client_name']}"
        )
    set_seed(int(client_config.get("seed", 42)))
    device = choose_device(args.device)

    train_loader, val_loader, pos_weight = load_client_dataloaders(client_config, device)

    encoder = ClientEncoder(
        input_dim=int(client_config.get("input_dim", 266)),
        hidden_dim=int(client_config.get("encoder_hidden_dim", 64)),
    ).to(device)
    encoder_path = load_encoder_weights_path(client_config)
    encoder.load_state_dict(torch.load(encoder_path, map_location=device))

    server_url = args.server_url or experiment_config["server_url"]
    api = SFLServerClient(server_url, auth_token=str(experiment_config.get("auth_token", "")))
    print(f"Server health: {api.health()}")

    client_id = int(client_config["client_id"])
    client_name = str(client_config["client_name"])
    api.register_client(
        client_id=client_id,
        client_name=client_name,
        train_size=len(train_loader.dataset),
        pos_weight=pos_weight,
    )
    print(f"Registered client {client_id} ({client_name}) with {len(train_loader.dataset):,} samples")

    mode = args.mode or str(experiment_config.get("mode", "option_b"))
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
        saved_filename = "encoder_option_b_latest.pth"
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    rounds = args.rounds or int(experiment_config.get("num_rounds", 1))
    fedavg_enabled = bool(experiment_config.get("fedavg_every_round", True)) and not args.skip_fedavg

    for round_idx in range(1, rounds + 1):
        metrics = trainer.train_round(round_idx, max_batches=args.max_batches)
        loss_value = metrics.get("avg_server_loss", metrics.get("avg_local_loss", 0.0))
        print(
            f"Round {round_idx}: loss={loss_value:.4f}, "
            f"batches={metrics['batches']}, "
            f"activation_bytes={metrics['activation_bytes']}"
        )
        val_metrics = trainer.validate(round_idx, max_batches=args.max_val_batches)
        if "auroc" in val_metrics:
            print(
                f"Round {round_idx}: val_auroc={val_metrics['auroc']:.4f}, "
                f"val_auprc={val_metrics['auprc']:.4f}, "
                f"val_f1={val_metrics['f1']:.4f}"
            )
        if fedavg_enabled:
            status = trainer.submit_encoder_for_fedavg()
            print(f"FedAvg submit: {status}")
            trainer.load_global_encoder()
            print("Loaded global encoder")

    saved_path = trainer.save_encoder(saved_filename)
    print(f"Saved updated encoder to {saved_path}")


def main() -> None:
    run_client(build_parser().parse_args())


if __name__ == "__main__":
    main()
