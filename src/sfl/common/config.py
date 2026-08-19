from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

# Directory holding the active federation's client configs. Every component that
# discovers clients by globbing (client runner, server FLTrust roster, FLTrust
# root builder) resolves through client_config_dir(), so a single environment
# variable switches the whole stack between federation profiles:
#
#     configs/clients      n=3, care-unit trio  (main paper results)
#     configs/clients_n6   n=6, one silo per ICU care unit (supplementary)
#
# Each profile's YAMLs carry their own data_path/model_dir, so switching the
# config directory switches the dataset and encoders with it.
CLIENT_CONFIG_DIR_ENV = "SFL_CLIENT_CONFIG_DIR"
DEFAULT_CLIENT_CONFIG_DIR = "configs/clients"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def client_config_dir() -> Path:
    """Active client-config directory, overridable via SFL_CLIENT_CONFIG_DIR."""
    return resolve_path(os.environ.get(CLIENT_CONFIG_DIR_ENV, DEFAULT_CLIENT_CONFIG_DIR))


def client_config_paths() -> list[Path]:
    """All client configs in the active profile, ordered by client_id."""
    paths = sorted(client_config_dir().glob("*.yaml"))
    if not paths:
        raise FileNotFoundError(
            f"No client configs found in {client_config_dir()}. "
            f"Set {CLIENT_CONFIG_DIR_ENV} to a directory containing client YAMLs."
        )
    return sorted(paths, key=lambda p: int(load_yaml(p)["client_id"]))


def profile_suffix() -> str:
    """Suffix identifying the active profile ('' for n=3, '_n6' for n=6).

    Derived from the client-config directory name, so profile-scoped artifacts
    stay paired with the federation that produced them.
    """
    name = client_config_dir().name
    return name[len("clients"):] if name.startswith("clients") else f"_{name}"


def fltrust_root_indices_path() -> Path:
    """Root-index file for the active profile.

    The indices point at row positions inside a specific profile's CSVs, so
    n=3 and n=6 must not share one file — reusing the wrong one silently
    slices the wrong patients into FLTrust's reference set.
    """
    return repo_root() / "configs" / f"fltrust_root_indices{profile_suffix()}.json"


def resolve_path(path: str | Path) -> Path:
    value = Path(path)
    if value.is_absolute():
        return value
    return repo_root() / value


def load_yaml(path: str | Path) -> dict[str, Any]:
    with resolve_path(path).open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping config in {path}")
    return data


def ensure_dir(path: str | Path) -> Path:
    target = resolve_path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target

