from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sfl.client.multi_runner import build_parser, run_all_clients


def main() -> None:
    parser = build_parser()
    parser.set_defaults(experiment_config="configs/experiment_label_private_splitfed.yaml")
    run_all_clients(parser.parse_args())


if __name__ == "__main__":
    main()
