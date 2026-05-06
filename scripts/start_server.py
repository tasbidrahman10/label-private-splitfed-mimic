from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import uvicorn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sfl.common.config import load_yaml
from sfl.server.app import create_app


def main() -> None:
    parser = argparse.ArgumentParser(description="Start the SFL FastAPI server.")
    parser.add_argument("--config", default="configs/server.yaml")
    args = parser.parse_args()

    os.chdir(ROOT)
    config = load_yaml(args.config)
    app = create_app(args.config)
    uvicorn.run(
        app,
        host=str(config.get("host", "0.0.0.0")),
        port=int(config.get("port", 8000)),
    )


if __name__ == "__main__":
    main()

