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
    # Clients hold pooled keep-alive connections idle across long gaps: the
    # async submission window (async_window_seconds, 25s by default) plus
    # per-client delay sleeps. Uvicorn's 5s default keep-alive would close those
    # connections underneath the client, which then reuses a dead socket and
    # fails with ConnectionReset/ConnectionAborted. Keep connections alive well
    # past the longest expected idle gap.
    keep_alive = max(120, int(float(config.get("async_window_seconds", 25.0)) * 4))
    uvicorn.run(
        app,
        host=str(config.get("host", "0.0.0.0")),
        port=int(config.get("port", 8000)),
        timeout_keep_alive=keep_alive,
    )


if __name__ == "__main__":
    main()

